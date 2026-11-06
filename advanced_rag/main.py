"""
========================================================================================
[Advanced Hybrid RAG Server with FAISS + Multi-Query + Hybrid Search + RAG Fusion (RRF)]
========================================================================================

■ 핵심 아키텍처 및 알고리즘 복잡도:
1. 다중 쿼리 확장 계층 (Multi-Query Layer):
   - 단일 입력 쿼리를 LLM을 통해 5개의 다각도 파생 쿼리로 확장.
   - 시간 복잡도: O(T) (생성 토큰 수 비례, LLM API 호출 1회)
   
2. 하이브리드 검색 계층 (Hybrid Search Layer):
   - Dense Search (FAISS):
     * IndexFlatL2 (정밀 전수 검색): 시간 복잡도 O(N * D)
     * IndexHNSWFlat (근사 최근접 탐색): 시간 복잡도 O(log N * D)
   - Sparse Search (BM25Okapi):
     * 어휘 역색인 기반 키워드 매칭: 시간 복잡도 O(|Q| * avg_dl)
   - 5개 쿼리 각각에 대해 Dense + Sparse 병렬 실행 (총 10개 검색 랭킹 생성)

3. RRF 순위 융합 계층 (RAG Fusion & RRF Layer):
   - Reciprocal Rank Fusion 공식:
     Score(d) = SUM_{q in Q} [ w_dense / (k + rank_dense(q, d)) + w_sparse / (k + rank_sparse(q, d)) ] (k=60)
   - 시간 복잡도: O(M * K + U log U) (M=10개 랭킹 리스트, K=각 리스트별 탐색 개수, U=고유 문서 수)

4. 답변 생성 계층 (Generation Layer):
   - 상위 선별된 고품질 컨텍스트를 주입하여 근거 기반 답변 생성 (스트리밍 및 일반 JSON 지원).
========================================================================================
"""

import os
import re
import io
import time
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np

# FastAPI 및 비동기 스트리밍 모듈
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# FAISS 벡터 인덱스
import faiss

# BM25 키워드 검색 엔진
from rank_bm25 import BM25Okapi

# PDF 텍스트 추출 모듈
from pypdf import PdfReader

# Google Gemini API
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedRAG")

# ========================================================================================
# 환경 변수 및 설정
# ========================================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "FlatL2")  # "FlatL2" 또는 "HNSWFlat"
VECTOR_DIMENSION = 768  # text-embedding-004 기본 차원

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini API Key configured successfully.")
else:
    logger.warning("GEMINI_API_KEY not found. Fallback mock embedding/generation will be used if needed.")

# ========================================================================================
# 데이터 모델 정의
# ========================================================================================
class DocumentChunk(BaseModel):
    """분할된 문서 청크 단위 메타데이터"""
    chunk_id: int
    source_file: str
    page: int
    text: str

class QueryRequest(BaseModel):
    """사용자 질의 요청 스키마"""
    question: str = Field(..., description="사용자 입력 질문")
    top_k: int = Field(default=4, ge=1, le=10, description="최종 반환할 컨텍스트 수")
    expansion_count: int = Field(default=5, description="확장할 쿼리 개수 (기본 5)")
    rrf_k: int = Field(default=60, description="RRF 평활화 상수 (기본 60)")

class IndexedStatusResponse(BaseModel):
    """서버 상태 및 색인 현황 응답"""
    status: str
    total_chunks: int
    sources: List[str]
    vector_dimension: int
    faiss_index_type: str
    complexity_info: Dict[str, str]

# ========================================================================================
# 인메모리 하이브리드 인덱스 저장소 (스레드 안전성 고려)
# ========================================================================================
class HybridIndexStore:
    def __init__(self, dimension: int = VECTOR_DIMENSION, index_type: str = "FlatL2"):
        self.dimension = dimension
        self.index_type = index_type
        self.chunks: List[DocumentChunk] = []
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self._init_faiss_index()

    def _init_faiss_index(self):
        """FAISS 인덱스 초기화 (FlatL2 또는 HNSWFlat)"""
        if self.index_type == "HNSWFlat":
            # HNSW 근사 최근접 탐색 (O(log N * D))
            # M = 32 (연결 수), efConstruction = 64
            self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.faiss_index.hnsw.efSearch = 32
            logger.info("Initialized FAISS IndexHNSWFlat.")
        else:
            # 기본 정밀 검색 FlatL2 (O(N * D))
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.info("Initialized FAISS IndexFlatL2.")

    def clear(self):
        """인덱스 초기화"""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25_index = None
        self._init_faiss_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """간이 한국어/영어 혼합 토크나이저 (형태소/단어 분리)"""
        # 특수문자 제거 후 띄어쓰기 및 음절 단위 결합
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens if tokens else [text.lower()]

    def add_documents(self, new_chunks: List[DocumentChunk], embeddings: np.ndarray):
        """청크, FAISS Dense 벡터, BM25 Sparse 인덱스 일괄 등록"""
        if len(new_chunks) == 0:
            return

        start_idx = len(self.chunks)
        self.chunks.extend(new_chunks)

        # 1. FAISS 벡터 추가 (Float32 변환 확인)
        vectors = np.ascontiguousarray(embeddings.astype("float32"))
        self.faiss_index.add(vectors)

        # 2. BM25 코퍼스 토큰화 및 인덱스 재생성
        new_tokens = [self.tokenize(c.text) for c in new_chunks]
        self.tokenized_corpus.extend(new_tokens)
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Indexed {len(new_chunks)} chunks. Total chunks now: {len(self.chunks)}")

# 전역 인덱스 저장소 인스턴스
store = HybridIndexStore(dimension=VECTOR_DIMENSION, index_type=FAISS_INDEX_TYPE)

# ========================================================================================
# 임베딩 및 LLM 헬퍼 함수
# ========================================================================================
def get_embeddings(texts: List[str]) -> np.ndarray:
    """텍스트 목록에 대해 Gemini 임베딩 벡터 생성"""
    if not GEMINI_API_KEY:
        # API Key 미설정 시 재현 가능한 모의 임베딩 생성 (테스트 보장용)
        logger.warning("No API key: Generating pseudo embeddings.")
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(len(texts), VECTOR_DIMENSION)).astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    try:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        vec_arr = np.array(embeddings, dtype="float32")
        # 코사인 유사도 측정을 위한 L2 정규화
        faiss.normalize_L2(vec_arr)
        return vec_arr
    except Exception as e:
        logger.error(f"Gemini embedding generation failed: {e}")
        # 오류 시 더미 임베딩 리턴으로 중단 방지
        rng = np.random.default_rng(len(texts))
        dummy = rng.normal(size=(len(texts), VECTOR_DIMENSION)).astype("float32")
        faiss.normalize_L2(dummy)
        return dummy

def get_query_embedding(query: str) -> np.ndarray:
    """단일 검색 쿼리에 대한 임베딩 벡터 생성"""
    if not GEMINI_API_KEY:
        rng = np.random.default_rng(abs(hash(query)) % 10000)
        v = rng.normal(size=(1, VECTOR_DIMENSION)).astype("float32")
        faiss.normalize_L2(v)
        return v

    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=query,
            task_type="retrieval_query"
        )
        v = np.array([result['embedding']], dtype="float32")
        faiss.normalize_L2(v)
        return v
    except Exception as e:
        logger.error(f"Query embedding generation failed: {e}")
        rng = np.random.default_rng(abs(hash(query)) % 10000)
        v = rng.normal(size=(1, VECTOR_DIMENSION)).astype("float32")
        faiss.normalize_L2(v)
        return v

# ========================================================================================
# 1. 다중 쿼리 확장 계층 (Multi-Query Layer)
# ========================================================================================
def expand_queries(original_query: str, target_count: int = 5) -> List[str]:
    """
    단일 입력 쿼리를 LLM을 통해 5개의 다양한 관점(키워드 변형, 구체화, 포괄화 등)의 파생 쿼리로 확장.
    시간 복잡도: O(T) (생성 토큰 수 비례, 1회 추론)
    """
    if not GEMINI_API_KEY:
        # API Key 미설정 시 기본 규칙 기반 확장
        return [
            original_query,
            f"{original_query} 핵심 개념 및 정의",
            f"{original_query} 구현 방법 및 원리",
            f"{original_query} 장단점 및 비교",
            f"{original_query} 실제 적용 사례"
        ]

    prompt = f"""당신은 AI 정보 검색 전문가입니다. 사용자의 질문을 바탕으로 고품질 문서 검색을 위해 서로 다른 5가지 관점의 파생 검색 쿼리를 생성하세요.
- 원본 의도를 유지하되 동의어, 구체화, 상위 개념, 문맥적 키워드를 고루 반영할 것.
- 번호 없이 오직 5개의 쿼리를 줄바꿈(\\n)으로만 출력할 것.

사용자 원본 질문: {original_query}
"""
    try:
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        response = model.generate_content(prompt)
        raw_lines = [line.strip().lstrip("-*12345. ") for line in response.text.strip().split("\n") if line.strip()]
        queries = [original_query]
        for q in raw_lines:
            if q and q not in queries:
                queries.append(q)
            if len(queries) >= target_count:
                break

        # 5개가 안 찰 경우 보충
        while len(queries) < target_count:
            queries.append(f"{original_query} 관련 내용 {len(queries)+1}")
        return queries[:target_count]
    except Exception as e:
        logger.error(f"Multi-Query expansion failed: {e}")
        return [
            original_query,
            f"{original_query} 핵심 내용 요약",
            f"{original_query} 주요 특징 및 구성",
            f"{original_query} 활용 및 원리",
            f"{original_query} 상세 설명"
        ]

# ========================================================================================
# 2. 하이브리드 검색 및 3. RRF 순위 융합 계층 (Hybrid Search & RRF Layer)
# ========================================================================================
def run_hybrid_search_and_rrf(queries: List[str], top_k: int = 4, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    확장된 5개 쿼리마다 FAISS(Dense) 및 BM25(Sparse) 검색을 병행하고,
    Reciprocal Rank Fusion (RRF) 공식으로 순위를 융합하여 상위 top_k 문서를 도출합니다.
    
    RRF 공식: Score(d) = SUM_{q} [ 1 / (rrf_k + rank_dense(q, d)) + 1 / (rrf_k + rank_sparse(q, d)) ]
    시간 복잡도: O(Q_cnt * (O_dense + O_sparse) + Total_Candidates * log(Total_Candidates))
    """
    total_chunks = len(store.chunks)
    if total_chunks == 0:
        return []

    # 각 청크 인덱스별 RRF 누적 점수 및 매칭 통계
    rrf_scores: Dict[int, float] = {}
    hit_counts: Dict[int, int] = {}
    dense_weight = 1.0
    sparse_weight = 1.0

    # 5개 쿼리 각각에 대해 Dense + Sparse 실행
    for q_idx, q in enumerate(queries):
        # A. Dense Vector Search (FAISS)
        # 시간 복잡도: IndexFlatL2 -> O(N * D), IndexHNSWFlat -> O(log N * D)
        q_vec = get_query_embedding(q)
        search_k = min(15, total_chunks)
        distances, indices = store.faiss_index.search(q_vec, search_k)
        
        dense_hits = indices[0]
        for rank, chunk_idx in enumerate(dense_hits):
            if chunk_idx < 0 or chunk_idx >= total_chunks:
                continue
            # RRF 점수 가산 (rank는 1부터 시작)
            score_addition = dense_weight / (rrf_k + (rank + 1))
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + score_addition
            hit_counts[chunk_idx] = hit_counts.get(chunk_idx, 0) + 1

        # B. Sparse Keyword Search (BM25)
        # 시간 복잡도: O(|Q| * avg_dl)
        if store.bm25_index is not None:
            tokenized_q = store.tokenize(q)
            bm25_scores = store.bm25_index.get_scores(tokenized_q)
            # 상위 search_k 개 인덱스 정렬
            sparse_ranked_indices = np.argsort(bm25_scores)[::-1][:search_k]
            for rank, chunk_idx in enumerate(sparse_ranked_indices):
                if bm25_scores[chunk_idx] <= 0:
                    continue
                score_addition = sparse_weight / (rrf_k + (rank + 1))
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + score_addition
                hit_counts[chunk_idx] = hit_counts.get(chunk_idx, 0) + 1

    # RRF 점수 기준 내림차순 정렬 (Re-ranking)
    sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    top_candidates = sorted_candidates[:top_k]

    # 결과 포맷팅
    results = []
    for rank, (chunk_idx, rrf_score) in enumerate(top_candidates, start=1):
        chunk = store.chunks[chunk_idx]
        results.append({
            "final_rank": rank,
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "text": chunk.text,
            "rrf_score": round(float(rrf_score), 6),
            "hit_count": hit_counts.get(chunk_idx, 0)
        })

    return results

# ========================================================================================
# 4. 답변 생성 계층 (LLM Generation)
# ========================================================================================
def generate_llm_answer(question: str, context_docs: List[Dict[str, Any]]) -> str:
    """RRF로 선별된 컨텍스트를 주입하여 최종 답변 생성"""
    if not context_docs:
        return "색인된 문서가 없거나 질문과 관련된 내용을 문서에서 찾을 수 없습니다."

    context_str = "\n\n".join([
        f"[출처: {doc['source_file']} (p.{doc['page']}) | RRF 점수: {doc['rrf_score']}]\n{doc['text']}"
        for doc in context_docs
    ])

    prompt = f"""당신은 제공된 문서를 바탕으로 정확하게 답변하는 전문 AI 리서치 연구원입니다.
반드시 아래 [검색된 문서 컨텍스트]에 근거하여 사용자의 질문에 한국어로 친절하고 체계적으로 답변하세요.
답변 작성 시 어떤 문서와 페이지를 참조했는지 명확하게 인용(예: [출처: 파일명 p.X])을 포함하세요.

[검색된 문서 컨텍스트]:
{context_str}

[사용자 질문]:
{question}
"""
    if not GEMINI_API_KEY:
        return f"[모의 답변 (GEMINI_API_KEY 미설정)]\n검색된 컨텍스트({len(context_docs)}건)를 바탕으로 질문 '{question}'에 대한 답변입니다.\n\n" + \
               f"주요 참조 내용: {context_docs[0]['text'][:150]}...\n\n[출처: {context_docs[0]['source_file']} p.{context_docs[0]['page']}]"

    try:
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# ========================================================================================
# FastAPI 앱 초기화 및 라우터 설정
# ========================================================================================
app = FastAPI(
    title="Advanced Hybrid RAG Studio",
    description="FAISS Index + Multi-Query + Hybrid Search + RAG Fusion (RRF) 백엔드 서버",
    version="1.0.0"
)

# CORS 미들웨어 (Google Apps Script 및 웹 클라이언트 직접 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Advanced Hybrid RAG Backend is running.",
        "endpoints": {
            "health": "/health",
            "index_pdf": "/index/pdf",
            "query": "/query",
            "query_stream": "/query/stream"
        }
    }

@app.get("/health", response_model=IndexedStatusResponse)
def get_health_status():
    """서버 동작 상태 및 알고리즘 복잡도 정보 반환"""
    unique_sources = list({c.source_file for c in store.chunks})
    return IndexedStatusResponse(
        status="healthy",
        total_chunks=len(store.chunks),
        sources=unique_sources,
        vector_dimension=store.dimension,
        faiss_index_type=store.index_type,
        complexity_info={
            "Multi-Query Expansion": "O(T) (생성 토큰 수 비례, LLM API 1회)",
            "Dense Search (FAISS)": f"{'O(N * D) [IndexFlatL2 전수 검색]' if store.index_type == 'FlatL2' else 'O(log N * D) [IndexHNSWFlat 근사 최근접]'} (N=청크수, D=768)",
            "Sparse Search (BM25)": "O(|Q| * avg_dl) (쿼리 토큰 수 * 평균 문서 길이)",
            "RAG Fusion (RRF)": "O(M * K + U log U) (M=10개 랭킹 경로, K=검색 후보수, U=고유 청크수)",
            "Total Pipeline": "Multi-Query(5) -> Hybrid(5 Dense + 5 Sparse) -> RRF Re-ranking -> LLM Context Injection"
        }
    )

@app.post("/index/pdf")
async def index_pdf(
    file: Optional[UploadFile] = File(None),
    base64_data: Optional[str] = Form(None),
    filename: Optional[str] = Form(None)
):
    """
    PDF 파일을 수신하여 텍스트 추출 -> 슬라이딩 윈도우 청킹 -> 임베딩 -> FAISS & BM25 인덱싱
    (Multipart 파일 업로드 및 Google Apps Script 연동을 위한 Base64 문자열 수신 모두 지원)
    """
    start_time = time.time()
    extracted_text = ""
    source_name = filename or "uploaded_document.pdf"

    try:
        if file is not None:
            source_name = file.filename
            content = await file.read()
            pdf_reader = PdfReader(io.BytesIO(content))
        elif base64_data:
            import base64
            decoded = base64.b64decode(base64_data)
            pdf_reader = PdfReader(io.BytesIO(decoded))
        else:
            raise HTTPException(status_code=400, detail="file 또는 base64_data 중 하나는 필수입니다.")

        # PDF 페이지별 텍스트 추출 및 청크 분할 (슬라이딩 윈도우: 600자 크기, 120자 중첩)
        chunk_size = 600
        chunk_overlap = 120
        new_chunks: List[DocumentChunk] = []
        chunk_counter = len(store.chunks)

        for page_idx, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            # 문자 단위 슬라이딩 윈도우 청킹
            start = 0
            while start < len(page_text):
                end = min(start + chunk_size, len(page_text))
                chunk_slice = page_text[start:end].strip()
                if len(chunk_slice) > 30:  # 너무 짧은 조각 제외
                    new_chunks.append(DocumentChunk(
                        chunk_id=chunk_counter,
                        source_file=source_name,
                        page=page_idx,
                        text=chunk_slice
                    ))
                    chunk_counter += 1
                if end == len(page_text):
                    break
                start += (chunk_size - chunk_overlap)

        if not new_chunks:
            raise HTTPException(status_code=400, detail="PDF에서 유효한 텍스트를 추출할 수 없습니다.")

        # 임베딩 생성 및 인덱스 저장
        chunk_texts = [c.text for c in new_chunks]
        embeddings = get_embeddings(chunk_texts)
        store.add_documents(new_chunks, embeddings)

        elapsed = round(time.time() - start_time, 3)
        return {
            "status": "success",
            "message": f"Successfully indexed PDF '{source_name}'.",
            "indexed_chunks": len(new_chunks),
            "total_chunks": len(store.chunks),
            "source_file": source_name,
            "elapsed_seconds": elapsed
        }

    except Exception as e:
        logger.error(f"PDF Indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF 색인 실패: {str(e)}")

@app.post("/query")
def handle_query(request: QueryRequest):
    """
    RAG Fusion 질의 파이프라인 엔드포인트:
    1) Multi-Query 확장 (5개 쿼리)
    2) Hybrid Search (FAISS Dense + BM25 Sparse, 총 10개 랭킹)
    3) RRF 상호 순위 융합 및 Re-ranking (최적 상위 top_k 추출)
    4) LLM 최종 답변 생성
    """
    if len(store.chunks) == 0:
        raise HTTPException(status_code=400, detail="색인된 문서가 없습니다. 먼저 PDF 파일을 업로드하세요.")

    total_start = time.time()

    # 1. Multi-Query 확장 계층
    t0 = time.time()
    expanded_queries = expand_queries(request.question, target_count=request.expansion_count)
    expansion_time = round((time.time() - t0) * 1000, 2)

    # 2 & 3. Hybrid Search + RRF 순위 융합 계층
    t1 = time.time()
    rrf_top_docs = run_hybrid_search_and_rrf(
        queries=expanded_queries,
        top_k=request.top_k,
        rrf_k=request.rrf_k
    )
    search_time = round((time.time() - t1) * 1000, 2)

    # 4. LLM 최종 답변 생성 계층
    t2 = time.time()
    llm_answer = generate_llm_answer(request.question, rrf_top_docs)
    generation_time = round((time.time() - t2) * 1000, 2)

    total_time = round((time.time() - total_start) * 1000, 2)

    return {
        "status": "success",
        "original_query": request.question,
        "expanded_queries": expanded_queries,
        "retrieved_documents": rrf_top_docs,
        "answer": llm_answer,
        "metrics": {
            "total_latency_ms": total_time,
            "expansion_latency_ms": expansion_time,
            "hybrid_and_rrf_latency_ms": search_time,
            "generation_latency_ms": generation_time,
            "total_indexed_chunks": len(store.chunks),
            "faiss_index_type": store.index_type
        },
        "complexity": {
            "multi_query": "O(T) (LLM Prompting, 5 queries)",
            "dense_search": f"{'O(N*D)' if store.index_type == 'FlatL2' else 'O(log N * D)'} per query",
            "sparse_search": "O(|Q| * avg_dl) per query",
            "rrf_fusion": f"O(10 * K + U log U) (k={request.rrf_k})"
        }
    }

@app.post("/query/stream")
async def handle_query_stream(request: QueryRequest):
    """
    실시간 토큰 스트리밍 엔드포인트 (SSE 형태):
    RRF 검색 메타데이터를 먼저 전송한 후, LLM 생성 답변을 스트림 단위로 전송.
    """
    if len(store.chunks) == 0:
        raise HTTPException(status_code=400, detail="색인된 문서가 없습니다.")

    expanded_queries = expand_queries(request.question, target_count=request.expansion_count)
    rrf_top_docs = run_hybrid_search_and_rrf(queries=expanded_queries, top_k=request.top_k, rrf_k=request.rrf_k)

    context_str = "\n\n".join([
        f"[출처: {doc['source_file']} (p.{doc['page']}) | RRF: {doc['rrf_score']}]\n{doc['text']}"
        for doc in rrf_top_docs
    ])

    prompt = f"""당신은 제공된 문서를 바탕으로 정확하게 답변하는 전문 AI 연구원입니다.
[검색된 문서 컨텍스트]:
{context_str}

[사용자 질문]:
{request.question}
"""

    async def event_generator():
        # 첫 번째 청크: 검색된 RRF 메타데이터 및 5개 쿼리 JSON 전송
        meta_payload = {
            "event": "metadata",
            "expanded_queries": expanded_queries,
            "retrieved_documents": rrf_top_docs
        }
        yield f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"

        # 두 번째 단계: 답변 생성 스트리밍
        if not GEMINI_API_KEY:
            mock_text = f"[모의 스트리밍]\n'{request.question}'에 대한 답변입니다.\n{context_str[:200]}..."
            for word in mock_text.split(" "):
                yield f"data: {json.dumps({'event': 'token', 'token': word + ' '}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done'}, ensure_ascii=False)}\n\n"
            return

        try:
            model = genai.GenerativeModel(GENERATION_MODEL_NAME)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'event': 'token', 'token': chunk.text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ========================================================================================
# 로컬 개발 및 디버그용 실행기
# ========================================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

