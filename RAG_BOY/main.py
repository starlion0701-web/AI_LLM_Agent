import os
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # CORS 모듈 추가
from pydantic import BaseModel

# LangChain 및 AI 모델 관련 임포트
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

app = FastAPI(title="Google Cloud Run RAG Server")

# ==============================================================================
# 0. CORS 미들웨어 설정 (프론트엔드 통신 허용)
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 구글 앱스 스크립트를 포함한 모든 도메인 허용
    allow_credentials=False,
    allow_methods=["*"],          # 모든 통신 방식(GET, POST, OPTIONS 등) 허용
    allow_headers=["*"],          # 모든 헤더 허용
)

# ==============================================================================
# 1. 환경 변수 로드 및 초기화
# ==============================================================================
# Cloud Run 환경 변수에서 값들을 가져옵니다.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not all([GOOGLE_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("서버 시작 실패: 필요한 환경 변수(API 키 및 Supabase URL)가 설정되지 않았습니다.")

# Supabase 클라이언트 및 OpenAI 임베딩 초기화
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
except Exception as e:
    raise RuntimeError(f"서버 초기화 중 외부 서비스 연결 실패. 상세 오류: {str(e)}")

# ==============================================================================
# 2. 데이터 모델 (Pydantic)
# ==============================================================================
class ChatRequest(BaseModel):
    query: str
    # 이전 대화 내역 (예: [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "반갑습니다"}])
    chat_history: List[Dict[str, str]]

# ==============================================================================
# 3. PDF 업로드 및 벡터 DB 저장 엔드포인트
# ==============================================================================
@app.post("/upload-pdf")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="업로드 에러: PDF 파일만 업로드 가능합니다.")
    
    temp_file_path = ""
    try:
        # 1. 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 2. PDF 텍스트 추출
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        if not documents:
            raise ValueError("문서 추출 실패: PDF 내에 읽을 수 있는 텍스트가 없습니다.")

        # 3. 텍스트 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # 4. Supabase 벡터 DB에 적재
        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        
        return {"message": f"성공: '{file.filename}' 파일이 {len(chunks)}개의 청크로 나뉘어 벡터 DB에 저장되었습니다."}

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"문서 처리 오류: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 (벡터 DB 저장 실패): {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ==============================================================================
# 4. RAG 및 실시간 스트리밍 대화 엔드포인트
# ==============================================================================
@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    try:
        # 1. LLM 초기화 (요청하신 gemini-3.5-flash-lite 모델)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2, # 명확하고 일관된 답변을 위해 낮게 설정
            streaming=True # 실시간 스트리밍 활성화
        )

        # 2. 대화 기록(Chat History)을 LangChain Message 객체로 변환
        formatted_history = []
        for msg in request.chat_history:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content")))

        # 3. Query Augmentation (질의 증강) 체인 구성
        # 시스템 메시지: 사용자의 의도 파악 및 질문 변환
        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            # 대화 기록 전달 공간 (동적 빈칸)
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        # 프롬프트 -> LLM -> 일반 텍스트 파서 연결
        query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()

      # 4. 질의 증강 체인 실행 (데이터 주입)
        try:
            # invoke -> ainvoke 로 변경하고 앞에 await 추가
            augmented_query = await query_augmentation_chain.ainvoke({
                "chat_history": formatted_history,
                "query": request.query
            })
        except Exception as aug_err:
            raise RuntimeError(f"질의 확장(Query Augmentation) 중 오류 발생: {str(aug_err)}")

        # 5. 증강된 질의로 벡터 DB 검색 (Retriever)
        try:
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            # similarity_search -> asimilarity_search 로 변경하고 앞에 await 추가
            retrieved_docs = await vector_store.asimilarity_search(augmented_query, k=4)
            # 검색된 문서들을 하나의 문자열로 결합
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 5. 증강된 질의로 벡터 DB 검색 (Retriever)
        try:
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            retrieved_docs = vector_store.similarity_search(augmented_query, k=4)
            # 검색된 문서들을 하나의 문자열로 결합
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as ret_err:
            raise RuntimeError(f"Supabase 문서 검색(Retrieval) 중 오류 발생: {str(ret_err)}")

        # 6. 최종 응답 생성용 프롬프트 설계 (도시 정책 전문가)
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n[검색된 문서 내용]\n{context}"),
            ("human", "{augmented_query}")
        ])

        rag_chain = rag_prompt | llm | StrOutputParser()

        # 7. 실시간 스트리밍 핸들러 (Generator 방식)
        # LangChain의 최신 astream()을 활용해 chunk 단위로 데이터를 클라이언트에 전송합니다.
        async def stream_generator():
            try:
                # 증강된 질문(augmented_query)과 검색된 문서(context) 주입
                async for chunk in rag_chain.astream({
                    "context": context_text,
                    "augmented_query": augmented_query
                }):
                    yield chunk
            except Exception as stream_err:
                # 스트리밍 도중 끊기거나 에러 발생 시 클라이언트에게 에러 내용 전달
                yield f"\n\n[스트리밍 오류 발생]: 답변을 생성하는 중 통신 문제가 발생했습니다. 상세 내역: {str(stream_err)}"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except RuntimeError as re:
        # 백엔드 로직 중 발생한 런타임 오류
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        # 예상치 못한 시스템 오류
        raise HTTPException(status_code=500, detail=f"서버 내부 알 수 없는 오류 발생: {str(e)}")
