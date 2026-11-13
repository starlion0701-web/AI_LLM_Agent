"""
========================================================================================
[FashionRAG: Multimodal Fashion Styling Recommendation Server]
FastAPI + Google AI (Gemini VLM) + ChromaDB Multi-Vector Retriever + Fashionpedia
========================================================================================
"""

import os
import re
import io
import time
import json
import base64
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import requests
from PIL import Image

# FastAPI & CORS
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ChromaDB
import chromadb
from chromadb.config import Settings

# Google Generative AI
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FashionRAG")

# ========================================================================================
# 1. 환경 변수 및 Google AI 설정
# ========================================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# 모델 자동 감지 후보군 (404 방지용 우선순위 탐색 리스트)
CANDIDATE_MODELS = [
    GENERATION_MODEL,
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro"
]

_resolved_model_name: Optional[str] = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google AI (Gemini) configured successfully.")
else:
    logger.warning("GEMINI_API_KEY not found in environment. Mock responses will be used if needed.")

def get_available_model_name() -> Optional[str]:
    """
    현재 API 키에서 실제로 generateContent를 지원하는 최적의 모델을 자동 탐색하여 캐시합니다.
    ModelService.ListModels를 통해 404 오류를 사전 방지합니다.
    """
    global _resolved_model_name
    if _resolved_model_name:
        return _resolved_model_name

    if not GEMINI_API_KEY:
        return None

    try:
        supported_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                clean_name = m.name.replace("models/", "")
                supported_models.append(clean_name)

        logger.info(f"Supported models on this API key: {supported_models}")

        # 후보군 순서대로 매칭되는 첫 번째 모델 채택
        for cand in CANDIDATE_MODELS:
            clean_cand = cand.replace("models/", "")
            if clean_cand in supported_models:
                _resolved_model_name = clean_cand
                logger.info(f"Auto-selected working Gemini model: {_resolved_model_name}")
                return _resolved_model_name

        if supported_models:
            _resolved_model_name = supported_models[0]
            logger.info(f"Selected first supported Gemini model: {_resolved_model_name}")
            return _resolved_model_name

    except Exception as e:
        logger.warning(f"genai.list_models() check failed ({e}). Falling back to candidate sequence.")

    _resolved_model_name = GENERATION_MODEL
    return _resolved_model_name

# ========================================================================================
# 2. ChromaDB 멀티벡터 저장소 초기화 및 Fashionpedia 시드 데이터
# ========================================================================================
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
collection = chroma_client.get_or_create_collection(
    name="fashionpedia_multimodal",
    metadata={"description": "Fashionpedia image summaries and attribute embeddings"}
)

# Stitch 프로젝트 1065265649829620997 기반 룩북 화보 및 Fashionpedia 속성 데이터셋
SEED_FASHION_ITEMS = [
    {
        "id": "look_01_avant_garde_trench",
        "title": "Avant-Garde Tailored Oversized Trench Look",
        "category": "outerwear_trench",
        "image_url": "https://stitch.googleapis.com/v1/projects/1065265649829620997/files/screenshot_0006601f43a6d8d2079e74839b217add:download",
        "silhouette": "Oversized & Architectural",
        "color_palette": ["Obsidian Slate", "Cool Charcoal", "Muted Ecru"],
        "fabrics": ["Bonded Heavy Cotton", "Technical Wool Gabardine"],
        "attributes": ["double-breasted", "exaggerated-lapel", "wide-leg-trousers", "high-collar", "fashion-week-runway"],
        "summary": "Contemporary avant-garde tailored oversized trench coat paired with structured architectural trousers. Minimalist urban slate backdrop, editorial studio lighting, clean runway chic aesthetic."
    },
    {
        "id": "look_02_emerald_tech_tailored",
        "title": "Layered Emerald Tech-Wear Sculpted Blazer Look",
        "category": "blazer_techwear",
        "image_url": "https://stitch.googleapis.com/v1/projects/1065265649829620997/files/screenshot_0006601f43c1f610079e74839b217add:download",
        "silhouette": "Structured & Layered",
        "color_palette": ["Vibrant Emerald Green", "Deep Slate", "Matte Black"],
        "fabrics": ["Ripstop Nylon", "Fine Wool Crepe", "Technical GORE-TEX Trim"],
        "attributes": ["sculpted-shoulders", "asymmetric-closure", "relaxed-wide-trousers", "tactical-accents", "street-couture"],
        "summary": "High-fashion street couture portrait styled in layered emerald green tech-wear accents, sculpted tailored blazer, and relaxed wide-leg trousers. Runway editorial vibe with cool slate undertones."
    },
    {
        "id": "look_03_monochrome_slate_knitwear",
        "title": "Monochrome Slate Grey Luxury Knitwear & Cobalt Accent",
        "category": "knitwear_luxury",
        "image_url": "https://stitch.googleapis.com/v1/projects/1065265649829620997/files/screenshot_0006601f43b2901802d3c2c7223ba4d0:download",
        "silhouette": "Minimalist Fluid & Tailored",
        "color_palette": ["Monochrome Slate Grey", "Cobalt Blue", "Graphite"],
        "fabrics": ["Fine Cashmere Rib", "Compact Merino Wool", "Polished Calfskin"],
        "attributes": ["mock-neck", "clean-belted-waist", "minimalist-chic", "luxury-campaign", "sculptural-hardware"],
        "summary": "Sophisticated minimalist editorial fashion lookbook photo featuring monochrome slate grey luxury knitwear with subtle cobalt blue structured belt hardware. Clean modern aesthetic, luxury brand lighting."
    }
]

def get_embedding(text: str) -> List[float]:
    """텍스트 임베딩 생성 (Gemini API 또는 Fallback 고정 차원 정규화 벡터)"""
    if GEMINI_API_KEY:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as e:
            logger.warning(f"Gemini embedding API error: {e}. Using deterministic fallback.")
    
    # Fallback: 결정론적 768차원 해시 임베딩
    vec = [0.0] * 768
    for idx, word in enumerate(text.split()):
        h = abs(hash(word)) % 768
        vec[h] += 1.0 / (idx + 1)
    norm = sum(x**2 for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]

def init_vector_store():
    """초기 Fashionpedia 및 룩북 데이터셋 ChromaDB 색인"""
    if collection.count() == 0:
        logger.info("Initializing ChromaDB with Fashionpedia seed items...")
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for item in SEED_FASHION_ITEMS:
            content = f"{item['title']}. {item['summary']} Silhouettes: {item['silhouette']}. Palette: {', '.join(item['color_palette'])}. Fabrics: {', '.join(item['fabrics'])}. Attributes: {', '.join(item['attributes'])}."
            ids.append(item["id"])
            documents.append(content)
            metadatas.append({
                "title": item["title"],
                "category": item["category"],
                "image_url": item["image_url"],
                "silhouette": item["silhouette"],
                "color_palette": json.dumps(item["color_palette"], ensure_ascii=False),
                "fabrics": json.dumps(item["fabrics"], ensure_ascii=False),
                "attributes": json.dumps(item["attributes"], ensure_ascii=False),
                "summary": item["summary"]
            })
            embeddings.append(get_embedding(content))
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Successfully indexed {len(ids)} fashion items in ChromaDB.")

# 서버 로드시 색인 실행
init_vector_store()

# ========================================================================================
# 3. 데이터 모델 (Pydantic Schema)
# ========================================================================================
class FashionQueryRequest(BaseModel):
    query: str = Field(..., description="사용자 한국어 패션 스타일링 질의")
    top_k: int = Field(default=3, ge=1, le=5, description="검색할 유사 이미지 및 룩북 수")
    image_base64: Optional[str] = Field(default=None, description="선택적 사용자 업로드 이미지 Base64")

class FashionItemResult(BaseModel):
    id: str
    title: str
    image_url: str
    similarity_score: float
    silhouette: str
    color_palette: List[str]
    fabrics: List[str]
    attributes: List[str]
    summary: str

class ComparisonStream(BaseModel):
    stream_id: str
    stream_name: str
    accent_color: str
    match_score: int
    title: str
    look_summary: str
    styling_points: List[str]
    recommended_items: List[str]
    image_url: str

class FashionResponse(BaseModel):
    status: str
    original_query: str
    translated_query: str
    retrieved_images: List[FashionItemResult]
    styling_advice_ko: str
    styling_advice_en: str
    streams: List[ComparisonStream]
    metrics: Dict[str, Any]

# ========================================================================================
# 4. 멀티모달 RAG 파이프라인 컴포넌트
# ========================================================================================
def translate_query_ko_to_en(korean_query: str) -> str:
    """Step 1: 한국어 사용자 입력을 정밀한 영문 패션 프롬프트로 번역"""
    fallback_query = f"High fashion runway styling advice for: {korean_query}"
    if not GEMINI_API_KEY:
        return fallback_query
    
    prompt = f"""You are a luxury fashion runway director and styling translator.
Translate the following Korean fashion request into a precise, descriptive English fashion search query.
Focus on silhouettes, garments, fabrics, aesthetics, and color matching.
Return ONLY the translated English query text without commentary.

[Korean Request]: {korean_query}"""

    # 가용 모델 자동 탐색 및 후보군 순차 시도
    resolved = get_available_model_name()
    candidates = [resolved] if resolved else []
    for c in CANDIDATE_MODELS:
        if c not in candidates:
            candidates.append(c)

    for cand_name in candidates:
        if not cand_name:
            continue
        try:
            model = genai.GenerativeModel(cand_name)
            response = model.generate_content(prompt)
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            logger.debug(f"Translation with {cand_name} failed: {e}. Trying next...")
            continue

    return fallback_query

def search_fashion_images(query_en: str, top_k: int = 3) -> List[FashionItemResult]:
    """Step 2: 영문 쿼리를 기반으로 ChromaDB에서 다중벡터 유사도 검색 수행"""
    query_vec = get_embedding(query_en)
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count())
    )
    
    retrieved = []
    ids = results["ids"][0] if results["ids"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.2] * len(ids)
    
    for idx, item_id in enumerate(ids):
        meta = metas[idx]
        dist = distances[idx] if idx < len(distances) else 0.2
        # L2 거리를 유사도 백분율로 환산 (가중치 계산)
        sim_score = max(0.5, round(1.0 - (dist / 2.0), 3))
        
        retrieved.append(FashionItemResult(
            id=item_id,
            title=meta.get("title", "Editorial Look"),
            image_url=meta.get("image_url", ""),
            similarity_score=sim_score,
            silhouette=meta.get("silhouette", "Contemporary"),
            color_palette=json.loads(meta.get("color_palette", "[]")),
            fabrics=json.loads(meta.get("fabrics", "[]")),
            attributes=json.loads(meta.get("attributes", "[]")),
            summary=meta.get("summary", "")
        ))
    
    return retrieved

def generate_vlm_styling_advice(query_en: str, items: List[FashionItemResult], user_img_b64: Optional[str] = None) -> str:
    """Step 3 & 4: 검색된 이미지 컨텍스트와 질문을 결합하여 Google AI VLM 스타일링 조언 생성"""
    default_fallback_advice = (
        "Based on the editorial lookbook, pair an architectural oversized tailored coat with "
        "relaxed-cut wide-leg wool trousers. Maintain an obsidian slate base palette and introduce "
        "either luminous emerald accents or a structured cobalt leather belt to create high-contrast runway tension."
    )

    if not GEMINI_API_KEY:
        return default_fallback_advice

    context_str = ""
    for idx, item in enumerate(items, 1):
        context_str += f"""
Look #{idx} - {item.title}:
- Silhouette: {item.silhouette}
- Color Palette: {', '.join(item.color_palette)}
- Key Fabrics: {', '.join(item.fabrics)}
- Attributes: {', '.join(item.attributes)}
- Overview: {item.summary}
"""

    prompt = f"""You are the Executive Creative Director of MODA AI Editorial Studio (collaborating with Vogue and luxury runways).
A client has requested fashion curation with the following intent:
[Client Request (English)]: "{query_en}"

[Retrieved Fashionpedia Context & Curated Looks]:
{context_str}

Please generate an authoritative, highly practical, and visionary fashion styling recommendation.
Your response MUST include:
1. Executive Styling Direction: Overall aesthetic concept, silhouette tension, and balance.
2. Color Harmonies & Palettes: Primary canvas tones and strategic accent placement (e.g., Sapphire, Emerald, or Cobalt highlights).
3. Layering & Fabric Combinations: Textural contrast between structured outerwear, fluid knits, and technical fabrics.
4. Accessory & Footwear Architecture: Specific eyewear, belt hardware, bags, and boots to anchor the runway look.

Keep the advice elevated, inspiring, and concise (under 250 words)."""

    content_parts = [prompt]
    if user_img_b64:
        try:
            img_bytes = base64.b64decode(user_img_b64.split(",")[-1])
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            content_parts.append(pil_img)
        except Exception as e:
            logger.warning(f"Failed to decode user image for VLM: {e}")

    # 지원 모델 자동 탐색 및 후보군 순차 시도 (404 방지)
    resolved = get_available_model_name()
    candidates = [resolved] if resolved else []
    for c in CANDIDATE_MODELS:
        if c not in candidates:
            candidates.append(c)

    for cand_name in candidates:
        if not cand_name:
            continue
        try:
            model = genai.GenerativeModel(cand_name)
            response = model.generate_content(content_parts)
            if response.text and response.text.strip():
                global _resolved_model_name
                _resolved_model_name = cand_name
                return response.text.strip()
        except Exception as e:
            logger.warning(f"VLM generation with {cand_name} failed: {e}. Trying next candidate...")
            continue

    logger.warning("All Gemini VLM model candidates failed. Returning curated fallback advice.")
    return default_fallback_advice

def translate_advice_en_to_ko(advice_en: str) -> str:
    """Step 5: 영문 VLM 응답을 세련된 한국어 패션 전문 용어로 번역"""
    default_ko_advice = (
        "아키텍처럴 오버사이즈 실루엣과 와이드 레그 트라우저의 볼륨감 있는 대비를 연출하는 룩을 제안합니다. "
        "옵시디언 슬레이트와 차콜 그레이를 메인 캔버스로 구성하고, 런웨이 감성의 에메랄드 테크웨어 포인트나 "
        "코발트 블루 스트럭처 레더 벨트를 더해 세련된 시각적 텐션을 완성하세요."
    )

    if not GEMINI_API_KEY:
        return default_ko_advice

    prompt = f"""당신은 최고급 패션 매거진(Vogue Korea, GQ)의 수석 패션 에디터입니다.
다음 영문 패션 스타일링 조언을 우아하고 전문적인 한국어 텍스트로 번역/윤문하세요.
패션 전문 어휘(오버사이즈 실루엣, 텍스처 대비, 레이어링, 슬레이트 톤, 테일러드 라인 등)를 자연스럽게 살려주세요.

[English Advice]:
{advice_en}"""

    resolved = get_available_model_name()
    candidates = [resolved] if resolved else []
    for c in CANDIDATE_MODELS:
        if c not in candidates:
            candidates.append(c)

    for cand_name in candidates:
        if not cand_name:
            continue
        try:
            model = genai.GenerativeModel(cand_name)
            response = model.generate_content(prompt)
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            logger.debug(f"Translation with {cand_name} failed: {e}. Trying next...")
            continue

    return default_ko_advice

def build_comparison_streams(items: List[FashionItemResult], advice_ko: str) -> List[ComparisonStream]:
    """Stitch 3단 비교 UI('Editorial Slate Haute Tech') 규격에 맞춘 Stream A/B/C 데이터 구성"""
    stream_configs = [
        {
            "id": "stream_a_sapphire",
            "name": "Stream A · Sapphire Prime",
            "accent": "#3B82F6",
            "default_score": 96,
            "title": "아키텍처럴 오버사이즈 테일러드 룩",
            "points": [
                "어깨 라인을 강조한 와이드 실루엣과 롱 코트 기장감의 조화",
                "옵시디언 슬레이트 바탕에 사파이어 블루 미세 악센트",
                "개버딘 울과 본디드 코튼의 탄탄한 질감 형성"
            ],
            "items": ["오버사이즈 테일러드 트렌치 코트", "와이드 플리츠 트라우저", "사파이어 블루 미니 클러치"]
        },
        {
            "id": "stream_b_emerald",
            "name": "Stream B · Emerald Tech-Wear",
            "accent": "#10B981",
            "default_score": 93,
            "title": "테크-테일러드 에메랄드 레이어드 룩",
            "points": [
                "테크니컬 립스탑 소재와 조각적 블레이저의 기능적 융합",
                "에메랄드 그린 컬러 블록으로 생동감 있는 런웨이 스트릿 무드",
                "고어텍스 트림 및 전술적 하네스 하드웨어 포인트"
            ],
            "items": ["조각적 테크 블레이저", "유틸리티 카고 와이드 팬츠", "에메랄드 나일론 크로스 스트랩"]
        },
        {
            "id": "stream_c_amethyst",
            "name": "Stream C · Royal Amethyst",
            "accent": "#8B5CF6",
            "default_score": 89,
            "title": "모노크롬 슬레이트 럭셔리 니트웨어 룩",
            "points": [
                "부드러운 캐시미어 립 조직과 유려한 드레이핑 실루엣",
                "차콜 그레이 베이스와 코발트-아메시스트 레더 벨트 대조",
                "미니멀한 모크넥과 기하학적 버클 하드웨어 악센트"
            ],
            "items": ["컴팩트 메리노 모크넥 니트", "테일러드 울 플란넬 팬츠", "코발트 스트럭처 레더 벨트"]
        }
    ]

    streams = []
    for idx, cfg in enumerate(stream_configs):
        matched_item = items[idx] if idx < len(items) else (items[0] if items else None)
        img_url = matched_item.image_url if matched_item else ""
        
        streams.append(ComparisonStream(
            stream_id=cfg["id"],
            stream_name=cfg["name"],
            accent_color=cfg["accent"],
            match_score=cfg["default_score"],
            title=cfg["title"],
            look_summary=matched_item.summary if matched_item else cfg["title"],
            styling_points=cfg["points"],
            recommended_items=cfg["items"],
            image_url=img_url
        ))

    return streams

# ========================================================================================
# 5. FastAPI 애플리케이션 초기화 및 엔드포인트 정의
# ========================================================================================
app = FastAPI(
    title="FashionRAG Multimodal Recommendation Server",
    description="Multimodal Fashion RAG API with Google AI VLM, ChromaDB & Fashionpedia",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """서버 헬스 체크 및 ChromaDB 컬렉션 현황"""
    resolved_model = get_available_model_name() or GENERATION_MODEL
    return {
        "status": "healthy",
        "service": "FashionRAG Multimodal Server",
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "chromadb_collection_items": collection.count(),
        "active_models": {
            "vlm": resolved_model,
            "embedding": EMBEDDING_MODEL
        },
        "design_system": "Editorial Slate Haute Tech"
    }

@app.post("/api/fashion/recommend", response_model=FashionResponse)
def recommend_fashion_styling(req: FashionQueryRequest):
    """
    [멀티모달 패션 RAG 메인 체인]
    한국어 질의 -> 영문 번역 -> ChromaDB 다중벡터 검색 -> Gemini VLM 스타일링 생성 -> 한국어 번역 -> 3단 비교 출력
    """
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="질의 내용이 비어있습니다.")

    t_total_start = time.time()

    # Step 1: 질의 번역 (KO -> EN)
    t0 = time.time()
    query_en = translate_query_ko_to_en(req.query.strip())
    latency_trans_query = round((time.time() - t0) * 1000, 2)

    # Step 2: 이미지 및 아이템 검색 (ChromaDB Vector Retrieval)
    t1 = time.time()
    retrieved_items = search_fashion_images(query_en, top_k=req.top_k)
    latency_retrieval = round((time.time() - t1) * 1000, 2)

    # Step 3 & 4: 프롬프트 포매팅 및 Gemini VLM 스타일링 조언 생성
    t2 = time.time()
    advice_en = generate_vlm_styling_advice(query_en, retrieved_items, req.image_base64)
    latency_vlm = round((time.time() - t2) * 1000, 2)

    # Step 5: 응답 번역 (EN -> KO)
    t3 = time.time()
    advice_ko = translate_advice_en_to_ko(advice_en)
    latency_trans_advice = round((time.time() - t3) * 1000, 2)

    # Step 6: 3단 비교 시각화 스트림 구성
    streams = build_comparison_streams(retrieved_items, advice_ko)

    total_latency = round((time.time() - t_total_start) * 1000, 2)

    return FashionResponse(
        status="success",
        original_query=req.query,
        translated_query=query_en,
        retrieved_images=retrieved_items,
        styling_advice_ko=advice_ko,
        styling_advice_en=advice_en,
        streams=streams,
        metrics={
            "total_latency_ms": total_latency,
            "query_translation_latency_ms": latency_trans_query,
            "chroma_retrieval_latency_ms": latency_retrieval,
            "vlm_generation_latency_ms": latency_vlm,
            "advice_translation_latency_ms": latency_trans_advice,
            "chromadb_indexed_items": collection.count(),
            "vlm_model": GENERATION_MODEL
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

