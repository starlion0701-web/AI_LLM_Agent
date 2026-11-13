"""
========================================================================================
[FashionRAG: Multimodal Fashion Styling Recommendation Server]
FastAPI + Google AI (Gemini) + ChromaDB + Fashionpedia 스타일 룩북
========================================================================================
"""

import os
import io
import time
import json
import base64
import hashlib
import logging
from typing import List, Dict, Any, Optional

from PIL import Image
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from chromadb.config import Settings

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FashionRAG")

# ========================================================================================
# 1. 환경 변수 및 Google AI 설정
#    (키는 Cloud Run 환경 변수 또는 Secret Manager에만 넣고, 코드/깃에는 넣지 않습니다)
# ========================================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
EMBED_DIM = 768

# 사용 가능한 모델을 자동으로 찾기 위한 후보군 (앞쪽일수록 우선)
CANDIDATE_MODELS = list(dict.fromkeys([
    GENERATION_MODEL,
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
]))

_resolved_model_name: Optional[str] = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google AI (Gemini) 키가 설정되었습니다.")
else:
    logger.warning(
        "GEMINI_API_KEY(또는 GOOGLE_API_KEY) 환경 변수가 없습니다. "
        "AI 호출 없이 기본 문구로 응답합니다."
    )


def get_available_model_name() -> Optional[str]:
    """이 API 키에서 실제로 generateContent를 지원하는 모델을 찾아 캐시합니다."""
    global _resolved_model_name
    if _resolved_model_name:
        return _resolved_model_name
    if not GEMINI_API_KEY:
        return None

    try:
        supported = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                supported.append(m.name.replace("models/", ""))
        logger.info(f"사용 가능한 모델: {supported}")

        for cand in CANDIDATE_MODELS:
            if cand in supported:
                _resolved_model_name = cand
                logger.info(f"선택된 모델: {cand}")
                return cand

        # 후보에 없으면 flash 계열(이미지/음성 전용 제외)을 우선 선택
        flash = [m for m in supported if "flash" in m and "image" not in m and "tts" not in m]
        pick = (flash or supported or [None])[0]
        if pick:
            _resolved_model_name = pick
            logger.info(f"후보 외 모델 선택: {pick}")
            return pick
    except Exception as e:
        logger.warning(f"모델 목록 조회 실패({e}). 후보를 순서대로 시도합니다.")

    return None


def generate_text(content: Any) -> Optional[str]:
    """후보 모델을 순서대로 시도해서 첫 성공 응답을 반환합니다. 모두 실패하면 None."""
    global _resolved_model_name
    if not GEMINI_API_KEY:
        return None

    resolved = get_available_model_name()
    candidates = ([resolved] if resolved else []) + [c for c in CANDIDATE_MODELS if c != resolved]

    for name in candidates:
        try:
            response = genai.GenerativeModel(name).generate_content(content)
            text = (response.text or "").strip()
            if text:
                _resolved_model_name = name
                return text
        except Exception as e:
            logger.warning(f"Gemini 모델 {name} 호출 실패: {e}")
    return None


# ========================================================================================
# 2. 룩북 데이터 (이미지 주소는 브라우저에서 바로 열리는 공개 주소만 사용)
#    index.html 의 DEFAULT_LOOKS 와 같은 주소를 유지하세요.
# ========================================================================================
SEED_FASHION_ITEMS = [
    {
        "id": "look_01_avant_garde_trench",
        "title": "Avant-Garde Tailored Oversized Trench Look",
        "category": "outerwear_trench",
        "image_url": "https://lh3.googleusercontent.com/aida/ALJJyrCzTv0O9NAIrFJJwaEaFnwwF46yfbAWuiN_erY42nrYfuhMB27vXIl3a10DysetKRYhMY6yhfhjJhAwiIEtvS36Y72j1xCS-iE4L9xJczRU1s3r-IwnqeR6Q8o1bJJzXX0csKrAJ7w7lcwnxtPDD5CYyAVHNxTmvaGydmb3gpRgDB5_IKw3iEwNNykgkHAOz8xlusmlJjDJR6ypIgo-_tkf05SRfob4IwA1agLPuc6GYLFdz86KLS_n3FU",
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
        "image_url": "https://images.unsplash.com/photo-1509631179647-0177331693ae?auto=format&fit=crop&w=800&q=80",
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
        "image_url": "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?auto=format&fit=crop&w=800&q=80",
        "silhouette": "Minimalist Fluid & Tailored",
        "color_palette": ["Monochrome Slate Grey", "Cobalt Blue", "Graphite"],
        "fabrics": ["Fine Cashmere Rib", "Compact Merino Wool", "Polished Calfskin"],
        "attributes": ["mock-neck", "clean-belted-waist", "minimalist-chic", "luxury-campaign", "sculptural-hardware"],
        "summary": "Sophisticated minimalist editorial fashion lookbook photo featuring monochrome slate grey luxury knitwear with subtle cobalt blue structured belt hardware. Clean modern aesthetic, luxury brand lighting."
    }
]

# 룩마다 붙는 한국어 카드 문구 (검색 순위가 바뀌어도 이미지와 문구가 항상 같은 룩을 가리키게 함)
ITEM_COPY: Dict[str, Dict[str, Any]] = {
    "look_01_avant_garde_trench": {
        "title": "아키텍처럴 오버사이즈 테일러드 룩",
        "points": [
            "어깨 라인을 강조한 와이드 실루엣과 롱 코트 기장감의 조화",
            "옵시디언 슬레이트 바탕에 사파이어 블루 미세 악센트",
            "개버딘 울과 본디드 코튼의 탄탄한 질감 형성"
        ],
        "items": ["오버사이즈 테일러드 트렌치 코트", "와이드 플리츠 트라우저", "사파이어 블루 미니 클러치"]
    },
    "look_02_emerald_tech_tailored": {
        "title": "테크-테일러드 에메랄드 레이어드 룩",
        "points": [
            "테크니컬 립스탑 소재와 조각적 블레이저의 기능적 융합",
            "에메랄드 그린 컬러 블록으로 생동감 있는 런웨이 스트릿 무드",
            "고어텍스 트림 및 전술적 하네스 하드웨어 포인트"
        ],
        "items": ["조각적 테크 블레이저", "유틸리티 카고 와이드 팬츠", "에메랄드 나일론 크로스 스트랩"]
    },
    "look_03_monochrome_slate_knitwear": {
        "title": "모노크롬 슬레이트 럭셔리 니트웨어 룩",
        "points": [
            "부드러운 캐시미어 립 조직과 유려한 드레이핑 실루엣",
            "차콜 그레이 베이스와 코발트-아메시스트 레더 벨트 대조",
            "미니멀한 모크넥과 기하학적 버클 하드웨어 악센트"
        ],
        "items": ["컴팩트 메리노 모크넥 니트", "테일러드 울 플란넬 팬츠", "코발트 스트럭처 레더 벨트"]
    }
}

# 화면의 카드 자리(A/B/C)별 색상과 표시용 점수
STREAM_SLOTS = [
    {"id": "stream_a_sapphire", "name": "Stream A · Sapphire Prime", "accent": "#3B82F6", "score": 96},
    {"id": "stream_b_emerald", "name": "Stream B · Emerald Tech-Wear", "accent": "#10B981", "score": 93},
    {"id": "stream_c_amethyst", "name": "Stream C · Royal Amethyst", "accent": "#8B5CF6", "score": 89},
]

# ========================================================================================
# 3. ChromaDB 저장소 (서버가 시작될 때마다 위 룩북으로 새로 색인)
# ========================================================================================
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
collection = chroma_client.get_or_create_collection(
    name="fashionpedia_multimodal",
    metadata={"description": "Fashionpedia image summaries and attribute embeddings"}
)


def hash_embedding(text: str) -> List[float]:
    """Gemini 임베딩을 쓸 수 없을 때의 대체 벡터 (같은 문장은 항상 같은 벡터)"""
    vec = [0.0] * EMBED_DIM
    for idx, word in enumerate(text.split()):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % EMBED_DIM
        vec[h] += 1.0 / (idx + 1)
    norm = sum(x ** 2 for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def get_embedding(text: str) -> List[float]:
    """텍스트 임베딩 (Gemini 우선, 실패하면 대체 벡터). 항상 EMBED_DIM 차원."""
    if GEMINI_API_KEY:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query"
            )
            vec = result["embedding"]
            if len(vec) == EMBED_DIM:
                return vec
            logger.warning(f"임베딩 차원이 {len(vec)}라서 대체 벡터를 사용합니다. (필요: {EMBED_DIM})")
        except Exception as e:
            logger.warning(f"Gemini 임베딩 실패: {e}. 대체 벡터를 사용합니다.")
    return hash_embedding(text)


def init_vector_store():
    """룩북 데이터를 ChromaDB에 색인"""
    if collection.count() > 0:
        return
    logger.info("ChromaDB에 룩북 데이터를 색인합니다...")
    ids, documents, metadatas, embeddings = [], [], [], []

    for item in SEED_FASHION_ITEMS:
        content = (
            f"{item['title']}. {item['summary']} Silhouettes: {item['silhouette']}. "
            f"Palette: {', '.join(item['color_palette'])}. Fabrics: {', '.join(item['fabrics'])}. "
            f"Attributes: {', '.join(item['attributes'])}."
        )
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

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info(f"{len(ids)}개 룩을 색인했습니다.")


init_vector_store()

# ========================================================================================
# 4. 데이터 모델
# ========================================================================================
class FashionQueryRequest(BaseModel):
    query: str = Field(..., description="사용자 한국어 패션 스타일링 질의")
    top_k: int = Field(default=3, ge=1, le=5, description="검색할 룩북 수")
    image_base64: Optional[str] = Field(default=None, description="선택: 사용자 업로드 이미지 Base64")


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
# 5. RAG 파이프라인
# ========================================================================================
def translate_query_ko_to_en(korean_query: str) -> str:
    """Step 1: 한국어 질의를 영문 패션 검색 문장으로 번역"""
    fallback_query = f"High fashion runway styling advice for: {korean_query}"
    prompt = f"""You are a luxury fashion runway director and styling translator.
Translate the following Korean fashion request into a precise, descriptive English fashion search query.
Focus on silhouettes, garments, fabrics, aesthetics, and color matching.
Return ONLY the translated English query text without commentary.

[Korean Request]: {korean_query}"""
    return generate_text(prompt) or fallback_query


def search_fashion_images(query_en: str, top_k: int = 3) -> List[FashionItemResult]:
    """Step 2: ChromaDB에서 유사한 룩 검색"""
    query_vec = get_embedding(query_en)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count())
    )

    ids = results["ids"][0] if results.get("ids") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else [0.2] * len(ids)

    retrieved = []
    for idx, item_id in enumerate(ids):
        meta = metas[idx]
        dist = distances[idx] if idx < len(distances) else 0.2
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


def generate_vlm_styling_advice(query_en: str, items: List[FashionItemResult],
                                user_img_b64: Optional[str] = None) -> str:
    """Step 3~4: 검색된 룩과 질의를 합쳐 스타일링 조언 생성"""
    default_fallback_advice = (
        "Based on the editorial lookbook, pair an architectural oversized tailored coat with "
        "relaxed-cut wide-leg wool trousers. Maintain an obsidian slate base palette and introduce "
        "either luminous emerald accents or a structured cobalt leather belt to create high-contrast runway tension."
    )

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

    content_parts: List[Any] = [prompt]
    if user_img_b64:
        try:
            img_bytes = base64.b64decode(user_img_b64.split(",")[-1])
            content_parts.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        except Exception as e:
            logger.warning(f"사용자 이미지를 읽지 못했습니다: {e}")

    text = generate_text(content_parts)
    if not text:
        logger.warning("Gemini 응답을 받지 못해 기본 조언을 반환합니다.")
    return text or default_fallback_advice


def translate_advice_en_to_ko(advice_en: str) -> str:
    """Step 5: 영문 조언을 한국어 패션 문체로 번역"""
    default_ko_advice = (
        "아키텍처럴 오버사이즈 실루엣과 와이드 레그 트라우저의 볼륨감 있는 대비를 연출하는 룩을 제안합니다. "
        "옵시디언 슬레이트와 차콜 그레이를 메인 캔버스로 구성하고, 런웨이 감성의 에메랄드 테크웨어 포인트나 "
        "코발트 블루 스트럭처 레더 벨트를 더해 세련된 시각적 텐션을 완성하세요."
    )
    prompt = f"""당신은 최고급 패션 매거진(Vogue Korea, GQ)의 수석 패션 에디터입니다.
다음 영문 패션 스타일링 조언을 우아하고 전문적인 한국어 텍스트로 번역/윤문하세요.
패션 전문 어휘(오버사이즈 실루엣, 텍스처 대비, 레이어링, 슬레이트 톤, 테일러드 라인 등)를 자연스럽게 살려주세요.

[English Advice]:
{advice_en}"""
    return generate_text(prompt) or default_ko_advice


def build_comparison_streams(items: List[FashionItemResult]) -> List[ComparisonStream]:
    """검색 순위대로 A/B/C 카드 데이터를 구성 (이미지와 문구는 항상 같은 룩 기준)"""
    streams = []
    for idx, slot in enumerate(STREAM_SLOTS):
        item = items[idx % len(items)] if items else None
        copy = ITEM_COPY.get(item.id, {}) if item else {}

        streams.append(ComparisonStream(
            stream_id=slot["id"],
            stream_name=slot["name"],
            accent_color=slot["accent"],
            match_score=slot["score"],
            title=copy.get("title", item.title if item else ""),
            look_summary=item.summary if item else "",
            styling_points=copy.get("points", []),
            recommended_items=copy.get("items", []),
            image_url=item.image_url if item else ""
        ))
    return streams


# ========================================================================================
# 6. FastAPI 앱
# ========================================================================================
app = FastAPI(
    title="FashionRAG Multimodal Recommendation Server",
    description="Multimodal Fashion RAG API with Google AI, ChromaDB & Fashionpedia",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """서버 상태 확인 (gemini_api_configured 가 false면 AI 호출 없이 기본 문구로 응답 중)"""
    return {
        "status": "healthy",
        "service": "FashionRAG Multimodal Server",
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "chromadb_collection_items": collection.count(),
        "active_models": {
            "vlm": get_available_model_name() or GENERATION_MODEL,
            "embedding": EMBEDDING_MODEL
        },
        "design_system": "Editorial Slate Haute Tech"
    }


@app.post("/api/fashion/recommend", response_model=FashionResponse)
def recommend_fashion_styling(req: FashionQueryRequest):
    """한국어 질의 -> 영문 번역 -> 룩 검색 -> 스타일링 생성 -> 한국어 번역 -> 3단 비교 출력"""
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="질의 내용이 비어있습니다.")

    t_total_start = time.time()

    t0 = time.time()
    query_en = translate_query_ko_to_en(req.query.strip())
    latency_trans_query = round((time.time() - t0) * 1000, 2)

    t1 = time.time()
    retrieved_items = search_fashion_images(query_en, top_k=req.top_k)
    latency_retrieval = round((time.time() - t1) * 1000, 2)

    t2 = time.time()
    advice_en = generate_vlm_styling_advice(query_en, retrieved_items, req.image_base64)
    latency_vlm = round((time.time() - t2) * 1000, 2)

    t3 = time.time()
    advice_ko = translate_advice_en_to_ko(advice_en)
    latency_trans_advice = round((time.time() - t3) * 1000, 2)

    streams = build_comparison_streams(retrieved_items)

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
            "vlm_model": get_available_model_name() or GENERATION_MODEL
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
