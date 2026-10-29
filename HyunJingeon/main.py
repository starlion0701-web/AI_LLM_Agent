import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pypdf import PdfReader

# Langchain 관련 모듈
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain import hub

# ---------------------------------------------------------
# 환경 변수 및 설정
# (실제 클라우드 런 배포 시 환경변수에 아래 값들을 셋팅해야 합니다)
# ---------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 구글 임베딩 모델 설정 (models/text-embedding-004)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# FastAPI 앱 생성
app = FastAPI(title="현진건 작가 AI 서버 (Google GenAI + Supabase)")

# ---------------------------------------------------------
# 커스텀 스트리밍 핸들러 (StreamHandler)
# ---------------------------------------------------------
class MyStreamHandler(BaseCallbackHandler):
    """
    LLM이 생성하는 토큰을 실시간으로 가로채어 처리하는 핸들러.
    콘솔에 출력하거나 별도의 로직을 태울 때 사용됩니다.
    FastAPI 응답용으로는 하단의 astream() 제너레이터를 병행 사용합니다.
    """
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 생성되는 텍스트 조각을 콘솔에 실시간 출력
        print(token, end="", flush=True)


# ---------------------------------------------------------
# 엔드포인트 1: PDF 업로드 및 텍스트 청크를 DB에 저장
# ---------------------------------------------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF를 입력받아 텍스트로 변환 후, 청크로 쪼개어 Supabase Vector DB에 저장합니다."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        # 1. PDF 텍스트 추출
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # 2. 텍스트 청크 단위로 쪼개기 (RecursiveCharacterTextSplitter)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # 3. Supabase Vector DB에 저장 (구글 임베딩 적용)
        # documents 테이블과 match_documents 함수가 DB에 준비되어 있어야 합니다.
        vector_store = SupabaseVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        
        return {"message": f"성공적으로 {len(chunks)}개의 청크를 분할하여 DB에 저장했습니다."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 엔드포인트 2: 질의응답 (스트리밍 및 다중 질문 검색기 적용)
# ---------------------------------------------------------
@app.get("/ask/")
async def ask_question(query: str):
    """사용자의 질문을 받아 다중 쿼리로 변환 후 검색하고, 현진건 작가 페르소나로 스트리밍 답변을 반환합니다."""
    
    # 1. Supabase VectorStore 인스턴스 생성
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    
    # 2. LLM 초기화 (답변 생성용) - 스트리밍 설정 및 핸들러 추가
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", # 또는 gemini-1.5-flash
        temperature=0.7,
        streaming=True,
        callbacks=[MyStreamHandler()] # 요구사항: StreamHandler 적용
    )
    
    # 다중 질문 생성을 위한 별도의 LLM (온도값을 높여 다양한 쿼리 생성 유도)
    query_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # 3. 다중 질문 검색기 개발 (MultiQueryRetriever 적용)
    # 사용자의 단일 질문을 다양한 형태로 확장하여 검색 성능(재현율)을 높입니다.
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        llm=query_llm
    )

    # 4. 프롬프트 허브 사용 및 시스템 역할 적용
    # 랭체인 허브에서 RAG 기본 프롬프트를 다운로드
    prompt = hub.pull("rlm/rag-prompt")
    
    # 요구사항: 시스템 역할(현진건 작가) 부여 (프롬프트 템플릿 수정)
    # "근대 한국 문학의 현실주의 문체(문제)로 답변"하도록 지시
    persona_system_prompt = (
        "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
        "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요.\n\n"
        "다음 문맥(Context)을 바탕으로 질문에 답하십시오: {context}"
    )
    # 다운받은 프롬프트의 첫 번째 메시지(System)를 변경
    prompt.messages[0].prompt.template = persona_system_prompt

    # 5. 질의 확장 구현 (RunnablePassthrough 적용)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 질의 내용이 그대로 통과(Passthrough)하며, 컨텍스트와 함께 프롬프트로 전달됨
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. 실시간 스트리밍 핸들러 (FastAPI StreamingResponse 응답)
    async def generate_response():
        # astream을 사용하여 청크 단위로 클라이언트에 실시간 전송
        async for chunk in rag_chain.astream(query):
            yield chunk

    # 텍스트 스트리밍 반환 (text/event-stream 등을 원하면 media_type 변경 가능)
    return StreamingResponse(generate_response(), media_type="text/plain")
