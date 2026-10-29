import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

# FastAPI 앱 생성
app = FastAPI(title="현진건 작가 AI 서버 (Google GenAI + Supabase)")

# CORS 설정 추가 (Apps Script 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [수정] 서버가 켜질 때 전역에서 바로 에러를 뿜지 않고, 요청이 올 때 안전하게 불러오도록 함수 처리
def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase 환경 변수가 설정되지 않았습니다.")
    return create_client(supabase_url, supabase_key)

def get_embeddings():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

# ---------------------------------------------------------
# 커스텀 스트리밍 핸들러
# ---------------------------------------------------------
class MyStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

# ---------------------------------------------------------
# 엔드포인트 1: PDF 업로드 및 벡터 DB 저장
# ---------------------------------------------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # 함수를 통해 클라이언트와 임베딩 호출
        vector_store = SupabaseVectorStore.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            client=get_supabase_client(),
            table_name="documents",
            query_name="match_documents"
        )
        
        return {"message": f"성공적으로 {len(chunks)}개의 청크를 분할하여 DB에 저장했습니다."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 엔드포인트 2: 질의응답 (스트리밍)
# ---------------------------------------------------------
@app.get("/ask/")
async def ask_question(query: str):
    try:
        vector_store = SupabaseVectorStore(
            client=get_supabase_client(),
            embedding=get_embeddings(),
            table_name="documents",
            query_name="match_documents"
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            streaming=True,
            callbacks=[MyStreamHandler()]
        )
        
        query_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            llm=query_llm
        )

        prompt = hub.pull("rlm/rag-prompt")
        
        persona_system_prompt = (
            "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
            "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요.\n\n"
            "다음 문맥(Context)을 바탕으로 질문에 답하십시오: {context}"
        )
        prompt.messages[0].prompt.template = persona_system_prompt

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async def generate_response():
            async for chunk in rag_chain.astream(query):
                yield chunk

        return StreamingResponse(generate_response(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
