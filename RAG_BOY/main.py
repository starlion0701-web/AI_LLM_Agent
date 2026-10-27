import os
import tempfile
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

app = FastAPI(title="Google Cloud Run RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not all([GOOGLE_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("환경 변수가 설정되지 않았습니다.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []

# 🔥 강제 CORS 헤더 (FastAPI 스트리밍 버그 원천 차단)
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}

@app.post("/upload-pdf")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 가능합니다.")
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        return {"message": "업로드 성공"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            streaming=True
        )

        history_str = ""
        if request.chat_history:
            for msg in request.chat_history:
                role = "질문" if msg.get("role") == "user" else "답변"
                history_str += f"{role}: {msg.get('content')}\n"

        augmented_query = request.query
        try:
            # 🔥 무한 로딩 방지: 10초 내에 응답 없으면 강제 패스
            if history_str:
                aug_prompt = PromptTemplate.from_template("이전 대화:\n{chat_history}\n\n질문: {query}\n\n위 질문을 대화 문맥에 맞게 한 문장으로 변환해:")
                chain = aug_prompt | llm | StrOutputParser()
                augmented_query = await asyncio.wait_for(
                    chain.ainvoke({"chat_history": history_str, "query": request.query}),
                    timeout=10.0
                )
        except Exception:
            pass

        context_text = ""
        try:
            # 🔥 무한 로딩 방지: 10초 내에 DB 검색 안 되면 강제 종료 후 에러 출력
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            retrieved_docs = await asyncio.wait_for(
                vector_store.asimilarity_search(augmented_query, k=4),
                timeout=10.0
            )
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as db_err:
            context_text = f"[DB 검색 에러 (타임아웃 또는 연결 문제): {str(db_err)}]"

        rag_prompt = PromptTemplate.from_template(
            "아래 문서를 바탕으로 질문에 답해.\n\n[문서]\n{context}\n\n[질문]\n{augmented_query}\n\n답변:"
        )
        rag_chain = rag_prompt | llm | StrOutputParser()

        async def stream_generator():
            try:
                async for chunk in rag_chain.astream({"context": context_text, "augmented_query": augmented_query}):
                    yield chunk
            except Exception as stream_err:
                yield f"\n\n[AI 생성 오류]: {str(stream_err)}"

        # 🔥 headers=CORS_HEADERS 를 강제 주입하여 무조건 통과시킴
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=CORS_HEADERS)

    except Exception as e:
        async def fallback_stream():
            yield f"서버 내부 오류: {str(e)}"
        return StreamingResponse(fallback_stream(), media_type="text/event-stream", headers=CORS_HEADERS)
