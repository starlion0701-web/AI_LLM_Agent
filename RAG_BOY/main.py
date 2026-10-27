import os
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 충돌을 일으키는 SystemMessage를 없애고 단순 PromptTemplate로 교체
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
    chat_history: List[Dict[str, str]]

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
    # 최후의 방어선: 어떤 에러가 나더라도 500 에러(Failed to fetch) 대신 텍스트로 원인을 출력
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            streaming=True
        )

        # 1. 역할 충돌 방지: 대화 기록을 단순 문자열로 뭉쳐서 처리
        history_str = ""
        for msg in request.chat_history:
            role = "질문" if msg.get("role") == "user" else "답변"
            history_str += f"{role}: {msg.get('content')}\n"

        # 2. 질의 증강 시도 (실패하더라도 뻗지 않고 원본 질문을 그대로 사용)
        augmented_query = request.query
        try:
            aug_prompt = PromptTemplate.from_template(
                "이전 대화:\n{chat_history}\n\n질문: {query}\n\n위 질문을 대화 문맥에 맞게 한 문장으로 변환해:"
            )
            chain = aug_prompt | llm | StrOutputParser()
            augmented_query = await chain.ainvoke({"chat_history": history_str, "query": request.query})
        except Exception:
            pass # 실패 시 조용히 넘어가고 원본 query 사용

        # 3. DB 검색 시도 (실패 시 서버 다운을 막고 화면에 에러 내용 출력)
        context_text = ""
        try:
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            retrieved_docs = await vector_store.asimilarity_search(augmented_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as db_err:
            context_text = f"[DB 검색 에러 발생. Supabase 설정을 확인하세요: {str(db_err)}]"

        # 4. 최종 답변 생성 (SystemMessage 객체를 완전히 제거하여 충돌 차단)
        rag_prompt = PromptTemplate.from_template(
            "아래 문서를 바탕으로 질문에 답해.\n\n[문서]\n{context}\n\n[질문]\n{augmented_query}\n\n답변:"
        )
        rag_chain = rag_prompt | llm | StrOutputParser()

        # 5. 스트리밍 반환
        async def stream_generator():
            try:
                async for chunk in rag_chain.astream({
                    "context": context_text,
                    "augmented_query": augmented_query
                }):
                    yield chunk
            except Exception as stream_err:
                yield f"\n\n[AI 생성 오류]: {str(stream_err)}"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        # 이 블록에 도달하더라도 통신 끊김(Failed to fetch) 대신 친절하게 텍스트 반환
        async def fallback_stream():
            yield f"서버 내부 치명적 오류가 발생했습니다: {str(e)}"
        return StreamingResponse(fallback_stream(), media_type="text/event-stream")
