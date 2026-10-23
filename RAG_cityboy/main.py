import os
import json
import traceback
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

import PyPDF2
from supabase import create_client, Client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# 프론트엔드(GAS 웹앱)에서 접근 가능하도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수 검증 (서버 시작 전 필수 키 확인)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY, GOOGLE_API_KEY]):
    raise RuntimeError("⚠️ 필수 환경변수가 설정되지 않았습니다. Cloud Run 환경변수를 확인하세요.")

# Supabase 클라이언트 및 임베딩 모델 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]

def format_history(history_data: List[Dict[str, str]]):
    """대화 기록을 LangChain 메시지 객체로 변환"""
    messages = []
    for msg in history_data:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    return messages

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """1. PDF 파일 업로드 및 텍스트 추출"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="⚠️ 업로드된 파일이 PDF 형식이 아닙니다.")
    
    try:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        
        if not text.strip():
            raise ValueError("PDF에서 추출된 텍스트가 없습니다. 이미지 기반 PDF일 수 있습니다.")
            
        return {"filename": file.filename, "text": text}
        
    except Exception as e:
        error_msg = f"⚠️ PDF 파싱 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

class EmbedRequest(BaseModel):
    text: str
    filename: str

@app.post("/api/embed")
async def create_vector_db(req: EmbedRequest):
    """2. 텍스트 청크 분할 및 Supabase 벡터 DB 저장"""
    try:
        # 청크 단위 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(req.text)
        
        # 메타데이터 생성
        metadatas = [{"source": req.filename, "chunk_index": i} for i in range(len(chunks))]
        
        # Supabase에 임베딩 후 저장 (SupabaseVectorStore.from_texts 사용)
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        
        return {"message": "✅ RAG 벡터 DB 생성이 완료되었습니다.", "chunk_count": len(chunks)}
        
    except Exception as e:
        error_msg = f"⚠️ 벡터 DB 생성 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """3. 질의 확장, 검색 및 실시간 스트리밍 답변 생성 (SSE 방식)"""
    
    async def event_generator():
        try:
            # 1. LLM 초기화 (요청하신 모델명 사용, 에러 발생 시 gemini-1.5-flash 등으로 변경 필요)
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
            
            history_msgs = format_history(req.history)
            
            # 2. 질의 확장(Query Augmentation) 체인
            aug_prompt = ChatPromptTemplate.from_messages([
                ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            aug_chain = aug_prompt | llm | StrOutputParser()
            
            # 체인 실행 (대화 기록과 현재 질문 주입)
            augmented_query = await aug_chain.ainvoke({"question": req.query, "history": history_msgs})
            
            # 클라이언트에 확장된 질문 전달 (디버깅/표시 용도)
            yield {"event": "augmented_query", "data": augmented_query}
            
            # 3. 증강된 질의로 벡터 DB 검색
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(augmented_query)
            
            # 검색 결과(Context) 포맷팅 및 클라이언트로 출처 우선 전달
            context_data = []
            context_str = ""
            for doc in docs:
                source = doc.metadata.get("source", "알 수 없음")
                content = doc.page_content
                context_data.append({"source": source, "content": content[:200] + "..."}) # 클라이언트 표시용 요약
                context_str += f"[출처: {source}]\n{content}\n\n"
            
            yield {"event": "context", "data": json.dumps(context_data, ensure_ascii=False)}
            
            # 4. RAG 답변 생성 및 스트리밍 (StreamHandler 역할)
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n참고 문서:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            rag_chain = rag_prompt | llm | StrOutputParser()
            
            # astream을 활용한 실시간 청크 전송
            async for chunk in rag_chain.astream({
                "context": context_str,
                "history": history_msgs,
                "question": augmented_query
            }):
                yield {"event": "message", "data": chunk}
                
            yield {"event": "done", "data": "[DONE]"}
            
        except Exception as e:
            error_msg = f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"
            print(traceback.format_exc())
            yield {"event": "error", "data": error_msg}

    return EventSourceResponse(event_generator())
