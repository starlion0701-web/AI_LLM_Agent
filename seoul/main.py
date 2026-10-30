import os
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Langchain 라이브러리 임포트
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler

# FastAPI 앱 초기화
app = FastAPI(title="Seoul City Planner RAG Server")

# 환경 변수에서 설정값 로드 (구글 클라우드 런 환경 변수에 등록 필요)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# 필수 환경 변수 누락 시 세밀한 에러 반환
if not all([GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError(
        "서버 초기화 실패: 필수 환경 변수가 설정되지 않았습니다. "
        "GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY 값을 확인해주세요."
    )

# Supabase 클라이언트 연결
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    raise RuntimeError(f"Supabase 클라이언트 초기화 중 에러가 발생했습니다. 상세 내역: {str(e)}")

# 구글 AI 임베딩 모델 설정
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 실시간 스트리밍 핸들러 (StreamHandler) 정의
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # 터미널이나 서버 로그에 생성되는 토큰을 실시간으로 출력 (디버깅/모니터링 용도)
        print(token, end="", flush=True)

# 구글 AI LLM (gemini-3.5-flash-lite) 설정 및 스트리밍 활성화
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    streaming=True,
    callbacks=[StreamHandler()]
)

# API 요청 모델 정의
class ChatRequest(BaseModel):
    query: str
    # chat_history 예시: [{"role": "user", "content": "안녕"}, {"role": "ai", "content": "안녕하세요"}]
    chat_history: List[Dict[str, str]] = []


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """입력한 PDF 파일을 텍스트로 변환하고 청크 단위로 분할하여 벡터 DB에 저장합니다."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다. 업로드된 파일명: {file.filename}")
    
    tmp_path = None
    try:
        # 1. 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # 2. PDF 텍스트 변환 및 로드
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 3. 텍스트를 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="PDF 파일에서 추출할 텍스트를 찾지 못했습니다.")

        # 4. 벡터 DB와 리트리버 개발: Supabase에 적용 및 임베딩 저장
        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )

        return {"detail": f"성공적으로 {len(chunks)}개의 청크가 분할되어 Supabase 벡터 DB에 적재되었습니다."}
        
    except Exception as e:
        # 세밀한 에러 메시지 반환
        raise HTTPException(
            status_code=500, 
            detail=f"PDF 변환 및 DB 적재 프로세스 중 심각한 오류가 발생했습니다. 에러 내역: {str(e)}"
        )
    finally:
        # 안전한 임시 파일 정리
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def generate_rag_response(query: str, chat_history_raw: List[Dict[str, str]]):
    """사용자 질의를 확장하고, 검색된 문서를 바탕으로 스트리밍 답변을 생성하는 제너레이터 함수"""
    try:
        # Langchain 객체로 대화 기록 변환
        chat_history = []
        for msg in chat_history_raw:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "ai":
                chat_history.append(AIMessage(content=msg.get("content")))

        # 1. 시스템 메시지 및 프롬프트 설계
        contextualize_q_system_prompt = (
            "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
            "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"
        )
        
        # 2. 대화 기록(Chat History) 전달 공간 마련
        # MessagesPlaceholder를 통해 과거의 대화 내역이 통째로 들어갈 수 있는 동적 빈칸 마련
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        # 3. Query Augmentation Chain 구성
        # RunnablePassthrough를 적용하여 질의와 과거 대화 내역을 투명하게 통과시킴
        query_augmentation_chain = (
            RunnablePassthrough() 
            | contextualize_q_prompt 
            | llm 
            | StrOutputParser()
        )
        
        # 4. 체인 실행 (데이터 주입)
        # 현재 사용자의 질문과 이전까지의 대화 기록(chat 객체)을 딕셔너리 형태로 묶어서 주입
        if chat_history:
            augmented_query = query_augmentation_chain.invoke(
                {"query": query, "chat_history": chat_history}
            )
        else:
            augmented_query = query

        # 5. 증강된 질의로 벡터 DB 검색
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 완성된 augmented_query를 리트리버에 전달하여 문서 검색 수행
        docs = retriever.invoke(augmented_query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # 6. RAG 프롬프트 설계 및 답변 생성 체인 연결
        qa_system_prompt = (
            "당신은 도시 계획 설계자 입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.\n"
            "주어진 문맥(Context)을 기반으로 답변을 작성하세요.\n\n"
            "Context: {context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        # RAG 응답 체인 구성
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: context_text)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # 실시간 스트리밍 핸들러(StreamHandler)와 연결된 astream을 통해 프론트엔드로 청크 전송
        async for chunk in rag_chain.astream({
            "query": augmented_query, 
            "chat_history": chat_history
        }):
            yield chunk
            
    except Exception as e:
        # 답변 생성 파이프라인 에러 처리
        yield f"\n\n[서버 내부 오류: 답변 생성 체인 실행 중 문제가 발생했습니다. 상세 내역: {str(e)}]"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """클라이언트와 상호작용하는 메인 챗봇 엔드포인트"""
    return StreamingResponse(
        generate_rag_response(request.query, request.chat_history),
        media_type="text/event-stream"
    )
