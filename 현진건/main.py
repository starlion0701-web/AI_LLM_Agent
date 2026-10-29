import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain 및 관련 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI(title="현진건 RAG 챗봇 서버")

class QuestionRequest(BaseModel):
    question: str

# 서버 시작 시 RAG 파이프라인을 한 번만 초기화하는 함수
def setup_rag_pipeline():
    # 1. 입력한 PDF 파일을 텍스트로 변환
    # PyPDFLoader를 사용하여 로컬에 있는 '운수좋은날.pdf' 파일을 읽어옵니다.
    loader = PyPDFLoader("운수좋은날.pdf")
    docs = loader.load()

    # 2. 청크(Chunk) 단위로 쪼개기
    # 문맥이 끊기지 않도록 chunk_overlap을 주어 1000자 단위로 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # 3. 구글 임베딩 모델 사용 (models/text-embedding-004)
    # 환경변수에 저장된 GOOGLE_API_KEY를 자동으로 참조하여 임베딩을 수행합니다.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 4. 벡터 DB 적용: Chroma DB에 적용
    # 분할된 청크와 임베딩 모델을 사용하여 로컬 디렉토리(chroma_db)에 벡터를 저장합니다.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 5. 실시간 스트리밍 핸들러 적용을 위한 콜백 및 LLM 설정
    # 서버 콘솔 출력용 StreamingStdOutCallbackHandler 적용 및 streaming=True 활성화
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    # 6. 검색기 개발: 다중 질문 검색기 (MultiQueryRetriever)
    # 사용자의 단일 질문을 LLM이 여러 개의 유사 질문으로 확장하여 벡터 DB를 다각도로 검색합니다.
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # 7. 프롬프트 허브 사용 및 시스템 역할 부여
    # LangChain Hub에서 표준 RAG 프롬프트를 가져온 후, 요구된 페르소나를 시스템 메시지로 덮어씌웁니다.
    # 허브 베이스 프롬프트 참조용 당김 (선택적 활용)
    # base_prompt = hub.pull("rlm/rag-prompt") 
    
    system_role = (
        "당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다. "
        "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요.\n\n"
        "주어진 문맥(Context)을 바탕으로 답변을 구성하십시오:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_role),
        ("human", "{question}")
    ])

    # 8. 질의 확장 구현 (RunnablePassthrough) 및 체인(Chain) 연결
    # RunnablePassthrough를 통해 입력된 질문을 그대로 전달하고, retriever를 통해 context를 채워 넣습니다.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 애플리케이션 메모리에 체인 할당
chain = setup_rag_pipeline()

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    # 비동기 제너레이터를 사용하여 실시간 스트리밍 응답을 클라이언트로 전송합니다.
    async def stream_generator():
        # astream 메서드를 통해 답변이 생성되는 즉시 청크 단위로 클라이언트에 전달됩니다.
        async for chunk in chain.astream(req.question):
            yield chunk

    # 텍스트 이벤트 스트림 형태로 스트리밍 응답 반환
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
