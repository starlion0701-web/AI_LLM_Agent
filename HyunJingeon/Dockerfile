# 파이썬 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# pip 자체를 최신 버전으로 업데이트
RUN pip install --no-cache-dir --upgrade pip

# [핵심 해결책] 의존성 무한 루프를 막기 위해 패키지를 3단계로 분할 설치합니다.
# 1. 기본 웹 서버 및 유틸리티
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pypdf

# 2. DB 연결 패키지 (충돌의 주범이므로 단독 설치)
RUN pip install --no-cache-dir supabase

# 3. AI 및 Langchain 패키지
RUN pip install --no-cache-dir langchain langchain-community langchain-google-genai langchainhub

# 서버 소스 코드 복사 (requirements.txt는 이제 사용하지 않으므로 무시됨)
COPY . .

# 포트 개방
EXPOSE 8080

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
