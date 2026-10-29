# 파이썬 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# pip 최신화 및 패키지 분할 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pypdf
RUN pip install --no-cache-dir supabase
RUN pip install --no-cache-dir langchain langchain-community langchain-google-genai langchainhub

# 소스 코드 복사
COPY . .

# 포트 개방
EXPOSE 8080

# [수정] 클라우드 런이 부여하는 $PORT 환경변수를 동적으로 받도록 변경
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
