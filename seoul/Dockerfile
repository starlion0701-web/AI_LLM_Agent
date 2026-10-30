# 파이썬 3.12 경량화 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 소스 코드 복사
COPY . .

# 구글 클라우드 런 기본 포트 설정
ENV PORT=8080
EXPOSE 8080

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
