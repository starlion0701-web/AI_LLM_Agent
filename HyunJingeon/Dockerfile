# 파이썬 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 소스 코드 복사
COPY . .

# 포트 개방 (Cloud Run은 기본적으로 8080 포트를 사용)
EXPOSE 8080

# FastAPI 서버 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
