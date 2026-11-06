# Python 3.11 슬림 이미지 기반
FROM python:3.11-slim

# 작업 디렉터리 설정
WORKDIR /app

# 기본 빌드 도구 및 라이브러리 설치 (필요시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# Google Cloud Run 기본 환경변수 (PORT 기본값: 8080)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# uvicorn 실행 (Cloud Run 포트 바인딩)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}

