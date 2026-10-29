FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# pip 최신화 후 패키지 설치 (의존성해결 속도 대폭 향상)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
