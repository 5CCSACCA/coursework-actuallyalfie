FROM python:3.10-slim

WORKDIR /app

ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY firebase_key.json firebase_key.json

COPY . .

CMD ["fastapi", "run", "main.py", "--host=0.0.0.0", "--port=9999"]