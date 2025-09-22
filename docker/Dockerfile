FROM python:3.10-slim

RUN apt-get update && \ 
    apt-get install -y ffmpeg libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir opencv-contrib-python && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e /app

ENTRYPOINT ["python3", "/app/fish_tracker/core/main.py"]