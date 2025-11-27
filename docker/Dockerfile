FROM python:3.10-slim

RUN apt-get update && \ 
    apt-get install -y ffmpeg libgl1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu130

RUN pip install ultralytics==8.3.229 && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e /app

ENTRYPOINT ["python", "-m", "fish_tracker.main"]