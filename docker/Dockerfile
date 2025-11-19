FROM python:3.10-slim

RUN apt-get update && \ 
    apt-get install -y ffmpeg libgl1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir ultralytics
RUN pip uninstall -y opencv-python
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git
RUN pip install --no-cache-dir d

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e /app

ENTRYPOINT ["python", "-m", "fish_tracker.core.main"]