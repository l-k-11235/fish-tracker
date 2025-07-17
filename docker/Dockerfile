FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier le code
WORKDIR /app
COPY requirements.txt .
COPY fish_tracker/ /fish_tracker

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e /fish_tracker/

ENTRYPOINT ["python3", "/app/fish_tracker/core/main.py"]


docker run --rm -it \
-e PYTHONPATH=/app \
-v "/home/lina/Documents/CV/fish-tracker/fish_tracker":/app/fish_tracker \
-v "/home/lina/Documents/CV/test_fish-tracker/inputs":/app/data/inputs \
-v "/home/lina/Documents/CV/test_fish-tracker/outputs":/app/data/outputs \
fish-tracker:latest \
--input_video_name WellsDam_1_East_20170627_13_0-30.mp4 \
--log_level DEBUG

docker run --rm -it \
-e PYTHONPATH=/app \
-v "/home/lina/Documents/CV/fish-tracker/fish_tracker":/app/fish_tracker \
-v "/home/lina/Documents/CV/test_fish-tracker/inputs":/app/data/inputs \
-v "/home/lina/Documents/CV/test_fish-tracker/outputs":/app/data/outputs \
fish-tracker:latest \
--input_video_name WellsDam_1_East_20170627_13_210-240.mp4 \
--first_frame WellsDam_1_East_20170627_13_first_frame.png \
--log_level DEBUG

