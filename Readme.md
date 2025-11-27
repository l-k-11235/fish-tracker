# Fish Tracker


Fish Tracker is a modular and extensible pipeline for tracking fish in underwater videos captured by fixed cameras.  
It combines **motion-based ROI detection**, **neural detection (YOLO)**, and **object tracking** using Kalman filtering and multiple similarity metrics.

The pipeline is implemented in Python using OpenCV and PyTorch, and is designed to be easily extended with new detectors, matchers, or trackers.

The code is written in Python with OpenCV and is designed to be easily extended with other tracking strategies.

Example runs are provided using data from the EyeSea project (e.g., video DCPUD_Wellsdam).


## Project structure

```
.
├── fish_tracker
│   ├── __init__.py
│   ├── main.py
│   ├── config
|   |   ├── full_config.py
|   |   ├── __init__.py
|   |   ├──  tracker_manager.py
|   |   ├── video_params.py
|   |   ├──  detector
|   |   │   ├── __init__.py
|   |   │   ├── base.py
|   |   │   ├── selective_search.py
|   |   │   └── yolo_seg.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── tracker_manager.py
│   │   └── tracker_matcher.py
│   │   └── output_writer.py
│   ├── detectors
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── selective_search.py
│   │   ├── yolo_seg.py
│   │   ├── worker.py
│   │   ├── roi_detection.py
│   ├── trackers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── kalman_tracker.py
│   └── utils
│       ├── __init__.py
│       ├── background.py
│       ├── roi_processor.py
│       ├── logger.py
├── docker
│   ├── build.sh
│   ├── Dockerfile
│   ├── .dockerignore
├── .github/workflows
│   ├── push.yaml
├── data
│   ├── examples
│   │   ├── inputs
│   │   │   ├── WellsDam_1_East_20170627_13_0-30.mp4
│   │   │   ├── WellsDam_1_East_20170627_13_210-240.mp4
│   │   │   └── WellsDam_1_East_20170627_13_first_frame.png
│   │   ├── outputs_WellsDam_1_East_20170627_13_0-30
│   │   │   ├── output.json
│   │   │   └── output.mp4
│   │   └── outputs_WellsDam_1_East_20170627_13_210-240
│   │       ├── output.json
│   │       └── output.mp4
│   ├── models
│   │   └── yolov8n-seg.pt
│   ├── configs
│   │   ├── test1_selective_search.yaml
│   │   ├── test2_selective_search.yaml
│   │   └── test3_yolo_seg.yaml
├── Readme.md
├── setup.py
└── requirements.txt
```

## Features

    Two ROI detection backends (selectable):
        - Motion-masked Selective Search (as in R-CNN)
        - YOLO-based neural detector filtered by motion masking.

    Non-Maximum Suppression to remove redundant overlapping boxes

    Multiple similarity metrics for data association: Euclidean, cosine, and hybrid (with MobileNet-based embeddings)

    Kalman filter tracker (extensible).

    Track lifecycle management: initialized → active → finished → trashed

    JSON export of tracking results

    Video rendering with annotated trajectories.

## Quick Start

### 1. Install dependencies

```bash
bash docker/build.sh
```

### 2. Run the tracker

The tracker now runs entirely from a YAML config file.

```bash
docker run --rm -it \
-v "$(pwd)/inputs":/app/data/inputs \
-v "$(pwd)/outputs":/app/data/outputs \
-v "$(pwd)/config.yaml: /app/config.yaml
fish-tracker:latest \
--config /app/config.yaml
```

### Configuration
You can provide minimal configs or full configs.
At runtime, the full resolved configuration is dumped automatically for reproducibility.

**Minimal config example:**:
```yaml
input_video_path: /app/data/inputs/WellsDam_1_East_20170627_13_0-30.mp4
detector_opts:
  dump_masked_frames: true
detector_type: yolo_seg
tracker_manager_opts:
  method: hybrid
```

**Full config example (self generated):**
```yaml
video_opts:
  frame_width: 1280
  frame_height: 960
  nb_frames: 960
  fps: 32
detector_type: yolo_seg
detector_opts:
  roi_processor_opts:
    embedding_model_name: mobilenetv3small
    pad_ratio: 0.1
    target_size: null
    device: cpu
  overlap_threshold: 0.1
  max_frame_coverage: 0.1
  min_frame_coverage: 0.005
  max_width_ratio: 0.3
  max_heigth_ratio: 0.3
  dump_masked_frames: true
  chunk_size: 8
  yolo_model_path: /app/data/models/yolov8n-seg.pt
  yolo_conf_thresh: 0.0005
  iou_threshold: 0.2
tracker_manager_opts:
  video_opts:
    frame_width: 1280
    frame_height: 960
    nb_frames: 960
    fps: 32
  method: hybrid
  alpha: 0.5
  distance_threshold: null
  max_absences: 2
  min_tracking_duration: 0.0
  tracking_method: KalmanFilter
  roi_filter_name: null
input_video_path: /app/data/inputs/WellsDam_1_East_20170627_13_0-30.mp4
ref_frame_path: null
output_json_path: /app/data/outputs/output.json
output_video_path: /app/data/outputs/output.mp4
log_level: INFO
start: 0
step: 5
end: 960
```

### 3. Outputs

    wdir/outputs/output.json: serialized tracker data

    wdir/outputs/output.mp4: video with visualized trajectories

    PNG frames used for the video generation

    Log file with detailed debug information

## Parameters

| Argument                  | Description                           | Default            |
| ------------------------- | --------------------------------------|--------------------|
| `--input_video_name`      | Input video path                      | *Required*         |
| `--first_frame`           | First frame                           | `None`             |
| `--output_video_name`     | Output annotated video                | `output.mp4`       |
| `--output_json_name`      | Output JSON file                      | `output.json`      |
| `--dump_masked_frames`    | Dump frames with motion mask (flag)   | `False`            |
| `--detector_type`         | Method used for ROI detection.        | `selective_search` |
| `--matching_method`       | Method used to match trackers and ROI | `geometric`           |
| `--alpha`                 | Weight of the geometric distance<br>in the hybrid method | `0.5`           |
| `--distance_threshold`    | Matching distance threshold           |               |
| `--max_absences`          | Max frame absences for a tracker      | `2`                |
| `--min_tracking_duration` | Minimum duration (s) to keep tracker  | `0`                |
| `--step`                  | Process every `step` frames           | `5`                |
| `--start`, `--end`        | Frame range                           | `0`, `None`        |
| `--num_workers`         | Nb of parallel processes used for<br>detection (selective search) | `8` |                            |
| `--chunk_size`            | Nb of frames processed together       | `8`                |
| `--log_level`             | Level of logging                      | `INFO`             |
