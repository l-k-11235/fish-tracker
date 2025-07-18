# Fish Tracker

This project provides a modular video processing pipeline for tracking fish in underwater footage captured by a fixed camera. It combines contour-based detection with object tracking, currently using Kalman filters.

The code is written in Python with OpenCV and is designed to be easily extended with other tracking strategies.

Example runs are provided using data from the EyeSea project (e.g., video DCPUD_Wellsdam).


## Project structure

```
.
├── docker
│   ├── build.sh
│   ├── Dockerfile
│   └── run.sh
├── fish_tracker
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── tracker_manager.py
│   │   └── tracker_matcher.py
│   ├── detection
│   │   ├── __init__.py
│   │   ├── object_detector.py
│   ├── trackers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── kalman_tracker.py
│   └── utils
│       ├── __init__.py
│       ├── background.py
│       ├── logger.py
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
├── Readme.md
├── setup.py
└── requirements.txt
```

## Features

    Contour-based fish detection via background subtraction.

    Kalman filter tracker (extensible).

    Track management with lifecycle: initialized, active, finished, trashed.

    Exports tracking results to JSON.

    Video rendering with annotated trajectories.

## Quick Start

### 1. Install dependencies

```bash
bash docker/build.sh
```

### 2. Run the tracker
```bash
docker run --rm -it \
-v "$(pwd)/inputs":/app/data/inputs \
-v "$(pwd)/outputs":/app/data/outputs \
fish-tracker:latest \
--input_video_name my-video.mp4
```

Or if you need to provide the first frame (for correct background substraction):

```bash
docker run --rm -it \
-v "$(pwd)/inputs":/app/data/inputs \
-v "$(pwd)/outputs":/app/data/outputs \
fish-tracker:latest \
--input_video_name my-video.mp4 \
--first_frame first_frame.png
```

### 3. Outputs

    wdir/outputs/output.json: serialized tracker data

    wdir/outputs/output.mp4: video with visualized trajectories

    PNG frames used for the video generation

    Log file with detailed debug information

## Parameters

| Argument                  | Description                      | Default       |
| ------------------------- | -------------------------------- | ------------- |
| `--input_video_name`      | Input video path                 | *Required*    |
| `--first_frame`           | First frame                      |  None         |
| `--output_video_name`     | Output annotated video           | `output.mp4`  |
| `--output_json_name`      | Output JSON file                 | `output.json` |
| `--min_area`              | Minimum contour area             | `2000`        |
| `--distance_threshold`    | Matching distance threshold      | `100`         |
| `--max_absences`          | Max frame absences for a tracker | `3`           |
| `--min_tracking_duration` | Minimum duration to keep tracker | `0.5`         |
| `--step`                  | Process every `step` frames      | `1`           |
| `--start`, `--end`        | Frame range                      | `0`, `None`   |
| `--log_level`             | Level of logging                 | `INFO`        |
