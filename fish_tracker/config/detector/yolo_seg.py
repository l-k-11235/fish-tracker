from pathlib import Path
from pydantic import ConfigDict

from .base import ObjectDetectorConfig


class YOLOSegDetectorConfig(ObjectDetectorConfig):
    yolo_model_path: Path = Path("/app/data/models/yolov8n-seg.pt")
    yolo_conf_thresh: float = 0.0005
    iou_threshold: float = 0.2

    model_config = ConfigDict(extra="ignore")
