from .base import ObjectDetectorConfig, ROIProcessorConfig
from .selective_search import SelectiveSearchDetectorConfig
from .yolo_seg import YOLOSegDetectorConfig

DetectorConfig = SelectiveSearchDetectorConfig | YOLOSegDetectorConfig

__all__ = [
    "ObjectDetectorConfig",
    "SelectiveSearchDetectorConfig",
    "YOLOSegDetectorConfig",
    "DetectorConfig",
    "ROIProcessorConfig",
]
