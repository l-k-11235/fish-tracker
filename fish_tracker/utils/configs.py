# utils/configs.py
import argparse
import warnings

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Tuple, Union


# ##### #
# Video #
# ##### #
class VideoParams(BaseModel):
    frame_width: int = Field(..., description="Width of the video frames")
    frame_height: int = Field(..., description="Height of the video frames")
    nb_frames: int = Field(..., description="Total number of frames in the video")
    fps: int = Field(..., description="Frames per second of the video")

    @classmethod
    def from_video(cls, video_path: str) -> "VideoParams":
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            warnings.warn(f"Unable to open video {video_path}")
            return cls(frame_width=0, frame_height=0, nb_frames=0, fps=0)

        fps: int = int(cap.get(cv2.CAP_PROP_FPS))
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            warnings.warn(f"Unable to read first frame from {video_path}")
            frame_height, frame_width = 0, 0
        else:
            frame_height, frame_width = frame.shape[:2]

        cap.release()
        return cls(
            frame_width=frame_width,
            frame_height=frame_height,
            nb_frames=nb_frames,
            fps=fps,
        )

    model_config = ConfigDict(extra="ignore")


# ### #
# ROI #
# ### #
class ROIProcessorConfig(BaseModel):
    embedding_model_name: Literal[None, "mobilenetv3small"] = Field(
        default=None, description="Name of the embedding model for ROI representation"
    )
    pad_ratio: float = Field(default=0.1, description="Padding ratio around the ROI")
    target_size: Tuple[int, int] | None = Field(
        default=None, description="Target size for ROI resizing (width, height)"
    )
    device: str = Field(default="cpu", description="Device to run the ROI processor on")

    model_config = ConfigDict(extra="ignore")
    # Ignore unknown keys instead of raising an error.


# ######### #
# Detection #
# ######### #
class ObjectDetectorConfig(BaseModel):
    roi_processor_opts: ROIProcessorConfig = ROIProcessorConfig()
    overlap_threshold: float = 0.1  # NMS
    max_frame_coverage: float = 0.10  # NMS
    min_frame_coverage: float = 0.005  # NMS
    max_width_ratio: float = 0.3  # NMS
    max_heigth_ratio: float = 0.3  # NMS
    dump_masked_frames: bool = False
    chunk_size: int = 8

    model_config = ConfigDict(extra="ignore")


class SelectiveSearchDetectorConfig(ObjectDetectorConfig):
    resize_factor: float = 0.25
    num_workers: int = 8

    model_config = ConfigDict(extra="ignore")


class YOLOSegDetectorConfig(ObjectDetectorConfig):
    yolo_model_path: str = "data/models/yolov8n-seg.pt"
    yolo_conf_thresh: float = 0.0005
    iou_threshold: float = 0.2

    model_config = ConfigDict(extra="ignore")


DetectorConfig = Union[SelectiveSearchDetectorConfig, YOLOSegDetectorConfig]


# ######## #
# Tracking #
# ######## #
class TrackerConfig(BaseModel):
    start_frame: int
    start_time: float
    x0: float
    y0: float
    embedding: list[float] = []


class TrackerMatcherConfig(BaseModel):
    frame_height: int
    frame_width: int
    method: str = "geometric"
    alpha: float = 0.5
    distance_threshold: float | None = None

    model_config = ConfigDict(extra="ignore")


class TrackerManagerConfig(TrackerMatcherConfig):
    max_absences: int = 2
    min_tracking_duration: float = 0.0
    fps: float
    input_video_name: str
    output_json_name: str
    tracking_method: str = "KalmanFilter"
    roi_filter_name: str | None = None

    model_config = ConfigDict(extra="ignore")


# ########### #
# Full Config #
# ########### #
class FullConfig(BaseModel):
    video_opts: VideoParams
    detector_type: Literal["selective_search", "yolo_seg"] = "selective_search"
    detector_opts: DetectorConfig
    tracker_manager_opts: TrackerManagerConfig
    input_video_path: str
    ref_frame_path: str | None
    output_json_path: str
    output_video_path: str
    log_level: Literal["CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"] = "INFO"
    start: int = 0
    step: int = 5
    end: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FullConfig":
        input_video_path: str = f"/app/data/inputs/{args.input_video_name}"
        ref_frame_path: str | None = (
            f"/app/data/inputs/{args.first_frame}" if args.first_frame else None
        )
        output_video_path: str = f"/app/data/outputs/{args.output_video_name}"
        output_json_path: str = f"/app/data/outputs/{args.output_json_name}"

        video_opts: VideoParams = VideoParams.from_video(input_video_path)

        if args.detector_type == "selective_search":
            from fish_tracker.utils.configs import SelectiveSearchDetectorConfig

            detector_opts = SelectiveSearchDetectorConfig(**vars(args))
        else:
            from fish_tracker.utils.configs import YOLOSegDetectorConfig

            detector_opts = YOLOSegDetectorConfig(**vars(args))
        video_params: VideoParams = VideoParams.from_video(input_video_path)
        tracker_manager_opts: TrackerManagerConfig = TrackerManagerConfig(
            **vars(args), **vars(video_params)
        )

        return cls(
            video_opts=video_opts,
            detector_type=args.detector_type,
            detector_opts=detector_opts,
            tracker_manager_opts=tracker_manager_opts,
            input_video_path=input_video_path,
            ref_frame_path=ref_frame_path,
            output_json_path=output_json_path,
            output_video_path=output_video_path,
            log_level=args.log_level,
            start=args.start,
            step=args.step,
            end=args.end if args.end > args.start else video_opts.nb_frames,
        )

    def save_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, sort_keys=False)

    model_config = ConfigDict(extra="ignore")
