import yaml
import warnings

from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal

from fish_tracker.config.detector.base import ROIProcessorConfig

from .video_params import VideoParams
from .tracker_manager import TrackerManagerConfig
from .detector import (
    DetectorConfig,
    SelectiveSearchDetectorConfig,
    YOLOSegDetectorConfig,
)


class FullConfig(BaseModel):
    video_opts: VideoParams = Field(default_factory=VideoParams)
    detector_type: Literal["selective_search", "yolo_seg"] = Field(
        default="selective_search", description="Method used for ROI detection."
    )
    detector_opts: DetectorConfig = Field(default_factory=SelectiveSearchDetectorConfig)
    tracker_manager_opts: TrackerManagerConfig = Field(
        default_factory=TrackerManagerConfig
    )
    input_video_path: Path = Field(
        default=Path("/app/data/inputs/input.mp4"), description="Input video file"
    )
    ref_frame_path: Path | None = Field(
        default=None, description="Path to the first frame (image file)"
    )
    output_json_path: Path = Field(
        default=Path("/app/data/outputs/output.json"), description="JSON output file."
    )
    output_video_path: Path = Field(
        default=Path("/app/data/outputs/output.mp4"), description="Video output file."
    )
    log_level: Literal["CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"] = "INFO"
    start: int = 0
    step: int = 5
    end: int = 0

    @model_validator(mode="after")
    def ensure_embedding_model(self) -> "FullConfig":
        """Ensure embedding model if needed"""
        if (
            self.tracker_manager_opts.method != "geometric"
            and self.detector_opts.roi_processor_opts.embedding_model_name is None
        ):
            warnings.warn(
                f"Embedding model name must be specified for matching method "
                f"{self.tracker_manager_opts.method}. Set to mobilenetv3small."
            )
            opts: ROIProcessorConfig = self.detector_opts.roi_processor_opts
            opts.embedding_model_name = "mobilenetv3small"
        return self

    @model_validator(mode="after")
    def configure(self) -> "FullConfig":
        self.video_opts = VideoParams.from_video(self.input_video_path)

        if self.end <= self.start:
            self.end = self.video_opts.nb_frames

        self.tracker_manager_opts = TrackerManagerConfig(
            **self.tracker_manager_opts.model_dump()
        )
        self.tracker_manager_opts.video_opts = self.video_opts

        if self.detector_type == "selective_search":
            self.detector_opts = SelectiveSearchDetectorConfig(
                **self.detector_opts.model_dump()
            )
        else:
            self.detector_opts = YOLOSegDetectorConfig(
                **self.detector_opts.model_dump()
            )

        return self

    @classmethod
    def from_file(cls, path: Path) -> "FullConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self) -> None:
        yml_path: Path = Path("/app/data/outputs/config.yaml")
        data = self.model_dump(mode="json")
        with open(yml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    model_config = ConfigDict(extra="ignore")
