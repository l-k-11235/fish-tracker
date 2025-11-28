from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class ROIProcessorConfig(BaseModel):
    embedding_model_name: Literal[None, "mobilenetv3small"] = Field(
        default=None, description="Name of the embedding model for ROI representation"
    )
    pad_ratio: float = Field(default=0.1, description="Padding ratio around the ROI")
    target_size: tuple[int, int] | None = Field(
        default=None, description="Target size for ROI resizing (width, height)"
    )
    device: str = Field(default="cpu", description="Device to run the ROI processor on")

    model_config = ConfigDict(extra="ignore")
    # Ignore unknown keys instead of raising an error.


class ObjectDetectorConfig(BaseModel):
    roi_processor_opts: ROIProcessorConfig = ROIProcessorConfig()
    overlap_threshold: float = 0.1  # NMS
    max_frame_coverage: float = 0.10  # NMS
    min_frame_coverage: float = 0.005  # NMS
    max_width_ratio: float = 0.3  # NMS
    max_heigth_ratio: float = 0.3  # NMS
    dump_masked_frames: bool = False
    chunk_size: int = Field(
        default=8,
        description=(
            "Number of frames processed together per iteration. "
            "Used as the chunk size for Selective Search (CPU) or "
            "as the batch size for YOLO segmentation."
        ),
    )

    model_config = ConfigDict(extra="ignore")
