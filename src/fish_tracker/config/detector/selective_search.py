from pydantic import Field, ConfigDict

from .base import ObjectDetectorConfig


class SelectiveSearchDetectorConfig(ObjectDetectorConfig):
    resize_factor: float = 0.25
    num_workers: int = Field(
        default=8,
        description="Number of parallel worker processes used for fish detection. ",
    )

    model_config = ConfigDict(extra="ignore")
