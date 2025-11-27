from pydantic import BaseModel, Field, ConfigDict

from .video_params import VideoParams


class TrackerConfig(BaseModel):
    start_frame: int
    start_time: float
    x0: float
    y0: float
    embedding: list[float] = []


class TrackerMatcherConfig(BaseModel):
    video_opts: VideoParams = Field(default_factory=VideoParams)
    method: str = "geometric"
    alpha: float = 0.5
    distance_threshold: float | None = None

    model_config = ConfigDict(extra="ignore")


class TrackerManagerConfig(TrackerMatcherConfig):
    max_absences: int = 2
    min_tracking_duration: float = 0.0
    tracking_method: str = "KalmanFilter"
    roi_filter_name: str | None = None

    model_config = ConfigDict(extra="ignore")
