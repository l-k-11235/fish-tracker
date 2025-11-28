from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class VideoParams(BaseModel):
    frame_width: int = Field(default=0, description="Width of the video frames")
    frame_height: int = Field(default=0, description="Height of the video frames")
    nb_frames: int = Field(default=0, description="Total number of frames in the video")
    fps: int = Field(default=0, description="Frames per second of the video")

    @classmethod
    def from_video(cls, video_path: Path) -> "VideoParams":
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video {video_path}")

        fps: int = int(cap.get(cv2.CAP_PROP_FPS))
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Unable to read first frame from {video_path}")
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
