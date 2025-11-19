import hashlib
import numpy as np
import random
import time

from abc import ABC, abstractmethod

from fish_tracker.utils.configs import TrackerConfig


class BaseTracker(ABC):
    """Abstract base class for all object trackers."""

    def __init__(self, config: TrackerConfig) -> None:
        """
        Initialize a tracker.
        Args:
            start_frame (int): Frame number where the object was first detected.
            start_time (float): Timestamp corresponding to the start_frame.
            x0 (float): Initial x-coordinate of the object.
            y0 (float): Initial y-coordinate of the object.
        """
        self.start_frame: int = config.start_frame
        self.start_time: float = config.start_time
        self.absences = 0
        self.x: float = config.x0
        self.y: float = config.y0
        self.embedding: list[float] = config.embedding
        self.trajectory: dict[int, tuple[int, int]] = {
            config.start_frame: (int(self.x), int(self.y))
        }
        self.end_time: float = 0
        self.duration: float = 0
        self.end_frame: float | None = None
        self.color: tuple[int, int, int] = self._pick_random_color()
        self._id: str = hashlib.sha1(str(time.time()).encode()).hexdigest()[
            :8
        ]  # 8-char hash
        self.box = None

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def predict(self) -> None:
        """
        Predict the next position of the tracked object.
        """
        pass

    @abstractmethod
    def correct_prediction(
        self, bbox: tuple[int, int, int, int], frame_num: int
    ) -> None:
        """
        Correct the tracker's prediction using the given bounding box
        (observation).

        Args:
            bbox (tuple): Bounding box as (x, y, w, h).
            frame_num (int): Frame number associated with the detection.
        """
        pass

    def update_trajectory(self, frame_num: int) -> None:
        self.trajectory[frame_num] = (int(self.x), int(self.y))

    def terminate(self, end_frame: int, end_time: float) -> None:
        """
        Finalize the track and compute duration.

        Args:
            end_frame (int): Last frame where the object was seen.
            end_time (float): Timestamp of the last frame.
        """
        self.end_frame = end_frame
        self.end_time = end_time
        self.duration = round(end_time - self.start_time, 2)
        self.trajectory[end_frame] = (int(self.x), int(self.y))

    def distance_to_box(self, box: tuple[int, int, int, int]) -> float:
        """
        Compute the distance between the current tracker position
        and a given bounding box.

        Args:
            box (tuple): Bounding box as (x, y, w, h).

        Returns:
            float: Euclidean distance between the tracker's position and the box center.
        """
        x, y, w, h = box
        center = np.array([x + w / 2, y + h / 2])
        pos = np.array([self.x, self.x])
        return float(np.linalg.norm(pos - center))

    @staticmethod
    def _pick_random_color() -> tuple[int, int, int]:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
