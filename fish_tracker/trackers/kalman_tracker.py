import cv2
import numpy as np
import random

from fish_tracker.trackers.base import BaseTracker


class KalmanObjectTracker(BaseTracker):
    """
    A tracker using a Kalman filter to estimate object position over time.
    """

    def __init__(
        self, start_frame: int, start_time: float, x0: float, y0: float
    ) -> None:
        super().__init__(start_frame, start_time, x0, y0)
        self._build_Kalman_filter()

    def _build_Kalman_filter(self):
        """Initialize the Kalman filter."""
        # Kalman filter
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix (takes [x, y, vx, vy])
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        # Observation matrix (we observe [x, y])
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Observation noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Initialize with first position (zero speed).
        self.kf.statePost = np.array([[self.x], [self.y], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    @staticmethod
    def _pick_random_color():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return color

    def predict(self):
        prediction = self.kf.predict()
        self.x, self.y = int(prediction[0].item()), int(prediction[1].item())

    def correct_prediction(
        self, bbox: tuple[int, int, int, int] | None, frame_num: int
    ) -> None:
        """
        Correct the predicted position using a new detection.

        Args:
            bbox (tuple): Detected bounding box as (x, y, w, h).
                          If None, no correction is applied.
            frame_num (int): Frame number of the current detection.
        """
        if bbox is None:
            return
        detected_x = bbox[0] + bbox[2] // 2
        detected_y = bbox[1] + bbox[3] // 2
        measurement = np.array([[np.float32(detected_x)], [np.float32(detected_y)]])
        self.kf.correct(measurement)
        self.x = measurement[0, 0]
        self.y = measurement[1, 0]
        self.box = bbox
