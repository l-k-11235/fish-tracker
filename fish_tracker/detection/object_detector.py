import cv2

from fish_tracker.utils.background import BackgroundSubtractor
from fish_tracker.utils.logger import get_logger


class ObjectDetector(BackgroundSubtractor):

    def __init__(self, min_contour_area=2000, log_level="INFO", **kwargs):

        super().__init__(**kwargs)

        self.logger = get_logger("ObjectDetector")
        self.logger.info("Detector initialization")

        self.min_contour_area = min_contour_area
        self.contours = []
        self.contour_areas = []

    def detect_contours(self):
        contours, _ = cv2.findContours(
            self.delta_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = [
            _c for _c in contours if cv2.contourArea(_c) > self.min_contour_area
        ]

    def process_frame(self, frame):
        self.set_delta_frame(frame)
        self.detect_contours()
        self.logger.debug(f"Detected {len(self.contours)} contours")
        return self.contours
