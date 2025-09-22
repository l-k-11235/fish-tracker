import cv2
import numpy as np

from fish_tracker.utils.logger import get_logger


class BackgroundSubtractor:

    def __init__(self):
        self.logger = get_logger("BackgroundSubstractor")
        self.logger.info("BackgroundSubstractor Initialization")
        self.ref = None
        self.motion_mask = None
        self.mask_sum = None
        self.boxes = []

    def get_motion_mask(self, frame, thresh_val=30, min_area=500):
        """
        Calculates a binary mask of the motion areas.
        Args:
            frame: current image (BGR).
            thresh_val: intensity threshold for detecting movement.
            min_area: minimum surface to keep a blob.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_gray = (
            cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
            if len(self.ref.shape) == 3
            else self.ref
        )

        diff = cv2.absdiff(gray, background_gray)

        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, motion_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        self.motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        self.mask_sum = self.motion_mask.sum()

        # Detect contours.
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.boxes = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                self.boxes.append((x, y, w, h))
