import cv2
import numpy as np

from fish_tracker.utils.logger import get_logger


class BackgroundSubtractor:

    def __init__(self):
        self.logger = get_logger("BackgroundSubstractor")
        self.logger.info("BackgroundSubstractor Initialization")

    def apply(self, frame, ref_frame):
        background_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_image = cv2.absdiff(gray, background_gray)
        thresh_val = np.percentile(diff_image, 85)
        _, mask_motion = cv2.threshold(diff_image, thresh_val, 255, cv2.THRESH_BINARY)
        coverage_ratio = np.count_nonzero(mask_motion) / mask_motion.size
        self.logger.debug(f"Motion mask coverage ratio : {coverage_ratio * 100:.2f}%")
        diff_image_rgb = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
        return mask_motion, diff_image_rgb
