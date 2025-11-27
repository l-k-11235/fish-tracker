from logging import Logger
import cv2
import numpy as np

from numpy.typing import NDArray
from typing import Any

from fish_tracker.utils.logger import get_logger


class BackgroundSubtractor:

    def __init__(self) -> None:
        self.logger: Logger = get_logger("BackgroundSubstractor")
        self.logger.info("BackgroundSubstractor Initialization")

    def apply(
        self, frame: NDArray[Any], ref_frame: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any]]:

        if frame.shape != ref_frame.shape:
            self.logger.error(
                f"Frame shape mismatch: "
                f"frame={frame.shape}, "
                f"ref_frame={ref_frame.shape}"
            )
            raise ValueError("Input frames must have the same shape")
        background_gray: NDArray[Any] = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_image: NDArray[Any] = cv2.absdiff(gray, background_gray)
        thresh_val: float = float(np.percentile(diff_image, 85))
        _, mask_motion = cv2.threshold(diff_image, thresh_val, 255, cv2.THRESH_BINARY)
        coverage_ratio: float = float(np.count_nonzero(mask_motion) / mask_motion.size)
        self.logger.debug(f"Motion mask coverage ratio : {coverage_ratio * 100:.2f}%")
        diff_image_rgb = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
        return mask_motion, diff_image_rgb
