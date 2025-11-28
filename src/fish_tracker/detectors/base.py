# detectors/base.py

from __future__ import annotations
from typing import Generic, TypeVar, Any, Optional
from logging import Logger
import numpy as np
from numpy.typing import NDArray
from fish_tracker.utils.logger import get_logger
from fish_tracker.config import ObjectDetectorConfig
from fish_tracker.utils.roi_processor import ROIProcessor, ROIResult
from fish_tracker.utils.background import BackgroundSubtractor


C = TypeVar("C", bound=ObjectDetectorConfig)


class ObjectDetector(Generic[C]):
    """
    Base classes for detectors.
    """

    def __init__(self, config: C) -> None:
        self.logger: Logger = get_logger(self.__class__.__name__)
        self.config = config
        self.background_subtractor = BackgroundSubtractor()
        self.roi_processor = ROIProcessor(config.roi_processor_opts)
        # # these are per-frame values filled before calling process
        self.diff_image_rgb: Optional[NDArray[Any]] = None

    def compute_region_prosals(self) -> None:
        raise NotImplementedError

    def process(self, frame: NDArray[Any]) -> list[ROIResult]:
        """Top-level processing for a single frame. compute_region_prosals must
        set self.region_proposals and self.diff_image_rgb.
        """
        self.compute_region_prosals()
        self.non_max_suppression()
        self.logger.debug("%d proposals after nms", len(self.region_proposals))
        return self.roi_processor.process(frame, self.region_proposals)

    def non_max_suppression(self) -> None:
        """
        Non maximum suppression.
        - Keeps only the largest box in a group of overlapping boxes.
        - Removes boxes that cover more than `max_frame_coverage` of the frame.
        """
        boxes = np.array(self.region_proposals, dtype=float)
        if boxes.size == 0:
            self.region_proposals = []
            return

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        height = y2 - y1
        width = x2 - x1
        area = height * width

        assert self.diff_image_rgb is not None
        frame_h, frame_w = self.diff_image_rgb.shape[:2]
        max_area = frame_h * frame_w * self.config.max_frame_coverage
        min_area = frame_h * frame_w * self.config.min_frame_coverage
        max_width = frame_w * self.config.max_width_ratio
        max_height = frame_h * self.config.max_heigth_ratio

        mask = (
            (area > min_area)
            & (area < max_area)
            & (width < max_width)
            & (height < max_height)
        )
        if not np.any(mask):
            self.region_proposals = []
            return

        boxes = boxes[mask]
        x1, y1, x2, y2, area = x1[mask], y1[mask], x2[mask], y2[mask], area[mask]
        order = area.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ratio = inter / area[order[1:]]
            inds = np.where(ratio <= self.config.overlap_threshold)[0]
            order = order[inds + 1]

        self.region_proposals = [
            (int(boxes[k][0]), int(boxes[k][1]), int(boxes[k][2]), int(boxes[k][3]))
            for k in keep
        ]

    def dump_processed_frame(self, dump_dir: str, frame_num: int) -> None:
        assert self.diff_image_rgb is not None
        import cv2
        import os

        annotated_frame = self.diff_image_rgb.copy()
        for x, y, w, h in self.region_proposals:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_path = os.path.join(dump_dir, f"frame{frame_num:04d}.png")
        mask = (annotated_frame == [0, 0, 0]).all(axis=-1)
        annotated_frame[mask] = [255, 0, 255]
        cv2.imwrite(img_path, annotated_frame)
