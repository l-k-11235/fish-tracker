# detectors/selective_search.py
"""
Selective Search detector implementation.
"""
import cv2

from pathlib import Path
from typing import Sequence

from .base import ObjectDetector
from fish_tracker.utils.configs import SelectiveSearchDetectorConfig
from fish_tracker.utils.roi_processor import ROIResult


class SelectiveSearchDetector(ObjectDetector[SelectiveSearchDetectorConfig]):
    def __init__(self, config: SelectiveSearchDetectorConfig) -> None:
        super().__init__(config)

    def compute_region_prosals(self) -> None:
        assert self.diff_image_rgb is not None
        frame = self.diff_image_rgb
        if self.config.resize_factor != 1.0:
            small_frame = cv2.resize(
                frame,
                (0, 0),
                fx=self.config.resize_factor,
                fy=self.config.resize_factor,
            )
        else:
            small_frame = frame

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(small_frame)
        ss.switchToSelectiveSearchFast()
        region_proposals: Sequence[Sequence[int]] = ss.process()

        self.region_proposals: list[tuple[int, int, int, int]] = [
            (
                int(x / self.config.resize_factor),
                int(y / self.config.resize_factor),
                int(w / self.config.resize_factor),
                int(h / self.config.resize_factor),
            )
            for (x, y, w, h) in region_proposals
        ]

    def process_chunk(
        self,
        video_path: Path,
        start: int,
        end: int,
        step: int,
        ref_frame_path: Path | None,
        dump_dir: Path | None,
    ) -> dict[int, list[ROIResult]]:
        from .worker import frames_generator

        detections: dict[int, list[ROIResult]] = {}
        for frame_num, frame, ref_frame in frames_generator(
            video_path, start, end, step, ref_frame_path
        ):
            _, diff = self.background_subtractor.apply(frame, ref_frame)
            self.diff_image_rgb = diff
            detections[frame_num] = self.process(frame)
            if dump_dir is not None:
                self.dump_processed_frame(dump_dir, frame_num)
        return detections
