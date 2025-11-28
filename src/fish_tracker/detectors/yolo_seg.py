# detectors/yolo_seg.py
from pathlib import Path

from .base import ObjectDetector
from fish_tracker.config import YOLOSegDetectorConfig
from fish_tracker.utils.roi_processor import ROIResult


class YOLOSegDetector(ObjectDetector[YOLOSegDetectorConfig]):
    """
    YOLO segmentation detector.
    """

    def __init__(self, config: YOLOSegDetectorConfig) -> None:
        super().__init__(config)
        # lazy import to avoid heavy cost at module import time
        from ultralytics import YOLO

        self.yolo_model = YOLO(self.config.yolo_model_path)
        # placeholders used in compute
        self.frame_results = None
        self.mask_motion = None

    def compute_region_prosals(self) -> None:
        assert self.frame_results is not None
        assert self.mask_motion is not None
        self.region_proposals: list[tuple[int, int, int, int]] = []
        import cv2

        for mask, box, _ in zip(
            self.frame_results.masks.data,
            self.frame_results.boxes.xyxy,
            self.frame_results.boxes.conf,
        ):
            mask_box = (mask.cpu().numpy() * 255).astype("uint8")
            mask_box = cv2.resize(
                mask_box,
                (self.mask_motion.shape[1], self.mask_motion.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            iou = (
                (self.mask_motion & mask_box).sum() / mask_box.sum()
                if mask_box.sum() > 0
                else 0.0
            )
            if iou > self.config.iou_threshold:
                x1, y1, x2, y2 = map(int, box)
                self.region_proposals.append((x1, y1, x2 - x1, y2 - y1))

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

        batch_frames = []
        batch_frame_nums = []
        batch_diff = []
        batch_masks_motion = []
        detections: dict[int, list[ROIResult]] = {}

        for frame_num, frame, ref_frame in frames_generator(
            video_path, start, end, step, ref_frame_path
        ):
            mask_motion, diff_image_rgb = self.background_subtractor.apply(
                frame, ref_frame
            )
            batch_mask = mask_motion
            batch_frames.append(frame)
            batch_diff.append(diff_image_rgb)
            batch_masks_motion.append(batch_mask)
            batch_frame_nums.append(frame_num)

        # do a batch predict on the diffs
        batch_results = self.yolo_model.predict(
            source=batch_diff,
            batch=len(batch_diff),
            conf=self.config.yolo_conf_thresh,
            device="cpu",
            half=False,
            verbose=False,
        )
        lengths = {
            "batch_results": len(batch_results),
            "batch_frames": len(batch_frames),
            "batch_frame_nums": len(batch_frame_nums),
            "batch_masks_motion": len(batch_masks_motion),
            "batch_diff": len(batch_diff),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(f"Batch size mismatch: {lengths}")

        for frame, frame_num, frame_results, mask_motion, diff in zip(
            batch_frames,
            batch_frame_nums,
            batch_results,
            batch_masks_motion,
            batch_diff,
        ):
            self.mask_motion = mask_motion
            self.diff_image_rgb = diff
            self.frame_results = frame_results
            detections[frame_num] = self.process(frame)
            if dump_dir is not None:
                self.dump_processed_frame(dump_dir, frame_num)
        return detections
