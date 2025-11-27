from logging import Logger
import cv2
import inspect
import multiprocessing as mp
from cv2.ximgproc.segmentation import SelectiveSearchSegmentation
import numpy as np
import os

from numpy.typing import NDArray
from typing import cast, Any, Generator, Generic, Literal, Optional, Sequence, TypeVar

from ultralytics.engine.results import Results

from fish_tracker.utils.background import BackgroundSubtractor
from fish_tracker.utils.configs import FullConfig, ObjectDetectorConfig, SelectiveSearchDetectorConfig, YOLOSegDetectorConfig
from fish_tracker.utils.roi_processor import ROIProcessor, ROIResult
from fish_tracker.utils.logger import get_logger

C = TypeVar("C", bound=ObjectDetectorConfig)

class ObjectDetector(Generic[C]):

    def __init__(self, config: C) -> None:
        self.logger: Logger = get_logger("ObjectDetector")

        self.background_subtractor = BackgroundSubtractor()
        self.roi_processor = ROIProcessor(config.roi_processor_opts)
        self.diff_image_rgb: NDArray[Any] | None = None
        self.region_proposals: list[tuple[int, int, int, int]] = []
        self.config: C = config

    def compute_region_prosals(self) ->  None:
        raise NotImplementedError

    def process(self, frame: NDArray[Any]) -> list[ROIResult]:
        self.compute_region_prosals()
        self.non_max_suppression()
        self.logger.debug(f"{len(self.region_proposals)} after non max suppression")
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

        # corners
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]

        # box dimensions
        height = y2 - y1
        width = x2 - x1
        area = height * width

        # compute constraints
        assert self.diff_image_rgb is not None
        frame_h, frame_w = self.diff_image_rgb.shape[:2]
        max_area = frame_h * frame_w * self.config.max_frame_coverage
        min_area = frame_h * frame_w * self.config.min_frame_coverage
        max_width = frame_w * self.config.max_width_ratio
        max_height = frame_h * self.config.max_heigth_ratio

        # filtering mask
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
        # sort boxes by descending area
        order = area.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # intersections with the other (smaller) boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            # coverage ratio based on the small boxes
            ratio = inter / area[order[1:]]
            inds = np.where(ratio <= self.config.overlap_threshold)[0]
            self.logger.debug(f"# inds {inds}")
            # update remaining boxes
            order = order[inds + 1]
            self.logger.debug(f"# order {order}")
        # cast back to int
        self.region_proposals = [
            (int(boxes[k][0]), int(boxes[k][1]), int(boxes[k][2]), int(boxes[k][3]))
            for k in keep
        ]

    def dump_processed_frame(self, dump_dir: str, frame_num: int):
        assert self.diff_image_rgb is not None
        annotated_frame = self.diff_image_rgb.copy()
        for x, y, w, h in self.region_proposals:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_path = os.path.join(dump_dir, f"frame{frame_num:04d}.png")
        mask = np.all(annotated_frame == [0, 0, 0], axis=-1)
        annotated_frame[mask] = [255, 0, 255]
        cv2.imwrite(img_path, annotated_frame)

    def process_chunk(
        self,
        video_path: str,
        start: int,
        end: int,
        step: int,
        ref_frame_path: str | None,
        dump_dir: str | None,
    ) -> dict[int, list[ROIResult]]:
        raise NotImplementedError


class SelectiveSearchDetector(ObjectDetector[SelectiveSearchDetectorConfig]):

    def __init__(self, config: SelectiveSearchDetectorConfig) -> None:
        super().__init__(config)

    def compute_region_prosals(self) -> None:
        self.logger.debug("Selective search...")
        assert self.diff_image_rgb is not None
        frame: NDArray[Any] = self.diff_image_rgb
        if self.config.resize_factor != 1.0:
            small_frame: NDArray[Any] = cv2.resize(
                frame, (0, 0), fx=self.config.resize_factor, fy=self.config.resize_factor
            )
        else:
            small_frame = frame
        ss: SelectiveSearchSegmentation = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(small_frame)
        ss.switchToSelectiveSearchFast()
        region_proposals: Sequence[Sequence[int]]= ss.process()
 
        self.region_proposals: list[tuple[int, int, int, int]]= [
            (
                int(x / self.config.resize_factor),
                int(y / self.config.resize_factor),
                int(w / self.config.resize_factor),
                int(h / self.config.resize_factor),
            )
            for (x, y, w, h) in region_proposals
        ]
        self.logger.debug("done!")

    def process_chunk(
        self,
        video_path: str,
        start: int,
        end: int,
        step: int,
        ref_frame_path: str | None,
        dump_dir: str | None,
    ) -> dict[int, list[ROIResult]]:
        detections: dict[int, list[ROIResult]] = {}
        for frame_num, frame, ref_frame in frames_generator(
            video_path,
            start,
            end,
            step,
            ref_frame_path,
        ):
            _, diff_image_rgb = self.background_subtractor.apply(frame, ref_frame)
            self.diff_image_rgb = diff_image_rgb
            detections[frame_num] = self.process(frame)
            if dump_dir is not None:
                self.dump_processed_frame(dump_dir, frame_num)
        return detections


class YOLOSegDetector(ObjectDetector[YOLOSegDetectorConfig]):
    def __init__(
        self, config: YOLOSegDetectorConfig) -> None:
        super().__init__(config)
        from ultralytics import YOLO
        self.yolo_model: "YOLO" = YOLO(self.config.yolo_model_path)
        from ultralytics.engine.results import Results
        self.frame_results: Results = cast(Results, None)
    
    def compute_region_proposal(self) -> None:
        self.region_proposals = []
        for mask, box, _ in zip(
            self.frame_results.masks.data,
            self.frame_results.boxes.xyxy,
            self.frame_results.boxes.conf,
        ):
            mask_box = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_box = cv2.resize(
                mask_box,
                (self.mask_motion.shape[1], self.mask_motion.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            iou = (self.mask_motion & mask_box).sum() / mask_box.sum()
            if iou > self.config.iou_threshold:
                x1, y1, x2, y2 = map(int, box)
                self.region_proposals.append((x1, y1, x2 - x1, y2 - y1))

    def process_chunk(
        self,
        video_path: str,
        start: int,
        end: int,
        step: int,
        ref_frame_path: str | None,
        dump_dir: str | None,
    ) -> dict[int, list[ROIResult]]:

        detections: dict[int, list[ROIResult]] = {}
        batch_frames: list[NDArray[Any]] = []
        batch_frame_nums: list[int] = []
        batch_diff: list[NDArray[Any]] = []
        batch_masks_motion: list[NDArray[Any]] = []

        for frame_num, frame, ref_frame in frames_generator(
            video_path=video_path,
            start=start,
            end=end,
            step=step,
            ref_frame_path=ref_frame_path
        ):
            batch_frames.append(frame)
            mask_motion, diff_image_rgb = self.background_subtractor.apply(
                frame, ref_frame
            )
            batch_diff.append(diff_image_rgb)
            batch_masks_motion.append(mask_motion)
            batch_frame_nums.append(frame_num)

        batch_results = self.yolo_model.predict(
            source=batch_diff,
            batch=len(batch_frames),
            conf=self.config.yolo_conf_thresh,
            device="cpu",
            half=False,
            verbose=False,
        )

        for frame, frame_num, frame_results, mask_motion, diff in zip(
            batch_frames,
            batch_frame_nums,
            batch_results,
            batch_masks_motion,
            batch_diff,
        ):
            self.mask_motion = mask_motion
            self.diff_image_rgb = diff
            self.frame_results: Results = frame_results
            detections[frame_num] = self.process(frame)
            if dump_dir is not None:
                self.dump_processed_frame(dump_dir, frame_num)

        return detections


def frames_generator(
    video_path: str,
    start: int,
    end: int,
    step: int,
    ref_frame_path: Optional[str] = None,
) -> Generator[tuple[int, NDArray[Any], NDArray[Any]], None, None]:
    """
    Generator that yields frames from a video between frame indices [start, end),
    sampled every `step` frames. Each yielded item is a tuple:
        (frame_num, frame, ref_frame)
    """
    # Load reference frame
    if ref_frame_path is not None:
        ref_frame = cv2.imread(ref_frame_path)
        if ref_frame is None:
            raise ValueError(
                f"Could not read reference frame from {ref_frame_path}"
            )
    else:
        cap_ref = cv2.VideoCapture(video_path)
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, start)
        success, ref_frame = cap_ref.read()
        cap_ref.release()
        if not success:
            raise ValueError(
                f"Could not read reference frame at index {start}"
            )

    # Read main video frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + step)
    frame_num = start + step

    while frame_num < end:
        success, frame = cap.read()
        if not success:
            break
        if frame_num % step == 0:
            yield frame_num, frame, ref_frame
        frame_num += 1

    cap.release()


# def _process_chunk(
#         detector_type: Literal["selective_search", "yolo_seg"],
#         detector_opts: ObjectDetectorConfig,
#         video_path: str,
#         start: int,
#         end: int,
#         step: int,
#         ref_frame_path: str | None,
#         dump_dir: str | None,
#     )-> dict[int, list[ROIResult]]:
#     if detector_type == "selective_search":
#         assert type(detector_opts) == SelectiveSearchDetectorConfig
#         detector = SelectiveSearchDetector(detector_opts)
#     else:
#         assert type(detector_opts) == YOLOSegDetectorConfig
#         detector = YOLOSegDetector(detector_opts)
#     return detector.process_chunk(video_path, start, end, step, ref_frame_path, dump_dir)


# def run_roi_detection(config: FullConfig) -> dict[str, list[ROIResult]]:

#     reference_frame_num: int | None = config.step if config.ref_frame_path is None else None
#     dump_dir = None
#     if config.detector_opts.dump_masked_frames:
#         dump_dir = '/app/data/outputs/masked_frames'
#         os.makedirs(dump_dir, exist_ok=True)

#     chunks: list[tuple[str, int, int, int, int | None, str | None, str | None]] = []
#     start, step, end = config.start, config.step, config.end
#     chunk_size: int = config.detector_opts.chunk_size
#     for chunk_start in range(start, end, step * chunk_size):
#         chunk_end: int = min(chunk_start + step * chunk_size, end)
#         chunks.append(
#             (
#                 config.input_video_path,
#                 chunk_start,
#                 chunk_end,
#                 step,
#                 reference_frame_num,
#                 config.ref_frame_path,
#                 dump_dir,
#             )
#         )
#     all_detections: dict[int, list[ROIResult]] = {}

#     if config.detector_type == "selective_search":
#         assert type(config.detector_opts) == SelectiveSearchDetectorConfig
#         with mp.Pool(config.detector_opts.num_workers) as pool:
#             results: list[dict[int, list[ROIResult]]] = pool.starmap(
#                 _process_chunk,
#                 [(config.detector_type, config.detector_opts, *chunk) for chunk in chunks],
#             )
#             for d in results:
#                 all_detections.update(d)
#     else:
#         assert type(config.detector_opts) == YOLOSegDetectorConfig
#         for chunk in chunks:
#             all_detections.update(
#                 _process_chunk(config.detector_type, config.detector_opts, *chunk)
#             )

#     return all_detections
