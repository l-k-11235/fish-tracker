import cv2
import multiprocessing as mp
import numpy as np
import os

from typing import Generator, Tuple, Optional
from ultralytics import YOLO

from fish_tracker.utils.background import BackgroundSubtractor
from fish_tracker.utils.embeddings import build_embedding_model
from fish_tracker.utils.logger import get_logger


class ObjectDetector:

    def __init__(
        self,
        embedding_model_name="mobilenetv3small",
        overlap_threshold=0.1,
        resize_factor=0.25,
        max_frame_coverage=0.20,
    ):
        self.logger = get_logger("ObjectDetector")

        self.background_subtractor = BackgroundSubtractor()
        self.embedding_generator = build_embedding_model(name=embedding_model_name)

        self.overlap_threshold = overlap_threshold
        self.resize_factor = resize_factor
        self.max_frame_coverage = max_frame_coverage
        self.diff_image_rgb = None
        self.region_proposals = []

    def process(self):
        raise NotImplementedError

    def non_max_suppression(
        self,
        max_frame_coverage=0.10,
        min_frame_coverage=0.005,
        max_width_ratio=0.3,
        max_heigth_ratio=0.3,
    ):
        """
        Non maximum suppression.
        - Keeps only the largest box in a group of overlapping boxes.
        - Removes boxes that cover more than `max_frame_coverage` of the frame.
        """
        boxes = np.array(self.region_proposals, dtype=float)
        if boxes.size == 0:
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
        frame_h, frame_w = self.diff_image_rgb.shape[:2]
        max_area = frame_h * frame_w * max_frame_coverage
        min_area = frame_h * frame_w * min_frame_coverage
        max_width = frame_w * max_width_ratio
        max_height = frame_h * max_heigth_ratio

        # filtering mask
        mask = (
            (area > min_area)
            & (area < max_area)
            & (width < max_width)
            & (height < max_height)
        )
        if not np.any(mask):
            self.region_proposals = []
        boxes = boxes[mask]
        x1, y1, x2, y2, area = x1[mask], y1[mask], x2[mask], y2[mask], area[mask]

        # sort boxes by descending area
        order = area.argsort()[::-1]
        keep = []
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
            inds = np.where(ratio <= self.overlap_threshold)[0]
            self.logger.debug(f"# inds {inds}")
            # update remaining boxes
            order = order[inds + 1]
            self.logger.debug(f"# order {order}")

        # cast back to int
        self.region_proposals = [tuple(map(int, boxes[k])) for k in keep]
        self.logger.debug(f"{len(self.region_proposals)} after non max suppression")

        return self.region_proposals

    def dump_processed_frame(self, dump_dir, frame_num):
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
        reference_frame_num: Optional[int],
        reference_frame_path: Optional[str],
        dump_dir: Optional[str],
    ):
        raise NotImplementedError


class SelectiveSearchDetector(ObjectDetector):

    def selective_search(self):
        self.logger.debug("Selective search...")
        frame = self.diff_image_rgb
        if self.resize_factor != 1.0:
            small_frame = cv2.resize(
                frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor
            )
        else:
            small_frame = frame
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(small_frame)
        ss.switchToSelectiveSearchFast()
        region_proposals = ss.process()

        self.region_proposals = [
            (
                int(x / self.resize_factor),
                int(y / self.resize_factor),
                int(w / self.resize_factor),
                int(h / self.resize_factor),
            )
            for (x, y, w, h) in region_proposals
        ]
        self.logger.debug("done!")

    def process(self):
        self.selective_search()
        self.non_max_suppression()
        self.embedding_generator.get_crops_embeddings(
            self.diff_image_rgb, self.region_proposals
        )

    def process_chunk(
        self,
        video_path: str,
        start: int,
        end: int,
        step: int,
        reference_frame_num: Optional[int],
        reference_frame_path: Optional[str],
        dump_dir: Optional[str],
    ):
        detections = {}
        draw_roi = True if dump_dir is not None else False

        for frame_num, frame, ref_frame in video_frames_generator(
            video_path=video_path,
            start=start,
            end=end,
            step=step,
            reference_frame_num=reference_frame_num,
            reference_frame_path=reference_frame_path,
        ):
            _, diff_image_rgb = self.background_subtractor.apply(frame, ref_frame)
            self.diff_image_rgb = diff_image_rgb
            self.process()
            detections[frame_num] = [
                {"box": _box, "embedding": _emb}
                for _box, _emb in zip(
                    self.region_proposals, self.embedding_generator.crops_embeddings
                )
            ]
            if draw_roi:
                self.dump_processed_frame(dump_dir, frame_num)
        return detections


class YOLOSegDetector(ObjectDetector):
    def __init__(
        self,
        embedding_model_name,
        yolo_model_path,
        yolo_conf_thresh=0.0005,
        iou_threshold=0.2,
    ):
        super().__init__(embedding_model_name)
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_conf_thresh = yolo_conf_thresh
        self.iou_threshold = iou_threshold
        self.frame_results = None

    def yolo_seg_region_proposal(self):
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
            if iou > self.iou_threshold:
                x1, y1, x2, y2 = map(int, box)
                self.region_proposals.append((x1, y1, x2 - x1, y2 - y1))

    def process(self):
        self.yolo_seg_region_proposal()
        self.non_max_suppression()
        self.embedding_generator.get_crops_embeddings(
            self.diff_image_rgb, self.region_proposals
        )

    def process_chunk(
        self,
        video_path: str,
        start: int,
        end: int,
        step: int,
        reference_frame_num: Optional[int],
        reference_frame_path: Optional[str],
        dump_dir: Optional[str],
    ):

        detections = {}
        draw_roi = True if dump_dir is not None else False
        batch_frames, batch_frame_nums = [], []
        batch_diff, batch_masks_motion = [], []

        for frame_num, frame, ref_frame in video_frames_generator(
            video_path=video_path,
            start=start,
            end=end,
            step=step,
            reference_frame_num=reference_frame_num,
            reference_frame_path=reference_frame_path,
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
            conf=self.yolo_conf_thresh,
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
            self.frame_results = frame_results
            self.process()
            detections[frame_num] = [
                {"box": _box, "embedding": _emb}
                for _box, _emb in zip(
                    self.region_proposals, self.embedding_generator.crops_embeddings
                )
            ]
            if draw_roi:
                self.dump_processed_frame(dump_dir, frame_num)

        return detections


def video_frames_generator(
    video_path: str,
    start: int,
    end: int,
    step: int = 1,
    reference_frame_num: Optional[int] = None,
    reference_frame_path: Optional[str] = None,
) -> Generator[Tuple[int, any, any], None, None]:
    """
    Generator that yields frames from a video between frame indices [start, end),
    sampled every `step` frames. Each yielded item is a tuple:
        (frame_num, frame, ref_frame)

    Parameters
    ----------
    video_path : str
        Path to the video file.
    start : int
        Starting frame index (inclusive).
    end : int
        Ending frame index (exclusive).
    step : int, optional
        Interval between consecutive frames (default: 1).
    reference_frame_num : int, optional
        Index of the reference frame to use. If None, defaults to `step`.
    reference_frame_path : str, optional
        Path to an image file used as the reference frame instead of extracting
        one from the video.

    Yields
    ------
    (frame_num, frame, ref_frame) : tuple
        frame_num : int
            The current frame index.
        frame : np.ndarray
            The current frame (BGR).
        ref_frame : np.ndarray
            The reference frame (BGR).
    """
    # Load reference frame
    if reference_frame_path is not None:
        ref_frame = cv2.imread(reference_frame_path)
        if ref_frame is None:
            raise ValueError(
                f"Could not read reference frame from {reference_frame_path}"
            )
    else:
        cap_ref = cv2.VideoCapture(video_path)
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num or step)
        success, ref_frame = cap_ref.read()
        cap_ref.release()
        if not success:
            raise ValueError(
                f"Could not read reference frame at index {reference_frame_num or step}"
            )

    # Read main video frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_num = start

    while frame_num < end:
        success, frame = cap.read()
        if not success:
            break
        if frame_num % step == 0:
            yield frame_num, frame, ref_frame
        frame_num += 1

    cap.release()


def _process_chunk(detector_class, detector_kwargs, *args):
    detector = detector_class(**detector_kwargs)
    return detector.process_chunk(*args)


def run_roi_detection(
    video_path,
    detector_type,
    start,
    end,
    step,
    output_dir,
    reference_frame_path=None,
    dump_masked_frames=False,
    num_workers=8,
    chunk_size=8,
    detector_kwargs=None,
):
    reference_frame_num = step if reference_frame_path is None else None
    dump_dir = None
    if dump_masked_frames:
        dump_dir = os.path.join(output_dir, "masked_frames")
        os.makedirs(dump_dir, exist_ok=True)

    detector_class = (
        SelectiveSearchDetector
        if detector_type == "selective_search"
        else YOLOSegDetector
    )

    chunks = []
    for chunk_start in range(start, end, step * chunk_size):
        chunk_end = min(chunk_start + step * chunk_size, end)
        chunks.append(
            (
                video_path,
                chunk_start,
                chunk_end,
                step,
                reference_frame_num,
                reference_frame_path,
                dump_dir,
            )
        )
    all_detections = {}

    if detector_type == "selective_search":
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                _process_chunk,
                [(detector_class, detector_kwargs, *chunk) for chunk in chunks],
            )
            for d in results:
                all_detections.update(d)
    else:
        for chunk in chunks:
            all_detections.update(
                _process_chunk(detector_class, detector_kwargs, *chunk)
            )

    return all_detections
