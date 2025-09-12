import cv2
import multiprocessing as mp
import numpy as np
import os

from fish_tracker.utils.background import BackgroundSubtractor
from fish_tracker.utils.logger import get_logger


class ObjectDetector(BackgroundSubtractor):

    def __init__(
        self,
        overlap_threshold=0.1,
        resize_factor=0.25,
        max_frame_coverage=0.20,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.logger = get_logger("ObjectDetector")
        self.logger.info("Detector initialization")
        self.overlap_threshold = overlap_threshold
        self.resize_factor = resize_factor
        self.max_frame_coverage = max_frame_coverage
        self.motion_mask = None
        self.region_proposals = []
        self.masked_frame = None

    def selective_search(self, frame):
        self.logger.debug("Selective search...")
        if self.resize_factor != 1.0:
            # Resize frames to speed up selective search.
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
        self.logger.info(f"{len(self.region_proposals)} proposals")

    def non_max_suppression(self, frame, max_frame_coverage=0.20):
        """
        Non maximum suppression.
        - Keeps only the largest box in a group of overlapping boxes.
        - Removes boxes that cover more than `max_frame_coverage` of the frame.
        """
        boxes = np.array(self.region_proposals, dtype=float)
        if boxes.size == 0:
            return []

        # corners
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]

        # areas
        areas = (x2 - x1) * (y2 - y1)

        # frame coverage filtering
        frame_area = frame.shape[0] * frame.shape[1]
        mask = areas < max_frame_coverage * frame_area
        if not np.any(mask):
            self.region_proposals = []
            return []

        boxes = boxes[mask]
        x1, y1, x2, y2, areas = x1[mask], y1[mask], x2[mask], y2[mask], areas[mask]

        # sort boxes by descending area
        order = areas.argsort()[::-1]
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
            ratio = inter / np.minimum(areas[i], areas[order[1:]])
            inds = np.where(ratio <= self.overlap_threshold)[0]

            # update remaining boxes
            order = order[inds + 1]

        # cast back to int
        self.region_proposals = [tuple(map(int, boxes[k])) for k in keep]
        self.logger.debug(f"{len(self.region_proposals)} after non max suppression")

        return self.region_proposals

    def process_frame(self, frame):
        # get the motion mask
        self.get_motion_mask(frame)
        # apply the motion mask to the frame
        mask_RGB = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2BGR)
        self.masked_frame = cv2.bitwise_and(frame, mask_RGB)
        mask_black = np.all(self.masked_frame < 5, axis=-1)
        self.masked_frame[mask_black] = (203, 192, 255)  # lightpink
        # apply selective search to the masked frame
        self.selective_search(self.masked_frame)
        # non maximum suppression
        self.non_max_suppression(self.masked_frame)


def init_worker(ref_frame):
    global detector
    detector = ObjectDetector()
    detector.ref = ref_frame


def detect_fishes_worker(frame_args):
    frame_num, frame = frame_args
    detector.process_frame(frame)

    frame = detector.masked_frame

    for x, y, w, h in detector.region_proposals:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame_num, detector.region_proposals, frame


def detect_fishes_parallel(
    cap,
    first_frame,
    start,
    end,
    step,
    dump_masked_frames=False,
    out_dir="/app/data/outputs",
    in_dir="/app/data/inputs",
    num_workers=8,
):

    tasks = []
    detections = {}

    frame_num = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    if dump_masked_frames:
        dump_dir = os.path.join(out_dir, "masked_frame")
        os.makedirs(dump_dir, exist_ok=True)

    while frame_num < end:
        success, frame = cap.read()
        if not success:
            break

        if frame_num == start:
            if first_frame is None:
                reference_frame = frame.copy()
            else:
                reference_frame = cv2.imread(f"{in_dir}/{first_frame}")

        elif frame_num % step == 0:
            tasks.append((frame_num, frame.copy()))

        frame_num += 1

    with mp.Pool(
        num_workers, initializer=init_worker, initargs=(reference_frame,)
    ) as pool:
        for frame_num, region_proposals, processed_frame in pool.imap_unordered(
            detect_fishes_worker, tasks
        ):

            detections[frame_num] = region_proposals
            if dump_masked_frames:
                img_path = os.path.join(dump_dir, f"masked_frame{frame_num:04d}.png")
                cv2.imwrite(img_path, processed_frame)
    return detections


# 49s // 20 frames resize factor 0.5
# 20s // 20 frames resize factor 0.25
