import json
from torch import Tensor

from fish_tracker.core.tracker_matcher import Match, TrackerMatcher
from fish_tracker.utils.configs import TrackerConfig, TrackerManagerConfig
from fish_tracker.utils.logger import get_logger
from fish_tracker.utils.roi_processor import ROIResult
from fish_tracker.trackers.base import BaseTracker

class TrackerManager(TrackerMatcher):

    def __init__(self, config:TrackerManagerConfig) -> None:
        super().__init__(config)
        self.logger = get_logger("TrackerManager")
        self.logger.info("TrackerManager Initialization")
        self.max_absences: int = config.max_absences
        self.min_tracking_duration = config.min_tracking_duration
        self.fps: float = config.fps
        self.input_dir = "/app/data/input"
        self.output_dir = "/app/data/outputs"
        self.input_video_path: str = f"/app/data/inputs/{config.input_video_name}"
        self.output_json_path: str = f"/app/data/outputs/{config.output_json_name}"
        self.motion_boxes: list[tuple[int, int, int, int]] = []
        self.embeddings: list[Tensor] = []
        self.trackers: list[BaseTracker] = []
        self.finished: list[BaseTracker] = []
        self.trashed: list[BaseTracker] = []
    
        if config.tracking_method.lower() == 'kalmanfilter':
            from fish_tracker.trackers.kalman_tracker import KalmanObjectTracker
            self.tracker_class = KalmanObjectTracker
        else:
            raise ValueError(
                f"Tracking method {config.tracking_method} is not implemented yet."
            )
        # if config.roi_filter_name is not None:
        #     self.roi_filter = build_roi_filter(config.roi_filter_name)
        # else:
        #     self.roi_filter = None


    # def filter_roi(self, roi_list, frame_num):
    #     from pathlib import Path
    #     from PIL import Image
    #     output_dir = Path(f"/app/data/outputs/crops/frame_{frame_num}")
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     filtered = []
    #     model, preprocess = clip.load("ViT-B/32", device="cpu")
    #     # text = clip.tokenize(["a photo of a fish.", "a photo of something which is not a fish."]).to("cpu")
    #     text = clip.tokenize([
    #         "A small low-quality underwater image region showing a moving living organism like a fish, even if only partly visible",
    #         "A small low-quality underwater region showing a still or non-living object",
    #     ])
    #     text = clip.tokenize([
    #         "A small low-quality underwater image region showing a living animal", #showing a living being, such as a fish, even if partly hidden",
    #         "A small low-quality underwater image region showing only water or a non-living object"
    #     ])

    #     with torch.no_grad():
    #         text_features = model.encode_text(text)
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #     for i, item in enumerate(roi_list):
    #         filename = output_dir / f"crop_{i}.png"
    #         item["crop"].save(filename)
    #         image = preprocess(item["crop"]).unsqueeze(0)
    #         with torch.no_grad():
    #             image_features = model.encode_image(image)
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #         similarity = (image_features @ text_features.T).squeeze(0) #.softmax(dim=-1)
    #         similarity = similarity.cpu().numpy()
    #         label = int(np.argmax(similarity))
    #         self.logger.debug(f'{i} label: {label}, similarity: {similarity},  ROI size: {item["crop"].size}')
    #         if label == 0:
    #             filtered.append(item)
    #     return filtered

    def handle_matched_trackers(
            self,
            matches: list[Match],
            frame_num: int
        ) -> None:

        for match in matches:
            match["tracker"].absences = 0
            match["tracker"].correct_prediction(match["detection"].bbox, frame_num)
            match["tracker"].embedding = match["detection"].embedding

    def handle_unmatched_trackers(
            self,
            unassigned_trackers: list[BaseTracker],
            frame_num: int,
            curr_time: float
        ):

        for _tracker in unassigned_trackers:
            _tracker.absences += 1
            self.logger.debug(f"{_tracker.absences} absences, max {self.max_absences}")
            if _tracker.absences >= self.max_absences:
                _tracker.terminate(frame_num, curr_time)
                self.trackers = [t for t in self.trackers if t.id != _tracker.id]
                self.logger.debug("terminated")
                if _tracker.duration > self.min_tracking_duration:
                    self.finished.append(_tracker)
                else:
                    self.logger.debug("trashed")
                    self.trashed.append(_tracker)

    def initialize_new_trackers(
            self,
            unassigned_detections: list[ROIResult],
            frame_num: int,
            curr_time: float
        ):

        for _detection in unassigned_detections:
            x0 = _detection.bbox[0] + _detection.bbox[2] / 2
            y0 = _detection.bbox[1] + _detection.bbox[3] / 2
            tracker_config = TrackerConfig(
                start_frame=frame_num,
                start_time=curr_time,
                x0=x0,
                y0=y0,
                embedding=_detection.embedding
            ) 
            tracker = self.tracker_class(tracker_config)
            self.logger.debug(f"Initialized tracker {tracker.id}")
            self.trackers.append(tracker)

    def process(
        self,
        frame_num: int,
        frame_roi: list[ROIResult],
        curr_time: float,
        step: int
    ) -> None:
    
        # if frame_roi and self.roi_filter is not None:
        #     frame_roi = self.roi_filter.process(frame_roi, frame_num)

        self.frame_num = frame_num
        self.curr_time = curr_time

        # Predictions.
        for _tracker in self.trackers:
            _tracker.predict()

        if (frame_num % step) == 0:
            self.logger.debug(
                "Running trackers: %s",
                [(_t.id, _t.absences) for _t in self.trackers],
            )
            self.logger.debug(f"{len(frame_roi)} ROI")

            # Map boxes with trackers.
            (matches, unassigned_trackers, unassigned_detections) = (
                self.make_associations(self.trackers, frame_roi)
            )

            self.handle_matched_trackers(matches, frame_num)

            # Handle unassigned trackers.
            self.handle_unmatched_trackers(unassigned_trackers, frame_num, curr_time)

            # Initialize new trackers.
            self.initialize_new_trackers(unassigned_detections, frame_num, curr_time)

        for _tracker in self.trackers:
            _tracker.update_trajectory(frame_num)

    def terminate(
        self,
        frame_num: int,
        curr_time: float
    ) -> None:

        for _tracker in self.trackers:
            self.logger.debug(f"Terminated tracker {_tracker.id}")
            _tracker.terminate(frame_num, curr_time)
            if _tracker.duration > self.min_tracking_duration:
                self.finished.append(_tracker)
            else:
                self.trashed.append(_tracker)

    def save_results(
            self,
            motion_boxes: dict[int, list[tuple[int, int, int, int]]],
            time: float
        ) -> None:
        """
        Saves tracking results (valid and rejected trackers).
        """

        def serialize_tracker(tracker: BaseTracker):
            return {
                "_id": getattr(tracker, "_id", None),
                "duration": getattr(tracker, "duration", None),
                "start_frame": getattr(tracker, "start_frame", None),
                "end_frame": getattr(tracker, "end_frame", None),
                "nb_absences": getattr(tracker, "absences", None),
                "trajectory": getattr(tracker, "trajectory", None),
                "color": getattr(tracker, "color", None),
            }

        result_data = {
            "nb_detected": len(self.finished),
            "nb_trashed": len(self.trashed),
            "current_time": time,
            "motion_boxes": motion_boxes,
        }

        if self.finished:
            result_data["Detection"] = [serialize_tracker(t) for t in self.finished]
        else:
            result_data["Detection"] = []

        if self.trashed:
            result_data["Trash"] = [serialize_tracker(t) for t in self.trashed]
        else:
            result_data["Trash"] = []

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
