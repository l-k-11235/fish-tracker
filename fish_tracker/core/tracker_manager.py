import json

from fish_tracker.utils.logger import get_logger
from fish_tracker.core.tracker_matcher import TrackerMatcher


class TrackerManager(TrackerMatcher):

    def __init__(
        self,
        max_absences,
        min_tracking_duration,
        fps,
        input_video_name,
        output_json_name,
        tracker_class=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.logger = get_logger("TrackerManager")
        self.logger.info("TrackerManager Initialization")

        self.max_absences = max_absences
        self.min_tracking_duration = min_tracking_duration
        self.fps = fps

        self.input_dir = "/app/data/input"
        self.output_dir = "/app/data/outputs"
        self.input_video_path = f"/app/data/inputs/{input_video_name}"
        self.output_json_path = f"/app/data/outputs/{output_json_name}"

        self.motion_boxes = []
        self.embeddings = []
        self.trackers = []
        self.finished = []
        self.trashed = []

        if tracker_class is None:
            from fish_tracker.trackers.kalman_tracker import KalmanObjectTracker

            self.tracker_class = KalmanObjectTracker
        else:
            self.tracker_class = tracker_class

    def handle_matched_trackers(self, matches, frame_num):

        for match in matches:
            match["tracker"].absences = 0
            match["tracker"].correct_prediction(match["detection"]["box"], frame_num)
            match["tracker"].embedding = match["detection"]["embedding"]

    def handle_unmatched_trackers(self, unassigned_trackers, frame_num, curr_time):

        for _tracker in unassigned_trackers:
            _tracker.absences += 1
            self.logger.debug(f"{_tracker.absences} absences, max {self.max_absences}")
            if _tracker.absences >= self.max_absences:
                _tracker.terminate(frame_num, curr_time)
                self.trackers = [t for t in self.trackers if t._id != _tracker._id]
                self.logger.debug("terminated")
                if _tracker.duration > self.min_tracking_duration:
                    self.finished.append(_tracker)
                else:
                    self.logger.debug("trashed")
                    self.trashed.append(_tracker)

    def initialize_new_trackers(self, unassigned_detections, frame_num, curr_time):

        for _detection in unassigned_detections:
            bbox = _detection["box"]
            x0 = bbox[0] + bbox[2] / 2
            y0 = bbox[1] + bbox[3] / 2
            tracker = self.tracker_class(
                start_frame=frame_num,
                start_time=curr_time,
                x0=x0,
                y0=y0,
                embedding=_detection["embedding"],
            )
            self.logger.debug(f"Initialized tracker {tracker._id}")
            self.trackers.append(tracker)

    def process(self, frame_num, curr_time, detections, step):

        self.frame_num = frame_num
        self.curr_time = curr_time

        # Predictions.
        for _tracker in self.trackers:
            _tracker.predict()

        if (frame_num % step) == 0:
            self.logger.debug(
                "Running trackers: %s",
                [(_t._id, _t.absences) for _t in self.trackers],
            )
            self.logger.debug(f"{len(detections)} detections")

            # Map boxes with trackers.
            (matches, unassigned_trackers, unassigned_detections) = (
                self.make_associations(self.trackers, detections)
            )

            self.handle_matched_trackers(matches, frame_num)

            # Handle unassigned trackers.
            self.handle_unmatched_trackers(unassigned_trackers, frame_num, curr_time)

            # Initialize new trackers.
            self.initialize_new_trackers(unassigned_detections, frame_num, curr_time)

        for _tracker in self.trackers:
            _tracker.update_trajectory(frame_num)

    def terminate(self, frame_num, curr_time):

        for _tracker in self.trackers:
            self.logger.debug(f"Terminated tracker {_tracker._id}")
            _tracker.terminate(frame_num, curr_time)
            if _tracker.duration > self.min_tracking_duration:
                self.finished.append(_tracker)
            else:
                self.trashed.append(_tracker)

    def save_results(self, motion_boxes, time):
        """
        Saves tracking results (valid and rejected trackers).
        """

        def serialize_tracker(tracker):
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
