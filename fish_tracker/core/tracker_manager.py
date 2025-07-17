import cv2
import json
import numpy as np
import os
import subprocess
from pathlib import Path

from fish_tracker.utils.logger import get_logger
from fish_tracker.core.tracker_matcher import TrackerMatcher


class TrackerManager(TrackerMatcher):

    def __init__(
        self,
        max_absences,
        min_tracking_duration,
        input_video_name,
        output_video_name,
        output_json_name,
        tracker_class=None,
        log_level='INFO',
        **kwargs
    ):

        super().__init__(log_level=log_level, **kwargs)

        self.logger = get_logger("TrackerManager")
        self.logger.info("TrackerManager Initialization")

        self.max_absences = max_absences
        self.min_tracking_duration = min_tracking_duration

        self.input_dir = '/app/data/input'
        self.output_dir = '/app/data/outputs'
        self.input_video_path = f'/app/data/inputs/{input_video_name}'
        self.output_video_path =f'/app/data/outputs/{output_video_name}'
        self.output_json_path = f'/app/data/outputs/{output_json_name}'

        self.trackers = []
        self.finished = []
        self.trashed = []
        self.contours = {}

        if tracker_class is None:
            from fish_tracker.trackers.kalman_tracker import KalmanObjectTracker
            self.tracker_class = KalmanObjectTracker
        else:
            self.tracker_class = tracker_class

    def handle_matched_trackers(self, associations, detected_boxes, frame_num):

        matched_trackers = []

        for _box_id, _tracker in associations.items():
            _tracker.absences = 0
            _tracker.correct_prediction(detected_boxes[_box_id], frame_num)
            matched_trackers.append(_tracker)

        return matched_trackers

    def handle_unmatched_trackers(self, unmatched_trackers_indices, frame_num, curr_time):
        unmatched_trackers = []
        for r in unmatched_trackers_indices:
            tracker = self.trackers[r]
            self.logger.debug(f'unmatched_tracker {tracker._id}')
            if tracker.absences  > self.max_absences:
                tracker.terminate(frame_num, curr_time)
                self.logger.debug("terminated")
                if  tracker.duration > self.min_tracking_duration:
                    self.finished.append(tracker)
                else:
                    self.logger.debug("trashed")
                    self.trashed.append(tracker)
            else:
                tracker.absences += 1
                unmatched_trackers.append(tracker)
        return unmatched_trackers

    def initialize_new_trackers(
        self, unmatched_detections, detected_boxes, frame_num, curr_time
    ):

        new_trackers = []
        for c in unmatched_detections:
            bbox = detected_boxes[c]
            x0 = bbox[0] + bbox[2] / 2
            y0 = bbox[1] + bbox[3] / 2
            tracker = self.tracker_class(frame_num, curr_time, x0, y0)
            self.logger.debug(f"Initialized tracker {tracker._id}")
            new_trackers.append(tracker)

        return new_trackers

    def process(self, frame_num, frame, curr_time, detected_boxes):

        self.frame_num = frame_num
        self.curr_time = curr_time

        # Predictions.
        for _tracker in self.trackers:
            _tracker.predict()

        # Map boxes with trackers.
        (associations, unmatched_trackers_indices, unmatched_detections) = (
            self.make_associations(self.trackers, detected_boxes)
        )
        associations = self.merge_multiple_associations(associations, self.trackers)
        self.logger.debug(f'associations: {[(_item[0], _item[1]._id) for _item in associations.items()]}')
        self.logger.debug(f'unmatched_detections: {unmatched_detections}')

        # Handle assigned trackers.
        matched_trackers = self.handle_matched_trackers(associations, detected_boxes, frame_num)
    
        # Handle unassigned trackers.
        unmatched_trackers = self.handle_unmatched_trackers(unmatched_trackers_indices, frame_num, curr_time)

        # Initialize new trackers.
        new_trackers = self.initialize_new_trackers(
            unmatched_detections, detected_boxes, frame_num, curr_time
        )
        self.trackers = matched_trackers
        self.trackers.extend(unmatched_trackers)
        self.trackers.extend(new_trackers)

        for _tracker in self.trackers:
            _tracker.update_trajectory(frame_num)

    def terminate(self, frame_num, curr_time):

        for _tracker in self.trackers:
            self.logger.debug(f'Terminated tracker {_tracker._id}')
            _tracker.terminate(frame_num, curr_time)
            if _tracker.duration > self.min_tracking_duration:
                self.finished.append(_tracker)
            else:
                self.trashed.append(_tracker)

    def save_results(self, time):
        """
        Saves tracking results (valid and rejected trackers).
        """

        def serialize_tracker(tracker):
            return {
                "_id": getattr(tracker, "_id", None),
                "duration": getattr(tracker, "duration", None),
                "start_frame": getattr(tracker, "start_frame", None),
                "end_frame": getattr(tracker, "end_frame", None),
                "trajectory": getattr(tracker, "trajectory", None),
                "color": getattr(tracker, "color", None),
            }

        result_data = {
            "nb_detected": len(self.finished),
            "nb_trashed": len(self.trashed),
            "current_time": time,
            "contours": self.contours,
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

    def save_output_video(self, start, end, step):

        with open(self.output_json_path, "r") as f:
            res = json.load(f)

        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current_trajectories = {_tracker["_id"]: [] for _tracker in res["Detection"]}

        for frame_num in range(start, end, step):
            _, frame = cap.read()

            # Reconstruct contours
            serialized_contours = res["contours"][str(frame_num)]
            if serialized_contours:
                contours = [
                    np.array(_c, dtype=np.int32).reshape((-1, 1, 2))
                    for _c in serialized_contours
                ]
                cv2.drawContours(frame, contours, -1, (255, 0, 0), 4)

            # Draw trajectories for “validated” trackers
            for _tracker in res["Detection"]:
                if _tracker["trajectory"].get(str(frame_num), None) is not None:
                    current_trajectories[_tracker["_id"]].append(
                        _tracker["trajectory"][str(frame_num)]
                    )
                if frame_num <= _tracker["end_frame"]:
                    trajectory = current_trajectories[_tracker["_id"]]
                    if trajectory:
                        pts = np.array(trajectory, dtype=np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(
                            frame,
                            [pts],
                            isClosed=False,
                            color=_tracker["color"],
                            thickness=5,
                        )
                        cv2.putText(
                            frame,
                            str(_tracker["_id"]),
                            trajectory[-1],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            _tracker["color"],
                            2,
                        )

            img_path = os.path.join(self.output_dir, f"out_frame{frame_num:04d}.png")
            cv2.imwrite(img_path, frame)

        cap.release()

        self.concat_png_to_video(folder=self.output_dir,
                                 output_video_path=self.output_video_path,
                                 logger=self.logger)

    @staticmethod
    def concat_png_to_video(folder,
                            output_video_path,
                            logger,
                            fps=30):

        cmd = [
            "ffmpeg",
            "-y", # force overwriting
            "-i",
            os.path.join(folder, "out_frame%4d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_video_path
        ]
        print(cmd)

        subprocess.run(cmd)

        logger.info(f"Created video : {os.path.join(folder, output_video_path)}")
