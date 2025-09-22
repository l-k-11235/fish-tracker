import numpy as np

from scipy.optimize import linear_sum_assignment
from fish_tracker.utils.logger import get_logger


class TrackerMatcher:

    def __init__(self, frame_height, frame_width, distance_threshold=200, **kwargs):

        self.logger = get_logger("TrackerMatcher")
        self.logger.info("TrackerMatcher Initialization")

        self.frame_height = frame_height
        self.frame_width = frame_width
        self.distance_threshold = distance_threshold

    @staticmethod
    def compute_cost_matrix(trackers, detected_boxes):
        """Cost matrix based on the euclidian distance"""

        if len(trackers) == 0 or len(detected_boxes) == 0:
            return np.zeros((len(trackers), len(detected_boxes)))

        tracker_positions = np.array([[t.x, t.y] for t in trackers])
        box_centers = np.array(
            [[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in detected_boxes]
        )
        diff = tracker_positions[:, np.newaxis, :] - box_centers[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diff, axis=2)

        return cost_matrix

    def make_associations(self, trackers, detected_boxes):

        if len(trackers) == 0 or len(detected_boxes) == 0:
            return {}, set(range(len(trackers))), set(range(len(detected_boxes)))

        # Cost Matrix
        cost_matrix = self.compute_cost_matrix(trackers, detected_boxes)

        # Hungarian (global optimal assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # valid matches
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.distance_threshold:
                matches[j] = [i]

        for i, j in zip(row_ind, col_ind):
            self.logger.debug(
                "tracker=%d, box=%d, cost=%.3f (th=%.3f)",
                i,
                j,
                cost_matrix[i, j],
                self.distance_threshold,
            )

        assigned_trackers = set(i for v in matches.values() for i in v)
        assigned_boxes = set(matches.keys())
        unassigned_trackers = set(range(len(trackers))) - assigned_trackers
        unassigned_detections = set(range(len(detected_boxes))) - assigned_boxes

        return matches, unassigned_trackers, unassigned_detections

    def merge_multiple_associations(self, associations, trackers):
        for box_idx, tracker_indices in associations.items():
            if len(tracker_indices) > 1:
                self.logger.debug("Merged trackers")
                group = [
                    _tracker
                    for j, _tracker in enumerate(trackers)
                    if j in tracker_indices
                ]
                starts = [_tracker.start_time for _tracker in group]
                start_time = min(starts)
                tracker = group[starts.index(start_time)]
                tracker.x = np.mean([_t.x for _t in group])
                tracker.y = np.mean([_t.y for _t in group])
                associations[box_idx] = tracker
            else:
                associations[box_idx] = trackers[tracker_indices[0]]
        return associations
