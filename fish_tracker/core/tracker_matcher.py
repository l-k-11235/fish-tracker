import numpy as np

from scipy.optimize import linear_sum_assignment
from fish_tracker.utils.logger import get_logger


class TrackerMatcher:

    def __init__(self,
                 distance_threshold,
                 **kwargs):

        self.logger = get_logger("TrackerMatcher")
        self.logger.info("TrackerMatcher Initialization")
    
        self.distance_threshold = distance_threshold

    @staticmethod
    def compute_cost_matrix(trackers, detected_boxes):
        """Cost matrix based on the euclidian"""

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
        # Cost Matrix
        cost_matrix = self.compute_cost_matrix(trackers, detected_boxes)

        # Global optimal assignment (one box-one tracker)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        base_associations = {
            int(j): [int(i)]
            for i, j in zip(row_ind, col_ind)
            if cost_matrix[i, j] < self.distance_threshold
        }
        self.logger.debug("Global optimal assignment")
        for i, j in zip(row_ind, col_ind):
            self.logger.debug(f'{i}, {j}, {cost_matrix[i, j]} {self.distance_threshold}')

        assigned_boxes = set(base_associations.keys())
        assigned_trackers = set().union(*base_associations.values())

        # For each box, look for other close trackers
        self.logger.debug("Search for other matches")
        for j, _box in enumerate(detected_boxes):
            for i, _tracker in enumerate(trackers):
                if i in assigned_trackers:
                    continue
                dist = _tracker.distance_to_box(_box)
                self.logger.debug(f'{i}, {j}, {dist} {self.distance_threshold}')

                if dist < self.distance_threshold:
                    if j not in base_associations:
                        base_associations[j] = []
                    base_associations[j].append(i)
                    assigned_trackers.add(i)
                    assigned_boxes.add(j)

        # Unassigned trackers
        unassigned_trackers = set(range(len(trackers))) - assigned_trackers

        # Unasigned boxes
        unassigned_detections = set(range(len(detected_boxes))) - assigned_boxes

        return base_associations, unassigned_trackers, unassigned_detections

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
