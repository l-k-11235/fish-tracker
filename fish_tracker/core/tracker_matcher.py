import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from fish_tracker.utils.logger import get_logger


class TrackerMatcher:

    def __init__(
        self,
        frame_height,
        frame_width,
        method="hybrid",
        alpha=0.5,
        distance_threshold=0.5,
    ):

        self.logger = get_logger("TrackerMatcher")
        self.logger.info(f"TrackerMatcher Initialization with method {method}")

        self.frame_height = frame_height
        self.frame_width = frame_width
        self.method = method
        self.alpha = alpha
        self.distance_threshold = distance_threshold

    @staticmethod
    def compute_cost_matrix(trackers, detections, method="hybrid", alpha=0.5):
        """
        Compute cost matrix combining geometric and/or embedding distance.

        Parameters
        ----------
        trackers : list
            List of tracker objects with attributes x, y, and optionally embedding.
        detections : list
            List of detections with keys 'box' and optionally 'embedding'.
        method : str
            'geometric', 'embedding', or 'hybrid' (both combined).
        alpha : float
            Weight for geometric cost (0 = only embedding, 1 = only geometric).

        Returns
        -------
        cost_matrix : np.ndarray
            Matrix of pairwise costs between trackers and detections.
        """
        if len(trackers) == 0 or len(detections) == 0:
            return np.zeros((len(trackers), len(detections)))

        # --- GEOMETRIC COST ---
        tracker_positions = np.array([[t.x, t.y] for t in trackers])
        box_centers = np.array(
            [
                [_d["box"][0] + _d["box"][2] / 2, _d["box"][1] + _d["box"][3] / 2]
                for _d in detections
            ]
        )
        geo_cost = np.linalg.norm(
            tracker_positions[:, np.newaxis, :] - box_centers[np.newaxis, :, :], axis=2
        )

        if method == "geometric":
            return geo_cost

        # --- EMBEDDING COST ---
        embed_cost = cdist(
            [_t.embedding for _t in trackers],
            [_b["embedding"] for _b in detections],
            metric="cosine",
        )

        if method == "embedding":
            return embed_cost

        # --- HYBRID COMBINATION ---
        if method == "hybrid":
            # Normalize
            geo_cost_norm = geo_cost / (geo_cost.max() + 1e-8)
            embed_cost_norm = embed_cost / (embed_cost.max() + 1e-8)

            cost_matrix = alpha * geo_cost_norm + (1 - alpha) * embed_cost_norm
            return cost_matrix

        raise ValueError(
            f"Unknown method '{method}' (use 'geometric', 'embedding', or 'hybrid')."
        )

    def make_associations(self, trackers, detections):

        if len(trackers) == 0 or len(detections) == 0:
            return {}, trackers, detections

        # Hungarian (global optimal assignment)
        cost_matrix = self.compute_cost_matrix(
            trackers, detections, method=self.method, alpha=self.alpha
        )
        self.logger.debug(f"cost matrix emb dissim: {cost_matrix}")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # row_ind ~ workers (trackers) // col_ind ~ tasks (bboxes)

        # Apply filter
        matches = []
        assigned_trackers_indices = []
        assigned_detections_indices = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.distance_threshold:
                self.logger.debug(
                    "tracker=%d, box=%d, cost=%.3f", i, j, cost_matrix[i, j]
                )
                assigned_trackers_indices.append(i)
                assigned_detections_indices.append(j)
                matches.append({"tracker": trackers[i], "detection": detections[j]})

        unassigned_trackers = [
            _tracker
            for i, _tracker in enumerate(trackers)
            if i not in assigned_trackers_indices
        ]
        self.logger.debug(
            f"unassigned trackers: {[_t._id for _t in unassigned_trackers]}"
        )
        unassigned_detections = [
            _bbox
            for j, _bbox in enumerate(detections)
            if j not in assigned_detections_indices
        ]
        return matches, unassigned_trackers, unassigned_detections
