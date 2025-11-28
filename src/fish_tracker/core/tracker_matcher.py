import numpy as np

from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import TypedDict

from fish_tracker.config import TrackerMatcherConfig
from fish_tracker.utils.logger import get_logger
from fish_tracker.utils.roi_processor import ROIResult
from fish_tracker.trackers.base import BaseTracker


class Match(TypedDict):
    tracker: BaseTracker
    detection: ROIResult


class TrackerMatcher:
    def __init__(self, config: TrackerMatcherConfig) -> None:

        self.logger = get_logger("TrackerMatcher")
        self.logger.info(f"TrackerMatcher Initialization with method {config.method}")

        self.frame_height: int = config.video_opts.frame_height
        self.frame_width: int = config.video_opts.frame_width
        self.method: str = config.method
        self.alpha: float = config.alpha
        if config.distance_threshold is None:
            self.distance_threshold: float = 0.9 if self.method == "hybrid" else 200.0
        else:
            self.distance_threshold: float = config.distance_threshold

    @staticmethod
    def compute_cost_matrix(
        trackers: list[BaseTracker],
        detections: list[ROIResult],
        method: str,
        alpha: float,
    ) -> NDArray[np.float64]:
        if len(trackers) == 0 or len(detections) == 0:
            return np.zeros((len(trackers), len(detections)))

        # GEOMETRIC COST
        tracker_positions: ArrayLike = np.array([[t.x, t.y] for t in trackers])
        box_centers: ArrayLike = np.array(
            [
                [_d.bbox[0] + _d.bbox[2] / 2, _d.bbox[1] + _d.bbox[3] / 2]
                for _d in detections
            ]
        )
        geo_cost: NDArray[np.float64] = np.linalg.norm(
            tracker_positions[:, np.newaxis, :] - box_centers[np.newaxis, :, :], axis=2
        )
        if method == "geometric":
            return geo_cost

        # EMBEDDING COST
        track_embeds: list[list[float]] = [t.embedding for t in trackers]
        track_embeds_np = np.asarray(track_embeds)
        det_embeds: list[list[float]] = [d.embedding for d in detections]
        det_embeds_np = np.asarray(det_embeds)

        embed_cost = cdist(track_embeds_np, det_embeds_np, metric="cosine")

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

    def make_associations(
        self, trackers: list[BaseTracker], detections: list[ROIResult]
    ) -> tuple[list[Match], list[BaseTracker], list[ROIResult]]:
        matches: list[Match] = []

        if len(trackers) == 0 or len(detections) == 0:
            return matches, trackers, detections

        # Hungarian (global optimal assignment)
        cost_matrix: NDArray[np.float64] = self.compute_cost_matrix(
            trackers, detections, method=self.method, alpha=self.alpha
        )
        self.logger.debug(f"cost matrix emb dissim: {cost_matrix}")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # row_ind ~ workers (trackers) // col_ind ~ tasks (bboxes)

        # Apply filter
        assigned_trackers_indices: list[int] = []
        assigned_detections_indices: list[int] = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.distance_threshold:
                self.logger.debug(
                    "tracker=%d, box=%d, cost=%.3f", i, j, cost_matrix[i, j]
                )
                assigned_trackers_indices.append(i)
                assigned_detections_indices.append(j)
                matches.append({"tracker": trackers[i], "detection": detections[j]})

        unassigned_trackers: list[BaseTracker] = [
            _tracker
            for i, _tracker in enumerate(trackers)
            if i not in assigned_trackers_indices
        ]
        self.logger.debug(
            f"unassigned trackers: {[_t.id for _t in unassigned_trackers]}"
        )
        unassigned_detections = [
            _bbox
            for j, _bbox in enumerate(detections)
            if j not in assigned_detections_indices
        ]
        return matches, unassigned_trackers, unassigned_detections
