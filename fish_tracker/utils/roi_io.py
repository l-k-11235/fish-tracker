# utils/roi_io.py
import numpy as np

from numpy.typing import NDArray
from pathlib import Path

from .roi_processor import ROIResult


def save_rois_npz(path: Path, detections: dict[int, list[ROIResult]]) -> None:
    frame_ids: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    embeddings: list[list[float]] = []
    crops: list[NDArray[np.uint8]] = []

    for frame_id, roi_list in detections.items():
        for roi in roi_list:
            frame_ids.append(frame_id)
            bboxes.append(roi.bbox)
            embeddings.append(roi.embedding)
            crops.append(roi.crop)

    np.savez_compressed(
        path,
        frame_ids=np.array(frame_ids),
        bboxes=np.array(bboxes),
        embeddings=np.array(embeddings),
        crops=np.array(crops, dtype=object),
    )


def load_rois_npz(path: Path) -> dict[int, list[ROIResult]]:
    data = np.load(path, allow_pickle=True)
    frame_ids = data["frame_ids"]
    bboxes = data["bboxes"]
    embeddings = data["embeddings"]
    crops = data["crops"]

    result: dict[int, list[ROIResult]] = {}

    for i, fid in enumerate(frame_ids):
        roi = ROIResult(
            bbox=tuple(bboxes[i]),
            embedding=embeddings[i].tolist(),
            crop=crops[i],
        )
        result.setdefault(int(fid), []).append(roi)

    return result
