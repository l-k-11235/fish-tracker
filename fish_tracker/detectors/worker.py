# detectors/worker.py
"""
Worker utilities: frames_generator and multiprocessing initializer + worker callable
"""
import cv2
import numpy as np

from numpy.typing import NDArray
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, List, Union

from fish_tracker.utils.roi_processor import ROIResult
from fish_tracker.utils.configs import (
    SelectiveSearchDetectorConfig,
    YOLOSegDetectorConfig,
)


def read_image(path: Path) -> Optional[NDArray[np.uint8]]:
    img = cv2.imread(str(path))
    if img is not None:
        img = img.astype(np.uint8)
    return img


def frames_generator(
    video_path: Path,
    start: int,
    end: int,
    step: int,
    ref_frame_path: Optional[Path] = None,
) -> Generator[Tuple[int, NDArray[np.uint8], NDArray[np.uint8]], None, None]:
    if ref_frame_path is not None:
        ref_frame = read_image(ref_frame_path)
        if ref_frame is None:
            raise ValueError(f"Could not read reference frame from {ref_frame_path}")
    else:
        cap_ref = cv2.VideoCapture(str(video_path))
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, start)
        success, ref_frame = cap_ref.read()
        ref_frame = ref_frame.astype(np.uint8)

        cap_ref.release()
        if not success:
            raise ValueError(f"Could not read reference frame at index {start}")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + step)
    frame_num: int = start + step

    while frame_num < end:
        success, frame = cap.read()
        frame = frame.astype(np.uint8)
        if not success:
            break
        if frame_num % step == 0:
            yield frame_num, frame, ref_frame
        frame_num += 1
    cap.release()


DetectorConfig = Union[SelectiveSearchDetectorConfig, YOLOSegDetectorConfig]

# multiprocess globals
_detector = None


def init_worker(detector_type: str, detector_opts: DetectorConfig) -> None:
    """Initialize a detector once per worker."""
    global _detector
    if detector_type == "selective_search":
        from fish_tracker.detectors.selective_search import SelectiveSearchDetector

        assert isinstance(detector_opts, SelectiveSearchDetectorConfig)
        _detector = SelectiveSearchDetector(detector_opts)
    elif detector_type == "yolo_seg":
        from fish_tracker.detectors.yolo_seg import YOLOSegDetector

        assert isinstance(detector_opts, YOLOSegDetectorConfig)
        _detector = YOLOSegDetector(detector_opts)
    else:
        raise ValueError("Unknown detector type")


def worker_process_chunk(
    args: Tuple[Path, int, int, int, Optional[Path], Optional[Path]],
) -> Dict[int, List[ROIResult]]:
    """Callable executed in a worker process. Receives a tuple of args (see run)."""
    if _detector is None:
        raise RuntimeError("Worker detector not initialised. Did you call init_worker?")
    video_path, start, end, step, ref_frame_path, dump_dir = args
    return _detector.process_chunk(
        video_path, start, end, step, ref_frame_path, dump_dir
    )
