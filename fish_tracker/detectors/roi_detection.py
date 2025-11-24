# File: detecors/roi_detection.py
"""
Entrypoint: run_roi_detection using a pool of workers. Uses init_worker pattern.
"""
import multiprocessing as mp
from pathlib import Path

from .worker import init_worker, worker_process_chunk
from fish_tracker.utils.configs import (
    FullConfig,
    SelectiveSearchDetectorConfig,
    YOLOSegDetectorConfig,
)
from fish_tracker.utils.roi_processor import ROIResult


def run_roi_detection(config: FullConfig) -> dict[int, list[ROIResult]]:
    dump_dir = None
    if config.detector_opts.dump_masked_frames:
        dump_dir = Path("/app/data/outputs/masked_frames")
        dump_dir.mkdir(parents=True, exist_ok=True)

    start, step, end = config.start, config.step, config.end
    chunk_size = config.detector_opts.chunk_size
    chunks: list[tuple[Path, int, int, int, Path | None, Path | None]] = []
    for chunk_start in range(start, end, step * chunk_size):
        chunk_end = min(chunk_start + step * chunk_size, end)
        chunks.append(
            (
                config.input_video_path,
                chunk_start,
                chunk_end,
                step,
                config.ref_frame_path,
                dump_dir,
            )
        )
    all_detections: dict[int, list[ROIResult]] = {}
    if config.detector_type == "selective_search":
        detector_opts = config.detector_opts
        assert isinstance(detector_opts, SelectiveSearchDetectorConfig)
        num_workers: int = detector_opts.num_workers
    else:
        detector_opts = config.detector_opts
        assert isinstance(detector_opts, YOLOSegDetectorConfig)
        # For YOLO on CPU we may prefer single process
        # unless GPU worker strategies available
        num_workers = 1

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config.detector_type, detector_opts),
    ) as pool:
        results = pool.map(worker_process_chunk, chunks)

    for res in results:
        all_detections.update(res)

    return all_detections
