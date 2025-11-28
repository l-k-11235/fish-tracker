# core/main.py

import argparse
import logging
import os
import random
import time

from datetime import datetime
from pathlib import Path

from fish_tracker.config import FullConfig
from fish_tracker.core.tracker_manager import TrackerManager
from fish_tracker.core.output_writer import save_output_frames, concat_frames_to_video
from fish_tracker.detectors.roi_detection import run_roi_detection
from fish_tracker.utils.logger import get_logger, set_global_log_level, set_log_file
from fish_tracker.utils.roi_io import save_rois_npz, load_rois_npz
from fish_tracker.utils.roi_processor import ROIResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to  the YAML configuration file"
    )
    args: argparse.Namespace = parser.parse_args()
    return args


def fish_tracking(config: FullConfig) -> None:
    log_level: int = getattr(logging, config.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)

    output_dir: Path = Path("/app/data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename: Path = output_dir / f"fish_tracking.{timestamp}.log"
    set_log_file(log_filename)
    logger: logging.Logger = get_logger("main")
    logger.info("Start tracking")

    config.save_yaml()

    random.seed(42)

    # #################### #
    # Regions of interest #
    # ################### #
    roi_path: Path = output_dir / "roi.npz"
    if not os.path.exists(roi_path):
        start_time: float = time.time()
        detections: dict[int, list[ROIResult]] = run_roi_detection(config)
        logger.info(
            (
                "Calculation of regions of interest took: "
                f"{round(time.time() - start_time)}s"
            )
        )
        save_rois_npz(roi_path, detections)
    else:
        logger.info("Detections have already been calculated.")

    # ######## #
    # Tracking #
    # ######## #
    roi: dict[int, list[ROIResult]] = load_rois_npz(roi_path)

    start_time: float = time.time()
    manager = TrackerManager(config.tracker_manager_opts)
    _frame_num: int = config.start
    curr_time: float = 0.0

    for _frame_num in range(config.start + 1, config.end):
        logger.debug(f"Frame {_frame_num}")
        frame_roi: list[ROIResult] = roi.get(_frame_num, [])
        curr_time: float = (_frame_num - config.start) / config.video_opts.fps
        manager.process(_frame_num, frame_roi, curr_time, config.step)

    manager.terminate(_frame_num, curr_time)
    logger.info(f"Tracking took: {round(time.time() - start_time, 2)}s")

    # ##########
    # # Result #
    # ##########
    beg = time.time()
    motion_boxes: dict[int, list[tuple[int, int, int, int]]] = {
        frame_id: [det.bbox for det in frame_dets]
        for frame_id, frame_dets in roi.items()
    }
    manager.save_results(config.output_json_path, motion_boxes, curr_time)
    save_output_frames(
        config.start,
        config.end,
        config.output_json_path,
        config.input_video_path,
        Path("/app/data/outputs"),
        manager.logger,
    )
    concat_frames_to_video(
        folder=Path("/app/data/outputs"),
        output_video_path=config.output_video_path,
        fps=config.video_opts.fps,
    )
    logger.info(f"Saving took: {round(time.time() - beg, 2)}s")


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    full_config: FullConfig = FullConfig.from_file(args.config)
    fish_tracking(full_config)
