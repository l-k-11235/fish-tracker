# core/main.py

import argparse
import logging
import os
import random
import time

from datetime import datetime
from pathlib import Path

from fish_tracker.core.tracker_manager import TrackerManager
from fish_tracker.core.output_writer import save_output_frames, concat_frames_to_video
from fish_tracker.detectors.roi_detection import run_roi_detection
from fish_tracker.utils.configs import FullConfig
from fish_tracker.utils.logger import get_logger, set_global_log_level, set_log_file
from fish_tracker.utils.roi_io import save_rois_npz, load_rois_npz
from fish_tracker.utils.roi_processor import ROIResult


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_name",
        "-vi",
        type=str,
        required=True,
        help="Input video file name",
    )
    parser.add_argument(
        "--first_frame",
        "-ff",
        type=str,
        required=False,
        default=None,
        help="Path to the first_frame (png file)",
    )
    parser.add_argument(
        "--output_video_name",
        "-vo",
        type=str,
        required=False,
        default="output.mp4",
        help="Output video file name",
    )
    parser.add_argument(
        "--output_json_name",
        "-jo",
        type=str,
        default="output.json",
        required=False,
        help="JSON output name",
    )
    parser.add_argument(
        "--dump_masked_frames",
        action="store_true",
        default=False,
        help="Dump masked frames (disabled by default).",
    )
    parser.add_argument(
        "--detector_type",
        "-detector_type",
        type=str,
        default="selective_search",
        choices=["selective_search", "yolo_seg"],
        required=False,
        help=("Method used for ROI detection."),
    )
    parser.add_argument(
        "--embedding_model_name",
        "-embedding_model_name",
        type=str,
        default=None,
        required=False,
        help=(
            "Name of the embedding model for ROI representation "
            "(currently supports 'mobilenetv3small')"
        ),
    )
    parser.add_argument(
        "--matching_method",
        "--matching_method",
        type=str,
        default="geometric",
        choices=["geometric", "embedding", "hybrid"],
        required=False,
        help=("Method used for ROI detection."),
    )
    parser.add_argument(
        "--alpha",
        "--alpha",
        type=float,
        default=0.5,
        required=False,
        help=("The weight of the geometric distance in the hybrid method"),
    )
    parser.add_argument(
        "--distance_threshold",
        "-dist",
        type=float,
        default=None,
        required=False,
        help=(
            "Minimum distance to preserve " "a match between a tracker and a contour."
        ),
    )
    parser.add_argument(
        "--max_absences",
        "-ab",
        type=int,
        default=2,
        required=False,
        help=("Maximum number of consecutive frames " "without a match for a tracker."),
    )
    parser.add_argument(
        "--min_tracking_duration",
        "-td",
        type=float,
        default=0,
        required=False,
        help="Minimum duration to preserve a tracking trajectory",
    )
    parser.add_argument(
        "--step",
        "-step",
        type=int,
        default=5,
        required=False,
        help="Process one frame every step frames.",
    )
    parser.add_argument(
        "--start",
        "-start",
        type=int,
        default=0,
        required=False,
        help="Index of the first frame to read from the video.",
    )
    parser.add_argument(
        "--end",
        "-end",
        type=int,
        default=0,
        required=False,
        help="Index of the last frame to read from the video.",
    )
    parser.add_argument(
        "--num_workers",
        "-num_workers",
        type=int,
        default=8,
        required=False,
        help=(
            "Number of parallel worker processes used for fish detection when using "
            "the Selective Search detector. Ignored when using YOLO segmentation."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        "-chunk_size",
        type=int,
        default=8,
        required=False,
        help=(
            "Number of frames processed together per iteration. "
            "Interpreted as the chunk size for Selective Search (CPU) "
            "or as the batch size for YOLO segmentation."
        ),
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level.",
    )
    args = parser.parse_args()
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
    manager.save_results(motion_boxes, curr_time)
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
    full_config: FullConfig = FullConfig.from_args(args)
    fish_tracking(full_config)
