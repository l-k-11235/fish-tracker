import argparse
import cv2
import json
import logging
import os
import random
import time

from datetime import datetime

from fish_tracker.utils.logger import get_logger, set_global_log_level, set_log_file
from fish_tracker.detection.object_detector import detect_fishes_parallel
from fish_tracker.core.tracker_manager import TrackerManager
from fish_tracker.core.output_writer import save_output_frames, concat_frames_to_video


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
        "--distance_threshold",
        "-dist",
        type=float,
        default=200,
        required=False,
        help=(
            "Minimum distance to preserve " "a match between a tracker and a contour."
        ),
    )
    parser.add_argument(
        "--max_absences",
        "-ab",
        type=int,
        default=1,
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
        default=None,
        required=False,
        help="Index of the last frame to read from the video.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level.",
    )
    args = parser.parse_args()
    return args


def get_video_settings(video_path, logger):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        logger.error(f"Error: Unable to open video {video_path}")
        nb_frames, frame_height, frame_width = 0, 0, 0
    else:
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
    return nb_frames, fps, frame_height, frame_width


def fish_tracking(
    input_video_name,
    first_frame,
    output_video_name,
    output_json_name,
    dump_masked_frames,
    distance_threshold,
    max_absences,
    min_tracking_duration,
    step,
    start,
    end,
    log_level,
):
    output_dir = "/app/data/outputs"
    input_dir = "/app/data/inputs"
    video_path = f"{input_dir}/{input_video_name}"
    ref_frame_path = None
    if first_frame is not None:
        ref_frame_path = f"{input_dir}/{first_frame}"
    output_video_path = f"/app/data/outputs/{output_video_name}"
    output_json_path = f"/app/data/outputs/{output_json_name}"

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{output_dir}/fish_tracking.{timestamp}.log"
    set_log_file(log_filename)
    logger = get_logger("main")
    logger.info("Start tracking")

    nb_frames, fps, frame_height, frame_width = get_video_settings(video_path, logger)

    if end is None:
        end = nb_frames

    random.seed(42)

    # ############## #
    # Fish detection #
    # ############## #
    detections_path = "/app/data/outputs/detections.json"
    if not os.path.exists(detections_path):
        beg = time.time()
        detections = detect_fishes_parallel(
            video_path, start, end, step, output_dir, ref_frame_path, dump_masked_frames
        )
        logger.info(f"Detection took: {round(time.time() - beg)}s")
        with open(detections_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)
    else:
        logger.info("Detections have already been calculated.")
    with open(detections_path, "r") as f:
        detections = json.load(f)

        # ######## #
        # Tracking #
        # ######## #

        beg = time.time()
        manager = TrackerManager(
            motion_boxes=detections,
            frame_height=frame_height,
            frame_width=frame_width,
            max_absences=max_absences,
            min_tracking_duration=min_tracking_duration,
            distance_threshold=distance_threshold,
            fps=fps,
            input_video_name=input_video_name,
            output_json_name=output_json_name,
            log_level=log_level,
        )

        for _frame_num in range(start + 1, end):
            logger.debug(f"Frame {_frame_num}")
            detected_boxes = detections.get(str(_frame_num), [])
            curr_time = (_frame_num - start) / fps
            manager.process(_frame_num, curr_time, detected_boxes)

        # ########################## #
        # Terminate running trackers #
        # ########################## #
        manager.terminate(_frame_num, curr_time)
        logger.info(f"Tracking took: {round(time.time() - beg, 2)}s")

        # ##########
        # # Result #
        # ##########
        beg = time.time()
        manager.save_results(curr_time)
        save_output_frames(
            start, end, output_json_path, video_path, output_dir, manager.logger
        )
        concat_frames_to_video(
            folder=output_dir,
            output_video_path=output_video_path,
            logger=manager.logger,
            fps=fps,
        )
    logger.info(f"Saving took: {round(time.time() - beg, 2)}s")


if __name__ == "__main__":
    args = parse_args()
    fish_tracking(
        input_video_name=args.input_video_name,
        first_frame=args.first_frame,
        output_video_name=args.output_video_name,
        output_json_name=args.output_json_name,
        dump_masked_frames=args.dump_masked_frames,
        distance_threshold=args.distance_threshold,
        max_absences=args.max_absences,
        min_tracking_duration=args.min_tracking_duration,
        step=args.step,
        start=args.start,
        end=args.end,
        log_level=args.log_level,
    )
