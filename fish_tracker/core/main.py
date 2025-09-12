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
        "--min_area",
        "-ar",
        type=float,
        default=2000,
        required=False,
        help="Minimum area to preserve a detected contour.",
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
        default=3,
        required=False,
        help=("Maximum number of consecutive frames " "without a match for a tracker."),
    )
    parser.add_argument(
        "--min_tracking_duration",
        "-td",
        type=float,
        default=2,
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


def read_video(video_path, logger):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Unable to open video {video_path}")
        nb_frames = 0
    else:
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, nb_frames


def fish_tracking(
    input_video_name,
    first_frame,
    output_video_name,
    output_json_name,
    dump_masked_frames,
    min_area,
    distance_threshold,
    max_absences,
    min_tracking_duration,
    step,
    start,
    end,
    log_level,
):

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"/app/data/outputs/fish_tracking.{timestamp}.log"
    set_log_file(log_filename)
    video_path = f"/app/data/inputs/{input_video_name}"

    logger = get_logger("main")
    logger.info("Start tracking")

    random.seed(42)

    cap, nb_frames = read_video(video_path, logger)
    logger.debug(f"{nb_frames} frames")

    if nb_frames:

        if end is None:
            end = nb_frames

        # ############## #
        # Fish detection #
        # ############## #
        detections_path = "/app/data/outputs/detections.json"
        if not os.path.exists(detections_path):
            beg = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            detections = detect_fishes_parallel(
                cap, first_frame, start, end, step, dump_masked_frames
            )
            logger.info(f"Detection took: {time.time() - beg}")
            with open(detections_path, "w", encoding="utf-8") as f:
                json.dump(detections, f, indent=2)
        else:
            logger.info("Detections have already been calculated.")
        with open(detections_path, "r") as f:
            detections = json.load(f)

        # ########################## #
        # Initialize Tracker Manager #
        # ########################## #
        manager = TrackerManager(
            motion_boxes=detections,
            max_absences=max_absences,
            min_tracking_duration=min_tracking_duration,
            distance_threshold=distance_threshold,
            input_video_name=input_video_name,
            output_video_name=output_video_name,
            output_json_name=output_json_name,
            log_level=log_level,
        )

        # ######## #
        # Tracking #
        # ######## #
        cap, _ = read_video(video_path, logger)

        frame_num = start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        while frame_num < end:
            success, frame = cap.read()
            if not success:
                break

            if frame_num == start:
                pass

            elif frame_num % step == 0:

                logger.debug(f"Frame {frame_num}")
                logger.debug(
                    "Running trackers: %s",
                    [(_t._id, _t.absences) for _t in manager.trackers],
                )
                detected_boxes = detections[str(frame_num)]
                logger.debug(f"{len(detected_boxes)} detected_boxes")
                curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                manager.process(frame_num, curr_time, detected_boxes)

            frame_num += 1

        cap.release()

        # ########################## #
        # Terminate running trackers #
        # ########################## #
        manager.terminate(frame_num, curr_time)

        # ##########
        # # Result #
        # ##########
        manager.save_results(curr_time)
        manager.save_output_video(start, end, step)


if __name__ == "__main__":
    args = parse_args()
    fish_tracking(
        input_video_name=args.input_video_name,
        first_frame=args.first_frame,
        output_video_name=args.output_video_name,
        output_json_name=args.output_json_name,
        dump_masked_frames=args.dump_masked_frames,
        min_area=args.min_area,
        distance_threshold=args.distance_threshold,
        max_absences=args.max_absences,
        min_tracking_duration=args.min_tracking_duration,
        step=args.step,
        start=args.start,
        end=args.end,
        log_level=args.log_level,
    )
