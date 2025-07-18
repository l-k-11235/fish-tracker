import argparse
import cv2
import logging
import random

from datetime import datetime

from fish_tracker.utils.logger import get_logger, set_global_log_level, set_log_file
from fish_tracker.detection.object_detector import ObjectDetector
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
        default=100,
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
        default=0.5,
        required=False,
        help="Minimum duration to preserve a tracking trajectory",
    )
    parser.add_argument(
        "--step",
        "-step",
        type=int,
        default=1,
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


def fish_tracking(
    input_video_name,
    first_frame,
    output_video_name,
    output_json_name,
    min_area=2000,
    distance_threshold=10,
    max_absences=3,
    min_tracking_duration=2,
    step=1,
    start=0,
    end=None,
    log_level="INFO",
):

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"/app/data/outputs/fish_tracking.{timestamp}.log"
    set_log_file(log_filename)

    logger = get_logger("main")
    logger.info("Start tracking")

    random.seed(42)
    video_path = f"/app/data/inputs/{input_video_name}"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Unable to open video {video_path}")
    else:

        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"{nb_frames} frames")

        if end is None:
            end = nb_frames

        if first_frame is None:
            _, first_frame = cap.read()
        else:
            first_frame = f"/app/data/inputs/{first_frame}"

        # ################### #
        # Initialize Detector #
        # ################### #
        detector = ObjectDetector(
            reference_frame=first_frame, min_contour_area=min_area, log_level=log_level
        )

        # ########################## #
        # Initialize Tracker Manager #
        # ########################## #
        manager = TrackerManager(
            max_absences=max_absences,
            min_tracking_duration=min_tracking_duration,
            distance_threshold=distance_threshold,
            input_video_name=input_video_name,
            output_video_name=output_video_name,
            output_json_name=output_json_name,
            log_level=log_level,
        )

        # ########## #
        # Play video #
        # ########## #
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_num in range(start, end, step):
            logger.debug(f"\n# frame {frame_num}")

            _, frame = cap.read()
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # ############### #
            # Detect contours #
            # ############### #
            contours = detector.process_frame(frame)
            manager.contours[frame_num] = [_c.squeeze().tolist() for _c in contours]
            detected_boxes = [cv2.boundingRect(_c) for _c in contours]

            # ######## #
            # Tracking #
            # ######## #
            logger.debug(
                "Running trackers: %s",
                [(_t._id, _t.absences) for _t in manager.trackers],
            )

            manager.process(frame_num, frame, curr_time, detected_boxes)

        cap.release()

        # ########################## #
        # Terminate running trackers #
        # ########################## #
        manager.terminate(frame_num, curr_time)

        ##########
        # Result #
        ##########
        manager.save_results(curr_time)
        manager.save_output_video(start, end, step)


if __name__ == "__main__":
    args = parse_args()
    fish_tracking(
        input_video_name=args.input_video_name,
        first_frame=args.first_frame,
        output_video_name=args.output_video_name,
        output_json_name=args.output_json_name,
        min_area=args.min_area,
        distance_threshold=args.distance_threshold,
        max_absences=args.max_absences,
        min_tracking_duration=args.min_tracking_duration,
        step=args.step,
        start=args.start,
        end=args.end,
        log_level=args.log_level,
    )
