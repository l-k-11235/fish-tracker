import cv2
import json
import multiprocessing as mp
import numpy as np
import os
import subprocess


def write_trajectories(frame, frame_num, motion_boxes, trackers, output_dir):

    for x, y, w, h in motion_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for _tracker in trackers:
        trajectory = [
            _tracker["trajectory"][str(i)]
            for i in range(
                _tracker["start_frame"], min(_tracker["end_frame"], frame_num + 1)
            )
        ]
        for x, y, w, h in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pts = np.array(trajectory, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=_tracker["color"],
            thickness=5,
        )
        cv2.putText(
            frame,
            str(_tracker["_id"]),
            trajectory[-1],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _tracker["color"],
            2,
        )
    img_path = os.path.join(output_dir, f"out_frame{frame_num:04d}.png")
    cv2.imwrite(img_path, frame)


def write_trajectories_chunk(start, end, json_path, video_path, output_dir):
    with open(json_path, "r") as f:
        res = json.load(f)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_num = start
    while frame_num < end:

        success, frame = cap.read()
        if not success:
            break

        if frame_num == start:
            pass
        motion_boxes = res["motion_boxes"].get(str(frame_num), [])
        trackers = [
            _t
            for _t in res["Detection"]
            if (_t["start_frame"] <= frame_num <= _t["end_frame"])
        ]
        if trackers:
            write_trajectories(frame, frame_num, motion_boxes, trackers, output_dir)
        frame_num += 1
    cap.release()


def save_output_frames(
    start, end, json_path, video_path, output_dir, logger, num_workers=8, chunk_size=16
):

    chunks = []
    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end)
        chunks.append((chunk_start, chunk_end, json_path, video_path, output_dir))
    logger.debug(f"{len(chunks)} chunks.")
    with mp.Pool(num_workers) as pool:
        pool.starmap(write_trajectories_chunk, chunks)


def concat_frames_to_video(folder, output_video_path, logger, fps=30):

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(folder, "out_frame*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_video_path,
    ]
    logger.debug(" ".join(cmd))

    subprocess.run(cmd)

    logger.info(f"Created video : {os.path.join(folder, output_video_path)}")
