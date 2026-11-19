"""Add a visible goal marker to an exported PointMaze video."""

from pathlib import Path

import cv2
import numpy as np


def mark_goal_video(video_stem, goal_visual):
    """
    Detect the green PointMaze particle in the goal image,
    then draw a yellow star at that pixel in the saved MP4.

    Does not modify observations or the original video.
    """
    src = Path(str(video_stem) + ".mp4")
    dst = Path(str(video_stem) + "_marked.mp4")

    if not src.is_file():
        raise FileNotFoundError(src)

    # Match the official make_video_mp4 image normalization.
    frame = goal_visual
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()

    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[-1]

    if frame.shape[0] == 3:  # CHW -> HWC
        frame = frame.transpose(1, 2, 0)

    frame = frame.astype(np.float32)
    lo, hi = float(frame.min()), float(frame.max())
    if hi <= lo:
        raise ValueError("Goal image has no visible content")

    goal_rgb = np.clip(
        (frame - lo) / (hi - lo) * 255, 0, 255
    ).astype(np.uint8)

    # The particle is green in the official PointMaze render.
    hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 50, 30], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)

    candidates = [
        i for i in range(1, count)
        if 10 <= stats[i, cv2.CC_STAT_AREA] <= 400
    ]
    if not candidates:
        raise ValueError(
            "Could not locate green goal particle; "
            "original video remains unchanged"
        )

    idx = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y = np.rint(centers[idx]).astype(int)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not (0 <= x < width and 0 <= y < height):
        cap.release()
        raise ValueError("Goal marker lies outside video frame")

    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write {dst}")

    # Yellow five-point star, BGR color format.
    points = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        radius = 11 if i % 2 == 0 else 5
        points.append((
            int(round(x + radius * np.cos(angle))),
            int(round(y + radius * np.sin(angle))),
        ))
    star = np.asarray(points, dtype=np.int32)

    frames = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break

            cv2.polylines(
                img, [star], True, (0, 0, 0), 5, cv2.LINE_AA
            )
            cv2.fillPoly(img, [star], (0, 225, 255))
            cv2.polylines(
                img, [star], True, (0, 0, 0), 1, cv2.LINE_AA
            )

            label_x = max(4, min(x + 13, width - 55))
            label_y = max(15, min(y - 12, height - 5))

            cv2.putText(
                img, "GOAL", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, "GOAL", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 225, 255), 1, cv2.LINE_AA
            )

            writer.write(img)
            frames += 1
    finally:
        cap.release()
        writer.release()

    print(f"[GoalOverlay] pixel=({x},{y}), frames={frames}")
    print(f"[GoalOverlay] saved={dst}")
    return str(dst)
