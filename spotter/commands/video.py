from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from spotter.commands.pipeline import run_pipeline
from spotter.counting.base import CounterState


def _video_frames(path: Path, fps: float) -> Iterator[tuple[np.ndarray, float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        idx = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield image, idx / fps
            idx += 1
    finally:
        cap.release()


def run_video(
    path: Path,
    exercise: str,
    counter: str = "fsm",
    model: str = "heavy",
    overlay: bool = True,
    show_window: bool = True,
    output_path: Path | None = None,
    dump_signals_path: Path | None = None,
    num_poses: int = 3,
    lock_frames: int = 30,
) -> CounterState:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    return run_pipeline(
        _video_frames(path, fps),
        exercise_name=exercise,
        counter_name=counter,
        model_variant=model,
        num_poses=num_poses,
        lock_frames=lock_frames,
        overlay=overlay,
        show_window=show_window,
        window_title=f"Spotter — {path.name}",
        output_path=output_path,
        output_fps=fps,
        dump_signals_path=dump_signals_path,
    )
