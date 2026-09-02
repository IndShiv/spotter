from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from spotter.commands.pipeline import run_pipeline
from spotter.counting.base import CounterState


def _camera_frames(camera_index: int) -> Iterator[tuple[np.ndarray, float]]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    start = time.monotonic()
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield image, time.monotonic() - start
    finally:
        cap.release()


def run_live(
    exercise: str,
    camera: int = 0,
    counter: str = "fsm",
    model: str = "lite",
    overlay: bool = True,
    dump_signals_path: Path | None = None,
    num_poses: int = 3,
    lock_frames: int = 30,
) -> CounterState:
    return run_pipeline(
        _camera_frames(camera),
        exercise_name=exercise,
        counter_name=counter,
        model_variant=model,
        num_poses=num_poses,
        lock_frames=lock_frames,
        overlay=overlay,
        show_window=True,
        window_title="Spotter — live",
        dump_signals_path=dump_signals_path,
    )
