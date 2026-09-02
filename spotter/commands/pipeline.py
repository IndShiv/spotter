"""Shared frame-processing loop behind both `spotter video` and `spotter live`.

Both commands only differ in where frames come from (a file vs. a camera)
and what "done" means (end of file vs. Ctrl-C / 'q'); everything else —
pose estimation, single-subject locking, counting, overlay, CSV dump — is
identical, so it lives here once.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from spotter.counting.base import CounterState, RepCounter
from spotter.counting.fsm import FSMCounter
from spotter.counting.peaks import PeakCounter
from spotter.exercises.loader import load_exercise
from spotter.overlay.renderer import OverlayRenderer
from spotter.pose.mediapipe_backend import MediaPipeBackend
from spotter.tracking.single_subject import SingleSubjectTracker


def build_counter(exercise, counter_name: str) -> RepCounter:
    if counter_name == "fsm":
        return FSMCounter(exercise)
    if counter_name == "peaks":
        return PeakCounter(exercise)
    raise ValueError(f"Unknown counter {counter_name!r}; choose 'fsm' or 'peaks'")


def run_pipeline(
    frame_source: Iterator[tuple[np.ndarray, float]],
    exercise_name: str,
    counter_name: str = "fsm",
    model_variant: str = "heavy",
    num_poses: int = 3,
    lock_frames: int = 30,
    overlay: bool = True,
    show_window: bool = True,
    window_title: str = "Spotter",
    output_path: Path | None = None,
    output_fps: float = 30.0,
    dump_signals_path: Path | None = None,
) -> CounterState:
    """Drive `frame_source` through the full pipeline; returns the final CounterState."""
    exercise = load_exercise(exercise_name)
    backend = MediaPipeBackend(model_variant=model_variant, num_poses=num_poses)
    tracker = SingleSubjectTracker(lock_frames=lock_frames)
    counter = build_counter(exercise, counter_name)
    renderer = OverlayRenderer() if overlay else None
    signal_names = [s.name for s in exercise.signals]

    writer: cv2.VideoWriter | None = None
    last_state = CounterState()

    with ExitStack() as stack:
        stack.callback(backend.close)
        if show_window:
            stack.callback(cv2.destroyAllWindows)

        csv_writer = None
        if dump_signals_path is not None:
            csv_file = stack.enter_context(open(dump_signals_path, "w", newline=""))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["timestamp", "state", "rep_count", "no_rep_count", "hold", *signal_names]
            )

        try:
            for image, timestamp in frame_source:
                detections = backend.process(image, timestamp)
                subject = tracker.select(detections)

                if subject is None:
                    cs = CounterState(
                        rep_count=last_state.rep_count,
                        no_rep_count=last_state.no_rep_count,
                        current_state="no_person_detected",
                        hold=True,
                        hold_reason="Spotter can't see you",
                    )
                    pose_frame = None
                else:
                    cs = counter.update(subject)
                    pose_frame = subject
                last_state = cs

                out_image = image
                if renderer is not None:
                    out_image = renderer.draw(image, pose_frame, cs, exercise)

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            f"{timestamp:.4f}",
                            cs.current_state,
                            cs.rep_count,
                            cs.no_rep_count,
                            int(cs.hold),
                            *[
                                f"{cs.signal_values[n]:.4f}" if n in cs.signal_values else ""
                                for n in signal_names
                            ],
                        ]
                    )

                if output_path is not None:
                    if writer is None:
                        h, w = out_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (w, h))
                    writer.write(out_image)

                if show_window:
                    cv2.imshow(window_title, out_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            if writer is not None:
                writer.release()

    return last_state
