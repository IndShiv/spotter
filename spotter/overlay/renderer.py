"""Debug overlay: skeleton, live signal + FSM state, rep count, and a
scrolling plot of the last 5s of the primary signal with transitions marked.

This is the primary tool for tuning thresholds by eye instead of by trial
and error — pair it with `--dump-signals` (spotter/eval or the CLI) when you
need exact numbers rather than a picture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from spotter.counting.base import CounterState
from spotter.exercises.schema import ExerciseDef
from spotter.pose.base import Frame
from spotter.pose.landmarks import POSE_CONNECTIONS

_SKELETON_COLOR = (80, 220, 80)
_JOINT_COLOR = (60, 180, 255)
_TEXT_COLOR = (255, 255, 255)
_HOLD_COLOR = (40, 40, 230)
_PLOT_BG = (30, 30, 30)
_PLOT_LINE = (80, 220, 80)
_PLOT_AXIS = (110, 110, 110)
_TRANSITION_COLOR = (0, 165, 255)


@dataclass
class _PlotSample:
    timestamp: float
    value: float
    state: str


class OverlayRenderer:
    def __init__(self, plot_window_seconds: float = 5.0, plot_height: int = 120):
        self.plot_window_seconds = plot_window_seconds
        self.plot_height = plot_height
        self._history: deque[_PlotSample] = deque()
        self._last_state: str | None = None

    def draw(
        self,
        image_bgr: np.ndarray,
        pose_frame: Frame | None,
        counter_state: CounterState,
        exercise: ExerciseDef,
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        out = image_bgr.copy()

        if pose_frame is not None:
            _draw_skeleton(out, pose_frame, w, h)

        primary_value = counter_state.signal_values.get(exercise.primary_signal)
        timestamp = pose_frame.timestamp if pose_frame is not None else None
        if primary_value is not None and timestamp is not None:
            self._history.append(_PlotSample(timestamp, primary_value, counter_state.current_state))
            cutoff = timestamp - self.plot_window_seconds
            while self._history and self._history[0].timestamp < cutoff:
                self._history.popleft()
            self._last_state = counter_state.current_state

        _draw_hud(out, counter_state, exercise, primary_value)
        _draw_plot(out, self._history, self.plot_height, exercise.primary_signal)
        return out


def _draw_skeleton(img: np.ndarray, frame: Frame, w: int, h: int) -> None:
    def px(name: str) -> tuple[int, int]:
        pt = frame.point(name)
        return int(pt[0] * w), int(pt[1] * h)

    for a, b in POSE_CONNECTIONS:
        va, vb = frame.vis(a), frame.vis(b)
        if va < 0.3 or vb < 0.3:
            continue
        cv2.line(img, px(a), px(b), _SKELETON_COLOR, 2, cv2.LINE_AA)

    for idx in range(len(frame.visibility)):
        if frame.visibility[idx] < 0.3:
            continue
        x, y = int(frame.landmarks[idx, 0] * w), int(frame.landmarks[idx, 1] * h)
        cv2.circle(img, (x, y), 3, _JOINT_COLOR, -1, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    counter_state: CounterState,
    exercise: ExerciseDef,
    primary_value: float | None,
) -> None:
    lines = [
        f"{exercise.display_name}",
        f"Reps: {counter_state.rep_count}   No-rep: {counter_state.no_rep_count}",
        f"State: {counter_state.current_state}"
        + ("  [calibrating]" if counter_state.calibrating else ""),
    ]
    if primary_value is not None:
        lines.append(f"{exercise.primary_signal}: {primary_value:.1f}")
    if counter_state.last_no_rep_reason:
        lines.append(f"Last no-rep: {counter_state.last_no_rep_reason}")

    y = 28
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 1, cv2.LINE_AA)
        y += 26

    if counter_state.hold:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), _HOLD_COLOR, 6)
        text = counter_state.hold_reason or "Spotter can't see you"
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx, ty = (w - size[0]) // 2, h // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _HOLD_COLOR, 2, cv2.LINE_AA)


def _draw_plot(img: np.ndarray, history: deque, plot_height: int, signal_name: str) -> None:
    if len(history) < 2:
        return
    h, w = img.shape[:2]
    plot_w = w
    y0 = h - plot_height
    overlay = img[y0:h, 0:plot_w]
    cv2.rectangle(overlay, (0, 0), (plot_w, plot_height), _PLOT_BG, -1)
    cv2.addWeighted(overlay, 0.7, img[y0:h, 0:plot_w], 0.3, 0, dst=img[y0:h, 0:plot_w])

    values = np.array([s.value for s in history])
    times = np.array([s.timestamp for s in history])
    v_min, v_max = float(values.min()), float(values.max())
    if v_max - v_min < 1e-6:
        v_max = v_min + 1.0
    t_min, t_max = times[0], times[-1]
    t_span = max(t_max - t_min, 1e-6)

    def to_xy(t: float, v: float) -> tuple[int, int]:
        x = int((t - t_min) / t_span * (plot_w - 1))
        y = int(y0 + plot_height - 1 - (v - v_min) / (v_max - v_min) * (plot_height - 20) - 5)
        return x, y

    cv2.line(img, (0, y0), (plot_w, y0), _PLOT_AXIS, 1, cv2.LINE_AA)

    points = [to_xy(s.timestamp, s.value) for s in history]
    for p0, p1 in pairwise(points):
        cv2.line(img, p0, p1, _PLOT_LINE, 2, cv2.LINE_AA)

    prev_state = history[0].state
    for sample, (x, _y) in zip(history, points):
        if sample.state != prev_state:
            cv2.line(img, (x, y0), (x, h), _TRANSITION_COLOR, 1, cv2.LINE_AA)
            prev_state = sample.state

    label = f"{signal_name}  [{v_min:.0f} - {v_max:.0f}] / {5:.0f}s"
    cv2.putText(
        img, label, (8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT_COLOR, 1, cv2.LINE_AA
    )
