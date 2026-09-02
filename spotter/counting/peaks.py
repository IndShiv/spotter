"""PCA + peak-finding alternative to the FSM counter, for head-to-head benchmarking.

Where the FSM encodes an explicit model of the movement (named states,
thresholds, form checks), this counter is deliberately dumb: project the
exercise's declared signals onto their dominant axis of variation and count
peaks in that 1D trace with `scipy.signal.find_peaks`. It shares the same
`SignalComputer` and confidence gate as the FSM so the two are comparable on
identical inputs — see `spotter/eval/run.py`.

This recomputes PCA and re-scans the whole rolling window on every frame
(O(window) per frame), which is fine for a prototype at video frame rates
but is not a true incremental/streaming implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from spotter.counting.base import CounterState, RepCounter
from spotter.counting.confidence import ConfidenceGate
from spotter.counting.signal_runtime import SignalComputer
from spotter.exercises.schema import ExerciseDef
from spotter.pose.base import Frame

# Peaks within this many samples of the live edge of the window are not
# trusted yet — they might still grow in prominence as more frames arrive.
_EDGE_GUARD_SAMPLES = 5


@dataclass
class _Sample:
    timestamp: float
    values: list[float]


class PeakCounter(RepCounter):
    def __init__(
        self,
        exercise: ExerciseDef,
        window_seconds: float = 30.0,
        prominence_ratio: float = 0.4,
    ):
        self.exercise = exercise
        self.window_seconds = window_seconds
        self.prominence_ratio = prominence_ratio
        self._signal_names = [s.name for s in exercise.signals]
        self._signals = SignalComputer(exercise)
        self._gate = ConfidenceGate(exercise.landmarks)
        self.reset()

    def reset(self) -> None:
        self._signals.reset()
        self._gate.reset()
        self.rep_count = 0
        self.no_rep_count = 0
        self._history: list[_Sample] = []
        self._last_counted_time: float = float("-inf")

    def update(self, frame: Frame) -> CounterState:
        should_hold, _mean_vis = self._gate.update(frame)
        values = self._signals.compute(frame)

        if should_hold:
            return CounterState(
                rep_count=self.rep_count,
                no_rep_count=self.no_rep_count,
                current_state="hold",
                signal_values=values,
                hold=True,
                hold_reason=self.exercise.confidence_gate.hold_message,
            )

        self._history.append(_Sample(frame.timestamp, [values[n] for n in self._signal_names]))
        cutoff = frame.timestamp - self.window_seconds
        self._history = [s for s in self._history if s.timestamp >= cutoff]

        min_samples = max(10, int(self.exercise.rep_duration.min_seconds * 2))
        just_completed_rep = False
        if len(self._history) >= min_samples:
            just_completed_rep = self._scan_for_new_peaks()

        return CounterState(
            rep_count=self.rep_count,
            no_rep_count=self.no_rep_count,
            current_state="tracking",
            signal_values=values,
            just_completed_rep=just_completed_rep,
        )

    def _project_to_1d(self) -> np.ndarray:
        matrix = np.array([s.values for s in self._history], dtype=float)  # (T, D)
        if matrix.shape[1] == 1:
            return matrix[:, 0]

        centered = matrix - matrix.mean(axis=0, keepdims=True)
        std = centered.std(axis=0, keepdims=True)
        std[std < 1e-9] = 1.0
        centered = centered / std

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        projected = centered @ principal

        # Orient so the sign matches the primary signal, for interpretability.
        primary_idx = self._signal_names.index(self.exercise.primary_signal)
        if np.corrcoef(projected, centered[:, primary_idx])[0, 1] < 0:
            projected = -projected
        return projected

    def _scan_for_new_peaks(self) -> bool:
        y = self._project_to_1d()
        timestamps = np.array([s.timestamp for s in self._history])

        avg_dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 1.0 / 30.0
        distance_samples = max(1, int(self.exercise.rep_duration.min_seconds / max(avg_dt, 1e-6)))
        prominence = self.prominence_ratio * (y.std() if y.std() > 1e-9 else 1.0)

        # Peaks correspond to the trough of the primary signal (e.g. squat
        # depth), which is a *minimum* since we oriented `y` to match it.
        peak_idx, _props = find_peaks(-y, distance=distance_samples, prominence=prominence)

        edge = len(y) - _EDGE_GUARD_SAMPLES
        counted_this_frame = False
        for idx in peak_idx:
            if idx >= edge:
                continue
            t = timestamps[idx]
            if t <= self._last_counted_time:
                continue
            self.rep_count += 1
            self._last_counted_time = t
            counted_this_frame = True
        return counted_this_frame
