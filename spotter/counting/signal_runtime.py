"""Turn an exercise's declared `SignalSpec`s into smoothed scalar values.

This is the one place that understands how the four signal *types* in
spotter/exercises/schema.py map onto the pure functions in
spotter/signals/geometry.py, and how `bilateral: average/left/right/none`
resolves generic point names (e.g. "knee") into frame-specific landmark
names (e.g. "left_knee"). Both `spotter.counting.fsm.FSMCounter` and
`spotter.counting.peaks.PeakCounter` build on this so the two counters are
guaranteed to see identical signal values when benchmarked head to head.
"""

from __future__ import annotations

from spotter.exercises.schema import ExerciseDef, SignalSpec
from spotter.pose.base import Frame
from spotter.signals.filters import OneEuroFilter
from spotter.signals.geometry import (
    bilateral_average,
    height_ratio,
    joint_angle,
    torso_lean,
    vertical_displacement,
)


def _resolve_sides(bilateral: str, base_names: list[str]) -> dict[str | None, list[str]]:
    if bilateral == "none":
        return {None: base_names}
    if bilateral == "average":
        return {
            "left": [f"left_{n}" for n in base_names],
            "right": [f"right_{n}" for n in base_names],
        }
    # "left" or "right"
    return {bilateral: [f"{bilateral}_{n}" for n in base_names]}


def _raw_value(sig: SignalSpec, frame: Frame, names: list[str]) -> float:
    points = [frame.point(n) for n in names]
    if sig.type == "joint_angle":
        a, b, c = points
        return joint_angle(a, b, c)
    if sig.type == "vertical_displacement":
        point, reference = points
        scale = 1.0
        if sig.scale_points:
            side_prefix = names[0].split("_")[0] if "_" in names[0] and len(names) else None
            scale_names = sig.scale_points
            if side_prefix in ("left", "right"):
                scale_names = [f"{side_prefix}_{n}" for n in sig.scale_points]
            s0, s1 = (frame.point(n) for n in scale_names)
            scale = float(((s0[:2] - s1[:2]) ** 2).sum() ** 0.5)
        return vertical_displacement(point, reference, scale)
    if sig.type == "height_ratio":
        a, b = points
        return height_ratio(a, b)
    if sig.type == "torso_lean":
        shoulder, hip = points
        return torso_lean(shoulder, hip)
    raise ValueError(f"Unhandled signal type: {sig.type}")


class SignalComputer:
    """Stateful (holds one OneEuroFilter per signal) — one instance per tracked subject."""

    def __init__(self, exercise: ExerciseDef):
        self.exercise = exercise
        self._filters: dict[str, OneEuroFilter] = {
            sig.name: OneEuroFilter(
                min_cutoff=sig.smoothing.min_cutoff,
                beta=sig.smoothing.beta,
                d_cutoff=sig.smoothing.d_cutoff,
            )
            for sig in exercise.signals
            if sig.smoothing.filter == "one_euro"
        }

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()

    def compute(self, frame: Frame) -> dict[str, float]:
        values: dict[str, float] = {}
        for sig in self.exercise.signals:
            sides = _resolve_sides(sig.bilateral, sig.points)
            side_values: dict[str | None, float] = {}
            side_vis: dict[str | None, float] = {}
            for side, names in sides.items():
                side_values[side] = _raw_value(sig, frame, names)
                side_vis[side] = frame.mean_visibility(names)

            if len(side_values) == 1:
                raw = next(iter(side_values.values()))
            else:
                raw = bilateral_average(
                    side_values["left"],
                    side_values["right"],
                    side_vis["left"],
                    side_vis["right"],
                )

            if sig.name in self._filters:
                values[sig.name] = self._filters[sig.name](raw, frame.timestamp)
            else:
                values[sig.name] = raw
        return values
