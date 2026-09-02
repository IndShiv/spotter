"""Shared confidence-gating logic, used identically by FSMCounter and PeakCounter.

Enforced globally per the project spec: if mean visibility of an exercise's
required landmarks drops below its threshold for N consecutive frames, the
counter must stop counting and surface that fact rather than guessing.
"""

from __future__ import annotations

from spotter.exercises.schema import LandmarksSpec
from spotter.pose.base import Frame


class ConfidenceGate:
    def __init__(self, landmarks: LandmarksSpec):
        self.landmarks = landmarks
        self._low_vis_frames = 0

    def reset(self) -> None:
        self._low_vis_frames = 0

    def update(self, frame: Frame) -> tuple[bool, float]:
        """Returns (should_hold, mean_visibility) for this frame."""
        vis = frame.mean_visibility(self.landmarks.required)
        if vis < self.landmarks.min_visibility:
            self._low_vis_frames += 1
        else:
            self._low_vis_frames = 0
        return self._low_vis_frames >= self.landmarks.frames_before_hold, vis
