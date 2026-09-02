"""Backend-agnostic pose data model and the `PoseBackend` protocol.

Every pose backend (MediaPipe today, Ultralytics YOLO-pose later) must
normalize its output into `Frame` objects indexed against the shared
33-point BlazePose topology in `spotter.pose.landmarks`. Nothing downstream
of this module may import a backend-specific type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np

from spotter.pose.landmarks import LANDMARK_NAMES, landmark_index

NUM_LANDMARKS = len(LANDMARK_NAMES)


@dataclass
class Frame:
    """One detected person, one instant in time.

    landmarks: (33, 3) array of normalized image-space (x, y, z) in [0, 1]
        for x/y; z is relative depth, roughly in image-width units.
    world_landmarks: (33, 3) array of metric world-space (x, y, z) in
        meters, centered on the hips. Used for signals that need real
        distances rather than perspective-distorted image coordinates.
    visibility: (33,) array in [0, 1]; MediaPipe's estimate of whether each
        landmark is visible (not occluded) in this frame.
    bbox: (x_min, y_min, x_max, y_max) in normalized image coordinates,
        derived from the landmark extent. Used for single-subject locking.
    """

    timestamp: float
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, arr in (
            ("landmarks", self.landmarks),
            ("world_landmarks", self.world_landmarks),
        ):
            if arr.shape != (NUM_LANDMARKS, 3):
                raise ValueError(f"{name} must have shape ({NUM_LANDMARKS}, 3), got {arr.shape}")
        if self.visibility.shape != (NUM_LANDMARKS,):
            raise ValueError(
                f"visibility must have shape ({NUM_LANDMARKS},), got {self.visibility.shape}"
            )

    def point(self, name: str, world: bool = False) -> np.ndarray:
        """xyz for a named landmark, image-space by default."""
        arr = self.world_landmarks if world else self.landmarks
        return arr[landmark_index(name)]

    def vis(self, name: str) -> float:
        return float(self.visibility[landmark_index(name)])

    def mean_visibility(self, names: list[str]) -> float:
        if not names:
            return 1.0
        idx = [landmark_index(n) for n in names]
        return float(np.mean(self.visibility[idx]))

    @staticmethod
    def bbox_from_landmarks(landmarks: np.ndarray) -> tuple[float, float, float, float]:
        xs, ys = landmarks[:, 0], landmarks[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class PoseBackend(Protocol):
    """A pose estimator that turns raw frames into `Frame` detections.

    A single call to `process` may return zero, one, or several detections
    (multiple people in view) — single-subject selection happens downstream
    in `spotter.tracking`, not inside the backend.
    """

    def process(self, image: np.ndarray, timestamp: float) -> list[Frame]:
        """Run pose estimation on one BGR image, return detections."""
        ...

    def close(self) -> None:
        """Release any backend resources (model handles, GPU context, ...)."""
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
