"""Stub for a future YOLO11/26-pose backend (Ultralytics).

Not implemented in Phase 1. Kept here so `spotter.pose.PoseBackend` has a
second concrete implementation on the roadmap and the CLI's `--pose-backend`
flag has somewhere to point once this lands — swapping backends should
never require touching spotter/signals, spotter/exercises, or
spotter/counting, since they only ever see `Frame`.

YOLO-pose uses a 17-point COCO keypoint layout, not BlazePose's 33 points —
implementing this means writing a COCO -> BlazePose-index remapper (most
joints have a direct match; a few, like heel and foot_index, don't exist in
COCO and would need to be left as zero-visibility/unavailable) inside
`process()`, so everything downstream keeps working unmodified.
"""

from __future__ import annotations

import numpy as np

from spotter.pose.base import Frame


class UltralyticsPoseBackend:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "UltralyticsPoseBackend is a Phase 2 stub. Use MediaPipeBackend for now."
        )

    def process(self, image: np.ndarray, timestamp: float) -> list[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
