"""Lock onto one athlete and ignore everyone else who walks through frame.

Strategy: for the first `lock_frames` frames, assume the largest person in
view (by bbox area) is the athlete being filmed — in a gym setting the
person the camera is pointed at is almost always closer/larger than someone
passing behind them. Once locked, track by bounding-box IoU continuity
frame to frame, falling back to nearest-centroid when the box moved too far
for IoU to overlap (fast motion, e.g. a burpee's jump).
"""

from __future__ import annotations

from spotter.pose.base import Frame

BBox = tuple[float, float, float, float]


def _area(box: BBox) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _iou(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 1e-9 else 0.0


def _centroid(box: BBox) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _centroid_dist(a: BBox, b: BBox) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class SingleSubjectTracker:
    def __init__(self, lock_frames: int = 30):
        self.lock_frames = lock_frames
        self.reset()

    def reset(self) -> None:
        self._frames_seen = 0
        self._locked = False
        self._locked_bbox: BBox | None = None

    @property
    def locked(self) -> bool:
        return self._locked

    def select(self, detections: list[Frame]) -> Frame | None:
        """Given all detections in a frame, return the one that's our athlete."""
        if not detections:
            return None

        if not self._locked:
            chosen = max(detections, key=lambda f: _area(f.bbox))
            self._frames_seen += 1
            self._locked_bbox = chosen.bbox
            if self._frames_seen >= self.lock_frames:
                self._locked = True
            return chosen

        assert self._locked_bbox is not None
        best = max(detections, key=lambda f: _iou(f.bbox, self._locked_bbox))
        if _iou(best.bbox, self._locked_bbox) <= 0.0:
            best = min(detections, key=lambda f: _centroid_dist(f.bbox, self._locked_bbox))
        self._locked_bbox = best.bbox
        return best
