"""Rolling buffer of recent `Frame`s, time-windowed rather than count-windowed.

Video frame rate isn't guaranteed constant (dropped frames, variable-rate
webcam capture), so windows for calibration, form checks, and the overlay's
scrolling plot are all defined in seconds and trimmed by timestamp.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from spotter.pose.base import Frame


@dataclass
class LandmarkBuffer:
    max_seconds: float = 10.0
    _frames: deque[Frame] = field(default_factory=deque, repr=False)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)
        cutoff = frame.timestamp - self.max_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def latest(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def since(self, t_start: float) -> list[Frame]:
        return [f for f in self._frames if f.timestamp >= t_start]

    def last_seconds(self, seconds: float) -> list[Frame]:
        if not self._frames:
            return []
        cutoff = self._frames[-1].timestamp - seconds
        return [f for f in self._frames if f.timestamp >= cutoff]

    def mean_visibility(self, required_landmarks: list[str]) -> float:
        frame = self.latest
        if frame is None:
            return 0.0
        return frame.mean_visibility(required_landmarks)
