"""Accuracy metrics for one clip and aggregated across a whole eval run.

MAE and OBO (off-by-one accuracy) are the headline numbers; false-positive
and missed-rep counts come from timestamp matching and are what actually
tell you *why* a clip's count was wrong, which raw counts alone don't.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClipMetrics:
    clip_name: str
    exercise: str
    pred_count: int
    true_count: int
    false_positives: int
    missed: int

    @property
    def abs_error(self) -> int:
        return abs(self.pred_count - self.true_count)

    @property
    def off_by_one_ok(self) -> bool:
        return self.abs_error <= 1


@dataclass
class AggregateMetrics:
    n_clips: int
    mae: float
    obo_accuracy: float
    total_false_positives: int
    total_missed: int

    def to_dict(self) -> dict:
        return {
            "n_clips": self.n_clips,
            "mae": self.mae,
            "obo_accuracy": self.obo_accuracy,
            "total_false_positives": self.total_false_positives,
            "total_missed": self.total_missed,
        }


def match_reps(
    pred_timestamps: list[float], true_timestamps: list[float], tolerance_seconds: float = 1.0
) -> tuple[int, int, int]:
    """Greedy nearest-neighbor match within tolerance.

    Returns (n_matched, n_false_positives, n_missed).
    """
    remaining_true = list(true_timestamps)
    false_positives = 0
    matched = 0

    for pred_t in sorted(pred_timestamps):
        if not remaining_true:
            false_positives += 1
            continue
        closest = min(remaining_true, key=lambda t: abs(t - pred_t))
        if abs(closest - pred_t) <= tolerance_seconds:
            remaining_true.remove(closest)
            matched += 1
        else:
            false_positives += 1

    missed = len(remaining_true)
    return matched, false_positives, missed


def evaluate_clip(
    clip_name: str,
    exercise: str,
    pred_timestamps: list[float],
    true_timestamps: list[float],
    tolerance_seconds: float = 1.0,
) -> ClipMetrics:
    _matched, false_positives, missed = match_reps(
        pred_timestamps, true_timestamps, tolerance_seconds
    )
    return ClipMetrics(
        clip_name=clip_name,
        exercise=exercise,
        pred_count=len(pred_timestamps),
        true_count=len(true_timestamps),
        false_positives=false_positives,
        missed=missed,
    )


def aggregate(clips: list[ClipMetrics]) -> AggregateMetrics:
    if not clips:
        return AggregateMetrics(0, 0.0, 0.0, 0, 0)
    mae = sum(c.abs_error for c in clips) / len(clips)
    obo = sum(1 for c in clips if c.off_by_one_ok) / len(clips)
    return AggregateMetrics(
        n_clips=len(clips),
        mae=mae,
        obo_accuracy=obo,
        total_false_positives=sum(c.false_positives for c in clips),
        total_missed=sum(c.missed for c in clips),
    )
