import numpy as np

from spotter.pose.base import Frame
from spotter.tracking.single_subject import SingleSubjectTracker


def _frame(x0, y0, x1, y1, t=0.0):
    lm = np.zeros((33, 3))
    world = lm.copy()
    vis = np.ones(33)
    return Frame(timestamp=t, landmarks=lm, world_landmarks=world, visibility=vis, bbox=(x0, y0, x1, y1))


def test_locks_onto_largest_bbox_during_lock_window():
    tracker = SingleSubjectTracker(lock_frames=3)
    big = _frame(0.1, 0.1, 0.9, 0.9)
    small = _frame(0.0, 0.0, 0.1, 0.1)
    for _ in range(3):
        chosen = tracker.select([small, big])
        assert chosen is big
    assert tracker.locked


def test_continuity_tracks_nearest_after_lock():
    tracker = SingleSubjectTracker(lock_frames=1)
    first = _frame(0.4, 0.4, 0.6, 0.6)
    tracker.select([first])
    assert tracker.locked

    same_person_moved = _frame(0.42, 0.42, 0.62, 0.62)
    stranger = _frame(0.0, 0.0, 0.05, 0.05)
    chosen = tracker.select([stranger, same_person_moved])
    assert chosen is same_person_moved


def test_returns_none_when_no_detections():
    tracker = SingleSubjectTracker(lock_frames=1)
    assert tracker.select([]) is None
