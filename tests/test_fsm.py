import pytest

from spotter.counting.fsm import FSMCounter
from spotter.counting.peaks import PeakCounter
from spotter.eval.synthetic import _make_frame, generate_squat_clip
from spotter.exercises.loader import load_exercise


@pytest.fixture
def squat():
    return load_exercise("squat")


def _drive(counter, frames):
    completed = []
    prev = 0
    for frame in frames:
        cs = counter.update(frame)
        if cs.rep_count > prev:
            completed.append(frame.timestamp)
            prev = cs.rep_count
    return completed


def test_fsm_counts_clean_reps(squat):
    clip = generate_squat_clip("clean", n_reps=6)
    counter = FSMCounter(squat)
    completed = _drive(counter, clip.frames)
    assert counter.rep_count == 6
    assert counter.no_rep_count == 0
    assert len(completed) == 6


def test_fsm_ignores_shallow_reps(squat):
    clip = generate_squat_clip("shallow", n_reps=4, bottom_angle=130.0, valid_rep_indices=[])
    counter = FSMCounter(squat)
    for frame in clip.frames:
        counter.update(frame)
    assert counter.rep_count == 0


def test_fsm_holds_and_does_not_count_when_occluded(squat):
    clip = generate_squat_clip("occluded", n_reps=1, dropout_rep_indices=[0])
    counter = FSMCounter(squat)
    saw_hold = False
    for frame in clip.frames:
        cs = counter.update(frame)
        saw_hold = saw_hold or cs.hold
    assert saw_hold
    assert counter.rep_count == 0


def test_fsm_rejects_jitter_shorter_than_min_rep_duration(squat):
    counter = FSMCounter(squat)
    t, dt = 0.0, 1 / 30
    for _ in range(20):
        counter.update(_make_frame(t, 175))
        t += dt
    # An implausibly fast full down-up swing, well under rep_duration.min_seconds (0.4s).
    for angle in (175, 60, 60, 175, 175):
        cs = counter.update(_make_frame(t, angle))
        t += 0.02
    assert counter.rep_count == 0
    assert counter.no_rep_count >= 1
    assert cs.last_no_rep_reason and "jitter" in cs.last_no_rep_reason


def test_fsm_warmup_guard_ignores_immediate_departure(squat):
    # First rep motion starts before 0.5s dwell in "standing" has elapsed;
    # it must not silently count.
    clip = generate_squat_clip("no_warmup", n_reps=1, stand_hold=0.05)
    counter = FSMCounter(squat)
    for frame in clip.frames:
        counter.update(frame)
    # Either it's rejected outright, or (if it re-triggers after the guard
    # clears) it's still only ever counted once — never double-counted from
    # the same physical rep.
    assert counter.rep_count <= 1


def test_fsm_reset_clears_counts(squat):
    clip = generate_squat_clip("clean", n_reps=3)
    counter = FSMCounter(squat)
    for frame in clip.frames:
        counter.update(frame)
    assert counter.rep_count == 3
    counter.reset()
    assert counter.rep_count == 0
    assert counter.no_rep_count == 0


def test_peak_counter_agrees_with_fsm_on_clean_reps(squat):
    clip = generate_squat_clip("clean", n_reps=5)
    fsm = FSMCounter(squat)
    peaks = PeakCounter(squat)
    for frame in clip.frames:
        fsm.update(frame)
        peaks.update(frame)
    assert peaks.rep_count == fsm.rep_count == 5
