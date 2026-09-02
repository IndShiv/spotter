from spotter.eval.metrics import aggregate, evaluate_clip, match_reps
from spotter.eval.run import run_synthetic_clips
from spotter.eval.synthetic import standard_synthetic_clips


def test_match_reps_exact():
    matched, fp, missed = match_reps([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert (matched, fp, missed) == (3, 0, 0)


def test_match_reps_false_positive_and_missed():
    matched, fp, missed = match_reps([1.0, 2.0, 10.0], [1.0, 2.0, 3.0])
    assert matched == 2
    assert fp == 1
    assert missed == 1


def test_evaluate_clip_off_by_one():
    m = evaluate_clip("c", "squat", [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert m.pred_count == 4
    assert m.true_count == 3
    assert m.abs_error == 1
    assert m.off_by_one_ok


def test_aggregate_empty_is_safe():
    agg = aggregate([])
    assert agg.n_clips == 0
    assert agg.mae == 0.0


def test_standard_synthetic_clips_are_well_formed():
    clips = standard_synthetic_clips()
    assert 3 <= len(clips) <= 5
    for clip in clips:
        assert clip.frames
        assert clip.exercise == "squat"


def test_synthetic_eval_meets_recorded_baseline():
    """The regression gate itself, as a pytest so `pytest` alone catches a
    counting regression without needing `make eval` or real video files."""
    import json
    from pathlib import Path

    results = run_synthetic_clips()
    agg = aggregate(results)

    baseline_path = Path(__file__).resolve().parents[1] / "eval" / "baseline.json"
    baseline = json.loads(baseline_path.read_text())
    assert agg.obo_accuracy >= baseline["obo_accuracy"] - 1e-9
