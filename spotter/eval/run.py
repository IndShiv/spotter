"""Eval harness entry point: `python -m spotter.eval.run` (also `make eval`).

Runs the fixed synthetic clip set (always) plus any annotated real clips
found under data/videos/ (only if the video files are actually present
locally — they're gitignored, so CI runs synthetic-only), then checks
aggregate OBO against eval/baseline.json as a regression gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spotter.counting.fsm import FSMCounter
from spotter.eval.metrics import ClipMetrics, aggregate, evaluate_clip
from spotter.eval.synthetic import standard_synthetic_clips
from spotter.exercises.loader import load_exercise
from spotter.pose.base import Frame

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_DIR = REPO_ROOT / "data" / "videos"
DEFAULT_BASELINE = REPO_ROOT / "eval" / "baseline.json"
DEFAULT_OUT = REPO_ROOT / "eval" / "results" / "latest.json"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def run_synthetic_clips() -> list[ClipMetrics]:
    results = []
    for clip in standard_synthetic_clips():
        exercise = load_exercise(clip.exercise)
        counter = FSMCounter(exercise)
        pred_timestamps = _drive_counter(counter, clip.frames)
        results.append(
            evaluate_clip(clip.name, clip.exercise, pred_timestamps, clip.true_rep_timestamps)
        )
    return results


def run_real_clips(videos_dir: Path) -> list[ClipMetrics]:
    """Run any real, annotated clips found in `videos_dir`.

    Each clip needs a `<stem>.json` sidecar (exercise, true_count,
    rep_timestamps, camera_angle, notes) and a matching video file with the
    same stem. Sidecars without a video present locally are skipped, not
    errored — that's the expected state for anyone who hasn't pulled the
    (gitignored) footage.
    """
    pending = []
    for json_path in sorted(videos_dir.glob("*.json")):
        video_path = next(
            (json_path.with_suffix(ext) for ext in VIDEO_EXTENSIONS if json_path.with_suffix(ext).exists()),
            None,
        )
        if video_path is not None:
            pending.append((json_path, video_path))

    if not pending:
        return []

    import cv2

    from spotter.pose.mediapipe_backend import MediaPipeBackend
    from spotter.tracking.single_subject import SingleSubjectTracker

    results = []
    for json_path, video_path in pending:
        meta = json.loads(json_path.read_text())
        exercise = load_exercise(meta["exercise"])
        backend = MediaPipeBackend(model_variant="heavy")
        tracker = SingleSubjectTracker()
        counter = FSMCounter(exercise)
        pred_timestamps: list[float] = []
        prev_rep_count = 0

        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                subject = tracker.select(backend.process(image, t))
                if subject is None:
                    continue
                cs = counter.update(subject)
                if cs.rep_count > prev_rep_count:
                    pred_timestamps.append(t)
                    prev_rep_count = cs.rep_count
        finally:
            cap.release()
            backend.close()

        results.append(
            evaluate_clip(json_path.stem, meta["exercise"], pred_timestamps, meta["rep_timestamps"])
        )
    return results


def _drive_counter(counter: FSMCounter, frames: list[Frame]) -> list[float]:
    pred_timestamps = []
    prev_rep_count = 0
    for frame in frames:
        cs = counter.update(frame)
        if cs.rep_count > prev_rep_count:
            pred_timestamps.append(frame.timestamp)
            prev_rep_count = cs.rep_count
    return pred_timestamps


def _print_report(clips: list[ClipMetrics]) -> None:
    header = f"{'clip':<28}{'exercise':<10}{'pred':>6}{'true':>6}{'err':>6}{'OBO':>6}{'FP':>5}{'missed':>8}"
    print(header)
    print("-" * len(header))
    for c in clips:
        print(
            f"{c.clip_name:<28}{c.exercise:<10}{c.pred_count:>6}{c.true_count:>6}"
            f"{c.abs_error:>6}{'ok' if c.off_by_one_ok else 'X':>6}{c.false_positives:>5}{c.missed:>8}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spotter eval harness")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-gate", action="store_true", help="Skip the regression gate check.")
    parser.add_argument(
        "--update-baseline", action="store_true", help="Overwrite the baseline with this run's result."
    )
    args = parser.parse_args(argv)

    clips = run_synthetic_clips()
    if args.videos_dir.exists():
        clips += run_real_clips(args.videos_dir)

    _print_report(clips)
    agg = aggregate(clips)
    print()
    print(
        f"Aggregate over {agg.n_clips} clips: MAE={agg.mae:.3f}  OBO={agg.obo_accuracy:.3f}  "
        f"false_positives={agg.total_false_positives}  missed={agg.total_missed}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "clips": [vars(c) | {"abs_error": c.abs_error, "off_by_one_ok": c.off_by_one_ok} for c in clips],
                "aggregate": agg.to_dict(),
            },
            indent=2,
        )
    )

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(agg.to_dict(), indent=2) + "\n")
        print(f"Baseline updated: {args.baseline}")
        return 0

    if args.no_gate:
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; run with --update-baseline to create one.")
        return 0

    baseline = json.loads(args.baseline.read_text())
    if agg.obo_accuracy < baseline["obo_accuracy"] - 1e-9:
        print(
            f"REGRESSION: OBO {agg.obo_accuracy:.3f} is below baseline "
            f"{baseline['obo_accuracy']:.3f} ({args.baseline})"
        )
        return 1

    print("OK: no regression against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
