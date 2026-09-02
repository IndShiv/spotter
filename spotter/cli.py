"""Typer CLI entry point: `spotter video`, `spotter live`, `spotter eval`."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    help="Spotter: a computer-vision rep counter for functional-fitness training.",
    add_completion=False,
)


@app.command()
def video(
    path: Path = typer.Argument(..., exists=True, help="Path to a recorded video clip."),
    exercise: str = typer.Option(..., "--exercise", "-e", help="Exercise name, e.g. squat."),
    counter: str = typer.Option("fsm", help="Counting strategy: fsm or peaks."),
    model: str = typer.Option("heavy", help="Pose model variant: lite, full, or heavy."),
    overlay: bool = typer.Option(True, help="Draw the debug overlay."),
    show: bool = typer.Option(True, help="Show a live window while processing."),
    output: Path | None = typer.Option(None, help="Write the annotated video here."),
    dump_signals: Path | None = typer.Option(None, help="Write a per-frame signal CSV here."),
    num_poses: int = typer.Option(3, help="Max people to detect per frame (before locking)."),
    lock_frames: int = typer.Option(30, help="Frames to observe before locking onto a subject."),
) -> None:
    """Count reps in a recorded video clip."""
    from spotter.commands.video import run_video

    result = run_video(
        path,
        exercise=exercise,
        counter=counter,
        model=model,
        overlay=overlay,
        show_window=show,
        output_path=output,
        dump_signals_path=dump_signals,
        num_poses=num_poses,
        lock_frames=lock_frames,
    )
    typer.echo(f"Reps: {result.rep_count}   No-rep: {result.no_rep_count}")


@app.command()
def live(
    exercise: str = typer.Option(..., "--exercise", "-e", help="Exercise name, e.g. squat."),
    camera: int = typer.Option(0, help="OpenCV camera index."),
    counter: str = typer.Option("fsm", help="Counting strategy: fsm or peaks."),
    model: str = typer.Option("lite", help="Pose model variant: lite, full, or heavy."),
    overlay: bool = typer.Option(True, help="Draw the debug overlay."),
    dump_signals: Path | None = typer.Option(None, help="Write a per-frame signal CSV here."),
    num_poses: int = typer.Option(3, help="Max people to detect per frame (before locking)."),
    lock_frames: int = typer.Option(30, help="Frames to observe before locking onto a subject."),
) -> None:
    """Count reps live from a webcam. Press 'q' or Esc to stop."""
    from spotter.commands.live import run_live

    result = run_live(
        exercise=exercise,
        camera=camera,
        counter=counter,
        model=model,
        overlay=overlay,
        dump_signals_path=dump_signals,
        num_poses=num_poses,
        lock_frames=lock_frames,
    )
    typer.echo(f"Reps: {result.rep_count}   No-rep: {result.no_rep_count}")


@app.command("list-exercises")
def list_exercises() -> None:
    """Print the names of all available exercise definitions."""
    from spotter.exercises.loader import available_exercises

    for name in available_exercises():
        typer.echo(name)


@app.command()
def eval(
    videos_dir: Path = typer.Option(Path("data/videos"), help="Directory of annotated real clips."),
    baseline: Path = typer.Option(Path("eval/baseline.json"), help="Regression baseline file."),
    out: Path = typer.Option(Path("eval/results/latest.json"), help="Where to write this run's results."),
    no_gate: bool = typer.Option(False, help="Skip the regression gate check."),
    update_baseline: bool = typer.Option(False, help="Overwrite the baseline with this run's result."),
) -> None:
    """Run the eval harness (synthetic clips + any real annotated clips found)."""
    from spotter.eval.run import main as eval_main

    argv = ["--videos-dir", str(videos_dir), "--baseline", str(baseline), "--out", str(out)]
    if no_gate:
        argv.append("--no-gate")
    if update_baseline:
        argv.append("--update-baseline")
    raise typer.Exit(eval_main(argv))


if __name__ == "__main__":
    app()
