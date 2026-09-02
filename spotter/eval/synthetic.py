"""Synthetic squat clips: parametric landmark sequences with known ground
truth, so the eval harness (and CI) can run without any real video files.

Each clip is a smooth knee-angle trajectory (stand -> depth -> stand,
repeated) turned into `Frame` objects via a two-segment-leg model, with
optional Gaussian landmark noise and visibility dropout windows layered on
top to probe specific failure modes (jitter robustness, "can't see you"
gating) rather than just the happy path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spotter.pose.base import Frame
from spotter.pose.landmarks import LANDMARK_NAMES, landmark_index

_NUM_LANDMARKS = len(LANDMARK_NAMES)
_THIGH = 0.25
_SHANK = 0.25


@dataclass
class SyntheticClip:
    name: str
    exercise: str
    frames: list[Frame]
    true_rep_timestamps: list[float]
    notes: str = ""

    @property
    def true_count(self) -> int:
        return len(self.true_rep_timestamps)


def _knee_ankle(hip: np.ndarray, knee: np.ndarray, angle_deg: float) -> np.ndarray:
    beta = np.radians(180 - angle_deg)
    return knee + _SHANK * np.array([np.sin(beta), np.cos(beta), 0.0])


def _rep_waveform(t_in_rep: float, rep_duration: float, top_angle: float, bottom_angle: float) -> float:
    """Smooth stand -> depth -> stand trajectory via a raised cosine, one full cycle per rep."""
    phase = (t_in_rep / rep_duration) * 2 * np.pi
    # cos(phase) goes 1 -> -1 -> 1 over the rep; map to top_angle -> bottom_angle -> top_angle
    frac = (1 - np.cos(phase)) / 2.0  # 0 -> 1 -> 0
    return top_angle - frac * (top_angle - bottom_angle)


def _make_frame(
    t: float,
    knee_angle_deg: float,
    torso_lean_deg: float = 0.0,
    visibility: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Frame:
    lm = np.zeros((_NUM_LANDMARKS, 3))
    lean_dx = _THIGH * np.tan(np.radians(torso_lean_deg))
    for side, x in (("left", 0.45), ("right", 0.55)):
        hip = np.array([x, 0.5, 0.0])
        shoulder = np.array([x + lean_dx, 0.2, 0.0])
        knee = np.array([x, hip[1] + _THIGH, 0.0])
        ankle = _knee_ankle(hip, knee, knee_angle_deg)
        lm[landmark_index(f"{side}_shoulder")] = shoulder
        lm[landmark_index(f"{side}_hip")] = hip
        lm[landmark_index(f"{side}_knee")] = knee
        lm[landmark_index(f"{side}_ankle")] = ankle

    if noise_std > 0 and rng is not None:
        lm[:, :2] += rng.normal(0, noise_std, size=(_NUM_LANDMARKS, 2))

    world = lm.copy()
    vis = np.full(_NUM_LANDMARKS, visibility)
    return Frame(
        timestamp=t,
        landmarks=lm,
        world_landmarks=world,
        visibility=vis,
        bbox=Frame.bbox_from_landmarks(lm),
    )


def generate_squat_clip(
    name: str,
    n_reps: int = 5,
    fps: int = 30,
    rep_duration: float = 1.5,
    stand_hold: float = 0.6,
    top_angle: float = 175.0,
    bottom_angle: float = 75.0,
    noise_std: float = 0.0,
    dropout_rep_indices: list[int] | None = None,
    valid_rep_indices: list[int] | None = None,
    seed: int = 0,
    notes: str = "",
) -> SyntheticClip:
    """Generate one synthetic squat clip.

    dropout_rep_indices: reps (0-indexed) during which visibility drops to
    near zero for the whole rep, simulating the athlete stepping out of
    frame / being occluded. Those reps are still included in
    `true_rep_timestamps` (they really happened) but Spotter is expected to
    miss them, which is the point of the scenario.

    valid_rep_indices: which reps count toward ground truth at all, matching
    how a human judge would score them — e.g. a rep that never reached depth
    isn't "one rep counted wrong", it's zero valid reps by definition.
    Defaults to every rep being valid.
    """
    rng = np.random.default_rng(seed)
    dropout = set(dropout_rep_indices or [])
    valid = set(range(n_reps) if valid_rep_indices is None else valid_rep_indices)
    frames: list[Frame] = []
    true_timestamps: list[float] = []
    t = 0.0
    dt = 1.0 / fps

    for rep_idx in range(n_reps):
        vis = 0.05 if rep_idx in dropout else 1.0

        for _ in range(int(stand_hold * fps)):
            frames.append(_make_frame(t, top_angle, visibility=vis, noise_std=noise_std, rng=rng))
            t += dt

        n_steps = int(rep_duration * fps)
        for i in range(n_steps):
            angle = _rep_waveform(i * dt, rep_duration, top_angle, bottom_angle)
            frames.append(_make_frame(t, angle, visibility=vis, noise_std=noise_std, rng=rng))
            t += dt

        # Ground truth: the rep "happened" at the return to standing.
        if rep_idx in valid:
            true_timestamps.append(t)

    return SyntheticClip(
        name=name, exercise="squat", frames=frames, true_rep_timestamps=true_timestamps, notes=notes
    )


def standard_synthetic_clips() -> list[SyntheticClip]:
    """The fixed set of synthetic clips used by `make eval` and CI."""
    return [
        generate_squat_clip(
            "synthetic_clean",
            n_reps=5,
            noise_std=0.0,
            notes="No noise, no dropout — the happy path.",
        ),
        generate_squat_clip(
            "synthetic_noisy",
            n_reps=5,
            noise_std=0.006,
            seed=1,
            notes="Landmark jitter comparable to a real MediaPipe stream; should not affect the count.",
        ),
        generate_squat_clip(
            "synthetic_heavy_noise",
            n_reps=5,
            noise_std=0.015,
            seed=2,
            notes="Noise well above typical MediaPipe jitter; a stress test for the One-Euro filter and hysteresis.",
        ),
        generate_squat_clip(
            "synthetic_dropout",
            n_reps=5,
            dropout_rep_indices=[2],
            seed=3,
            notes="Rep 3 occurs while visibility is near zero; Spotter should HOLD and miss it rather than guess.",
        ),
        generate_squat_clip(
            "synthetic_shallow_reps",
            n_reps=5,
            bottom_angle=130.0,
            valid_rep_indices=[],
            seed=4,
            notes=(
                "Never reaches real depth (knee_angle stays above the bottom threshold). "
                "A judge would give zero credit for these, so ground truth is 0 valid reps — "
                "this checks that Spotter agrees rather than counting motion as reps."
            ),
        ),
    ]
