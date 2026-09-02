"""Pure signal functions over landmark positions.

Every function here takes plain numpy points/scalars and returns a plain
float (or array) — no knowledge of `Frame`, exercises, or the FSM. That
keeps them trivially unit-testable and reusable from both the live
counting path and the eval harness's synthetic clip generator.
"""

from __future__ import annotations

import numpy as np


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex `b`, in degrees, formed by rays b->a and b->c.

    Uses only the (x, y) plane by default since callers pass 2D or 3D
    points transparently (z is ignored via slicing to the first two axes
    when present) — pass world_landmarks in for a metric 3D angle.
    """
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vertical_displacement(point: np.ndarray, reference: np.ndarray, scale: float) -> float:
    """Signed vertical displacement of `point` below `reference`, normalized.

    Positive means `point` is lower in the frame than `reference` (image-space
    y grows downward). `scale` should be a per-athlete length (e.g. torso
    length) so the signal is roughly comparable across camera distances.
    """
    if scale < 1e-9:
        return 0.0
    return float((point[1] - reference[1]) / scale)


def height_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Ratio of `a`'s height to `b`'s height above the bottom of the frame (y=1).

    Useful as e.g. hip-height / shoulder-height: near 1.0 when the athlete is
    curled up (hips and shoulders close together vertically), smaller when
    upright.
    """
    denom = 1.0 - b[1]
    if abs(denom) < 1e-9:
        return 0.0
    return float((1.0 - a[1]) / denom)


def torso_lean(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Forward/backward lean of the torso from vertical, in degrees.

    0 degrees = perfectly upright. Computed in the image (x, y) plane, so it
    reflects lean toward/away from the camera's horizontal axis, not true 3D
    lean — good enough for a form-check gate, not for biomechanics.
    """
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]  # positive: shoulders above hips, as expected
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-9))))


def bilateral_average(
    left: float, right: float, left_vis: float = 1.0, right_vis: float = 1.0
) -> float:
    """Visibility-weighted average of a left/right signal pair.

    A limb that's occluded (low visibility) contributes little to the
    combined signal, so a sandbag blocking one hip during a lunge doesn't
    silently corrupt the whole rep count.
    """
    total_vis = left_vis + right_vis
    if total_vis < 1e-6:
        return (left + right) / 2.0
    return (left * left_vis + right * right_vis) / total_vis
