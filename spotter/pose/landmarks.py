"""BlazePose 33-point landmark topology, shared by every backend.

Backends other than MediaPipe (e.g. a future YOLO-pose backend) must remap
their own keypoint sets onto these 33 indices in `Frame`, so everything
downstream (signals, exercises, overlay) only ever talks about names.
"""

from __future__ import annotations

LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

_NAME_TO_INDEX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

# Skeleton edges for overlay drawing, as (name_a, name_b) pairs.
POSE_CONNECTIONS: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
    ("right_heel", "right_foot_index"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
]


def landmark_index(name: str) -> int:
    try:
        return _NAME_TO_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Unknown landmark name: {name!r}") from exc
