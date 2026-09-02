"""Download and cache MediaPipe Pose Landmarker model weights.

Model files are large binaries and are never checked into the repo (see
.gitignore); they're fetched on first use into a local cache directory.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

MODEL_URLS = {
    "lite": f"{_BASE_URL}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": f"{_BASE_URL}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": f"{_BASE_URL}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def cache_dir() -> Path:
    path = Path.home() / ".cache" / "spotter" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model(variant: str = "heavy") -> Path:
    """Return a local path to the requested model, downloading if needed.

    variant: one of "lite" (fast, for live webcam), "full", or "heavy"
        (most accurate, for offline video evaluation).
    """
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {list(MODEL_URLS)}")

    dest = cache_dir() / f"pose_landmarker_{variant}.task"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = MODEL_URLS[variant]
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest
