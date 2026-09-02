"""MediaPipe Tasks Pose Landmarker backend — the default pose source.

Uses `_heavy` (most accurate) for offline video evaluation and `_lite`
(fastest) for live webcam use; both run through the same Tasks API in
VIDEO mode, which just needs monotonically increasing millisecond
timestamps, so it works fine driving a live camera loop too (we don't need
the LIVE_STREAM async-callback mode for a single-process CLI).
"""

from __future__ import annotations

import numpy as np

from spotter.pose.base import NUM_LANDMARKS, Frame
from spotter.pose.models import ensure_model


class MediaPipeBackend:
    def __init__(
        self,
        model_variant: str = "heavy",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        model_path = ensure_model(model_variant)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, image: np.ndarray, timestamp: float) -> list[Frame]:
        rgb = image[:, :, ::-1] if image.shape[-1] == 3 else image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        timestamp_ms = max(int(timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return []

        frames = []
        for i, norm_landmarks in enumerate(result.pose_landmarks):
            landmarks = _to_array(norm_landmarks)
            visibility = np.array(
                [lm.visibility for lm in norm_landmarks[:NUM_LANDMARKS]], dtype=float
            )
            if result.pose_world_landmarks and i < len(result.pose_world_landmarks):
                world_landmarks = _to_array(result.pose_world_landmarks[i])
            else:
                world_landmarks = landmarks.copy()

            frames.append(
                Frame(
                    timestamp=timestamp,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    visibility=visibility,
                    bbox=Frame.bbox_from_landmarks(landmarks),
                )
            )
        return frames

    def close(self) -> None:
        self._landmarker.close()


def _to_array(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks[:NUM_LANDMARKS]], dtype=float)
