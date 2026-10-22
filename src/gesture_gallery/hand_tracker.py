"""MediaPipe Hand Landmarker wrapper.

The app uses MediaPipe's video mode: each frame gets a monotonically
increasing timestamp, which lets the task reuse tracking between frames.
Image mode is available for tests and still images.

The ``.task`` model is intentionally not committed. Run
``scripts/download_model.py`` or follow the README.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


@dataclass(frozen=True)
class HandLandmarks:
    """Normalized landmarks for one detected hand."""

    points: list[tuple[float, float, float]]
    confidence: float
    handedness: str | None = None


class HandTracker:
    """Owns the MediaPipe task and releases it when done."""

    def __init__(
        self,
        model_path: str | Path,
        mode: str = "video",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {self.model_path}\n"
                "Run: .venv/bin/python scripts/download_model.py"
            )
        if mode not in {"image", "video"}:
            raise ValueError("mode must be 'image' or 'video'")

        self.mode = mode
        running_mode = vision.RunningMode.IMAGE if mode == "image" else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr, timestamp_ms: int | None = None) -> HandLandmarks | None:
        """Return landmarks for the most confident hand, or ``None``."""

        if frame_bgr is None:
            return None
        if getattr(frame_bgr, "ndim", 0) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 BGR image")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.mode == "image":
            result = self._landmarker.detect(image)
        else:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = int(timestamp_ms)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        best_index = 0
        best_score = -1.0
        for index, handedness in enumerate(result.handedness):
            score = handedness[0].score if handedness else 0.0
            if score > best_score:
                best_index = index
                best_score = score

        landmarks = result.hand_landmarks[best_index]
        points = [(float(item.x), float(item.y), float(item.z)) for item in landmarks]
        handedness = None
        if best_index < len(result.handedness) and result.handedness[best_index]:
            handedness = result.handedness[best_index][0].category_name
        return HandLandmarks(
            points=points,
            confidence=max(0.0, best_score),
            handedness=handedness,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
