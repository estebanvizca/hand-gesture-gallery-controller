"""Webcam capture with explicit resource cleanup."""

from __future__ import annotations

import sys
from typing import Any

import cv2


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""


class Camera:
    """Thin wrapper around ``cv2.VideoCapture``.

    Frames are kept in memory. Nothing is written to disk or sent over the
    network.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 960,
        height: int = 540,
        fps: int = 30,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: Any | None = None

    def open(self) -> None:
        if self._capture is not None:
            return

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.index, backend)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera {self.index}. Check that it is connected, "
                "not in use by another application, and that this terminal has "
                "camera permission."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            # Not every backend exposes this property; it is only a latency hint.
            pass
        self._capture = capture

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def actual_size(self) -> tuple[int, int] | None:
        if not self.is_open:
            return None
        assert self._capture is not None
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self):
        """Return the next BGR frame, or ``None`` when capture fails."""

        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
