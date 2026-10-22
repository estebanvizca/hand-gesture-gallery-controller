from __future__ import annotations

import numpy as np
import pytest

from gesture_gallery import camera as camera_module
from gesture_gallery.camera import Camera, CameraError


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.width = 0
        self.height = 0
        self.fps = 0

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, prop, value) -> bool:
        if prop == camera_module.cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif prop == camera_module.cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        elif prop == camera_module.cv2.CAP_PROP_FPS:
            self.fps = int(value)
        return True

    def get(self, prop) -> float:
        if prop == camera_module.cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == camera_module.cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def read(self):
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_camera_opens_reads_and_releases(monkeypatch) -> None:
    fake = FakeCapture()
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *args: fake)
    camera = Camera(0, 640, 480, 30)
    camera.open()
    assert camera.is_open is True
    assert camera.actual_size == (640, 480)
    frame = camera.read()
    assert frame is not None
    assert frame.shape == (10, 10, 3)
    camera.release()
    assert fake.released is True
    assert camera.is_open is False


def test_camera_raises_clear_error_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *args: FakeCapture(opened=False))
    camera = Camera(9)
    with pytest.raises(CameraError, match="Could not open camera 9"):
        camera.open()
    assert camera.is_open is False
