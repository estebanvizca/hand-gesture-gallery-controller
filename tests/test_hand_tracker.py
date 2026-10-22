from __future__ import annotations

import numpy as np
import pytest

from gesture_gallery.config import DEFAULT_MODEL_PATH
from gesture_gallery.hand_tracker import HandTracker


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Hand Landmarker model not downloaded")
def test_tracker_loads_and_returns_none_on_blank_frame() -> None:
    tracker = HandTracker(DEFAULT_MODEL_PATH, mode="image")
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert tracker.detect(frame) is None
    finally:
        tracker.close()
