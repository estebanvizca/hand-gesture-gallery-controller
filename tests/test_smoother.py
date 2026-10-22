from __future__ import annotations

from gesture_gallery.smoother import GestureSmoother


def test_smoother_requires_a_full_window() -> None:
    smoother = GestureSmoother(window_size=5, min_votes=3)
    assert smoother.update("fist", 0.9) is None
    assert smoother.update("fist", 0.9) is None
    assert smoother.update("fist", 0.9) is None
    assert smoother.update("fist", 0.9) is None
    assert smoother.update("fist", 0.9) == "fist"


def test_smoother_uses_majority_vote() -> None:
    smoother = GestureSmoother(window_size=5, min_votes=3)
    labels = ["fist", "fist", "open", "fist", "open"]
    result = None
    for label in labels:
        result = smoother.update(label, 0.9)
    assert result == "fist"


def test_smoother_rejects_low_confidence() -> None:
    smoother = GestureSmoother(window_size=5, min_votes=3, min_confidence=0.7)
    for _ in range(5):
        result = smoother.update("pinch", 0.4)
    assert result is None


def test_smoother_returns_none_for_ambiguous_window() -> None:
    smoother = GestureSmoother(window_size=5, min_votes=3)
    for label in ["fist", "fist", "open", "open", None]:
        result = smoother.update(label, 0.9)
    assert result is None


def test_smoother_reset_clears_history() -> None:
    smoother = GestureSmoother(window_size=3, min_votes=2)
    smoother.update("fist", 0.9)
    smoother.update("fist", 0.9)
    smoother.reset()
    assert smoother.update("fist", 0.9) is None
