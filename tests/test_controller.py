from __future__ import annotations

from gesture_gallery.controller import Event, to_event


def test_gesture_mapping() -> None:
    assert to_event("swipe_right") is Event.NEXT
    assert to_event("swipe_left") is Event.PREVIOUS
    assert to_event("pinch") is Event.SELECT
    assert to_event("fist") is Event.PAUSE


def test_non_events_are_ignored() -> None:
    assert to_event(None) is None
    assert to_event("open") is None
    assert to_event("neutral") is None
    assert to_event("unknown") is None


def test_mapping_is_case_and_whitespace_tolerant() -> None:
    assert to_event("  SWIPE_RIGHT  ") is Event.NEXT
