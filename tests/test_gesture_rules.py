from __future__ import annotations

from landmarks import make_hand
from gesture_gallery.gesture_rules import GestureEngine, Thresholds


def frame(engine: GestureEngine, pose: str, now_ms: float, *, x: float = 0.0, confidence: float = 1.0):
    return engine.update(make_hand(pose, x=x), now_ms, confidence)


def test_no_hand_never_emits_an_event() -> None:
    engine = GestureEngine()
    update = engine.update(None, 0.0)
    assert update.hand_present is False
    assert update.event is None
    assert update.label is None


def test_swipe_right_needs_open_hand_and_displacement() -> None:
    engine = GestureEngine()
    positions = [-0.25, -0.18, -0.11, -0.04, 0.03, 0.10, 0.17, 0.24, 0.31]
    event = None
    for index, x in enumerate(positions):
        update = frame(engine, "open", index * 50, x=x)
        event = update.event or event
    assert event == "swipe_right"


def test_swipe_left_maps_to_the_other_direction() -> None:
    engine = GestureEngine()
    positions = [0.25, 0.18, 0.11, 0.04, -0.03, -0.10, -0.17, -0.24, -0.31]
    event = None
    for index, x in enumerate(positions):
        update = frame(engine, "open", index * 50, x=x)
        event = update.event or event
    assert event == "swipe_left"


def test_closed_hand_does_not_swipe() -> None:
    engine = GestureEngine()
    event = None
    for index in range(9):
        update = frame(engine, "fist", index * 50, x=-0.25 + index * 0.07)
        event = update.event or event
    assert event in (None, "fist")


def test_pinch_fires_once_and_requires_release() -> None:
    engine = GestureEngine()
    event = None
    for index in range(8):
        update = frame(engine, "pinch", index * 50)
        event = update.event or event
    assert event == "pinch"

    for index in range(8, 20):
        update = frame(engine, "pinch", index * 50)
        assert update.event is None


def test_pinch_can_fire_again_after_release_and_neutral() -> None:
    engine = GestureEngine()
    for index in range(8):
        frame(engine, "pinch", index * 50)
    for index in range(8, 20):
        frame(engine, "open", index * 50)
    event = None
    for index in range(20, 28):
        update = frame(engine, "pinch", index * 50)
        event = update.event or event
    assert event == "pinch"


def test_fist_fires_once_and_can_toggle_again() -> None:
    engine = GestureEngine()
    event = None
    for index in range(5):
        update = frame(engine, "fist", index * 50)
        event = update.event or event
    assert event == "fist"

    for index in range(5, 15):
        update = frame(engine, "fist", index * 50)
        assert update.event is None

    for index in range(15, 20):
        frame(engine, "open", index * 50)
    for index in range(20, 30):
        frame(engine, "open", index * 50)

    event = None
    for index in range(30, 35):
        update = frame(engine, "fist", index * 50)
        event = update.event or event
    assert event == "fist"


def test_cooldown_suppresses_a_second_candidate_too_soon() -> None:
    engine = GestureEngine(Thresholds.from_sensitivity(1.0, cooldown_ms=600))
    for index in range(5):
        frame(engine, "fist", index * 50)
    for index in range(5, 10):
        frame(engine, "open", index * 50)
    for index in range(10, 16):
        frame(engine, "open", index * 50)
    # The person re-fists before 600 ms of cooldown has passed.
    event = None
    for index in range(16, 21):
        update = frame(engine, "fist", index * 50)
        event = update.event or event
    assert event is None


def test_sensitivity_changes_pinch_threshold() -> None:
    points = make_hand("open")
    points[4] = (0.42, 0.42, 0.0)  # thumb tip near the index tip, ratio ~0.3

    sensitive = GestureEngine(Thresholds.from_sensitivity(2.0))
    strict = GestureEngine(Thresholds.from_sensitivity(0.5))
    sensitive_events = []
    strict_events = []
    for index in range(10):
        sensitive_events.append(sensitive.update(points, index * 50).event)
        strict_events.append(strict.update(points, index * 50).event)
    assert "pinch" in sensitive_events
    assert "pinch" not in strict_events


def test_low_confidence_does_not_emit_or_report_hand() -> None:
    engine = GestureEngine(Thresholds.from_sensitivity(1.0, min_confidence=0.8))
    update = engine.update(make_hand("fist"), 0.0, 0.2)
    assert update.hand_present is True
    assert update.low_confidence is True
    assert update.event is None


def test_engine_resets_on_hand_loss() -> None:
    engine = GestureEngine()
    for index in range(5):
        frame(engine, "fist", index * 50)
    update = engine.update(None, 300)
    assert update.hand_present is False
    assert engine.armed is False
