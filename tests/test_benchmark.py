from __future__ import annotations

from gesture_gallery.benchmark import summarize


def row(gesture: str, correct: int, latency, false_activation: int = 0, fps: float = 25.0) -> dict:
    return {
        "gesture": gesture,
        "correct": correct,
        "latency_ms": latency,
        "false_activation": false_activation,
        "fps": fps,
    }


def test_summarize_computes_accuracy_and_medians() -> None:
    rows = [
        row("swipe_right", 1, 300),
        row("swipe_right", 1, 500),
        row("swipe_right", 0, ""),
        row("pinch", 1, 700),
        row("idle", 0, "", false_activation=1),
        row("idle", 0, "", false_activation=1),
    ]
    summary = summarize(rows, idle_minutes=5)
    assert summary["per_gesture"]["swipe_right"]["correct"] == 2
    assert summary["per_gesture"]["swipe_right"]["accuracy"] == 0.667
    assert summary["per_gesture"]["swipe_right"]["median_latency_ms"] == 400
    assert summary["per_gesture"]["pinch"]["accuracy"] == 1.0
    assert summary["false_activations"] == 2
    assert summary["false_activations_per_minute"] == 0.4
    assert summary["median_fps"] == 25.0


def test_summarize_without_idle_has_no_rate() -> None:
    summary = summarize([row("fist", 1, 200)], idle_minutes=0)
    assert summary["false_activations_per_minute"] is None
    assert summary["per_gesture"]["fist"]["median_latency_ms"] == 200


def test_summarize_handles_no_attempts() -> None:
    summary = summarize([], idle_minutes=0)
    assert summary["attempts"] == 0
    assert summary["per_gesture"] == {}
    assert summary["median_fps"] is None
