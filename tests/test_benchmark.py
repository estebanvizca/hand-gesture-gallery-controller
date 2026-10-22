from __future__ import annotations

from gesture_gallery import benchmark as benchmark_module
from gesture_gallery.benchmark import BenchmarkSession, summarize
from gesture_gallery.config import Settings
from gesture_gallery.gesture_rules import GestureUpdate


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
    assert summary["per_gesture"]["swipe_right"]["missed"] == 1
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


def test_abort_during_result_phase_stops_the_session(monkeypatch, tmp_path) -> None:
    class FakeClock:
        def tick(self, fps: int = 30) -> int:
            return 33

        def get_fps(self) -> float:
            return 25.0

    class FakeSession(BenchmarkSession):
        def __init__(self) -> None:
            super().__init__(
                Settings(),
                gesture="pinch",
                attempts=1,
                idle_minutes=0,
                output=tmp_path / "session.csv",
                lighting="test",
                distance_cm=60,
                notes="abort test",
            )
            self.pump_calls = 0

        def _pump_events(self) -> bool:
            self.pump_calls += 1
            return self.pump_calls < 3

        def _draw(self, *args, **kwargs) -> GestureUpdate:
            return GestureUpdate(
                label="pinch",
                event="pinch",
                confidence=0.9,
                hand_present=True,
                neutral=False,
                cooldown_remaining_ms=0,
            )

    class FakeMonotonic:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            self.value += 0.001
            return self.value

    monkeypatch.setattr(benchmark_module.time, "monotonic", FakeMonotonic())
    session = FakeSession()

    completed = session._run_gesture(None, None, FakeClock(), None, None, None, "pinch")

    assert completed is False
    assert session._aborted is True
    assert len(session.rows) == 1
    assert session.rows[0]["correct"] == 1
