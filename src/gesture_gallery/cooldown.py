"""Cooldown gate for confirmed gestures."""

from __future__ import annotations


class Cooldown:
    """Block new events for a fixed interval after a confirmed gesture.

    The clock is injected as milliseconds so the behavior is deterministic in
    tests and independent of wall-clock jumps.
    """

    def __init__(self, cooldown_ms: int = 600) -> None:
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms cannot be negative")
        self.cooldown_ms = cooldown_ms
        self._last_event_ms: float | None = None

    def allow(self, now_ms: float) -> bool:
        """Return ``True`` when enough time has passed since the last event."""

        if self._last_event_ms is None:
            return True
        return now_ms - self._last_event_ms >= self.cooldown_ms

    def mark(self, now_ms: float) -> None:
        """Record the time of a confirmed event."""

        self._last_event_ms = now_ms

    def remaining_ms(self, now_ms: float) -> int:
        """Return the remaining cooldown in whole milliseconds, never negative."""

        if self._last_event_ms is None:
            return 0
        remaining = self.cooldown_ms - (now_ms - self._last_event_ms)
        return max(0, int(round(remaining)))

    def reset(self) -> None:
        self._last_event_ms = None
