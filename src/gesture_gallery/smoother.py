"""Temporal smoothing of gesture predictions."""

from __future__ import annotations

from collections import Counter, deque


class GestureSmoother:
    """Majority vote over a short sliding window of predictions.

    The smoother intentionally returns ``None`` until the window is full. That
    one-frame cost buys a stable signal: a single noisy MediaPipe frame cannot
    trigger a gallery action.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_votes: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= min_votes <= window_size:
            raise ValueError("min_votes must be between 1 and window_size")
        self.window_size = window_size
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self._history: deque[tuple[str | None, float]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._history) == self.window_size

    def update(self, label: str | None, confidence: float) -> str | None:
        """Add one prediction and return the stable label, if any."""

        self._history.append((label, float(confidence)))
        if not self.is_ready:
            return None

        counts = Counter(item[0] for item in self._history)
        label, votes = counts.most_common(1)[0]
        if label is None or votes < self.min_votes:
            return None

        matching = [score for item_label, score in self._history if item_label == label]
        if sum(matching) / len(matching) < self.min_confidence:
            return None
        return label
