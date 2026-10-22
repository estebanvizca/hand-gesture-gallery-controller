"""Gesture rules and the temporal state machine.

The thresholds are the ones in the project specification. Every distance is
expressed in palm widths so the same gesture works at different distances
from the camera.

The state machine has four jobs:

* confirm noisy frame-level evidence over a short window;
* emit each gesture once, not once per frame;
* require a neutral pose between events;
* keep a cooldown after every confirmed event.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from . import features as hand_features
from .cooldown import Cooldown
from .features import HandFeatures, Point
from .smoother import GestureSmoother

SWIPE_WINDOW_MS = 400
SWIPE_MIN_PALM_WIDTHS = 0.5
FRAMES_REQUIRED = 6
FRAMES_WINDOW = 8
PINCH_ENTER_PALM_WIDTHS = 0.25
PINCH_EXIT_PALM_WIDTHS = 0.40
COOLDOWN_MS = 600
NEUTRAL_FRAMES = 6
PINCH_RELEASE_FRAMES = 3
FIST_RELEASE_FRAMES = 5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Thresholds:
    """Tunable thresholds, scaled by the user-facing sensitivity setting."""

    sensitivity: float = 1.0
    swipe_min_palm_widths: float = SWIPE_MIN_PALM_WIDTHS
    pinch_enter: float = PINCH_ENTER_PALM_WIDTHS
    pinch_exit: float = PINCH_EXIT_PALM_WIDTHS
    min_confidence: float = 0.5
    cooldown_ms: int = COOLDOWN_MS

    @classmethod
    def from_sensitivity(
        cls,
        sensitivity: float = 1.0,
        min_confidence: float = 0.5,
        cooldown_ms: int = COOLDOWN_MS,
    ) -> "Thresholds":
        sensitivity = _clamp(float(sensitivity), 0.5, 2.0)
        # Swipe: higher sensitivity means less hand travel is required.
        # Pinch: higher sensitivity means a wider thumb-index gap still counts.
        pinch_enter = _clamp(PINCH_ENTER_PALM_WIDTHS * sensitivity, 0.10, 0.60)
        pinch_exit = max(PINCH_EXIT_PALM_WIDTHS * sensitivity, pinch_enter * 1.25)
        return cls(
            sensitivity=sensitivity,
            swipe_min_palm_widths=SWIPE_MIN_PALM_WIDTHS / sensitivity,
            pinch_enter=pinch_enter,
            pinch_exit=pinch_exit,
            min_confidence=_clamp(float(min_confidence), 0.0, 1.0),
            cooldown_ms=max(0, int(cooldown_ms)),
        )


@dataclass(frozen=True)
class SwipeSample:
    ts_ms: float
    x: float
    width: float
    open: bool


class SwipeDetector:
    """Detect a horizontal palm-center displacement over 400 ms."""

    def __init__(
        self,
        min_palm_widths: float = SWIPE_MIN_PALM_WIDTHS,
        window_ms: int = SWIPE_WINDOW_MS,
    ) -> None:
        self.min_palm_widths = min_palm_widths
        self.window_ms = window_ms
        self._samples: deque[SwipeSample] = deque()
        self._open_flags: deque[bool] = deque(maxlen=FRAMES_WINDOW)

    def reset(self) -> None:
        self._samples.clear()
        self._open_flags.clear()

    def update(self, now_ms: float, features: HandFeatures) -> str | None:
        sample = SwipeSample(now_ms, features.center_x, features.width, features.open)
        self._samples.append(sample)
        self._open_flags.append(features.open)

        if not features.open:
            # Do not let movement made with a closed hand contribute to a later
            # swipe. The open/closed history is kept for the 6-of-8 test.
            self._samples.clear()
            return None

        cutoff = now_ms - self.window_ms
        while self._samples and self._samples[0].ts_ms < cutoff:
            self._samples.popleft()

        if len(self._open_flags) < FRAMES_WINDOW:
            return None
        if sum(self._open_flags) < FRAMES_REQUIRED:
            return None
        if len(self._samples) < 4:
            return None

        oldest = self._samples[0]
        duration = now_ms - oldest.ts_ms
        if duration < self.window_ms * 0.75:
            return None

        width = features.width
        if width <= 1e-9:
            return None
        displacement = features.center_x - oldest.x
        if abs(displacement) < self.min_palm_widths * width:
            return None

        self.reset()
        return "swipe_right" if displacement > 0 else "swipe_left"

    def recent_displacement(self, now_ms: float, features: HandFeatures, window_ms: int = 200) -> float:
        """Return the absolute horizontal movement over the recent window."""

        if features.width <= 1e-9:
            return 0.0
        cutoff = now_ms - window_ms
        reference = None
        for sample in self._samples:
            if sample.ts_ms >= cutoff:
                reference = sample
                break
        if reference is None:
            return 0.0
        return abs(features.center_x - reference.x) / features.width

    def progress(self, now_ms: float, features: HandFeatures) -> float:
        """Return 0..1 progress towards the swipe threshold, for the HUD."""

        if features.width <= 1e-9 or not self._samples:
            return 0.0
        oldest = self._samples[0]
        displacement = abs(features.center_x - oldest.x)
        return _clamp(displacement / (self.min_palm_widths * features.width), 0.0, 1.0)


class PinchDetector:
    """Pinch with a six-of-eight confirmation and 0.25/0.40 hysteresis."""

    def __init__(self, enter: float = PINCH_ENTER_PALM_WIDTHS, exit: float = PINCH_EXIT_PALM_WIDTHS) -> None:
        self.enter = enter
        self.exit = exit
        self.active = False
        self._enter_window: deque[bool] = deque(maxlen=FRAMES_WINDOW)
        self._release_count = 0

    def reset(self) -> None:
        self.active = False
        self._enter_window.clear()
        self._release_count = 0

    def update(self, ratio: float) -> bool:
        """Return ``True`` only on the frame the pinch is confirmed."""

        if self.active:
            if ratio > self.exit:
                self._release_count += 1
                if self._release_count >= PINCH_RELEASE_FRAMES:
                    self.reset()
            else:
                self._release_count = 0
            return False

        self._enter_window.append(ratio < self.enter)
        if len(self._enter_window) == FRAMES_WINDOW and sum(self._enter_window) >= FRAMES_REQUIRED:
            self.active = True
            self._enter_window.clear()
            self._release_count = 0
            return True
        return False


class FistDetector:
    """Fist confirmation through the shared five-frame smoother."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.smoother = GestureSmoother(
            window_size=5,
            min_votes=3,
            min_confidence=min_confidence,
        )
        self.active = False
        self._release_count = 0

    def reset(self) -> None:
        self.smoother.reset()
        self.active = False
        self._release_count = 0

    def update(self, is_fist: bool, confidence: float) -> bool:
        stable = self.smoother.update("fist" if is_fist else None, confidence)
        if self.active:
            if is_fist:
                self._release_count = 0
            else:
                self._release_count += 1
                if self._release_count >= FIST_RELEASE_FRAMES:
                    self.reset()
            return False
        if stable == "fist":
            self.active = True
            self._release_count = 0
            return True
        return False


@dataclass(frozen=True)
class GestureUpdate:
    """Everything the UI and controller need to know about one frame."""

    label: str | None
    event: str | None
    confidence: float
    hand_present: bool
    neutral: bool
    cooldown_remaining_ms: int
    pinch_ratio: float | None = None
    swipe_progress: float = 0.0
    low_confidence: bool = False


class GestureEngine:
    """Turn one hand's landmarks into stable labels and one-shot events."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds()
        self.cooldown = Cooldown(self.thresholds.cooldown_ms)
        self.swipe = SwipeDetector(self.thresholds.swipe_min_palm_widths)
        self.pinch = PinchDetector(self.thresholds.pinch_enter, self.thresholds.pinch_exit)
        self.fist = FistDetector(self.thresholds.min_confidence)
        self.armed = True
        self._neutral_frames = 0
        self._last_label: str | None = None

    def set_sensitivity(self, sensitivity: float) -> None:
        self.thresholds = Thresholds.from_sensitivity(
            sensitivity,
            min_confidence=self.thresholds.min_confidence,
            cooldown_ms=self.thresholds.cooldown_ms,
        )
        self.swipe = SwipeDetector(self.thresholds.swipe_min_palm_widths)
        self.pinch = PinchDetector(self.thresholds.pinch_enter, self.thresholds.pinch_exit)
        self.fist = FistDetector(self.thresholds.min_confidence)
        self.reset()

    def reset(self) -> None:
        self.swipe.reset()
        self.pinch.reset()
        self.fist.reset()
        self.armed = False
        self._neutral_frames = 0
        self._last_label = None

    def update(
        self,
        points: list[Point] | None,
        now_ms: float,
        confidence: float = 1.0,
    ) -> GestureUpdate:
        if points is None or len(points) < 21:
            self.swipe.reset()
            self.pinch.reset()
            self.fist.reset()
            self.armed = False
            self._neutral_frames = 0
            self._last_label = None
            return GestureUpdate(
                label=None,
                event=None,
                confidence=0.0,
                hand_present=False,
                neutral=False,
                cooldown_remaining_ms=self.cooldown.remaining_ms(now_ms),
            )

        if confidence < self.thresholds.min_confidence:
            return GestureUpdate(
                label=None,
                event=None,
                confidence=confidence,
                hand_present=True,
                neutral=False,
                cooldown_remaining_ms=self.cooldown.remaining_ms(now_ms),
                low_confidence=True,
            )

        hand = hand_features.extract_features(points, self.thresholds.pinch_exit)

        fist_edge = self.fist.update(hand.fist, confidence)

        pinch_edge = False
        if self.fist.active or hand.fist:
            self.pinch.reset()
        else:
            pinch_edge = self.pinch.update(hand.pinch_ratio)

        swipe_edge = None
        if not self.fist.active and not self.pinch.active:
            swipe_edge = self.swipe.update(now_ms, hand)
        else:
            self.swipe.reset()

        neutral = self._is_neutral(hand, now_ms)
        candidate = "fist" if fist_edge else "pinch" if pinch_edge else swipe_edge

        event = None
        if candidate is not None:
            if self.armed and self.cooldown.allow(now_ms):
                event = candidate
                self.cooldown.mark(now_ms)
            self.armed = False
            self._neutral_frames = 0

        if not self.armed:
            if neutral:
                self._neutral_frames += 1
                if self._neutral_frames >= NEUTRAL_FRAMES and self.cooldown.allow(now_ms):
                    self.armed = True
                    self._neutral_frames = 0
            else:
                self._neutral_frames = 0

        if candidate is not None:
            label = candidate
        elif self.fist.active:
            label = "fist"
        elif self.pinch.active:
            label = "pinch"
        elif hand.open:
            label = "open"
        else:
            label = "neutral"

        self._last_label = label
        return GestureUpdate(
            label=label,
            event=event,
            confidence=confidence,
            hand_present=True,
            neutral=neutral,
            cooldown_remaining_ms=self.cooldown.remaining_ms(now_ms),
            pinch_ratio=hand.pinch_ratio,
            swipe_progress=self.swipe.progress(now_ms, hand),
        )

    def _is_neutral(self, hand: HandFeatures, now_ms: float) -> bool:
        if not hand.open:
            return False
        if self.pinch.active or self.fist.active:
            return False
        return self.swipe.recent_displacement(now_ms, hand) < 0.20
