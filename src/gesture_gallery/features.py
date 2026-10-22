"""Scale-invariant features from MediaPipe hand landmarks.

MediaPipe returns 21 landmarks per hand with normalized coordinates. The
functions here reduce those points to the few measurements the gesture rules
need: palm center, palm width, pinch distance and finger extension.

All functions are pure and accept a sequence of ``(x, y, z)`` tuples, which
makes them straightforward to test without a camera or a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float, float]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

PALM_INDICES = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_PAIRS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)
DEFAULT_PINCH_EXIT = 0.40


def _require_landmarks(points: list[Point]) -> None:
    if len(points) < 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(points)}")


def _distance_2d(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def palm_center(points: list[Point]) -> tuple[float, float]:
    """Return the normalized ``(x, y)`` center of the palm.

    The center is the mean of the wrist and the four finger MCP joints. It is
    more stable than a single landmark and stays close to the physical palm
    while the fingers move.
    """

    _require_landmarks(points)
    xs = [points[index][0] for index in PALM_INDICES]
    ys = [points[index][1] for index in PALM_INDICES]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def palm_width(points: list[Point]) -> float:
    """Return the index-to-pinky knuckle distance in normalized units.

    This distance is used as the scale reference for every threshold, so the
    same gesture works at different distances from the camera.
    """

    _require_landmarks(points)
    return _distance_2d(points[INDEX_MCP], points[PINKY_MCP])


def pinch_ratio(points: list[Point]) -> float:
    """Return the thumb-tip to index-tip distance in palm widths."""

    _require_landmarks(points)
    width = palm_width(points)
    if width <= 1e-9:
        return math.inf
    return _distance_2d(points[THUMB_TIP], points[INDEX_TIP]) / width


def finger_states(points: list[Point]) -> tuple[bool, bool, bool, bool]:
    """Return whether index, middle, ring and pinky are extended.

    A finger is extended when its tip is farther from the wrist than its PIP
    (middle) joint. The thumb is excluded; it is noisier and the pinch rule
    uses it separately.
    """

    _require_landmarks(points)
    states = []
    for tip, pip in FINGER_PAIRS:
        tip_distance = _distance_2d(points[tip], points[WRIST])
        pip_distance = _distance_2d(points[pip], points[WRIST])
        states.append(tip_distance > pip_distance)
    return tuple(states)  # type: ignore[return-value]


def extended_count(points: list[Point]) -> int:
    """Return how many of the four non-thumb fingers are extended."""

    return sum(finger_states(points))


def is_fist(points: list[Point]) -> bool:
    """Return ``True`` when at least three fingertips are curled inward.

    The test is the one from the project specification: a fingertip closer to
    the wrist than its middle joint counts as curled.
    """

    return extended_count(points) <= 1


def is_open(points: list[Point], pinch_exit: float = DEFAULT_PINCH_EXIT) -> bool:
    """Return ``True`` for an open hand that is not pinching."""

    return extended_count(points) >= 3 and pinch_ratio(points) > pinch_exit


@dataclass(frozen=True)
class HandFeatures:
    """The measurements used by the gesture state machine."""

    center_x: float
    center_y: float
    width: float
    pinch_ratio: float
    extended: tuple[bool, bool, bool, bool]
    extended_count: int
    fist: bool
    open: bool


def extract_features(points: list[Point], pinch_exit: float = DEFAULT_PINCH_EXIT) -> HandFeatures:
    """Reduce 21 landmarks to the features consumed by the rules."""

    center = palm_center(points)
    width = palm_width(points)
    states = finger_states(points)
    ratio = pinch_ratio(points)
    return HandFeatures(
        center_x=center[0],
        center_y=center[1],
        width=width,
        pinch_ratio=ratio,
        extended=states,
        extended_count=sum(states),
        fist=sum(states) <= 1,
        open=sum(states) >= 3 and ratio > pinch_exit,
    )
