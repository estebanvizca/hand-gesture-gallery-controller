from __future__ import annotations

import math

import pytest

from landmarks import make_hand
from gesture_gallery import features


def test_palm_center_and_width_are_positive() -> None:
    points = make_hand("open")
    x, y = features.palm_center(points)
    assert 0.0 < x < 1.0
    assert 0.0 < y < 1.0
    assert 0.15 < features.palm_width(points) < 0.25


def test_open_hand_has_extended_fingers_and_no_pinch() -> None:
    points = make_hand("open")
    assert features.extended_count(points) == 4
    assert features.is_open(points) is True
    assert features.is_fist(points) is False
    assert features.pinch_ratio(points) > 1.0


def test_fist_is_recognised_by_curled_fingertips() -> None:
    points = make_hand("fist")
    assert features.extended_count(points) <= 1
    assert features.is_fist(points) is True
    assert features.is_open(points) is False


def test_pinch_has_a_small_thumb_index_ratio() -> None:
    points = make_hand("pinch")
    assert features.pinch_ratio(points) < 0.05
    assert features.is_open(points) is False


def test_features_are_scale_invariant() -> None:
    small = make_hand("open", scale=0.7)
    large = make_hand("open", scale=1.4)
    assert features.pinch_ratio(small) == pytest.approx(features.pinch_ratio(large), abs=1e-9)
    assert features.palm_width(large) == pytest.approx(features.palm_width(small) * 2, rel=1e-9)


def test_bad_landmark_count_fails_loudly() -> None:
    with pytest.raises(ValueError):
        features.palm_center([(0.0, 0.0, 0.0)])


def test_degenerate_palm_width_returns_infinite_pinch() -> None:
    points = [(0.5, 0.5, 0.0) for _ in range(21)]
    assert math.isinf(features.pinch_ratio(points))
