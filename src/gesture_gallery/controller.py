"""Translate stable gestures into gallery events."""

from __future__ import annotations

from enum import Enum


class Event(str, Enum):
    NEXT = "NEXT"
    PREVIOUS = "PREVIOUS"
    SELECT = "SELECT"
    PAUSE = "PAUSE"


GESTURE_EVENTS: dict[str, Event] = {
    "swipe_right": Event.NEXT,
    "swipe_left": Event.PREVIOUS,
    "pinch": Event.SELECT,
    "fist": Event.PAUSE,
}

EVENT_TITLES: dict[Event, str] = {
    Event.NEXT: "Next work",
    Event.PREVIOUS: "Previous work",
    Event.SELECT: "Details",
    Event.PAUSE: "Pause",
}

EVENT_ARROWS: dict[Event, str] = {
    Event.NEXT: "→",
    Event.PREVIOUS: "←",
    Event.SELECT: "◎",
    Event.PAUSE: "Ⅱ",
}


def to_event(gesture: str | None) -> Event | None:
    """Return the event for a stable gesture label.

    Unknown labels, ``None``, and neutral labels such as ``"open"`` are
    deliberately ignored. Absence of a hand never produces an action.
    """

    if gesture is None:
        return None
    return GESTURE_EVENTS.get(gesture.strip().lower())
