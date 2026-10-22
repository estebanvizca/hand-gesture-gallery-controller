"""Synthetic hand landmarks for tests.

The points are not a photograph; they are a small geometric hand that
satisfies the same relationships MediaPipe reports: a wrist, four MCP joints,
PIP joints and fingertips. Coordinates are normalized to a 0..1 frame.
"""

from __future__ import annotations

Point = tuple[float, float, float]

OPEN_HAND: tuple[Point, ...] = (
    (0.50, 0.88, 0.00),  # 0 wrist
    (0.46, 0.85, -0.01),  # 1 thumb CMC
    (0.41, 0.79, -0.02),  # 2 thumb MCP
    (0.37, 0.73, -0.02),  # 3 thumb IP
    (0.33, 0.66, -0.02),  # 4 thumb tip
    (0.43, 0.62, 0.00),  # 5 index MCP
    (0.42, 0.50, 0.00),  # 6 index PIP
    (0.42, 0.43, 0.00),  # 7 index DIP
    (0.42, 0.36, 0.00),  # 8 index tip
    (0.50, 0.60, 0.00),  # 9 middle MCP
    (0.50, 0.46, 0.00),  # 10 middle PIP
    (0.50, 0.38, 0.00),  # 11 middle DIP
    (0.50, 0.30, 0.00),  # 12 middle tip
    (0.57, 0.62, 0.00),  # 13 ring MCP
    (0.58, 0.49, 0.00),  # 14 ring PIP
    (0.58, 0.42, 0.00),  # 15 ring DIP
    (0.58, 0.35, 0.00),  # 16 ring tip
    (0.63, 0.65, 0.00),  # 17 pinky MCP
    (0.65, 0.55, 0.00),  # 18 pinky PIP
    (0.65, 0.49, 0.00),  # 19 pinky DIP
    (0.65, 0.43, 0.00),  # 20 pinky tip
)

FIST_POINTS: tuple[Point, ...] = (
    (0.50, 0.88, 0.00),
    (0.46, 0.85, -0.01),
    (0.41, 0.79, -0.02),
    (0.37, 0.73, -0.02),
    (0.34, 0.69, -0.02),
    (0.43, 0.62, 0.00),
    (0.42, 0.50, 0.00),
    (0.42, 0.44, 0.00),
    (0.44, 0.72, 0.00),  # index tip curled toward wrist
    (0.50, 0.60, 0.00),
    (0.50, 0.46, 0.00),
    (0.50, 0.40, 0.00),
    (0.50, 0.70, 0.00),  # middle tip curled
    (0.57, 0.62, 0.00),
    (0.58, 0.49, 0.00),
    (0.58, 0.43, 0.00),
    (0.56, 0.72, 0.00),  # ring tip curled
    (0.63, 0.65, 0.00),
    (0.65, 0.55, 0.00),
    (0.65, 0.50, 0.00),
    (0.61, 0.75, 0.00),  # pinky tip curled
)

PINCH_POINTS: tuple[Point, ...] = tuple(
    (0.42, 0.36, 0.00) if index == 4 else point
    for index, point in enumerate(OPEN_HAND)
)


def make_hand(
    pose: str = "open",
    *,
    x: float = 0.0,
    y: float = 0.0,
    scale: float = 1.0,
    origin: tuple[float, float] = (0.5, 0.88),
) -> list[Point]:
    """Return a translated and scaled synthetic hand."""

    if pose == "open":
        points = OPEN_HAND
    elif pose == "fist":
        points = FIST_POINTS
    elif pose == "pinch":
        points = PINCH_POINTS
    else:
        raise ValueError(f"Unknown pose: {pose}")

    ox, oy = origin
    return [
        (
            ox + (point[0] - ox) * scale + x,
            oy + (point[1] - oy) * scale + y,
            point[2],
        )
        for point in points
    ]
