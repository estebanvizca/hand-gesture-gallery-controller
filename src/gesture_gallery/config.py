"""Configuration for the gesture gallery demo."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
GALLERY_DIR = ASSETS_DIR / "gallery"
FONTS_DIR = ASSETS_DIR / "fonts"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    camera_fps: int = 30
    mirror: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    sensitivity: float = 1.0
    min_confidence: float = 0.5
    cooldown_ms: int = 600
    no_camera: bool = False
    show_landmarks: bool = True
    fullscreen: bool = False
    gallery_dir: Path = GALLERY_DIR

    def __post_init__(self) -> None:
        if not 0.5 <= self.sensitivity <= 2.0:
            raise ValueError("sensitivity must be between 0.5 and 2.0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms cannot be negative")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the options shared by the app and the benchmark."""

    parser.add_argument(
        "--camera",
        type=int,
        default=_env_int("GESTURE_CAMERA", 0),
        help="webcam index (default: 0, env GESTURE_CAMERA)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=_env_int("GESTURE_WIDTH", 960),
        help="requested camera width (default: 960)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=_env_int("GESTURE_HEIGHT", 540),
        help="requested camera height (default: 540)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=_env_int("GESTURE_FPS", 30),
        help="requested camera FPS (default: 30)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("GESTURE_MODEL", str(DEFAULT_MODEL_PATH))),
        help="path to hand_landmarker.task (default: models/hand_landmarker.task)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=_env_float("GESTURE_SENSITIVITY", 1.0),
        help="0.5-2.0; higher values make gestures easier to trigger (default: 1.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=_env_float("GESTURE_MIN_CONFIDENCE", 0.5),
        help="minimum MediaPipe confidence (default: 0.5)",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=_env_int("GESTURE_COOLDOWN_MS", 600),
        help="milliseconds between confirmed gestures (default: 600)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not mirror the camera preview (mirrored by default)",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="start in keyboard-only mode",
    )
    parser.add_argument(
        "--hide-landmarks",
        action="store_true",
        help="hide the landmark skeleton",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="start in fullscreen mode",
    )
    return parser


def load_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        description="Gesture-controlled gallery: swipe, pinch and fist.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    return Settings(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        mirror=not args.no_mirror,
        model_path=args.model.expanduser().resolve(),
        sensitivity=args.sensitivity,
        min_confidence=args.min_confidence,
        cooldown_ms=args.cooldown_ms,
        no_camera=args.no_camera,
        show_landmarks=not args.hide_landmarks,
        fullscreen=args.fullscreen,
        gallery_dir=GALLERY_DIR,
    )
