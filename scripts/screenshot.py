"""Render a reproducible UI screenshot from a still test frame.

The normal demo uses the live webcam. This script exists so the README can
show the interface without including anyone's face or a recorded video. The
still frame is a CC BY-SA 4.0 photograph of a hand; see
``docs/ATTRIBUTION.md``.

Run:

    .venv/bin/python scripts/screenshot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pygame

from gesture_gallery.config import GALLERY_DIR, Settings
from gesture_gallery.gallery import Gallery, load_gallery
from gesture_gallery.gesture_rules import GestureEngine, Thresholds
from gesture_gallery.hand_tracker import HandTracker
from gesture_gallery.ui import GalleryUI, WINDOW_HEIGHT, WINDOW_WIDTH


def main() -> int:
    frame_path = ROOT / "docs" / "test-frame.jpg"
    output_path = ROOT / "docs" / "screenshot.png"
    if not frame_path.exists():
        print(f"Missing still frame: {frame_path}", file=sys.stderr)
        return 2
    if not Settings().model_path.exists():
        print("Download the model first: .venv/bin/python scripts/download_model.py", file=sys.stderr)
        return 2

    settings = Settings(no_camera=False)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Could not read {frame_path}", file=sys.stderr)
        return 2
    if settings.mirror:
        frame = cv2.flip(frame, 1)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    gallery = Gallery(load_gallery(GALLERY_DIR))
    ui = GalleryUI(screen, settings, gallery)
    tracker = HandTracker(settings.model_path, mode="image")
    engine = GestureEngine(
        Thresholds.from_sensitivity(
            settings.sensitivity,
            min_confidence=settings.min_confidence,
            cooldown_ms=settings.cooldown_ms,
        )
    )

    try:
        landmarks = tracker.detect(frame)
        update = None
        for index in range(6):
            update = engine.update(
                landmarks.points if landmarks else None,
                index * 50.0,
                landmarks.confidence if landmarks else 0.0,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ui.draw(
            frame=rgb,
            landmarks=landmarks,
            update=update,
            fps=0.0,
            camera_size=(frame.shape[1], frame.shape[0]),
            camera_error=None,
            tracker_error=None,
        )
        pygame.image.save(screen, str(output_path))
    finally:
        tracker.close()
        pygame.quit()

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
