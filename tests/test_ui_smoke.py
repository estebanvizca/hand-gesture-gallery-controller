from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from gesture_gallery.config import GALLERY_DIR, Settings
from gesture_gallery.gallery import Gallery, load_gallery
from gesture_gallery.gesture_rules import GestureUpdate
from gesture_gallery.ui import GalleryUI, WINDOW_HEIGHT, WINDOW_WIDTH


def test_ui_draws_gallery_and_pause_overlay() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    gallery = Gallery(load_gallery(GALLERY_DIR))
    ui = GalleryUI(screen, Settings(no_camera=True), gallery)
    update = GestureUpdate(
        label=None,
        event=None,
        confidence=0.0,
        hand_present=False,
        neutral=False,
        cooldown_remaining_ms=0,
    )
    ui.draw(
        frame=None,
        landmarks=None,
        update=update,
        fps=30.0,
        camera_size=None,
        camera_error=None,
        tracker_error=None,
    )
    gallery.toggle_pause()
    ui.draw(
        frame=None,
        landmarks=None,
        update=update,
        fps=30.0,
        camera_size=None,
        camera_error=None,
        tracker_error=None,
    )
    assert screen.get_size() == (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.quit()
