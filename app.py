"""Entry point for the gesture-controlled gallery.

Run:

    .venv/bin/python app.py

Keyboard fallback is always available, even if the camera or model is
missing. Close the window or press Esc to release the camera.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pygame

from gesture_gallery.camera import Camera, CameraError
from gesture_gallery.config import Settings, load_settings
from gesture_gallery.controller import Event, to_event
from gesture_gallery.gallery import Gallery, load_gallery
from gesture_gallery.gesture_rules import GestureEngine, GestureUpdate, Thresholds
from gesture_gallery.hand_tracker import HandTracker
from gesture_gallery.ui import GalleryUI, WINDOW_HEIGHT, WINDOW_WIDTH


def _empty_update(engine: GestureEngine, now_ms: float) -> GestureUpdate:
    return engine.update(None, now_ms)


def _dispatch(ui: GalleryUI, gallery: Gallery, event: Event) -> None:
    if gallery.handle_event(event):
        ui.flash_event(event)


def run(settings: Settings) -> int:
    pygame.init()
    pygame.display.set_caption("Field Notes — Gesture Gallery")
    flags = pygame.FULLSCREEN if settings.fullscreen else 0
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    clock = pygame.time.Clock()

    try:
        artworks = load_gallery(settings.gallery_dir)
    except (OSError, ValueError) as error:
        pygame.quit()
        print(f"Could not load gallery: {error}", file=sys.stderr)
        return 2

    gallery = Gallery(artworks)
    ui = GalleryUI(screen, settings, gallery)
    engine = GestureEngine(
        Thresholds.from_sensitivity(
            settings.sensitivity,
            min_confidence=settings.min_confidence,
            cooldown_ms=settings.cooldown_ms,
        )
    )

    camera: Camera | None = None
    camera_error: str | None = None
    tracker: HandTracker | None = None
    tracker_error: str | None = None

    if not settings.no_camera:
        camera = Camera(
            settings.camera_index,
            settings.camera_width,
            settings.camera_height,
            settings.camera_fps,
        )
        try:
            camera.open()
        except CameraError as error:
            camera_error = str(error)
            camera = None
            print(f"Camera unavailable: {error}", file=sys.stderr)

        if camera is not None:
            try:
                tracker = HandTracker(settings.model_path)
            except Exception as error:  # model missing or MediaPipe load failure
                tracker_error = str(error)
                print(f"Hand tracker unavailable: {error}", file=sys.stderr)

    running = True
    fps = 0.0
    sensitivity = settings.sensitivity

    try:
        while running:
            clock.tick(30)
            now = time.monotonic()
            now_ms = now * 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    if gallery.detail_open:
                        gallery.detail_open = False
                    else:
                        running = False
                elif event.key == pygame.K_LEFT:
                    if gallery.previous():
                        ui.flash_event(Event.PREVIOUS)
                elif event.key == pygame.K_RIGHT:
                    if gallery.next():
                        ui.flash_event(Event.NEXT)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    if gallery.select():
                        ui.flash_event(Event.SELECT)
                elif event.key == pygame.K_p:
                    gallery.toggle_pause()
                    ui.flash_event(Event.PAUSE)
                elif event.key == pygame.K_l:
                    ui.toggle_landmarks()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    sensitivity = max(0.5, round(sensitivity - 0.1, 1))
                    engine.set_sensitivity(sensitivity)
                    settings = replace(settings, sensitivity=sensitivity)
                    ui.settings = settings
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                    sensitivity = min(2.0, round(sensitivity + 0.1, 1))
                    engine.set_sensitivity(sensitivity)
                    settings = replace(settings, sensitivity=sensitivity)
                    ui.settings = settings

            frame_bgr = None
            frame_rgb = None
            landmarks = None
            update = _empty_update(engine, now_ms)

            if camera is not None:
                frame_bgr = camera.read()
                if frame_bgr is not None:
                    if settings.mirror:
                        frame_bgr = cv2.flip(frame_bgr, 1)
                    if tracker is not None:
                        try:
                            landmarks = tracker.detect(frame_bgr, int(now_ms))
                            update = engine.update(
                                landmarks.points if landmarks else None,
                                now_ms,
                                landmarks.confidence if landmarks else 0.0,
                            )
                        except Exception as error:  # keep the UI alive if tracking fails
                            tracker_error = f"Tracking stopped: {error}"
                            print(tracker_error, file=sys.stderr)
                            tracker.close()
                            tracker = None
                            landmarks = None
                            update = _empty_update(engine, now_ms)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            event = to_event(update.event)
            if event is not None:
                _dispatch(ui, gallery, event)

            fps = clock.get_fps() or fps
            ui.draw(
                frame=frame_rgb,
                landmarks=landmarks,
                update=update,
                fps=fps,
                camera_size=camera.actual_size if camera else None,
                camera_error=camera_error,
                tracker_error=tracker_error,
            )
            pygame.display.flip()
    finally:
        if tracker is not None:
            tracker.close()
        if camera is not None:
            camera.release()
        pygame.quit()

    return 0


def main() -> int:
    try:
        settings = load_settings()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    return run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
