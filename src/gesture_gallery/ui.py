"""Pygame rendering for the gesture gallery.

The UI is deliberately a single, calm screen: one work on a paper mat, one
live camera monitor, one filmstrip. Gesture state is shown as text and a
confidence bar rather than hidden behind an icon.

No frames are saved by this module.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .config import FONTS_DIR, Settings
from .controller import EVENT_ARROWS, EVENT_TITLES, Event
from .features import HandFeatures
from .gallery import Gallery
from .gesture_rules import GestureUpdate
from .hand_tracker import HandLandmarks

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

# Editorial dark-room palette.
BG_TOP = (20, 18, 15)
BG_BOTTOM = (9, 8, 7)
PANEL = (24, 22, 18)
PANEL_RAISED = (31, 28, 23)
PAPER = (232, 224, 208)
PAPER_SHADOW = (5, 4, 3)
INK = (244, 237, 225)
MUTED = (158, 150, 135)
FAINT = (105, 99, 89)
LINE = (54, 49, 41)
ACCENT = (226, 102, 63)
SAGE = (148, 175, 145)
GOLD = (214, 181, 120)
PAUSE_OVERLAY = (8, 7, 6, 205)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

GESTURE_LABELS = {
    "swipe_right": "SWIPE  →",
    "swipe_left": "←  SWIPE",
    "pinch": "PINCH",
    "fist": "FIST",
    "open": "OPEN HAND",
    "neutral": "NEUTRAL",
}

STATUS_COLORS = {
    "LIVE": SAGE,
    "NO HAND": GOLD,
    "PAUSED": ACCENT,
    "KEYBOARD ONLY": MUTED,
    "MODEL MISSING": ACCENT,
    "LOW CONFIDENCE": GOLD,
}


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _lerp(first: int, second: int, amount: float) -> int:
    return int(first + (second - first) * amount)


def _ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1 - (1 - value) ** 3


def _fit(image: pygame.Surface, width: int, height: int) -> pygame.Surface:
    """Scale an image to fit inside a box without distortion."""

    source_w, source_h = image.get_size()
    scale = min(width / source_w, height / source_h)
    return pygame.transform.smoothscale(image, (max(1, int(source_w * scale)), max(1, int(source_h * scale))))


def _cover(image: pygame.Surface, width: int, height: int) -> pygame.Surface:
    """Scale and center-crop an image to fill a box."""

    source_w, source_h = image.get_size()
    scale = max(width / source_w, height / source_h)
    scaled = pygame.transform.smoothscale(
        image,
        (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
    )
    left = max(0, (scaled.get_width() - width) // 2)
    top = max(0, (scaled.get_height() - height) // 2)
    return scaled.subsurface((left, top, width, height)).copy()


def _wrap(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class GalleryUI:
    """Draw the gallery, camera monitor and gesture HUD."""

    def __init__(self, screen: pygame.Surface, settings: Settings, gallery: Gallery) -> None:
        self.screen = screen
        self.settings = settings
        self.gallery = gallery
        self.show_landmarks = settings.show_landmarks
        self._fonts = self._load_fonts()
        self._background = self._make_background()
        self._image_cache: dict[tuple[Path, int, int], pygame.Surface] = {}
        self._thumb_cache: dict[Path, pygame.Surface] = {}
        self._event_flash: tuple[Event, float] | None = None
        self._transition_from = gallery.index
        self._transition_start = 0.0
        self._last_index = gallery.index
        self._detail_alpha = 0.0
        self._last_handedness: str | None = None

    def _load_fonts(self) -> dict[str, pygame.font.Font]:
        display = str(FONTS_DIR / "Fraunces.ttf")
        sans = str(FONTS_DIR / "Manrope.ttf")
        mono = str(FONTS_DIR / "IBMPlexMono-Regular.ttf")
        return {
            "display_64": pygame.font.Font(display, 64),
            "display_48": pygame.font.Font(display, 48),
            "display_40": pygame.font.Font(display, 40),
            "display_30": pygame.font.Font(display, 30),
            "sans_22": pygame.font.Font(sans, 22),
            "sans_18": pygame.font.Font(sans, 18),
            "sans_16": pygame.font.Font(sans, 16),
            "sans_14": pygame.font.Font(sans, 14),
            "mono_18": pygame.font.Font(mono, 18),
            "mono_14": pygame.font.Font(mono, 14),
            "mono_12": pygame.font.Font(mono, 12),
            "mono_11": pygame.font.Font(mono, 11),
        }

    def _make_background(self) -> pygame.Surface:
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        for y in range(WINDOW_HEIGHT):
            amount = y / WINDOW_HEIGHT
            color = (
                _lerp(BG_TOP[0], BG_BOTTOM[0], amount),
                _lerp(BG_TOP[1], BG_BOTTOM[1], amount),
                _lerp(BG_TOP[2], BG_BOTTOM[2], amount),
            )
            pygame.draw.line(surface, color, (0, y), (WINDOW_WIDTH, y))
        # A faint vertical rhythm, like a gallery wall.
        for x in range(0, WINDOW_WIDTH, 80):
            pygame.draw.line(surface, (24, 22, 18), (x, 0), (x, WINDOW_HEIGHT))
        return surface

    def flash_event(self, event: Event) -> None:
        self._event_flash = (event, time.monotonic())

    def toggle_landmarks(self) -> None:
        self.show_landmarks = not self.show_landmarks

    def draw(
        self,
        *,
        frame,
        landmarks: HandLandmarks | None,
        update: GestureUpdate,
        fps: float,
        camera_size: tuple[int, int] | None,
        camera_error: str | None,
        tracker_error: str | None,
    ) -> None:
        now = time.monotonic()
        self._sync_transition(now)
        self._sync_detail(now)

        self.screen.blit(self._background, (0, 0))
        self._draw_header(update, camera_error, tracker_error)
        self._draw_artwork(now)
        self._draw_camera_panel(
            frame=frame,
            landmarks=landmarks,
            update=update,
            fps=fps,
            camera_size=camera_size,
            camera_error=camera_error,
            tracker_error=tracker_error,
        )
        self._draw_filmstrip()
        self._draw_footer()
        if self.gallery.paused:
            self._draw_pause_overlay()
        if self.gallery.detail_open:
            self._draw_detail_overlay(now)
        self._draw_event_flash(now)

    def _sync_transition(self, now: float) -> None:
        if self.gallery.index != self._last_index:
            self._transition_from = self._last_index
            self._transition_start = now
            self._last_index = self.gallery.index

    def _sync_detail(self, now: float) -> None:
        target = 1.0 if self.gallery.detail_open else 0.0
        speed = 8.0
        delta = target - self._detail_alpha
        self._detail_alpha += max(-speed * (1 / 30), min(speed * (1 / 30), delta))
        self._detail_alpha = max(0.0, min(1.0, self._detail_alpha))

    def _text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        position: tuple[int, int],
        *,
        alpha: int = 255,
        anchor: str = "topleft",
    ) -> pygame.Rect:
        surface = font.render(text, True, color)
        if alpha != 255:
            surface.set_alpha(alpha)
        rect = surface.get_rect()
        setattr(rect, anchor, position)
        self.screen.blit(surface, rect)
        return rect

    def _draw_header(self, update: GestureUpdate, camera_error: str | None, tracker_error: str | None) -> None:
        self._text("FIELD NOTES  /  2026", self._fonts["mono_12"], ACCENT, (40, 18))
        self._text("Field Notes", self._fonts["display_64"], INK, (38, 26))
        self._text(
            "Eight studies in colour, light and gesture.",
            self._fonts["sans_16"],
            MUTED,
            (42, 112),
        )

        if self.gallery.paused:
            status = "PAUSED"
        elif self.settings.no_camera or camera_error:
            status = "KEYBOARD ONLY"
        elif tracker_error:
            status = "MODEL MISSING"
        elif update.low_confidence:
            status = "LOW CONFIDENCE"
        elif update.hand_present:
            status = "LIVE"
        else:
            status = "NO HAND"
        color = STATUS_COLORS.get(status, MUTED)
        pill = pygame.Rect(1012, 36, 228, 36)
        pygame.draw.rect(self.screen, PANEL_RAISED, pill, border_radius=18)
        pygame.draw.rect(self.screen, color, pill, width=1, border_radius=18)
        pygame.draw.circle(self.screen, color, (pill.x + 20, pill.centery), 4)
        self._text(status, self._fonts["mono_12"], color, (pill.x + 34, pill.centery - 7))
        self._text("CAMERA INPUT  /  LOCAL ONLY", self._fonts["mono_11"], FAINT, (1014, 82))
        self._text("NO FRAME LEAVES THIS MACHINE", self._fonts["mono_11"], FAINT, (1014, 100))

    def _draw_artwork(self, now: float) -> None:
        rect = pygame.Rect(40, 136, 840, 500)
        shadow = rect.move(10, 12)
        pygame.draw.rect(self.screen, PAPER_SHADOW, shadow)
        pygame.draw.rect(self.screen, PAPER, rect)
        pygame.draw.rect(self.screen, (186, 174, 153), rect, width=1)

        inner = rect.inflate(-32, -32)
        image_size = (inner.width - 36, inner.height - 36)
        image = self._art_surface(self.gallery.index, image_size)

        transition = now - self._transition_start
        if 0 <= transition < 0.32 and self._transition_from != self.gallery.index:
            progress = _ease_out_cubic(transition / 0.32)
            previous = self._art_surface(self._transition_from, image_size)
            self._blit_centered(previous, inner.center, offset=(-24 * progress, 0), alpha=int(255 * (1 - progress)))
            self._blit_centered(image, inner.center, offset=(24 * (1 - progress), 0), alpha=int(255 * progress))
        else:
            self._blit_centered(image, inner.center)

        artwork = self.gallery.current
        self._text(
            f"{self.gallery.index + 1:02d} / {self.gallery.count:02d}",
            self._fonts["mono_12"],
            ACCENT,
            (42, 648),
        )
        self._text(artwork.title, self._fonts["display_30"], INK, (92, 640))
        metadata = "  ·  ".join(part for part in (artwork.year, artwork.medium) if part)
        if metadata:
            self._text(metadata.upper(), self._fonts["mono_11"], MUTED, (94, 680))
        self._text("FIST TO PAUSE", self._fonts["mono_11"], FAINT, (770, 652))

    def _blit_centered(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        *,
        offset: tuple[int, int] = (0, 0),
        alpha: int = 255,
    ) -> None:
        if alpha <= 0:
            return
        draw_surface = surface
        if alpha < 255:
            draw_surface = surface.copy()
            draw_surface.set_alpha(alpha)
        rect = draw_surface.get_rect(center=(center[0] + offset[0], center[1] + offset[1]))
        self.screen.blit(draw_surface, rect)

    def _art_surface(self, index: int, size: tuple[int, int]) -> pygame.Surface:
        path = self.gallery.artworks[index].path
        key = (path, size[0], size[1])
        if key not in self._image_cache:
            image = pygame.image.load(str(path)).convert()
            self._image_cache[key] = _fit(image, size[0], size[1])
        return self._image_cache[key]

    def _thumb_surface(self, index: int) -> pygame.Surface:
        path = self.gallery.artworks[index].path
        if path not in self._thumb_cache:
            image = pygame.image.load(str(path)).convert()
            self._thumb_cache[path] = _cover(image, 68, 44)
        return self._thumb_cache[path]

    def _draw_camera_panel(
        self,
        *,
        frame,
        landmarks: HandLandmarks | None,
        update: GestureUpdate,
        fps: float,
        camera_size: tuple[int, int] | None,
        camera_error: str | None,
        tracker_error: str | None,
    ) -> None:
        panel = pygame.Rect(904, 136, 336, 500)
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.rect(self.screen, LINE, panel, width=1)

        self._text("LIVE INPUT", self._fonts["mono_12"], ACCENT, (924, 154))
        cam_label = f"CAM {self.settings.camera_index}"
        if camera_size:
            cam_label += f"  ·  {camera_size[0]}×{camera_size[1]}"
        self._text(cam_label, self._fonts["mono_11"], FAINT, (1220 - self._fonts["mono_11"].size(cam_label)[0], 156))

        video_rect = pygame.Rect(924, 178, 296, 222)
        pygame.draw.rect(self.screen, (12, 11, 9), video_rect)
        pygame.draw.rect(self.screen, LINE, video_rect, width=1)

        if frame is not None:
            video = self._frame_surface(frame, video_rect.size)
            self.screen.blit(video, video_rect)
            if landmarks is not None and self.show_landmarks:
                self._draw_landmarks(landmarks, video_rect, (frame.shape[1], frame.shape[0]))
        else:
            self._draw_camera_placeholder(video_rect, camera_error, tracker_error)

        # Corner brackets give the preview a monitor feel.
        for corner in (
            (video_rect.left + 8, video_rect.top + 8, 1, 1),
            (video_rect.right - 8, video_rect.top + 8, -1, 1),
            (video_rect.left + 8, video_rect.bottom - 8, 1, -1),
            (video_rect.right - 8, video_rect.bottom - 8, -1, -1),
        ):
            x, y, dx, dy = corner
            pygame.draw.line(self.screen, (244, 237, 225), (x, y), (x + 14 * dx, y), 1)
            pygame.draw.line(self.screen, (244, 237, 225), (x, y), (x, y + 14 * dy), 1)

        if tracker_error:
            hint = "Run scripts/download_model.py"
        elif camera_error or self.settings.no_camera:
            hint = "Keyboard controls remain active."
        else:
            hint = "One hand, palm toward the camera."
        self._text(hint, self._fonts["mono_11"], FAINT, (924, 408))

        self._text("DETECTED", self._fonts["mono_11"], FAINT, (924, 430))
        label = GESTURE_LABELS.get(update.label or "", "—")
        label_color = ACCENT if update.event else INK
        self._text(label, self._fonts["display_30"], label_color, (922, 444))

        bar_rect = pygame.Rect(924, 492, 296, 4)
        pygame.draw.rect(self.screen, LINE, bar_rect)
        value = max(0.0, min(1.0, update.confidence if update.hand_present else 0.0))
        pygame.draw.rect(self.screen, SAGE, (bar_rect.x, bar_rect.y, int(bar_rect.width * value), bar_rect.height))
        self._text("CONFIDENCE", self._fonts["mono_11"], FAINT, (924, 504))
        self._text(f"{value * 100:3.0f}%", self._fonts["mono_11"], MUTED, (1196, 504))

        if tracker_error:
            status = "TRACKER OFFLINE"
        elif camera_error or self.settings.no_camera:
            status = "KEYBOARD MODE"
        elif self.gallery.paused:
            status = "PAUSED"
        elif update.hand_present:
            status = "HAND IN FRAME"
        else:
            status = "SEARCHING"
        status_color = ACCENT if status == "TRACKER OFFLINE" else SAGE if update.hand_present else MUTED
        self._text(status, self._fonts["mono_11"], status_color, (924, 524))

        self._text(f"FPS {fps:4.1f}" if fps > 0 else "FPS   --", self._fonts["mono_11"], MUTED, (924, 544))
        self._text(f"COOLDOWN {update.cooldown_remaining_ms / 1000:.1f}s", self._fonts["mono_11"], MUTED, (1058, 544))
        self._text(f"SENS {self.settings.sensitivity:.1f}", self._fonts["mono_11"], FAINT, (924, 562))

        pygame.draw.line(self.screen, LINE, (924, 572), (1220, 572), 1)
        legends = (
            ("SWIPE  ← →", "navigate"),
            ("PINCH", "open details"),
            ("FIST", "pause / resume"),
        )
        for row, (gesture, action) in enumerate(legends):
            y = 584 + row * 16
            self._text(gesture, self._fonts["mono_11"], INK, (924, y))
            self._text(action, self._fonts["mono_11"], FAINT, (1042, y))

    def _frame_surface(self, frame, size: tuple[int, int]) -> pygame.Surface:
        height, width = frame.shape[:2]
        surface = pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB").copy()
        return _cover(surface, size[0], size[1])

    def _draw_landmarks(
        self,
        landmarks: HandLandmarks,
        rect: pygame.Rect,
        frame_size: tuple[int, int],
    ) -> None:
        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            return
        # The preview uses cover-fit cropping; apply the same transform to the
        # normalized landmarks so the skeleton stays on the hand.
        scale = max(rect.width / frame_width, rect.height / frame_height)
        offset_x = (frame_width * scale - rect.width) / 2
        offset_y = (frame_height * scale - rect.height) / 2
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        points = [
            (
                int(point[0] * frame_width * scale - offset_x),
                int(point[1] * frame_height * scale - offset_y),
            )
            for point in landmarks.points
        ]
        for start, end in HAND_CONNECTIONS:
            pygame.draw.line(overlay, (*SAGE, 190), points[start], points[end], 2)
        for x, y in points:
            pygame.draw.circle(overlay, (*INK, 230), (x, y), 3)
            pygame.draw.circle(overlay, (*ACCENT, 255), (x, y), 1)
        self.screen.blit(overlay, rect.topleft)

    def _draw_camera_placeholder(self, rect: pygame.Rect, camera_error: str | None, tracker_error: str | None) -> None:
        for offset in range(-rect.height, rect.width, 24):
            pygame.draw.line(
                self.screen,
                (21, 19, 16),
                (rect.left + offset, rect.bottom),
                (rect.left + offset + rect.height, rect.top),
                1,
            )
        message = "NO CAMERA FRAME"
        if tracker_error:
            message = "MODEL REQUIRED"
        elif camera_error:
            message = "CAMERA OFFLINE"
        elif self.settings.no_camera:
            message = "CAMERA OFF"
        self._text(message, self._fonts["mono_12"], MUTED, rect.center, anchor="center")

    def _draw_filmstrip(self) -> None:
        y = 704
        x = 40
        width = 136
        gap = 13
        for index in range(self.gallery.count):
            rect = pygame.Rect(x + index * (width + gap), y, width, 54)
            thumb = self._thumb_surface(index)
            self.screen.blit(thumb, (rect.x, rect.y))
            if index == self.gallery.index:
                pygame.draw.rect(self.screen, ACCENT, rect.inflate(6, 6), width=2)
            else:
                pygame.draw.rect(self.screen, LINE, rect, width=1)
            self._text(f"{index + 1:02d}", self._fonts["mono_11"], ACCENT if index == self.gallery.index else FAINT, (rect.x + 76, rect.y + 6))
            title = self.gallery.artworks[index].title.split()[0]
            self._text(title.upper()[:9], self._fonts["mono_11"], INK if index == self.gallery.index else MUTED, (rect.x + 76, rect.y + 24))

    def _draw_footer(self) -> None:
        pygame.draw.line(self.screen, LINE, (40, 768), (1240, 768), 1)
        self._text(
            "← →  NAVIGATE     ENTER  SELECT     P  PAUSE     [ ]  SENSITIVITY     L  LANDMARKS     ESC  QUIT",
            self._fonts["mono_11"],
            MUTED,
            (40, 776),
        )
        self._text("NO VIDEO IS RECORDED", self._fonts["mono_11"], FAINT, (1064, 776))

    def _draw_pause_overlay(self) -> None:
        overlay = pygame.Surface((840, 500), pygame.SRCALPHA)
        overlay.fill(PAUSE_OVERLAY)
        self.screen.blit(overlay, (40, 136))
        self._text("PAUSED", self._fonts["display_64"], INK, (460, 350), anchor="center")
        self._text("Fist or P to resume", self._fonts["mono_14"], MUTED, (460, 416), anchor="center")

    def _draw_detail_overlay(self, now: float) -> None:
        if self._detail_alpha <= 0.01:
            return
        alpha = int(215 * self._detail_alpha)
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 4, 3, alpha))
        self.screen.blit(overlay, (0, 0))

        card = pygame.Rect(120, 110, 1040, 580)
        pygame.draw.rect(self.screen, PAPER, card)
        pygame.draw.rect(self.screen, (186, 174, 153), card, width=1)

        image = self._art_surface(self.gallery.index, (500, 420))
        self.screen.blit(image, (160, 190))

        artwork = self.gallery.current
        accent = _hex(artwork.accent)
        pygame.draw.line(self.screen, accent, (694, 190), (758, 190), 3)
        self._text(f"{self.gallery.index + 1:02d} / {self.gallery.count:02d}", self._fonts["mono_12"], ACCENT, (694, 206))
        self._text(artwork.title, self._fonts["display_40"], (42, 36, 29), (690, 230))
        metadata = "  ·  ".join(part for part in (artwork.year, artwork.medium) if part)
        if metadata:
            self._text(metadata.upper(), self._fonts["mono_11"], (105, 96, 82), (694, 282))
        description = _wrap(artwork.description, self._fonts["sans_18"], 420)
        for row, line in enumerate(description):
            self._text(line, self._fonts["sans_18"], (62, 54, 44), (694, 322 + row * 26))
        self._text("PINCH OR ENTER TO CLOSE", self._fonts["mono_11"], (126, 114, 96), (694, 620))
        self._text("ESC", self._fonts["mono_11"], (126, 114, 96), (694, 640))

    def _draw_event_flash(self, now: float) -> None:
        if self._event_flash is None:
            return
        event, started = self._event_flash
        elapsed = now - started
        if elapsed > 0.95:
            self._event_flash = None
            return
        alpha = int(255 * max(0.0, min(1.0, 1 - (elapsed - 0.55) / 0.4))) if elapsed > 0.55 else 255
        label = f"{EVENT_TITLES[event]}  {EVENT_ARROWS[event]}"
        surface = self._fonts["sans_18"].render(label, True, INK)
        surface.set_alpha(alpha)
        rect = surface.get_rect(topright=(1254, 96))
        background = pygame.Surface((rect.width + 24, 32), pygame.SRCALPHA)
        background.fill((226, 102, 63, min(220, alpha)))
        self.screen.blit(background, (rect.x - 12, 94))
        self.screen.blit(surface, (rect.x, 98))
