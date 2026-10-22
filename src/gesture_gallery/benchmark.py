"""Guided live benchmark for the gesture gallery.

The benchmark logs counts only. It never writes camera frames, video or
images. Run it with a person in front of the camera:

    .venv/bin/python benchmark.py --attempts 20 --idle-minutes 5

The CSV schema matches ``eval/session_template.csv``. A JSON summary is
written next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pygame

from gesture_gallery.camera import Camera
from gesture_gallery.config import FONTS_DIR, REPORTS_DIR, Settings, add_common_arguments, load_settings
from gesture_gallery.gesture_rules import GestureEngine, GestureUpdate, Thresholds
from gesture_gallery.hand_tracker import HandLandmarks, HandTracker

CSV_FIELDS = [
    "attempt_id",
    "timestamp",
    "gesture",
    "expected_event",
    "detected_event",
    "correct",
    "latency_ms",
    "fps",
    "false_activation",
    "lighting",
    "distance_cm",
    "camera",
    "notes",
]

GESTURE_EVENTS = {
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "pinch": "pinch",
    "fist": "fist",
}

GESTURE_TITLES = {
    "swipe_left": "SWIPE LEFT",
    "swipe_right": "SWIPE RIGHT",
    "pinch": "PINCH",
    "fist": "FIST",
}

GESTURE_HINTS = {
    "swipe_left": "Open hand, move left across the camera.",
    "swipe_right": "Open hand, move right across the camera.",
    "pinch": "Touch thumb and index finger, then release.",
    "fist": "Close at least three fingers, then open your hand.",
}

BG = (14, 13, 11)
PANEL = (24, 22, 18)
INK = (244, 237, 225)
MUTED = (158, 150, 135)
FAINT = (100, 94, 84)
LINE = (54, 49, 41)
ACCENT = (226, 102, 63)
SAGE = (148, 175, 145)
GOLD = (214, 181, 120)


@dataclass
class AttemptRow:
    attempt_id: int
    timestamp: str
    gesture: str
    expected_event: str
    detected_event: str
    correct: int
    latency_ms: float | str
    fps: float
    false_activation: int
    lighting: str
    distance_cm: int | str
    camera: str
    notes: str


def summarize(rows: list[dict], idle_minutes: float = 0.0) -> dict:
    """Compute the metrics promised in the README.

    Kept as a pure function so the arithmetic can be tested without a camera.
    """

    attempt_rows = [row for row in rows if row["gesture"] != "idle"]
    idle_rows = [row for row in rows if row["gesture"] == "idle"]
    per_gesture = {}
    for gesture in GESTURE_EVENTS:
        matching = [row for row in attempt_rows if row["gesture"] == gesture]
        if not matching:
            continue
        correct = sum(int(row["correct"]) for row in matching)
        latencies = [
            float(row["latency_ms"])
            for row in matching
            if row["latency_ms"] not in ("", None)
        ]
        per_gesture[gesture] = {
            "attempts": len(matching),
            "correct": correct,
            "missed": len(matching) - correct,
            "accuracy": round(correct / len(matching), 3),
            "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        }

    idle_minutes = max(0.0, float(idle_minutes))
    false_activations = len(idle_rows)
    fps_values = [
        float(row["fps"])
        for row in rows
        if row.get("fps") not in ("", None) and float(row["fps"]) > 0
    ]
    return {
        "per_gesture": per_gesture,
        "false_activations": false_activations,
        "false_activations_per_minute": round(false_activations / idle_minutes, 2) if idle_minutes else None,
        "median_fps": round(statistics.median(fps_values), 1) if fps_values else None,
        "attempts": len(attempt_rows),
    }


class BenchmarkUI:
    """Small dedicated HUD for the guided session."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.display = pygame.font.Font(str(FONTS_DIR / "Fraunces.ttf"), 48)
        self.title = pygame.font.Font(str(FONTS_DIR / "Fraunces.ttf"), 34)
        self.sans = pygame.font.Font(str(FONTS_DIR / "Manrope.ttf"), 19)
        self.sans_small = pygame.font.Font(str(FONTS_DIR / "Manrope.ttf"), 15)
        self.mono = pygame.font.Font(str(FONTS_DIR / "IBMPlexMono-Regular.ttf"), 13)
        self.mono_small = pygame.font.Font(str(FONTS_DIR / "IBMPlexMono-Regular.ttf"), 11)

    def text(self, value: str, font, color, position, anchor="topleft", alpha=255):
        surface = font.render(value, True, color)
        if alpha != 255:
            surface.set_alpha(alpha)
        rect = surface.get_rect()
        setattr(rect, anchor, position)
        self.screen.blit(surface, rect)
        return rect

    def draw(
        self,
        *,
        phase: str,
        prompt: str,
        detail: str,
        progress: str,
        result: str | None,
        result_color,
        frame,
        landmarks: HandLandmarks | None,
        update: GestureUpdate,
        fps: float,
        remaining: float | None = None,
    ) -> None:
        self.screen.fill(BG)
        pygame.draw.line(self.screen, LINE, (40, 108), (1060, 108), 1)
        self.text("BENCHMARK  /  GESTURE GALLERY", self.mono, ACCENT, (40, 30))
        self.text("Live session", self.display, INK, (38, 44))
        self.text(progress, self.mono, MUTED, (1060 - self.mono.size(progress)[0], 38))

        video = pygame.Rect(40, 140, 500, 375)
        pygame.draw.rect(self.screen, (10, 9, 8), video)
        pygame.draw.rect(self.screen, LINE, video, width=1)
        if frame is not None:
            height, width = frame.shape[:2]
            surface = pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB").copy()
            scale = max(video.width / width, video.height / height)
            scaled = pygame.transform.smoothscale(surface, (int(width * scale), int(height * scale)))
            left = (scaled.get_width() - video.width) // 2
            top = (scaled.get_height() - video.height) // 2
            self.screen.blit(scaled, video, (left, top, video.width, video.height))
            if landmarks is not None:
                overlay = pygame.Surface((video.width, video.height), pygame.SRCALPHA)
                points = [
                    (
                        int(p[0] * width * scale - left),
                        int(p[1] * height * scale - top),
                    )
                    for p in landmarks.points
                ]
                from gesture_gallery.ui import HAND_CONNECTIONS

                for start, end in HAND_CONNECTIONS:
                    pygame.draw.line(overlay, (*SAGE, 190), points[start], points[end], 2)
                for x, y in points:
                    pygame.draw.circle(overlay, (*INK, 230), (x, y), 3)
                self.screen.blit(overlay, video.topleft)
        else:
            self.text("WAITING FOR CAMERA", self.mono, MUTED, video.center, anchor="center")

        self.text("CURRENT FRAME", self.mono_small, FAINT, (40, 530))
        self.text(f"FPS {fps:4.1f}", self.mono, MUTED, (40, 550))
        self.text(f"DETECTED {update.label or '—'}", self.mono, INK, (150, 550))

        panel = pygame.Rect(580, 140, 480, 510)
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.rect(self.screen, LINE, panel, width=1)
        self.text(phase.upper(), self.mono, ACCENT, (604, 168))
        self.text(prompt, self.title, INK, (602, 196))
        for row, line in enumerate(self._wrap(detail, self.sans, 424)):
            self.text(line, self.sans, MUTED, (604, 250 + row * 26))

        if remaining is not None:
            self.text(f"{remaining:4.1f}s", self.mono, GOLD, (1008, 168))

        if result:
            self.text(result, self.title, result_color, (602, 350))
        if update.hand_present:
            self.text("HAND DETECTED", self.mono, SAGE, (604, 440))
        else:
            self.text("NO HAND", self.mono, FAINT, (604, 440))

        self.text("ESC  ABORT", self.mono_small, FAINT, (40, 690))
        self.text("COUNTS ONLY · NO VIDEO IS SAVED", self.mono_small, FAINT, (780, 690))

    def _wrap(self, text: str, font, width: int) -> list[str]:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines


class BenchmarkSession:
    def __init__(
        self,
        settings: Settings,
        *,
        gesture: str,
        attempts: int,
        idle_minutes: float,
        output: Path,
        lighting: str,
        distance_cm: int | None,
        notes: str,
        timeout_s: float = 10.0,
    ) -> None:
        self.settings = settings
        self.gesture = gesture
        self.attempts = attempts
        self.idle_minutes = idle_minutes
        self.output = output
        self.lighting = lighting
        self.distance_cm = distance_cm
        self.notes = notes
        self.timeout_s = timeout_s
        self.rows: list[dict] = []
        self._aborted = False
        self._clock: pygame.time.Clock | None = None

    def run(self) -> int:
        pygame.init()
        pygame.display.set_caption("Gesture Gallery — Benchmark")
        flags = pygame.FULLSCREEN if self.settings.fullscreen else 0
        screen = pygame.display.set_mode((1100, 720), flags)
        clock = pygame.time.Clock()
        self._clock = clock
        ui = BenchmarkUI(screen)

        camera = Camera(
            self.settings.camera_index,
            self.settings.camera_width,
            self.settings.camera_height,
            self.settings.camera_fps,
        )
        tracker = None
        try:
            camera.open()
            tracker = HandTracker(self.settings.model_path)
        except Exception as error:
            camera.release()
            pygame.quit()
            print(f"Cannot start benchmark: {error}", file=sys.stderr)
            return 2

        engine = GestureEngine(
            Thresholds.from_sensitivity(
                self.settings.sensitivity,
                min_confidence=self.settings.min_confidence,
                cooldown_ms=self.settings.cooldown_ms,
            )
        )

        try:
            if not self._wait_for_start(ui, screen, clock, camera, tracker, engine):
                return 0
            gestures = list(GESTURE_EVENTS) if self.gesture == "all" else [self.gesture]
            for gesture in gestures:
                if not self._run_gesture(ui, screen, clock, camera, tracker, engine, gesture):
                    break
            if self.idle_minutes > 0 and not self._aborted:
                self._run_idle(ui, screen, clock, camera, tracker, engine)
        finally:
            tracker.close()
            camera.release()
            pygame.quit()

        self._write_outputs()
        return 0

    def _read_state(self, camera, tracker, engine, now_ms):
        frame = camera.read()
        if frame is None:
            return None, None, engine.update(None, now_ms)
        if self.settings.mirror:
            frame = cv2.flip(frame, 1)
        landmarks = tracker.detect(frame, int(now_ms))
        update = engine.update(
            landmarks.points if landmarks else None,
            now_ms,
            landmarks.confidence if landmarks else 0.0,
        )
        return frame, landmarks, update

    def _pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _draw(self, ui, screen, camera, tracker, engine, *, phase, prompt, detail, progress, result, color, remaining=None):
        now_ms = time.monotonic() * 1000
        frame, landmarks, update = self._read_state(camera, tracker, engine, now_ms)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ui.draw(
            phase=phase,
            prompt=prompt,
            detail=detail,
            progress=progress,
            result=result,
            result_color=color,
            frame=frame,
            landmarks=landmarks if self.settings.show_landmarks else None,
            update=update,
            fps=(self._clock.get_fps() if self._clock else 0.0),
            remaining=remaining,
        )
        pygame.display.flip()
        return update

    def _wait_for_start(self, ui, screen, clock, camera, tracker, engine):
        start = time.monotonic()
        while time.monotonic() - start < 30:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        return True
            self._draw(
                ui,
                screen,
                camera,
                tracker,
                engine,
                phase="ready",
                prompt="Press SPACE to begin",
                detail=(
                    "Stand about 50–80 cm from the camera with one hand visible. "
                    "The session prompts each gesture; return to a still open hand between attempts."
                ),
                progress=f"{self.attempts} attempts × {len(GESTURE_EVENTS) if self.gesture == 'all' else 1} gestures",
                result=None,
                color=MUTED,
            )
            clock.tick(30)
        return False

    def _run_gesture(self, ui, screen, clock, camera, tracker, engine, gesture) -> bool:
        for attempt in range(1, self.attempts + 1):
            if not self._pump_events():
                self._aborted = True
                return False

            progress = f"{gesture.upper()}  {attempt:02d}/{self.attempts:02d}"
            prompt_start = time.monotonic()
            detected = ""
            while time.monotonic() - prompt_start < self.timeout_s:
                if not self._pump_events():
                    self._aborted = True
                    return False
                remaining = self.timeout_s - (time.monotonic() - prompt_start)
                update = self._draw(
                    ui,
                    screen,
                    camera,
                    tracker,
                    engine,
                    phase="perform",
                    prompt=GESTURE_TITLES[gesture],
                    detail=GESTURE_HINTS[gesture],
                    progress=progress,
                    result=None,
                    color=MUTED,
                    remaining=remaining,
                )
                if update.event is not None:
                    detected = update.event
                    break
                clock.tick(30)

            latency = (time.monotonic() - prompt_start) * 1000 if detected else ""
            correct = int(detected == GESTURE_EVENTS[gesture])
            row = AttemptRow(
                attempt_id=len([r for r in self.rows if r["gesture"] != "idle"]) + 1,
                timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                gesture=gesture,
                expected_event=GESTURE_EVENTS[gesture],
                detected_event=detected or "NONE",
                correct=correct,
                latency_ms=round(latency, 1) if latency != "" else "",
                fps=round(clock.get_fps(), 1),
                false_activation=0,
                lighting=self.lighting,
                distance_cm=self.distance_cm if self.distance_cm is not None else "",
                camera=f"camera {self.settings.camera_index}",
                notes=self.notes,
            )
            self.rows.append(asdict(row))

            result_text = "CORRECT" if correct else "MISSED"
            result_color = SAGE if correct else ACCENT
            result_detail = f"{detected or 'no event'}  ·  {latency:.0f} ms" if latency != "" else "no event before timeout"
            wait_until = time.monotonic() + 1.0
            while time.monotonic() < wait_until:
                if not self._pump_events():
                    self._aborted = True
                    return False
                self._draw(
                    ui,
                    screen,
                    camera,
                    tracker,
                    engine,
                    phase="result",
                    prompt=GESTURE_TITLES[gesture],
                    detail=result_detail,
                    progress=progress,
                    result=result_text,
                    color=result_color,
                )
                clock.tick(30)

            # Give the person time to relax before the next prompt.
            neutral_start = time.monotonic()
            while time.monotonic() - neutral_start < 0.8:
                if not self._pump_events():
                    self._aborted = True
                    return False
                self._draw(
                    ui,
                    screen,
                    camera,
                    tracker,
                    engine,
                    phase="reset",
                    prompt="Return to neutral",
                    detail="Open hand, hold still.",
                    progress=progress,
                    result=None,
                    color=MUTED,
                )
                clock.tick(30)
        return True

    def _run_idle(self, ui, screen, clock, camera, tracker, engine) -> None:
        duration = self.idle_minutes * 60
        started = time.monotonic()
        false_count = 0
        while time.monotonic() - started < duration:
            if not self._pump_events():
                self._aborted = True
                return
            remaining = duration - (time.monotonic() - started)
            update = self._draw(
                ui,
                screen,
                camera,
                tracker,
                engine,
                phase="idle",
                prompt="Do not gesture",
                detail=(
                    "Keep your hands out of frame or still. Any confirmed event is a "
                    "false activation. This measures the system, not the person."
                ),
                progress=f"IDLE  {int(duration - remaining)}s / {int(duration)}s",
                result=None,
                color=MUTED,
                remaining=remaining,
            )
            if update.event is not None:
                false_count += 1
                self.rows.append(
                    asdict(
                        AttemptRow(
                            attempt_id=0,
                            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                            gesture="idle",
                            expected_event="NONE",
                            detected_event=update.event,
                            correct=0,
                            latency_ms="",
                            fps=round(clock.get_fps(), 1),
                            false_activation=1,
                            lighting=self.lighting,
                            distance_cm=self.distance_cm if self.distance_cm is not None else "",
                            camera=f"camera {self.settings.camera_index}",
                            notes="idle false activation",
                        )
                    )
                )
            clock.tick(30)

    def _write_outputs(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = summarize(self.rows, self.idle_minutes)
        summary["output"] = str(self.output)
        summary["aborted"] = self._aborted
        summary["settings"] = {
            "sensitivity": self.settings.sensitivity,
            "cooldown_ms": self.settings.cooldown_ms,
            "min_confidence": self.settings.min_confidence,
            "lighting": self.lighting,
            "distance_cm": self.distance_cm,
            "notes": self.notes,
        }
        summary_path = self.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(f"\nWrote {self.output}")
        print(f"Wrote {summary_path}")
        for gesture, values in summary["per_gesture"].items():
            print(
                f"{gesture:12s} {values['correct']:2d}/{values['attempts']:2d} "
                f"({values['accuracy'] * 100:5.1f}%)  "
                f"median {values['median_latency_ms']} ms"
            )
        if summary["false_activations_per_minute"] is not None:
            print(
                f"false activations: {summary['false_activations']} "
                f"({summary['false_activations_per_minute']}/min)"
            )
        if summary["median_fps"] is not None:
            print(f"median FPS: {summary['median_fps']}")


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Guided live benchmark for the gesture gallery (counts only, no video).",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--gesture",
        choices=["all", *GESTURE_EVENTS],
        default="all",
        help="gesture to test (default: all)",
    )
    parser.add_argument("--attempts", type=int, default=20, help="attempts per gesture (default: 20)")
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=0.0,
        help="minutes of intentional inactivity for false-activation measurement (default: 0)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for each attempt")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    parser.add_argument("--lighting", default="normal", help="lighting description")
    parser.add_argument("--distance-cm", type=int, default=None, help="approximate distance from camera")
    parser.add_argument("--notes", default="", help="session notes")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.idle_minutes < 0:
        parser.error("--idle-minutes cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.no_camera:
        parser.error("the benchmark requires a camera; --no-camera is not supported")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    common = [
        "--camera", str(args.camera),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--model", str(args.model),
        "--sensitivity", str(args.sensitivity),
        "--min-confidence", str(args.min_confidence),
        "--cooldown-ms", str(args.cooldown_ms),
    ]
    if args.no_mirror:
        common.append("--no-mirror")
    if args.no_camera:
        common.append("--no-camera")
    if args.hide_landmarks:
        common.append("--hide-landmarks")
    if args.fullscreen:
        common.append("--fullscreen")
    try:
        settings = load_settings(common)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    session = BenchmarkSession(
        settings,
        gesture=args.gesture,
        attempts=args.attempts,
        idle_minutes=args.idle_minutes,
        output=args.output,
        lighting=args.lighting,
        distance_cm=args.distance_cm,
        notes=args.notes,
        timeout_s=args.timeout,
    )
    return session.run()


if __name__ == "__main__":
    raise SystemExit(main())
