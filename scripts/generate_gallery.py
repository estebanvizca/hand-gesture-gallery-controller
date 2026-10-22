"""Generate the original sample artworks used by the gallery.

The images are deterministic, created from code, and released with the
repository under CC BY 4.0. They are intentionally abstract: no people,
no third-party photographs, no licensing ambiguity.

Run from the project root:

    .venv/bin/python scripts/generate_gallery.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "gallery"
WIDTH, HEIGHT = 1400, 1050


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def gradient(top: str, bottom: str) -> Image.Image:
    top_rgb = np.array(hex_rgb(top), dtype=float)
    bottom_rgb = np.array(hex_rgb(bottom), dtype=float)
    rows = np.linspace(0.0, 1.0, HEIGHT)[:, None, None]
    array = top_rgb[None, None, :] * (1 - rows) + bottom_rgb[None, None, :] * rows
    array = np.repeat(array, WIDTH, axis=1)
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def add_grain(image: Image.Image, amount: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount, (HEIGHT, WIDTH, 1))
    array = np.asarray(image, dtype=float) + noise
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def add_vignette(image: Image.Image, strength: float = 0.32) -> Image.Image:
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    cx, cy = WIDTH / 2, HEIGHT / 2
    distance = np.sqrt(((x - cx) / (WIDTH * 0.62)) ** 2 + ((y - cy) / (HEIGHT * 0.62)) ** 2)
    mask = np.clip(1.0 - strength * np.clip(distance, 0, 1) ** 2.2, 0, 1)[:, :, None]
    array = np.asarray(image, dtype=float) * mask
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def shape_layer() -> Image.Image:
    return Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))


def compose(base: Image.Image, layer: Image.Image, blur: float = 0) -> Image.Image:
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def tidal_register() -> Image.Image:
    image = gradient("#0B1E2A", "#153845")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    bands = [
        (150, 290, "#E8DFC9", 44),
        (360, 470, "#7FA6A0", 58),
        (520, 690, "#D9C6A6", 34),
        (720, 780, "#E4572E", 220),
        (820, 930, "#0A151C", 120),
    ]
    for top, bottom, color, alpha in bands:
        draw.rectangle((0, top, WIDTH, bottom), fill=(*hex_rgb(color), alpha))
    for i in range(38):
        x = 80 + i * 34
        draw.line((x, 0, x - 48, HEIGHT), fill=(232, 223, 201, 16), width=1)
    return compose(image, layer)


def solar_interval() -> Image.Image:
    image = gradient("#E9D9BD", "#C98A58")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 610, WIDTH, HEIGHT), fill=(*hex_rgb("#8F3D28"), 210))
    draw.ellipse((770, 120, 1270, 620), fill=(*hex_rgb("#E4572E"), 245))
    draw.rectangle((120, 170, 620, 730), fill=(*hex_rgb("#F4EBDD"), 82))
    draw.rectangle((220, 260, 540, 640), outline=(*hex_rgb("#2B211B"), 180), width=3)
    for i in range(6):
        y = 720 + i * 36
        draw.line((0, y, WIDTH, y - 90), fill=(*hex_rgb("#F4EBDD"), 48), width=2)
    return compose(image, layer, blur=0.4)


def quiet_signal() -> Image.Image:
    image = gradient("#08120F", "#10231C")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    for i in range(23):
        x = 90 + i * 56
        alpha = 40 + (i % 5) * 18
        draw.line((x, 120, x, 930), fill=(*hex_rgb("#BFD8C5"), alpha), width=2)
    draw.rectangle((560, 190, 612, 860), fill=(*hex_rgb("#E8DFC9"), 230))
    draw.rectangle((612, 190, 632, 860), fill=(*hex_rgb("#E4572E"), 200))
    draw.ellipse((1040, 690, 1110, 760), fill=(*hex_rgb("#BFD8C5"), 120))
    return compose(image, layer, blur=0.6)


def after_rain() -> Image.Image:
    image = gradient("#48585C", "#AEB9B4")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 690, WIDTH, HEIGHT), fill=(*hex_rgb("#293B3C"), 150))
    draw.ellipse((-160, 310, 580, 850), fill=(*hex_rgb("#D9C6A6"), 82))
    draw.ellipse((430, 210, 1140, 790), fill=(*hex_rgb("#6F8F72"), 70))
    draw.ellipse((880, 420, 1460, 890), fill=(*hex_rgb("#E8DFC9"), 55))
    draw.line((0, 690, WIDTH, 690), fill=(*hex_rgb("#F4EBDD"), 90), width=2)
    for i in range(17):
        y = 760 + i * 14
        draw.line((0, y, WIDTH, y + 7), fill=(*hex_rgb("#D8E0DA"), 18), width=1)
    return compose(image, layer, blur=24)


def red_shift() -> Image.Image:
    image = gradient("#170D0D", "#3B1412")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    for i in range(10):
        offset = i * 130 - 350
        draw.polygon(
            [
                (offset, HEIGHT),
                (offset + 260, HEIGHT),
                (offset + 860, 0),
                (offset + 600, 0),
            ],
            fill=(*hex_rgb("#D64B2A"), 46 + i * 7),
        )
    draw.rectangle((0, 0, WIDTH, 92), fill=(*hex_rgb("#E8DFC9"), 220))
    draw.rectangle((0, 92, WIDTH, 100), fill=(*hex_rgb("#E4572E"), 255))
    for x in range(0, WIDTH, 70):
        draw.line((x, 0, x, HEIGHT), fill=(*hex_rgb("#F4EBDD"), 11), width=1)
    return compose(image, layer, blur=0.5)


def paper_moon() -> Image.Image:
    image = gradient("#F1E9DA", "#D9CDB8")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    draw.ellipse((300, 110, 1110, 920), fill=(*hex_rgb("#C9D2C1"), 140))
    draw.ellipse((300, 110, 1110, 920), outline=(*hex_rgb("#27332C"), 210), width=3)
    draw.rectangle((990, 600, 1240, 850), fill=(*hex_rgb("#284C63"), 230))
    draw.arc((420, 230, 990, 800), 200, 342, fill=(*hex_rgb("#E4572E"), 220), width=5)
    draw.arc((470, 280, 940, 750), 20, 160, fill=(*hex_rgb("#27332C"), 160), width=2)
    for i in range(9):
        y = 980 - i * 22
        draw.line((80, y, WIDTH - 80, y), fill=(*hex_rgb("#27332C"), 20), width=1)
    return compose(image, layer, blur=0.3)


def night_garden() -> Image.Image:
    image = gradient("#07120D", "#12281E")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    rng = random.Random(14)
    for i in range(26):
        cx = rng.randint(-80, WIDTH + 80)
        cy = rng.randint(120, HEIGHT + 80)
        rx = rng.randint(70, 260)
        ry = rng.randint(16, 62)
        angle = rng.uniform(-55, 55)
        color = rng.choice(["#6F8F72", "#C08B6E", "#BFD8C5", "#E4572E"])
        alpha = rng.randint(38, 120)
        leaf = Image.new("RGBA", (rx * 2, ry * 2), (0, 0, 0, 0))
        ImageDraw.Draw(leaf).ellipse((0, 0, rx * 2, ry * 2), fill=(*hex_rgb(color), alpha))
        leaf = leaf.rotate(angle, expand=True)
        layer.alpha_composite(leaf, (cx - leaf.width // 2, cy - leaf.height // 2))
    draw.ellipse((900, 120, 1010, 230), fill=(*hex_rgb("#E8DFC9"), 220))
    return compose(image, layer, blur=2.2)


def low_horizon() -> Image.Image:
    image = gradient("#D8B47C", "#8F5A37")
    layer = shape_layer()
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 560, WIDTH, HEIGHT), fill=(*hex_rgb("#3B2A20"), 220))
    draw.rectangle((0, 560, WIDTH, 580), fill=(*hex_rgb("#E4572E"), 230))
    draw.ellipse((470, 250, 930, 710), fill=(*hex_rgb("#E8DFC9"), 215))
    draw.ellipse((470, 250, 930, 710), outline=(*hex_rgb("#3B2A20"), 180), width=3)
    for i in range(8):
        y = 620 + i * 42
        draw.line((0, y, WIDTH, y), fill=(*hex_rgb("#D8B47C"), 28), width=2)
    return compose(image, layer, blur=0.4)


WORKS = [
    {
        "file": "01-tidal-register.jpg",
        "title": "Tidal Register",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "A study of horizontal time: layered bands of sea, sand and signal. "
            "The single orange rule is the moment the tide turns."
        ),
        "accent": "#7FA6A0",
        "render": tidal_register,
    },
    {
        "file": "02-solar-interval.jpg",
        "title": "Solar Interval",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "Warm geometry for a room at noon. Two rectangles overlap like shadows "
            "that have agreed to keep each other company."
        ),
        "accent": "#E4572E",
        "render": solar_interval,
    },
    {
        "file": "03-quiet-signal.jpg",
        "title": "Quiet Signal",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "A field of faint vertical carriers, interrupted by one clear transmission. "
            "Made to be read from across a quiet room."
        ),
        "accent": "#BFD8C5",
        "render": quiet_signal,
    },
    {
        "file": "04-after-rain.jpg",
        "title": "After Rain",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "Soft bodies in suspension after the weather has passed. The horizon is "
            "low enough to leave room for the sky."
        ),
        "accent": "#6F8F72",
        "render": after_rain,
    },
    {
        "file": "05-red-shift.jpg",
        "title": "Red Shift",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "Ten diagonal intervals of vermilion moving away from the viewer. "
            "A strict grid holds the drift in place."
        ),
        "accent": "#D64B2A",
        "render": red_shift,
    },
    {
        "file": "06-paper-moon.jpg",
        "title": "Paper Moon",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "A pale circle drawn on warm paper, with two arcs that never quite meet. "
            "The blue rectangle is a door left ajar."
        ),
        "accent": "#284C63",
        "render": paper_moon,
    },
    {
        "file": "07-night-garden.jpg",
        "title": "Night Garden",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "Leaves remembered rather than observed. A small moon keeps watch over "
            "a garden that exists only after dark."
        ),
        "accent": "#C08B6E",
        "render": night_garden,
    },
    {
        "file": "08-low-horizon.jpg",
        "title": "Low Horizon",
        "year": "2026",
        "medium": "Digital pigment print",
        "description": (
            "Ochre air, a dark ground and one circle holding the middle. An exercise "
            "in how little a landscape needs."
        ),
        "accent": "#D8B47C",
        "render": low_horizon,
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, work in enumerate(WORKS, start=1):
        image = work["render"]()
        image = add_vignette(image)
        image = add_grain(image, amount=2.2, seed=100 + index)
        path = OUT_DIR / work["file"]
        image.save(path, "JPEG", quality=92, optimize=True)
        metadata.append(
            {
                "file": work["file"],
                "title": work["title"],
                "year": work["year"],
                "medium": work["medium"],
                "description": work["description"],
                "accent": work["accent"],
            }
        )
        print(f"wrote {path.relative_to(ROOT)}")

    with (OUT_DIR / "works.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {(OUT_DIR / 'works.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
