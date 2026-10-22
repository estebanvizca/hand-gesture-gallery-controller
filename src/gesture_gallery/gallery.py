"""Gallery state and metadata loading.

The state machine is deliberately free of MediaPipe, OpenCV and Pygame. The
UI layer reads it; gesture and keyboard input both drive it through the same
``Event`` values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .controller import Event


@dataclass(frozen=True)
class Artwork:
    """One work in the gallery."""

    path: Path
    title: str
    year: str
    medium: str
    description: str
    accent: str = "#D9683A"


def load_gallery(directory: str | Path) -> list[Artwork]:
    """Load ``works.json`` plus images from *directory*.

    A metadata file is preferred, but a directory of images still works: the
    file stem becomes the title. This keeps the demo usable if someone drops
    in their own pictures.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {directory}")

    manifest = directory / "works.json"
    if manifest.exists():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        artworks = []
        for item in raw:
            image = directory / item["file"]
            if not image.exists():
                raise FileNotFoundError(f"Gallery image not found: {image}")
            artworks.append(
                Artwork(
                    path=image,
                    title=str(item.get("title") or image.stem),
                    year=str(item.get("year") or ""),
                    medium=str(item.get("medium") or ""),
                    description=str(item.get("description") or ""),
                    accent=str(item.get("accent") or "#D9683A"),
                )
            )
        if not artworks:
            raise ValueError(f"No works listed in {manifest}")
        return artworks

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        raise FileNotFoundError(f"No gallery images found in {directory}")
    return [
        Artwork(
            path=image,
            title=image.stem.replace("_", " ").replace("-", " ").title(),
            year="",
            medium="",
            description="",
            accent="#D9683A",
        )
        for image in images
    ]


class Gallery:
    """Ordered image gallery with pause and detail states."""

    def __init__(self, artworks: list[Artwork]) -> None:
        if not artworks:
            raise ValueError("The gallery needs at least one artwork")
        self.artworks = list(artworks)
        self.index = 0
        self.paused = False
        self.detail_open = False

    @property
    def current(self) -> Artwork:
        return self.artworks[self.index]

    @property
    def count(self) -> int:
        return len(self.artworks)

    def next(self) -> bool:
        """Advance one work. Returns ``False`` when the gallery is paused."""

        if self.paused:
            return False
        self.detail_open = False
        self.index = (self.index + 1) % self.count
        return True

    def previous(self) -> bool:
        """Go back one work. Returns ``False`` when the gallery is paused."""

        if self.paused:
            return False
        self.detail_open = False
        self.index = (self.index - 1) % self.count
        return True

    def select(self) -> bool:
        """Toggle the detail card for the current work."""

        if self.paused:
            return False
        self.detail_open = not self.detail_open
        return True

    def toggle_pause(self) -> bool:
        """Pause or resume input. Returns the new paused state."""

        self.paused = not self.paused
        if self.paused:
            self.detail_open = False
        return self.paused

    def handle_event(self, event: Event) -> bool:
        """Apply an event and report whether the gallery changed."""

        if event is Event.NEXT:
            return self.next()
        if event is Event.PREVIOUS:
            return self.previous()
        if event is Event.SELECT:
            return self.select()
        if event is Event.PAUSE:
            self.toggle_pause()
            return True
        return False
