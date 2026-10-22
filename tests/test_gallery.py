from __future__ import annotations

import json

import pytest

from gesture_gallery.controller import Event
from gesture_gallery.gallery import Gallery, load_gallery


def test_load_gallery_uses_metadata_and_order(tmp_path) -> None:
    for name in ("one.jpg", "two.jpg"):
        (tmp_path / name).write_bytes(b"not really a jpeg")
    (tmp_path / "works.json").write_text(
        json.dumps(
            [
                {
                    "file": "two.jpg",
                    "title": "Second",
                    "year": "2026",
                    "medium": "Print",
                    "description": "Two.",
                    "accent": "#123456",
                },
                {
                    "file": "one.jpg",
                    "title": "First",
                    "year": "2025",
                    "medium": "Ink",
                    "description": "One.",
                },
            ]
        ),
        encoding="utf-8",
    )
    artworks = load_gallery(tmp_path)
    assert [artwork.title for artwork in artworks] == ["Second", "First"]
    assert artworks[0].accent == "#123456"
    assert artworks[1].year == "2025"


def test_load_gallery_falls_back_to_image_files(tmp_path) -> None:
    (tmp_path / "quiet_signal.png").write_bytes(b"x")
    (tmp_path / "paper-moon.jpg").write_bytes(b"x")
    artworks = load_gallery(tmp_path)
    assert [artwork.title for artwork in artworks] == ["Paper Moon", "Quiet Signal"]


def test_load_gallery_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gallery(tmp_path)


def make_gallery() -> Gallery:
    # Gallery only needs paths and metadata; no image decoding happens here.
    artworks = load_gallery_from_names(["A", "B", "C"])
    return Gallery(artworks)


def load_gallery_from_names(names: list[str]):
    from gesture_gallery.gallery import Artwork
    from pathlib import Path

    return [
        Artwork(Path(f"{name}.jpg"), name, "2026", "Print", f"Work {name}")
        for name in names
    ]


def test_navigation_wraps() -> None:
    gallery = make_gallery()
    assert gallery.index == 0
    assert gallery.next() is True
    assert gallery.index == 1
    gallery.previous()
    gallery.previous()
    assert gallery.index == 2
    gallery.next()
    assert gallery.index == 0


def test_pause_blocks_navigation_and_select() -> None:
    gallery = make_gallery()
    assert gallery.toggle_pause() is True
    assert gallery.next() is False
    assert gallery.previous() is False
    assert gallery.select() is False
    assert gallery.index == 0
    assert gallery.toggle_pause() is False
    assert gallery.next() is True


def test_select_toggles_detail_and_navigation_closes_it() -> None:
    gallery = make_gallery()
    assert gallery.select() is True
    assert gallery.detail_open is True
    assert gallery.select() is True
    assert gallery.detail_open is False
    gallery.select()
    gallery.next()
    assert gallery.detail_open is False


def test_pausing_closes_detail() -> None:
    gallery = make_gallery()
    gallery.select()
    gallery.toggle_pause()
    assert gallery.detail_open is False


def test_handle_event_dispatches() -> None:
    gallery = make_gallery()
    assert gallery.handle_event(Event.NEXT) is True
    assert gallery.index == 1
    assert gallery.handle_event(Event.SELECT) is True
    assert gallery.detail_open is True
    assert gallery.handle_event(Event.PAUSE) is True
    assert gallery.paused is True
    assert gallery.detail_open is False
    assert gallery.handle_event(Event.PREVIOUS) is False
