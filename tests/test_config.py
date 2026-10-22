from __future__ import annotations

import pytest

from gesture_gallery.config import Settings, load_settings


def test_settings_validate_sensitivity() -> None:
    with pytest.raises(ValueError):
        Settings(sensitivity=2.5)


def test_settings_validate_confidence_and_cooldown() -> None:
    with pytest.raises(ValueError):
        Settings(min_confidence=1.2)
    with pytest.raises(ValueError):
        Settings(cooldown_ms=-1)


def test_cli_overrides_defaults() -> None:
    settings = load_settings(
        [
            "--camera",
            "2",
            "--sensitivity",
            "1.5",
            "--no-mirror",
            "--no-camera",
            "--hide-landmarks",
            "--cooldown-ms",
            "900",
        ]
    )
    assert settings.camera_index == 2
    assert settings.sensitivity == 1.5
    assert settings.mirror is False
    assert settings.no_camera is True
    assert settings.show_landmarks is False
    assert settings.cooldown_ms == 900
    assert settings.model_path.is_absolute()
