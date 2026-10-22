from __future__ import annotations

from gesture_gallery.cooldown import Cooldown


def test_first_event_is_allowed() -> None:
    cooldown = Cooldown(600)
    assert cooldown.allow(1000) is True
    assert cooldown.remaining_ms(1000) == 0


def test_cooldown_blocks_until_interval_has_passed() -> None:
    cooldown = Cooldown(600)
    cooldown.mark(1000)
    assert cooldown.allow(1599) is False
    assert cooldown.allow(1600) is True
    assert cooldown.remaining_ms(1200) == 400


def test_reset_clears_last_event() -> None:
    cooldown = Cooldown(600)
    cooldown.mark(1000)
    cooldown.reset()
    assert cooldown.allow(1001) is True


def test_negative_cooldown_is_rejected() -> None:
    try:
        Cooldown(-1)
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("expected ValueError")
