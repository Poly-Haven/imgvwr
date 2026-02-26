from __future__ import annotations

from pathlib import Path

from hdri_viewer.preferences import (
    AppPreferences,
    PreferredViewTransform,
    load_preferences,
    save_preferences,
)


def test_load_preferences_returns_defaults_when_missing_file(tmp_path: Path) -> None:
    prefs = load_preferences(tmp_path / "preferences.json")
    assert prefs == AppPreferences()


def test_save_then_load_preferences_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prefs" / "preferences.json"
    source = AppPreferences(preferred_view_transform=PreferredViewTransform(display="sRGB", view="Filmic"))

    save_preferences(source, path)
    loaded = load_preferences(path)

    assert loaded == source


def test_load_preferences_returns_defaults_for_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    path.write_text('{"preferred_view_transform": {"display": "sRGB"}}', encoding="utf-8")

    prefs = load_preferences(path)
    assert prefs == AppPreferences()


def test_save_preferences_clears_payload_when_no_preference(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    save_preferences(AppPreferences(), path)

    content = path.read_text(encoding="utf-8").strip()
    assert content == "{}"
