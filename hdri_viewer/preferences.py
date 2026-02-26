from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PreferredViewTransform:
    """Persisted OCIO display/view preference."""

    display: str
    view: str


@dataclass(frozen=True, slots=True)
class AppPreferences:
    """Persisted user preferences for imgvwr."""

    preferred_view_transform: PreferredViewTransform | None = None


def preferences_path() -> Path:
    """Returns platform-appropriate preferences file path."""

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "imgvwr" / "preferences.json"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "imgvwr" / "preferences.json"

    return Path.home() / ".config" / "imgvwr" / "preferences.json"


def load_preferences(path: Path | None = None) -> AppPreferences:
    """Loads preferences from disk, returning defaults when unavailable/invalid."""

    resolved_path = path or preferences_path()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return AppPreferences()

    return _decode_preferences(payload)


def save_preferences(preferences: AppPreferences, path: Path | None = None) -> None:
    """Saves preferences to disk using an atomic replace strategy."""

    resolved_path = path or preferences_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _encode_preferences(preferences)
    temporary_path = resolved_path.with_name(f"{resolved_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(resolved_path)


def _decode_preferences(payload: Any) -> AppPreferences:
    if not isinstance(payload, dict):
        return AppPreferences()

    preferred_payload = payload.get("preferred_view_transform")
    if not isinstance(preferred_payload, dict):
        return AppPreferences()

    display = preferred_payload.get("display")
    view = preferred_payload.get("view")
    if not isinstance(display, str) or not isinstance(view, str):
        return AppPreferences()

    if not display or not view:
        return AppPreferences()

    return AppPreferences(preferred_view_transform=PreferredViewTransform(display=display, view=view))


def _encode_preferences(preferences: AppPreferences) -> dict[str, Any]:
    if preferences.preferred_view_transform is None:
        return {}

    preferred = preferences.preferred_view_transform
    return {
        "preferred_view_transform": {
            "display": preferred.display,
            "view": preferred.view,
        }
    }
