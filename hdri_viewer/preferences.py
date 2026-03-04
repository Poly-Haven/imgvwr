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
    """Persisted user preferences for panoviewer."""

    preferred_view_transform_by_filetype: dict[str, PreferredViewTransform] | None = None


def preferences_path() -> Path:
    """Returns platform-appropriate preferences file path."""

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "panoviewer" / "preferences.json"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "panoviewer" / "preferences.json"

    return Path.home() / ".config" / "panoviewer" / "preferences.json"


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

    raw_by_filetype = payload.get("preferred_view_transform_by_filetype")
    if isinstance(raw_by_filetype, dict):
        decoded_by_filetype: dict[str, PreferredViewTransform] = {}
        for raw_extension, raw_transform in raw_by_filetype.items():
            if not isinstance(raw_extension, str):
                continue
            extension = raw_extension.strip().lower()
            if not extension:
                continue

            preferred_transform = _decode_view_transform(raw_transform)
            if preferred_transform is None:
                continue
            decoded_by_filetype[extension] = preferred_transform

        if decoded_by_filetype:
            return AppPreferences(preferred_view_transform_by_filetype=decoded_by_filetype)

    return AppPreferences()


def _encode_preferences(preferences: AppPreferences) -> dict[str, Any]:
    preferred_by_filetype = preferences.preferred_view_transform_by_filetype
    if not preferred_by_filetype:
        return {}

    payload_by_filetype: dict[str, Any] = {}
    for extension, preferred in preferred_by_filetype.items():
        normalized_extension = extension.strip().lower()
        if not normalized_extension:
            continue
        payload_by_filetype[normalized_extension] = {
            "display": preferred.display,
            "view": preferred.view,
        }

    if not payload_by_filetype:
        return {}

    return {
        "preferred_view_transform_by_filetype": payload_by_filetype,
    }


def _decode_view_transform(payload: Any) -> PreferredViewTransform | None:
    if not isinstance(payload, dict):
        return None

    display = payload.get("display")
    view = payload.get("view")
    if not isinstance(display, str) or not isinstance(view, str):
        return None

    if not display or not view:
        return None

    return PreferredViewTransform(display=display, view=view)
