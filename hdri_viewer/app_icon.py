from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from PyQt6.QtGui import QIcon


@lru_cache(maxsize=1)
def application_icon() -> QIcon:
    """Returns the application icon, or an empty icon when unavailable."""

    icon_dir = Path(__file__).resolve().parent / "resources" / "icons"
    candidates: tuple[str, ...]
    if os.name == "nt":
        candidates = ("app_icon.ico", "app_icon.png")
    else:
        candidates = ("app_icon.png", "app_icon.ico")

    for name in candidates:
        icon_path = icon_dir / name
        if icon_path.is_file():
            return QIcon(str(icon_path))

    return QIcon()
