from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FileInfo:
    """Simple metadata record for context menu display."""

    width: int = 0
    height: int = 0
    channels: int = 0
    dtype_name: str = "-"
