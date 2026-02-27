from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FileInfo:
    """Simple metadata record for context menu display."""

    source_name: str = "-"
    width: int = 0
    height: int = 0
    channels: int = 0
    dtype_name: str = "-"
    compression_name: str = "-"
    input_is_encoded_srgb: bool = False
