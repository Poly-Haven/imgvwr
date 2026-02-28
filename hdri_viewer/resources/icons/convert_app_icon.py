from __future__ import annotations

from pathlib import Path

from PIL import Image


def convert_app_icon() -> Path:
    icon_dir = Path(__file__).resolve().parent
    source_path = icon_dir / "app_icon.png"
    target_path = icon_dir / "app_icon.ico"

    if not source_path.is_file():
        raise FileNotFoundError(f"Source icon not found: {source_path}")

    with Image.open(source_path) as image:
        rgba = image.convert("RGBA")
        rgba.save(
            target_path,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )

    return target_path


if __name__ == "__main__":
    output_path = convert_app_icon()
    print(output_path)
