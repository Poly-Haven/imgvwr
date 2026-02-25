from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import OpenImageIO as oiio

_READ_PROGRESS_WEIGHT = 0.85


def _emit_progress(percent: int) -> None:
    print(f"PROGRESS:{max(0, min(100, percent))}", flush=True)


def run_loader(path: Path, meta_path: Path, pixels_path: Path) -> None:
    input_file = oiio.ImageInput.open(str(path))
    if input_file is None:
        raise RuntimeError(f"Failed to open image: {path}. {oiio.geterror()}")

    try:
        _emit_progress(0)

        spec = input_file.spec()
        width = int(spec.width)
        height = int(spec.height)
        channels = int(spec.nchannels)
        tile_width = int(spec.tile_width)
        tile_height = int(spec.tile_height)

        pixels_memmap = np.lib.format.open_memmap(
            pixels_path,
            mode="w+",
            dtype=np.float32,
            shape=(height, width, channels),
        )

        if tile_width > 0 and tile_height > 0:
            y_step = max(tile_height, 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_tiles(0, width, y_begin, y_end, 0, 1, 0, channels, oiio.FLOAT)
                if pixels_raw is None:
                    raise RuntimeError(input_file.geterror())

                chunk = np.asarray(pixels_raw, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                pixels_memmap[y_begin:y_end, :, :] = chunk
                progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)
                _emit_progress(progress)
        else:
            y_step = max(min(128, height), 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_scanlines(y_begin, y_end, 0, 0, channels, oiio.FLOAT)
                if pixels_raw is None:
                    raise RuntimeError(input_file.geterror())

                chunk = np.asarray(pixels_raw, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                pixels_memmap[y_begin:y_end, :, :] = chunk
                progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)
                _emit_progress(progress)

        del pixels_memmap
        _emit_progress(88)
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump({"width": width, "height": height, "channels": channels}, file)
        _emit_progress(90)
    finally:
        input_file.close()


def main() -> int:
    if len(sys.argv) != 4:
        raise RuntimeError("Expected arguments: <image_path> <meta_path> <pixels_path>")

    run_loader(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
