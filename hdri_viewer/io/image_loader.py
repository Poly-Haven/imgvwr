from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Final

import numpy as np

_SUPPORTED_EXTENSIONS: Final[set[str]] = {".exr", ".hdr", ".jpg", ".jpeg"}
_READ_PROGRESS_WEIGHT: Final[float] = 0.90

ProgressCallback = Callable[[float], None]


@dataclass(slots=True)
class ImageData:
    """Loaded image payload used for GPU upload."""

    source_path: Path
    width: int
    height: int
    channels: int
    dtype_name: str
    pixels: np.ndarray


def is_supported_image_path(path: Path) -> bool:
    """Checks if the file extension is supported by the viewer."""

    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def normalize_rgb_channels(pixels: np.ndarray) -> np.ndarray:
    """Normalizes image channels to RGB without changing precision."""

    return _normalize_rgb_channels_with_progress(pixels, None, 0.0, 1.0)


def _normalize_rgb_channels_with_progress(
    pixels: np.ndarray,
    progress_callback: ProgressCallback | None,
    start_progress: float,
    end_progress: float,
) -> np.ndarray:
    """Normalizes channels to RGB while emitting staged progress updates."""

    if pixels.ndim != 3:
        raise ValueError("Expected HxWxC image array.")

    height, width, channels = pixels.shape
    if height == 0 or width == 0:
        _emit_progress(progress_callback, end_progress)
        return np.zeros((height, width, 3), dtype=pixels.dtype)

    output = np.empty((height, width, 3), dtype=pixels.dtype)
    row_step = max(min(256, height), 1)
    progress_span = max(end_progress - start_progress, 0.0)

    for row_begin in range(0, height, row_step):
        row_end = min(row_begin + row_step, height)

        if channels == 1:
            gray = pixels[row_begin:row_end, :, 0:1]
            output[row_begin:row_end, :, :] = np.repeat(gray, repeats=3, axis=2)
        elif channels == 2:
            output[row_begin:row_end, :, 0:2] = pixels[row_begin:row_end, :, 0:2]
            output[row_begin:row_end, :, 2] = 0
        else:
            output[row_begin:row_end, :, :] = pixels[row_begin:row_end, :, :3]

        row_progress = row_end / height
        _emit_progress(progress_callback, start_progress + row_progress * progress_span)

    return output


def load_image(path: Path, progress_callback: ProgressCallback | None = None) -> ImageData:
    """Loads an HDR/LDR image via OpenImageIO as float32 linear pixels."""

    use_subprocess_loader = os.environ.get("IMGVWR_USE_SUBPROCESS_LOADER", "1") == "1"
    if os.name == "nt" and use_subprocess_loader:
        return _load_image_subprocess(path, progress_callback)

    return _load_image_direct(path, progress_callback)


def _load_image_direct(path: Path, progress_callback: ProgressCallback | None = None) -> ImageData:
    """Loads an image directly in-process via OpenImageIO."""

    if not is_supported_image_path(path):
        raise ValueError(f"Unsupported image format: {path.suffix}")

    import OpenImageIO as oiio

    oiio_module: Any = oiio
    input_file = oiio_module.ImageInput.open(str(path))
    if input_file is None:
        error_text = oiio_module.geterror()
        raise RuntimeError(f"Failed to open image: {path}. {error_text}")

    try:
        _emit_progress(progress_callback, 0.0)

        spec = input_file.spec()
        width = int(spec.width)
        height = int(spec.height)
        channels = int(spec.nchannels)
        tile_width = int(spec.tile_width)
        tile_height = int(spec.tile_height)

        pixels = np.empty((height, width, channels), dtype=np.float32)

        if tile_width > 0 and tile_height > 0:
            y_step = max(tile_height, 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_tiles(
                    0,
                    width,
                    y_begin,
                    y_end,
                    0,
                    1,
                    0,
                    channels,
                    oiio_module.FLOAT,
                )
                if pixels_raw is None:
                    error_text = input_file.geterror()
                    raise RuntimeError(f"Failed to read image data: {error_text}")

                chunk = np.asarray(pixels_raw, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                pixels[y_begin:y_end, :, :] = chunk
                read_progress = (y_end / height) * _READ_PROGRESS_WEIGHT
                _emit_progress(progress_callback, read_progress)
        else:
            y_step = max(min(128, height), 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_scanlines(
                    y_begin,
                    y_end,
                    0,
                    0,
                    channels,
                    oiio_module.FLOAT,
                )
                if pixels_raw is None:
                    error_text = input_file.geterror()
                    raise RuntimeError(f"Failed to read image data: {error_text}")

                chunk = np.asarray(pixels_raw, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                pixels[y_begin:y_end, :, :] = chunk
                read_progress = (y_end / height) * _READ_PROGRESS_WEIGHT
                _emit_progress(progress_callback, read_progress)

        rgb_pixels = np.ascontiguousarray(
            _normalize_rgb_channels_with_progress(pixels, progress_callback, 0.90, 0.99),
            dtype=np.float32,
        )
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=width,
            height=height,
            channels=channels,
            dtype_name="float32",
            pixels=rgb_pixels,
        )
    finally:
        input_file.close()


def _load_image_subprocess(path: Path, progress_callback: ProgressCallback | None = None) -> ImageData:
    """Loads image in a subprocess to isolate native-library crashes on Windows."""

    if not is_supported_image_path(path):
        raise ValueError(f"Unsupported image format: {path.suffix}")

    with tempfile.TemporaryDirectory(prefix="imgvwr_loader_") as directory:
        work_dir = Path(directory)
        meta_path = work_dir / "meta.json"
        pixels_path = work_dir / "pixels.npy"

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hdri_viewer.io.subprocess_loader",
                str(path),
                str(meta_path),
                str(pixels_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        progress_value = 0
        stdout_pipe = process.stdout
        if stdout_pipe is not None:
            for line in stdout_pipe:
                line_text = line.strip()
                if line_text.startswith("PROGRESS:"):
                    progress_text = line_text.split(":", 1)[1]
                    try:
                        progress_value = max(0, min(100, int(progress_text)))
                    except ValueError:
                        continue
                    _emit_progress(progress_callback, progress_value / 100.0)

        stderr_output = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()

        if return_code != 0:
            stderr_text = stderr_output.strip()
            raise RuntimeError("Image loader subprocess failed. " f"Exit code: {return_code}. Stderr: {stderr_text}")

        if not meta_path.exists() or not pixels_path.exists():
            raise RuntimeError("Image loader subprocess did not produce output files.")

        _emit_progress(progress_callback, 0.92)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _emit_progress(progress_callback, 0.94)
        source_pixels = np.load(pixels_path, mmap_mode="r")
        _emit_progress(progress_callback, 0.96)
        rgb_pixels = np.ascontiguousarray(
            _normalize_rgb_channels_with_progress(source_pixels, progress_callback, 0.96, 0.995),
            dtype=np.float32,
        )
        del source_pixels
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            channels=3,
            dtype_name="float32",
            pixels=rgb_pixels,
        )


def _emit_progress(progress_callback: ProgressCallback | None, value: float) -> None:
    """Emits bounded progress to a callback when provided."""

    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, value)))
