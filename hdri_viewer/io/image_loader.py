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

    progress_span = max(end_progress - start_progress, 0.0)
    _emit_progress(progress_callback, start_progress + progress_span * 0.25)

    if channels == 1:
        output = np.repeat(pixels[:, :, 0:1], repeats=3, axis=2)
    elif channels == 2:
        output = np.empty((height, width, 3), dtype=pixels.dtype)
        output[:, :, 0:2] = pixels[:, :, 0:2]
        output[:, :, 2] = 0
    else:
        output = pixels[:, :, :3]

    _emit_progress(progress_callback, end_progress)
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
        _emit_progress(progress_callback, 0.05)
        pixels_raw = input_file.read_image(oiio_module.FLOAT)
        if pixels_raw is None:
            error_text = input_file.geterror()
            raise RuntimeError(f"Failed to read image data: {error_text}")

        pixels = np.asarray(pixels_raw, dtype=np.float32)
        if pixels.ndim == 1:
            pixels = pixels.reshape((height, width, channels))

        _emit_progress(progress_callback, _READ_PROGRESS_WEIGHT)

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
        rgb_pixels = np.load(pixels_path)
        _emit_progress(progress_callback, 0.995)
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            channels=int(metadata["channels"]),
            dtype_name="float32",
            pixels=rgb_pixels,
        )


def _emit_progress(progress_callback: ProgressCallback | None, value: float) -> None:
    """Emits bounded progress to a callback when provided."""

    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, value)))
