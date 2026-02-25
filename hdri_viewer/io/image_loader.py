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
_READ_PROGRESS_WEIGHT: Final[float] = 0.95

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

    if pixels.ndim != 3:
        raise ValueError("Expected HxWxC image array.")

    channels = pixels.shape[2]
    if channels == 3:
        return pixels
    if channels == 1:
        return np.repeat(pixels, repeats=3, axis=2)
    if channels == 2:
        zeros = np.zeros((pixels.shape[0], pixels.shape[1], 1), dtype=pixels.dtype)
        return np.concatenate((pixels, zeros), axis=2)
    return pixels[:, :, :3]


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

        rgb_pixels = np.ascontiguousarray(normalize_rgb_channels(pixels), dtype=np.float32)
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

    loader_script = (
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "import json\n"
        "import numpy as np\n"
        "import OpenImageIO as oiio\n"
        "import sys\n"
        "_READ_PROGRESS_WEIGHT = 0.95\n"
        "path = Path(sys.argv[1])\n"
        "meta_path = Path(sys.argv[2])\n"
        "pixels_path = Path(sys.argv[3])\n"
        "inp = oiio.ImageInput.open(str(path))\n"
        "if inp is None:\n"
        "    raise RuntimeError(f'Failed to open image: {path}. {oiio.geterror()}')\n"
        "try:\n"
        "    print('PROGRESS:0', flush=True)\n"
        "    spec = inp.spec()\n"
        "    width = int(spec.width)\n"
        "    height = int(spec.height)\n"
        "    channels = int(spec.nchannels)\n"
        "    tile_width = int(spec.tile_width)\n"
        "    tile_height = int(spec.tile_height)\n"
        "    pixels = np.empty((height, width, channels), dtype=np.float32)\n"
        "    if tile_width > 0 and tile_height > 0:\n"
        "        y_step = max(tile_height, 1)\n"
        "        for y_begin in range(0, height, y_step):\n"
        "            y_end = min(y_begin + y_step, height)\n"
        "            raw = inp.read_tiles(0, width, y_begin, y_end, 0, 1, 0, channels, oiio.FLOAT)\n"
        "            if raw is None:\n"
        "                raise RuntimeError(inp.geterror())\n"
        "            chunk = np.asarray(raw, dtype=np.float32)\n"
        "            if chunk.ndim == 1:\n"
        "                chunk = chunk.reshape((y_end - y_begin, width, channels))\n"
        "            pixels[y_begin:y_end, :, :] = chunk\n"
        "            progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)\n"
        "            print(f'PROGRESS:{progress}', flush=True)\n"
        "    else:\n"
        "        y_step = max(min(128, height), 1)\n"
        "        for y_begin in range(0, height, y_step):\n"
        "            y_end = min(y_begin + y_step, height)\n"
        "            raw = inp.read_scanlines(y_begin, y_end, 0, 0, channels, oiio.FLOAT)\n"
        "            if raw is None:\n"
        "                raise RuntimeError(inp.geterror())\n"
        "            chunk = np.asarray(raw, dtype=np.float32)\n"
        "            if chunk.ndim == 1:\n"
        "                chunk = chunk.reshape((y_end - y_begin, width, channels))\n"
        "            pixels[y_begin:y_end, :, :] = chunk\n"
        "            progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)\n"
        "            print(f'PROGRESS:{progress}', flush=True)\n"
        "    np.save(pixels_path, pixels)\n"
        "    with meta_path.open('w', encoding='utf-8') as file:\n"
        "        json.dump({'width': width, 'height': height, 'channels': channels}, file)\n"
        "    print('PROGRESS:100', flush=True)\n"
        "finally:\n"
        "    inp.close()\n"
    )

    with tempfile.TemporaryDirectory(prefix="imgvwr_loader_") as directory:
        work_dir = Path(directory)
        meta_path = work_dir / "meta.json"
        pixels_path = work_dir / "pixels.npy"

        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                loader_script,
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

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        source_pixels = np.load(pixels_path)
        rgb_pixels = np.ascontiguousarray(normalize_rgb_channels(source_pixels), dtype=np.float32)
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
