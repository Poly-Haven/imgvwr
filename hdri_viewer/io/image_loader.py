from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Final

import numpy as np

_SUPPORTED_EXTENSIONS: Final[set[str]] = {".exr", ".hdr", ".jpg", ".jpeg"}


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


def load_image(path: Path) -> ImageData:
    """Loads an HDR/LDR image via OpenImageIO as float32 linear pixels."""

    use_subprocess_loader = os.environ.get("IMGVWR_USE_SUBPROCESS_LOADER", "1") == "1"
    if os.name == "nt" and use_subprocess_loader:
        return _load_image_subprocess(path)

    return _load_image_direct(path)


def _load_image_direct(path: Path) -> ImageData:
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
        spec = input_file.spec()
        width = int(spec.width)
        height = int(spec.height)
        channels = int(spec.nchannels)

        pixels_raw = input_file.read_image(format=oiio_module.FLOAT)
        if pixels_raw is None:
            error_text = input_file.geterror()
            raise RuntimeError(f"Failed to read image data: {error_text}")

        pixels = np.asarray(pixels_raw, dtype=np.float32)
        if pixels.ndim == 1:
            pixels = pixels.reshape((height, width, channels))

        rgb_pixels = np.ascontiguousarray(normalize_rgb_channels(pixels), dtype=np.float32)

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


def _load_image_subprocess(path: Path) -> ImageData:
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
        "path = Path(sys.argv[1])\n"
        "meta_path = Path(sys.argv[2])\n"
        "pixels_path = Path(sys.argv[3])\n"
        "inp = oiio.ImageInput.open(str(path))\n"
        "if inp is None:\n"
        "    raise RuntimeError(f'Failed to open image: {path}. {oiio.geterror()}')\n"
        "try:\n"
        "    spec = inp.spec()\n"
        "    width = int(spec.width)\n"
        "    height = int(spec.height)\n"
        "    channels = int(spec.nchannels)\n"
        "    raw = inp.read_image(format=oiio.FLOAT)\n"
        "    if raw is None:\n"
        "        raise RuntimeError(inp.geterror())\n"
        "    pixels = np.asarray(raw, dtype=np.float32)\n"
        "    if pixels.ndim == 1:\n"
        "        pixels = pixels.reshape((height, width, channels))\n"
        "    np.save(pixels_path, pixels)\n"
        "    with meta_path.open('w', encoding='utf-8') as file:\n"
        "        json.dump({'width': width, 'height': height, 'channels': channels}, file)\n"
        "finally:\n"
        "    inp.close()\n"
    )

    with tempfile.TemporaryDirectory(prefix="imgvwr_loader_") as directory:
        work_dir = Path(directory)
        meta_path = work_dir / "meta.json"
        pixels_path = work_dir / "pixels.npy"

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                loader_script,
                str(path),
                str(meta_path),
                str(pixels_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            stderr_text = completed.stderr.strip()
            raise RuntimeError(
                "Image loader subprocess failed. " f"Exit code: {completed.returncode}. Stderr: {stderr_text}"
            )

        if not meta_path.exists() or not pixels_path.exists():
            raise RuntimeError("Image loader subprocess did not produce output files.")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        source_pixels = np.load(pixels_path)
        rgb_pixels = np.ascontiguousarray(normalize_rgb_channels(source_pixels), dtype=np.float32)

        return ImageData(
            source_path=path,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            channels=int(metadata["channels"]),
            dtype_name="float32",
            pixels=rgb_pixels,
        )
