from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Final

import numpy as np

_READ_PROGRESS_WEIGHT: Final[float] = 0.90
_RAW_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".3fr",
        ".arw",
        ".cr2",
        ".cr3",
        ".crw",
        ".dng",
        ".fff",
        ".iiq",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".ptx",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".x3f",
    }
)

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
    input_is_encoded_srgb: bool = False
    compression_name: str = "-"


def is_supported_image_path(path: Path) -> bool:
    """Returns True because the loader now attempts to open any requested file path."""

    _ = path
    return True


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
    """Loads an image as float32 scene-linear RGB using metadata-guided heuristics."""

    if _should_use_encoded_fast_path(path):
        fast_encoded = _load_encoded_image_fast(path, progress_callback)
        if fast_encoded is not None:
            return fast_encoded

    use_subprocess_loader = os.environ.get("IMGVWR_USE_SUBPROCESS_LOADER", "1") == "1"
    if os.name == "nt" and use_subprocess_loader:
        return _load_image_subprocess(path, progress_callback)

    return _load_image_direct(path, progress_callback)


def _load_image_direct(path: Path, progress_callback: ProgressCallback | None = None) -> ImageData:
    """Loads an image directly in-process via OpenImageIO."""

    if _is_raw_image_path(path):
        raw_image = _load_raw_image_with_rawpy(path, progress_callback)
        if raw_image is not None:
            return raw_image

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
        bits_per_sample = _infer_bits_per_sample(spec)
        source_dtype_name = _infer_source_dtype_name(spec, bits_per_sample)
        compression_name = _infer_compression_name(spec)
        color_space_hint = _infer_color_space_hint(spec)
        icc_profile_bytes = _extract_icc_profile_bytes(spec)
        transfer_kind = _guess_transfer_kind(
            bits_per_sample=bits_per_sample,
            color_space_hint=color_space_hint,
            source_path=path,
        )
        is_fast_encoded_8bit = (
            transfer_kind == "encoded" and bits_per_sample is not None and bits_per_sample <= 8
        )
        _emit_progress(progress_callback, 0.05)

        read_format = oiio_module.UINT8 if is_fast_encoded_8bit else oiio_module.FLOAT
        pixels_raw = input_file.read_image(read_format)
        if pixels_raw is None:
            error_text = input_file.geterror()
            raise RuntimeError(f"Failed to read image data: {error_text}")

        target_dtype = np.uint8 if is_fast_encoded_8bit else np.float32
        pixels = np.asarray(pixels_raw, dtype=target_dtype)
        if pixels.ndim == 1:
            pixels = pixels.reshape((height, width, channels))

        _emit_progress(progress_callback, _READ_PROGRESS_WEIGHT)

        if is_fast_encoded_8bit:
            rgb_pixels = np.ascontiguousarray(
                _normalize_rgb_channels_with_progress(pixels, progress_callback, 0.90, 0.99),
                dtype=np.uint8,
            )
            if _should_apply_icc_transform(color_space_hint):
                icc_converted_u8 = _apply_icc_profile_to_srgb_u8(rgb_pixels, icc_profile_bytes)
                if icc_converted_u8 is not None:
                    rgb_pixels = icc_converted_u8

            _emit_progress(progress_callback, 1.0)
            return ImageData(
                source_path=path,
                width=width,
                height=height,
                channels=channels,
                dtype_name=source_dtype_name,
                pixels=rgb_pixels,
                input_is_encoded_srgb=True,
                compression_name=compression_name,
            )

        rgb_pixels = np.ascontiguousarray(
            _normalize_rgb_channels_with_progress(pixels, progress_callback, 0.90, 0.99),
            dtype=np.float32,
        )
        rgb_pixels = _maybe_decode_to_scene_linear(
            rgb_pixels,
            bits_per_sample=bits_per_sample,
            color_space_hint=color_space_hint,
            icc_profile_bytes=icc_profile_bytes,
            source_path=path,
        )
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=width,
            height=height,
            channels=channels,
            dtype_name=source_dtype_name,
            pixels=rgb_pixels,
            input_is_encoded_srgb=False,
            compression_name=compression_name,
        )
    finally:
        input_file.close()


def _load_image_subprocess(
    path: Path, progress_callback: ProgressCallback | None = None
) -> ImageData:
    """Loads image in a subprocess to isolate native-library crashes on Windows."""

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
            raise RuntimeError(
                "Image loader subprocess failed. "
                f"Exit code: {return_code}. Stderr: {stderr_text}"
            )

        if not meta_path.exists() or not pixels_path.exists():
            raise RuntimeError("Image loader subprocess did not produce output files.")

        _emit_progress(progress_callback, 0.92)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _emit_progress(progress_callback, 0.94)
        bits_per_sample = _coerce_optional_int(metadata.get("bits_per_sample"))
        color_space_hint = _coerce_optional_str(metadata.get("color_space_hint"))
        transfer_kind = _guess_transfer_kind(
            bits_per_sample=bits_per_sample,
            color_space_hint=color_space_hint,
            source_path=path,
        )
        is_fast_encoded_8bit = (
            transfer_kind == "encoded" and bits_per_sample is not None and bits_per_sample <= 8
        )

        if is_fast_encoded_8bit:
            rgb_pixels = np.asarray(np.load(pixels_path), dtype=np.uint8)
            if _should_apply_icc_transform(color_space_hint):
                icc_converted_u8 = _apply_icc_profile_to_srgb_u8(
                    rgb_pixels,
                    _decode_optional_base64(metadata.get("icc_profile_b64")),
                )
                if icc_converted_u8 is not None:
                    rgb_pixels = icc_converted_u8
            source_dtype_name = _coerce_optional_str(metadata.get("source_dtype_name"))
            if source_dtype_name is None:
                source_dtype_name = _coerce_optional_str(metadata.get("dtype_name")) or "uint8"
            input_is_encoded_srgb = True
        else:
            rgb_pixels = np.asarray(np.load(pixels_path), dtype=np.float32)
            rgb_pixels = _maybe_decode_to_scene_linear(
                rgb_pixels,
                bits_per_sample=bits_per_sample,
                color_space_hint=color_space_hint,
                icc_profile_bytes=_decode_optional_base64(metadata.get("icc_profile_b64")),
                source_path=path,
            )
            source_dtype_name = _coerce_optional_str(metadata.get("source_dtype_name"))
            if source_dtype_name is None:
                source_dtype_name = _coerce_optional_str(metadata.get("dtype_name")) or "float32"
            input_is_encoded_srgb = False

        compression_name = _coerce_optional_str(metadata.get("compression_name")) or "-"

        _emit_progress(progress_callback, 0.995)
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            channels=int(metadata["channels"]),
            dtype_name=source_dtype_name,
            pixels=rgb_pixels,
            input_is_encoded_srgb=input_is_encoded_srgb,
            compression_name=compression_name,
        )


def _load_encoded_image_fast(
    path: Path, progress_callback: ProgressCallback | None = None
) -> ImageData | None:
    """Fast in-process loader for 8-bit encoded images using Pillow."""

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            _emit_progress(progress_callback, 0.05)
            mode = str(image.mode).upper()
            if mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCBCR"}:
                return None

            rgb_image = image.convert("RGB")
            _emit_progress(progress_callback, 0.75)

            icc_profile_bytes = image.info.get("icc_profile")
            if isinstance(icc_profile_bytes, bytearray):
                icc_profile_bytes = bytes(icc_profile_bytes)
            if not isinstance(icc_profile_bytes, bytes):
                icc_profile_bytes = None

            rgb_pixels = np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8))
            if icc_profile_bytes and not _icc_profile_looks_srgb(icc_profile_bytes):
                icc_converted_u8 = _apply_icc_profile_to_srgb_u8(rgb_pixels, icc_profile_bytes)
                if icc_converted_u8 is not None:
                    rgb_pixels = icc_converted_u8

            _emit_progress(progress_callback, 1.0)
            return ImageData(
                source_path=path,
                width=int(rgb_pixels.shape[1]),
                height=int(rgb_pixels.shape[0]),
                channels=3,
                dtype_name="uint8",
                pixels=rgb_pixels,
                input_is_encoded_srgb=True,
                compression_name="-",
            )
    except Exception:
        return None


def _load_raw_image_with_rawpy(
    path: Path, progress_callback: ProgressCallback | None = None
) -> ImageData | None:
    """Loads full-resolution RAW files via rawpy as scene-linear float32 RGB."""

    if not _is_raw_image_path(path):
        return None

    try:
        import rawpy
    except ImportError:
        return None

    try:
        _emit_progress(progress_callback, 0.05)
        with rawpy.imread(str(path)) as raw_file:
            rgb_u16 = raw_file.postprocess(
                output_bps=16,
                gamma=(1.0, 1.0),
                no_auto_bright=True,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
            )

        _emit_progress(progress_callback, _READ_PROGRESS_WEIGHT)
        rgb_pixels = np.ascontiguousarray(
            np.asarray(rgb_u16, dtype=np.float32) / 65535.0, dtype=np.float32
        )
        _emit_progress(progress_callback, 1.0)

        return ImageData(
            source_path=path,
            width=int(rgb_pixels.shape[1]),
            height=int(rgb_pixels.shape[0]),
            channels=3,
            dtype_name="float32",
            pixels=rgb_pixels,
            input_is_encoded_srgb=False,
            compression_name="-",
        )
    except Exception:
        return None


def _maybe_decode_to_scene_linear(
    pixels: np.ndarray,
    *,
    bits_per_sample: int | None,
    color_space_hint: str | None,
    icc_profile_bytes: bytes | None,
    source_path: Path | None = None,
) -> np.ndarray:
    """Converts encoded RGB to scene-linear when metadata heuristics indicate it is needed."""

    transfer_kind = _guess_transfer_kind(
        bits_per_sample=bits_per_sample,
        color_space_hint=color_space_hint,
        source_path=source_path,
    )
    if transfer_kind == "linear":
        return np.ascontiguousarray(pixels, dtype=np.float32)

    source_encoded = _normalize_encoded_unit_range(pixels, bits_per_sample)
    if _should_apply_icc_transform(color_space_hint):
        icc_converted = _apply_icc_profile_to_srgb(source_encoded, icc_profile_bytes)
        if icc_converted is not None:
            source_encoded = icc_converted

    return np.ascontiguousarray(_srgb_to_linear(source_encoded), dtype=np.float32)


def _guess_transfer_kind(
    *,
    bits_per_sample: int | None,
    color_space_hint: str | None,
    source_path: Path | None = None,
) -> str:
    """Returns either "linear" or "encoded" based on source metadata heuristics."""

    if source_path is not None and source_path.suffix.lower() in _RAW_EXTENSIONS:
        return "linear"

    if color_space_hint is not None:
        hint = color_space_hint.strip().lower()
        if hint:
            if any(
                token in hint
                for token in ("scene_linear", "linear", "lin_", "raw", "acescg", "non-color")
            ):
                return "linear"
            if any(token in hint for token in ("srgb", "gamma", "adobe", "display p3", "p3")):
                return "encoded"

    if bits_per_sample is not None and bits_per_sample <= 8:
        return "encoded"

    return "linear"


def _should_apply_icc_transform(color_space_hint: str | None) -> bool:
    """Returns whether ICC conversion should run for encoded sources."""

    if color_space_hint is None:
        return True

    hint = color_space_hint.strip().lower()
    if not hint:
        return True

    if any(token in hint for token in ("srgb", "rec709", "bt.709", "g22", "gamma 2.2")):
        return False

    return True


def _normalize_encoded_unit_range(pixels: np.ndarray, bits_per_sample: int | None) -> np.ndarray:
    """Normalizes likely encoded pixels to [0, 1] without affecting already normalized arrays."""

    encoded = np.asarray(pixels, dtype=np.float32)
    max_value = float(np.nanmax(encoded)) if encoded.size > 0 else 0.0
    if max_value > 1.0001:
        if bits_per_sample is not None and 1 <= bits_per_sample <= 16:
            scale = float((1 << bits_per_sample) - 1)
            if scale > 0.0:
                encoded = encoded / scale
        elif bits_per_sample is not None and bits_per_sample <= 8:
            encoded = encoded / 255.0
        else:
            encoded = encoded / max_value
    return np.clip(encoded, 0.0, 1.0)


def _apply_icc_profile_to_srgb(
    pixels: np.ndarray, icc_profile_bytes: bytes | None
) -> np.ndarray | None:
    """Applies embedded ICC profile to encoded RGB pixels, returning sRGB-encoded pixels."""

    if not icc_profile_bytes:
        return None

    try:
        from PIL import Image, ImageCms
    except ImportError:
        return None

    try:
        encoded_uint8 = np.clip(np.rint(pixels * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
        source_image = Image.fromarray(encoded_uint8, mode="RGB")

        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_bytes))
        destination_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            source_image,
            source_profile,
            destination_profile,
            outputMode="RGB",
        )
        return np.asarray(converted, dtype=np.float32) / 255.0
    except Exception:
        return None


def _apply_icc_profile_to_srgb_u8(
    pixels: np.ndarray, icc_profile_bytes: bytes | None
) -> np.ndarray | None:
    """Applies embedded ICC profile to uint8 encoded RGB pixels."""

    if not icc_profile_bytes:
        return None

    try:
        from PIL import Image, ImageCms
    except ImportError:
        return None

    try:
        source_image = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), mode="RGB")
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_bytes))
        destination_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            source_image,
            source_profile,
            destination_profile,
            outputMode="RGB",
        )
        return np.ascontiguousarray(np.asarray(converted, dtype=np.uint8))
    except Exception:
        return None


def _icc_profile_looks_srgb(icc_profile_bytes: bytes) -> bool:
    """Returns whether embedded ICC profile already represents sRGB-like encoding."""

    try:
        from PIL import ImageCms
    except ImportError:
        return False

    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_bytes))
        profile_name = ImageCms.getProfileName(profile).strip().lower()
        return "srgb" in profile_name
    except Exception:
        return False


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Converts sRGB-encoded values in [0,1] to scene-linear Rec.709 values."""

    clipped = np.clip(values, 0.0, 1.0)
    lower = clipped <= 0.04045
    return np.where(lower, clipped / 12.92, ((clipped + 0.055) / 1.055) ** 2.4)


def _infer_bits_per_sample(spec: Any) -> int | None:
    """Returns source bit depth per channel when available."""

    for attribute_name in ("oiio:BitsPerSample", "BitsPerSample", "Exif:BitsPerSample"):
        value = spec.getattribute(attribute_name)
        parsed = _coerce_optional_int(value)
        if parsed is not None and parsed > 0:
            return parsed

    pixel_format = getattr(spec, "format", None)
    if pixel_format is not None:
        try:
            base_size = int(pixel_format.basesize())
            if base_size > 0:
                return base_size * 8
        except Exception:
            pass

    return None


def _infer_source_dtype_name(spec: Any, bits_per_sample: int | None) -> str:
    """Returns human-readable source pixel type from metadata."""

    pixel_format = getattr(spec, "format", None)
    if pixel_format is not None:
        format_text = str(pixel_format).strip().lower()
        if format_text:
            if "half" in format_text:
                return "half"
            if "float" in format_text:
                return "float32"
            if "double" in format_text:
                return "float64"
            if any(token in format_text for token in ("uint8", "uchar", "byte")):
                return "uint8"
            if any(token in format_text for token in ("uint16", "ushort")):
                return "uint16"
            if "uint" in format_text:
                return "uint32"
            if any(token in format_text for token in ("int8", "char")):
                return "int8"
            if any(token in format_text for token in ("int16", "short")):
                return "int16"
            if "int" in format_text:
                return "int32"

    if bits_per_sample is not None:
        if bits_per_sample <= 8:
            return "uint8"
        if bits_per_sample <= 16:
            return "uint16"
        if bits_per_sample <= 32:
            return "float32"
        return f"{bits_per_sample}-bit"

    return "-"


def _infer_compression_name(spec: Any) -> str:
    """Returns human-readable compression type from metadata when present."""

    for attribute_name in ("compression", "Compression"):
        compression = _coerce_optional_str(spec.getattribute(attribute_name))
        if compression is not None:
            return compression
    return "-"


def _infer_color_space_hint(spec: Any) -> str | None:
    """Returns best-effort color space hint from file metadata."""

    for attribute_name in ("oiio:ColorSpace", "ColorSpace"):
        text = _coerce_optional_str(spec.getattribute(attribute_name))
        if text:
            return text

    exif_color_space = spec.getattribute("Exif:ColorSpace")
    exif_parsed = _coerce_optional_int(exif_color_space)
    if exif_parsed == 1:
        return "sRGB"

    return None


def _extract_icc_profile_bytes(spec: Any) -> bytes | None:
    """Extracts embedded ICC profile bytes from image metadata when present."""

    icc_payload = spec.getattribute("ICCProfile")
    if isinstance(icc_payload, bytes):
        return icc_payload or None
    if isinstance(icc_payload, bytearray):
        data = bytes(icc_payload)
        return data or None

    if isinstance(icc_payload, np.ndarray):
        if icc_payload.size == 0:
            return None
        icc_bytes = np.asarray(icc_payload, dtype=np.uint8).tobytes()
        return icc_bytes or None

    return None


def _decode_optional_base64(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    if not value:
        return None
    try:
        decoded = base64.b64decode(value)
    except Exception:
        return None
    return decoded or None


def _should_use_encoded_fast_path(path: Path) -> bool:
    """Returns whether path is a known fast-path encoded format."""

    return path.suffix.lower() in {".jpg", ".jpeg"}


def _is_raw_image_path(path: Path) -> bool:
    """Returns whether path suffix is a known camera RAW format."""

    return path.suffix.lower() in _RAW_EXTENSIONS


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _emit_progress(progress_callback: ProgressCallback | None, value: float) -> None:
    """Emits bounded progress to a callback when provided."""

    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, value)))
