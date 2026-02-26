from __future__ import annotations

import base64
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
        bits_per_sample = _infer_bits_per_sample(spec)
        color_space_hint = _infer_color_space_hint(spec)
        icc_profile_bytes = _extract_icc_profile_bytes(spec)
        transfer_kind = _guess_transfer_kind(bits_per_sample=bits_per_sample, color_space_hint=color_space_hint)
        is_fast_encoded_8bit = transfer_kind == "encoded" and bits_per_sample is not None and bits_per_sample <= 8
        tile_width = int(spec.tile_width)
        tile_height = int(spec.tile_height)

        output_dtype = np.uint8 if is_fast_encoded_8bit else np.float32
        read_format = oiio.UINT8 if is_fast_encoded_8bit else oiio.FLOAT

        rgb_pixels = np.lib.format.open_memmap(
            pixels_path,
            mode="w+",
            dtype=output_dtype,
            shape=(height, width, 3),
        )

        if tile_width > 0 and tile_height > 0:
            y_step = max(tile_height, 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_tiles(0, width, y_begin, y_end, 0, 1, 0, channels, read_format)
                if pixels_raw is None:
                    raise RuntimeError(input_file.geterror())

                chunk = np.asarray(pixels_raw, dtype=output_dtype)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                if channels == 1:
                    gray = chunk[:, :, 0:1]
                    rgb_pixels[y_begin:y_end, :, :] = np.repeat(gray, repeats=3, axis=2)
                elif channels == 2:
                    rgb_pixels[y_begin:y_end, :, 0:2] = chunk[:, :, 0:2]
                    rgb_pixels[y_begin:y_end, :, 2] = 0
                else:
                    rgb_pixels[y_begin:y_end, :, :] = chunk[:, :, :3]

                progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)
                _emit_progress(progress)
        else:
            y_step = max(min(1024, height), 1)
            for y_begin in range(0, height, y_step):
                y_end = min(y_begin + y_step, height)
                pixels_raw = input_file.read_scanlines(y_begin, y_end, 0, 0, channels, read_format)
                if pixels_raw is None:
                    raise RuntimeError(input_file.geterror())

                chunk = np.asarray(pixels_raw, dtype=output_dtype)
                if chunk.ndim == 1:
                    chunk = chunk.reshape((y_end - y_begin, width, channels))

                if channels == 1:
                    gray = chunk[:, :, 0:1]
                    rgb_pixels[y_begin:y_end, :, :] = np.repeat(gray, repeats=3, axis=2)
                elif channels == 2:
                    rgb_pixels[y_begin:y_end, :, 0:2] = chunk[:, :, 0:2]
                    rgb_pixels[y_begin:y_end, :, 2] = 0
                else:
                    rgb_pixels[y_begin:y_end, :, :] = chunk[:, :, :3]

                progress = int((y_end / height) * _READ_PROGRESS_WEIGHT * 100.0)
                _emit_progress(progress)

        del rgb_pixels
        _emit_progress(88)
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "dtype_name": "uint8" if is_fast_encoded_8bit else "float32",
                    "bits_per_sample": bits_per_sample,
                    "color_space_hint": color_space_hint,
                    "icc_profile_b64": base64.b64encode(icc_profile_bytes).decode("ascii") if icc_profile_bytes else "",
                },
                file,
            )
        _emit_progress(90)
    finally:
        input_file.close()


def _infer_bits_per_sample(spec: oiio.ImageSpec) -> int | None:
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


def _infer_color_space_hint(spec: oiio.ImageSpec) -> str | None:
    for attribute_name in ("oiio:ColorSpace", "ColorSpace"):
        text = _coerce_optional_str(spec.getattribute(attribute_name))
        if text:
            return text

    exif_color_space = _coerce_optional_int(spec.getattribute("Exif:ColorSpace"))
    if exif_color_space == 1:
        return "sRGB"
    return None


def _coerce_optional_int(value: object) -> int | None:
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


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _extract_icc_profile_bytes(spec: oiio.ImageSpec) -> bytes | None:
    payload = spec.getattribute("ICCProfile")
    if isinstance(payload, bytes):
        return payload or None
    if isinstance(payload, bytearray):
        data = bytes(payload)
        return data or None
    if isinstance(payload, np.ndarray):
        if payload.size == 0:
            return None
        data = np.asarray(payload, dtype=np.uint8).tobytes()
        return data or None
    return None


def _guess_transfer_kind(*, bits_per_sample: int | None, color_space_hint: str | None) -> str:
    if color_space_hint is not None:
        hint = color_space_hint.strip().lower()
        if hint:
            if any(token in hint for token in ("srgb", "rec709", "gamma", "adobe", "display p3", "p3")):
                return "encoded"
            if any(token in hint for token in ("scene_linear", "linear", "raw", "acescg", "non-color")):
                return "linear"

    if bits_per_sample is not None and bits_per_sample <= 8:
        return "encoded"

    return "linear"


def main() -> int:
    if len(sys.argv) != 4:
        raise RuntimeError("Expected arguments: <image_path> <meta_path> <pixels_path>")

    run_loader(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
