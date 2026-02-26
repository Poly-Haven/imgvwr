from __future__ import annotations

import io
import base64
from pathlib import Path

import numpy as np
import pytest

from hdri_viewer.io import image_loader
from hdri_viewer.io.image_loader import is_supported_image_path, normalize_rgb_channels


def test_supported_extension_check() -> None:
    assert is_supported_image_path(Path("a.exr"))
    assert is_supported_image_path(Path("b.HDR"))
    assert is_supported_image_path(Path("c.png"))


def test_channel_normalization_from_one_channel() -> None:
    source = np.ones((2, 3, 1), dtype=np.float32)
    rgb = normalize_rgb_channels(source)
    assert rgb.shape == (2, 3, 3)
    assert np.allclose(rgb[:, :, 0], 1.0)
    assert np.allclose(rgb[:, :, 1], 1.0)
    assert np.allclose(rgb[:, :, 2], 1.0)


def test_channel_normalization_from_four_channels() -> None:
    source = np.zeros((1, 2, 4), dtype=np.float32)
    source[0, 0, 3] = 1.0
    rgb = normalize_rgb_channels(source)
    assert rgb.shape == (1, 2, 3)


def test_normalize_raises_for_non_image_array() -> None:
    source = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected HxWxC image array"):
        normalize_rgb_channels(source)


def test_load_image_dispatches_to_subprocess_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = image_loader.ImageData(
        source_path=Path("a.exr"),
        width=1,
        height=1,
        channels=3,
        dtype_name="float32",
        pixels=np.zeros((1, 1, 3), dtype=np.float32),
    )

    monkeypatch.setattr(image_loader.os, "name", "nt")
    monkeypatch.setenv("IMGVWR_USE_SUBPROCESS_LOADER", "1")
    monkeypatch.setattr(image_loader, "_load_image_subprocess", lambda path, cb: expected)
    monkeypatch.setattr(
        image_loader,
        "_load_image_direct",
        lambda path, cb: (_ for _ in ()).throw(AssertionError("Direct loader should not be used.")),
    )

    result = image_loader.load_image(Path("a.exr"))
    assert result is expected


def test_load_image_dispatches_to_fast_encoded_path_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = image_loader.ImageData(
        source_path=Path("a.jpg"),
        width=1,
        height=1,
        channels=3,
        dtype_name="float32",
        pixels=np.zeros((1, 1, 3), dtype=np.float32),
    )

    monkeypatch.setattr(image_loader.os, "name", "nt")
    monkeypatch.setenv("IMGVWR_USE_SUBPROCESS_LOADER", "1")
    monkeypatch.setattr(image_loader, "_should_use_encoded_fast_path", lambda path: True)
    monkeypatch.setattr(image_loader, "_load_encoded_image_fast", lambda path, cb: expected)
    monkeypatch.setattr(
        image_loader,
        "_load_image_subprocess",
        lambda path, cb: (_ for _ in ()).throw(AssertionError("Subprocess loader should not be used.")),
    )
    monkeypatch.setattr(
        image_loader,
        "_load_image_direct",
        lambda path, cb: (_ for _ in ()).throw(AssertionError("Direct loader should not be used.")),
    )

    result = image_loader.load_image(Path("a.jpg"))
    assert result is expected


def test_load_image_dispatches_to_direct_when_subprocess_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = image_loader.ImageData(
        source_path=Path("a.exr"),
        width=2,
        height=2,
        channels=3,
        dtype_name="float32",
        pixels=np.zeros((2, 2, 3), dtype=np.float32),
    )

    monkeypatch.setattr(image_loader.os, "name", "nt")
    monkeypatch.setenv("IMGVWR_USE_SUBPROCESS_LOADER", "0")
    monkeypatch.setattr(
        image_loader,
        "_load_image_subprocess",
        lambda path, cb: (_ for _ in ()).throw(AssertionError("Subprocess loader should not be used.")),
    )
    monkeypatch.setattr(image_loader, "_load_image_direct", lambda path, cb: expected)

    result = image_loader.load_image(Path("a.exr"))
    assert result is expected


def test_subprocess_loader_invokes_module_and_parses_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_pixels = np.array(
        [
            [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
            [[9.0, 10.0, 11.0], [13.0, 14.0, 15.0]],
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "pixels.npy", source_pixels)
    (tmp_path / "meta.json").write_text('{"width": 2, "height": 2, "channels": 3}', encoding="utf-8")

    class _FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    popen_calls: list[list[str]] = []

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = ["PROGRESS:10\n", "PROGRESS:85\n", "noise\n"]
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    def _fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        popen_calls.append(args)
        return _FakeProcess()

    monkeypatch.setattr(image_loader.tempfile, "TemporaryDirectory", lambda prefix: _FakeTemporaryDirectory())
    monkeypatch.setattr(image_loader.subprocess, "Popen", _fake_popen)

    progress_updates: list[float] = []
    loaded = image_loader._load_image_subprocess(Path("a.exr"), progress_callback=progress_updates.append)

    assert popen_calls
    assert popen_calls[0][1:3] == ["-m", "hdri_viewer.io.subprocess_loader"]
    assert loaded.width == 2
    assert loaded.height == 2
    assert loaded.channels == 3
    assert loaded.pixels.shape == (2, 2, 3)
    assert progress_updates[0] == 0.10
    assert progress_updates[-1] == 1.0


def test_emit_progress_clamps_to_bounds() -> None:
    values: list[float] = []

    image_loader._emit_progress(values.append, -5.0)
    image_loader._emit_progress(values.append, 0.5)
    image_loader._emit_progress(values.append, 2.0)

    assert values == [0.0, 0.5, 1.0]


def test_srgb_to_linear_matches_reference_points() -> None:
    source = np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)
    linear = image_loader._srgb_to_linear(source)

    expected = np.array(
        [
            0.0,
            0.04045 / 12.92,
            ((0.5 + 0.055) / 1.055) ** 2.4,
            1.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(linear, expected, atol=1e-6)


def test_guess_transfer_kind_prefers_hint_over_bit_depth() -> None:
    assert image_loader._guess_transfer_kind(bits_per_sample=16, color_space_hint="sRGB") == "encoded"
    assert image_loader._guess_transfer_kind(bits_per_sample=8, color_space_hint="Linear Rec.709") == "linear"
    assert image_loader._guess_transfer_kind(bits_per_sample=8, color_space_hint="srgb_rec709_scene") == "encoded"


def test_guess_transfer_kind_uses_bit_depth_when_hint_missing() -> None:
    assert image_loader._guess_transfer_kind(bits_per_sample=8, color_space_hint=None) == "encoded"
    assert image_loader._guess_transfer_kind(bits_per_sample=16, color_space_hint=None) == "linear"


def test_should_apply_icc_transform_skips_known_srgb_hints() -> None:
    assert image_loader._should_apply_icc_transform("srgb_rec709_scene") is False
    assert image_loader._should_apply_icc_transform("Rec709") is False
    assert image_loader._should_apply_icc_transform(None) is True
    assert image_loader._should_apply_icc_transform("Display P3") is True


def test_maybe_decode_to_scene_linear_for_encoded_8bit() -> None:
    encoded = np.array([[[128.0, 128.0, 128.0]]], dtype=np.float32)
    decoded = image_loader._maybe_decode_to_scene_linear(
        encoded,
        bits_per_sample=8,
        color_space_hint=None,
        icc_profile_bytes=None,
    )

    expected_channel = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4
    assert decoded.shape == (1, 1, 3)
    assert np.allclose(decoded[0, 0, :], expected_channel, atol=1e-6)


def test_extract_icc_profile_bytes_from_numpy_array() -> None:
    class _Spec:
        def getattribute(self, name: str) -> object:
            if name == "ICCProfile":
                return np.array([1, 2, 3], dtype=np.uint8)
            return None

    extracted = image_loader._extract_icc_profile_bytes(_Spec())
    assert extracted == b"\x01\x02\x03"


def test_decode_optional_base64_round_trip() -> None:
    source = b"abc123"
    encoded = base64.b64encode(source).decode("ascii")
    assert image_loader._decode_optional_base64(encoded) == source


def test_subprocess_loader_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = ["PROGRESS:10\n"]
            self.stderr = io.StringIO("boom")

        def wait(self) -> int:
            return 3

    monkeypatch.setattr(image_loader.tempfile, "TemporaryDirectory", lambda prefix: _FakeTemporaryDirectory())
    monkeypatch.setattr(image_loader.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())

    with pytest.raises(RuntimeError, match="Image loader subprocess failed"):
        image_loader._load_image_subprocess(Path("a.exr"))


def test_subprocess_loader_raises_when_outputs_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = ["PROGRESS:10\n"]
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(image_loader.tempfile, "TemporaryDirectory", lambda prefix: _FakeTemporaryDirectory())
    monkeypatch.setattr(image_loader.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())

    with pytest.raises(RuntimeError, match="did not produce output files"):
        image_loader._load_image_subprocess(Path("a.exr"))
