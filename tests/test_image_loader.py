from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from hdri_viewer.io import image_loader
from hdri_viewer.io.image_loader import is_supported_image_path, normalize_rgb_channels


def test_supported_extension_check() -> None:
    assert is_supported_image_path(Path("a.exr"))
    assert is_supported_image_path(Path("b.HDR"))
    assert not is_supported_image_path(Path("c.png"))


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
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "pixels.npy", source_pixels)
    (tmp_path / "meta.json").write_text('{"width": 2, "height": 2, "channels": 4}', encoding="utf-8")

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
