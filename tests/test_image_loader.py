from __future__ import annotations

from pathlib import Path

import numpy as np

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
