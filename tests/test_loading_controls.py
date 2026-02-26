from __future__ import annotations

from pathlib import Path

from hdri_viewer.viewer.widget.loading_controls import LoadingControlsMixin


def test_defaults_to_2d_for_non_2_to_1_exr() -> None:
    assert LoadingControlsMixin._should_default_to_2d_projection(Path("image.exr"), width=3000, height=2000)


def test_keeps_panorama_projection_for_exact_2_to_1_exr() -> None:
    assert not LoadingControlsMixin._should_default_to_2d_projection(Path("image.exr"), width=4000, height=2000)


def test_defaults_to_2d_for_non_2_to_1_hdr_case_insensitive_extension() -> None:
    assert LoadingControlsMixin._should_default_to_2d_projection(Path("image.HDR"), width=1999, height=1000)


def test_defaults_to_2d_for_non_2_to_1_jpg() -> None:
    assert LoadingControlsMixin._should_default_to_2d_projection(Path("image.jpg"), width=1502, height=1184)


def test_keeps_panorama_projection_for_exact_2_to_1_jpg() -> None:
    assert not LoadingControlsMixin._should_default_to_2d_projection(Path("image.jpg"), width=8192, height=4096)


def test_does_not_default_to_2d_for_invalid_height() -> None:
    assert not LoadingControlsMixin._should_default_to_2d_projection(Path("image.exr"), width=2000, height=0)
