from __future__ import annotations

import math

from hdri_viewer.viewer.camera import CameraController


def test_pitch_is_clamped() -> None:
    camera = CameraController()
    camera.rotate(0.0, 50000.0)
    assert camera.state.pitch_radians <= math.radians(90.0)


def test_fov_is_clamped() -> None:
    camera = CameraController()
    camera.adjust_fov(1000.0)
    assert camera.state.fov_degrees == 140.0
    camera.adjust_fov(-5000.0)
    assert camera.state.fov_degrees == 5.0


def test_fov_limit_can_be_expanded_for_fisheye() -> None:
    camera = CameraController()
    camera.set_max_fov_degrees(180.0)
    camera.adjust_fov(1000.0)
    assert camera.state.fov_degrees == 180.0
