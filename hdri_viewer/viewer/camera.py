from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class CameraState:
    """Stores camera orientation and projection settings for panorama viewing."""

    yaw_radians: float = 0.0
    pitch_radians: float = 0.0
    fov_degrees: float = 90.0


class CameraController:
    """Handles camera manipulation for equirectangular image inspection."""

    _MIN_PITCH_RAD = math.radians(-89.0)
    _MAX_PITCH_RAD = math.radians(89.0)
    _MIN_FOV_DEG = 20.0
    _MAX_FOV_DEG = 120.0

    def __init__(self, state: CameraState | None = None) -> None:
        self._state = state if state is not None else CameraState()

    @property
    def state(self) -> CameraState:
        """Returns the mutable camera state object."""

        return self._state

    def rotate(self, delta_x: float, delta_y: float, sensitivity: float = 0.005) -> None:
        """Updates yaw and pitch from pointer drag deltas in pixels."""

        self._state.yaw_radians += delta_x * sensitivity
        next_pitch = self._state.pitch_radians + delta_y * sensitivity
        self._state.pitch_radians = max(self._MIN_PITCH_RAD, min(self._MAX_PITCH_RAD, next_pitch))

    def adjust_fov(self, delta_degrees: float) -> None:
        """Adjusts field-of-view with clamping for a stable interactive range."""

        next_fov = self._state.fov_degrees + delta_degrees
        self._state.fov_degrees = max(self._MIN_FOV_DEG, min(self._MAX_FOV_DEG, next_fov))
