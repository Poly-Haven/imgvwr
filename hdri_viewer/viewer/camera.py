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

    _MIN_PITCH_RAD = math.radians(-90.0)
    _MAX_PITCH_RAD = math.radians(90.0)
    _MIN_FOV_DEG = 5.0
    _DEFAULT_MAX_FOV_DEG = 140.0

    def __init__(self, state: CameraState | None = None) -> None:
        self._state = state if state is not None else CameraState()
        self._max_fov_degrees = self._DEFAULT_MAX_FOV_DEG

    @property
    def state(self) -> CameraState:
        """Returns the mutable camera state object."""

        return self._state

    def rotate(self, delta_x: float, delta_y: float, sensitivity: float = 0.005) -> None:
        """Updates yaw and pitch from pointer drag deltas in pixels."""

        self.rotate_radians(delta_x * sensitivity, delta_y * sensitivity)

    def rotate_radians(self, delta_yaw: float, delta_pitch: float) -> None:
        """Updates yaw/pitch with direct radian deltas."""

        self._state.yaw_radians += delta_yaw
        next_pitch = self._state.pitch_radians + delta_pitch
        self._state.pitch_radians = max(self._MIN_PITCH_RAD, min(self._MAX_PITCH_RAD, next_pitch))

    def adjust_fov(self, delta_degrees: float) -> None:
        """Adjusts field-of-view with clamping for a stable interactive range."""

        next_fov = self._state.fov_degrees + delta_degrees
        self._state.fov_degrees = max(self._MIN_FOV_DEG, min(self._max_fov_degrees, next_fov))

    def set_max_fov_degrees(self, max_fov_degrees: float) -> None:
        """Updates and applies the maximum allowed FOV for the active lens mode."""

        self._max_fov_degrees = max(self._MIN_FOV_DEG, float(max_fov_degrees))
        self._state.fov_degrees = max(self._MIN_FOV_DEG, min(self._max_fov_degrees, self._state.fov_degrees))
