from __future__ import annotations

import math
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent, QMouseEvent, QWheelEvent

from hdri_viewer.io.image_loader import is_supported_image_path


class InputControlsMixin:
    """Mouse, keyboard, drag/drop, and camera interaction helpers."""

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Stores start position for drag-based camera rotation."""

        if event is not None and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Rotates/pans camera while dragging with left or middle mouse button."""

        if event is None or not (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton)):
            return

        pos = event.position().toPoint()
        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos

        viewport_width = max(self.width(), 1)
        viewport_height = max(self.height(), 1)
        if self._projection_2d_enabled:
            viewport_aspect = viewport_width / viewport_height
            image_aspect = max(self._renderer.image_aspect, 1e-6)
            inv_zoom = max(math.tan(math.radians(self._camera.state.fov_degrees) * 0.5), 0.02)

            scale_x = inv_zoom * (viewport_aspect / image_aspect)
            scale_y = inv_zoom

            pan_u_delta = -(float(delta.x()) * scale_x) / viewport_width
            pan_v_delta = (float(delta.y()) * scale_y) / viewport_height

            yaw_delta = pan_u_delta * (2.0 * math.pi)
            pitch_delta = pan_v_delta * math.pi
        else:
            tan_half_fov = math.tan(math.radians(self._camera.state.fov_degrees) * 0.5)
            aspect = viewport_width / viewport_height
            latitude_radians = abs(self._camera.state.pitch_radians)
            horizontal_pan_multiplier = min(2.5, 1.0 / math.cos(latitude_radians))

            yaw_radians_per_pixel = (2.0 * aspect * tan_half_fov) / viewport_width
            pitch_radians_per_pixel = (2.0 * tan_half_fov) / viewport_height

            yaw_delta = float(delta.x()) * yaw_radians_per_pixel * horizontal_pan_multiplier
            pitch_delta = float(delta.y()) * pitch_radians_per_pixel

        self._camera.rotate_radians(yaw_delta, pitch_delta)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        """Toggles fullscreen mode on left-button double click."""

        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return

        window = self.window()
        if window is None:
            return

        if window.isFullScreen():
            window.showNormal()
        else:
            window.showFullScreen()

        event.accept()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Adjusts FOV or exposure based on Ctrl modifier."""

        if event is None:
            return

        steps = event.angleDelta().y() / 20.0
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._exposure_stops += steps * 0.1
            self._renderer.set_exposure(self._exposure_stops)
        else:
            fov_scale = max(0.25, self._camera.state.fov_degrees / 90.0)
            self._camera.adjust_fov(-steps * 2.0 * fov_scale)
        self.update()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Handles exposure hotkeys and interaction mode toggles."""

        if event is None:
            return

        if event.key() == Qt.Key.Key_Comma:
            self._exposure_stops -= 1.0
            self._renderer.set_exposure(self._exposure_stops)
            self.update()
            return
        if event.key() == Qt.Key.Key_Period:
            self._exposure_stops += 1.0
            self._renderer.set_exposure(self._exposure_stops)
            self.update()
            return
        if event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            window = self.window()
            if window is not None and window.isFullScreen():
                window.showNormal()
            elif window is not None:
                window.close()
            return
        if event.key() == Qt.Key.Key_P:
            center_u, center_v = self._camera_center_uv(self._projection_2d_enabled)
            next_projection_2d_enabled = not self._projection_2d_enabled
            self._set_camera_from_center_uv(center_u, center_v, next_projection_2d_enabled)

            self._projection_2d_enabled = next_projection_2d_enabled
            self._renderer.set_projection_2d_enabled(self._projection_2d_enabled)
            self.update()
            return
        if event.key() == Qt.Key.Key_T:
            self._toggle_standard_view()
            return

        super().keyPressEvent(event)

    def _camera_center_uv(self, projection_2d_enabled: bool) -> tuple[float, float]:
        """Returns the current center pixel as equirect UV for the active projection mode."""

        yaw = self._camera.state.yaw_radians
        pitch = self._camera.state.pitch_radians

        if projection_2d_enabled:
            center_u = (0.5 + (yaw / (2.0 * math.pi))) % 1.0
            center_v = 0.5 - (pitch / math.pi)
        else:
            center_u = (0.25 - (yaw / (2.0 * math.pi))) % 1.0
            center_v = 0.5 - (pitch / math.pi)

        return center_u, max(0.0, min(1.0, center_v))

    def _set_camera_from_center_uv(self, center_u: float, center_v: float, projection_2d_enabled: bool) -> None:
        """Sets camera yaw/pitch so the specified UV remains centered for the target projection mode."""

        if projection_2d_enabled:
            base_yaw = (center_u - 0.5) * (2.0 * math.pi)
        else:
            base_yaw = (0.25 - center_u) * (2.0 * math.pi)
        target_pitch = (0.5 - center_v) * math.pi

        current_yaw = self._camera.state.yaw_radians
        while base_yaw - current_yaw > math.pi:
            base_yaw -= 2.0 * math.pi
        while base_yaw - current_yaw < -math.pi:
            base_yaw += 2.0 * math.pi

        pitch_limit = math.radians(89.0)
        self._camera.state.yaw_radians = base_yaw
        self._camera.state.pitch_radians = max(-pitch_limit, min(pitch_limit, target_pitch))

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Accepts drag operations for supported image files."""

        if event is None:
            return

        mime_data = event.mimeData()
        if mime_data is not None and mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if is_supported_image_path(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Starts loading for dropped image file."""

        if event is None:
            return

        mime_data = event.mimeData()
        if mime_data is None:
            return

        urls = mime_data.urls()
        if not urls:
            return

        path = Path(urls[0].toLocalFile())
        self.open_path(path)
