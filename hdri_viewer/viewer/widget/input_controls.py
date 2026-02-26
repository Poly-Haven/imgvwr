from __future__ import annotations

import math
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QCursor, QDragEnterEvent, QDropEvent, QKeyEvent, QMouseEvent, QWheelEvent

from hdri_viewer.io.image_loader import is_supported_image_path


class InputControlsMixin:
    """Mouse, keyboard, drag/drop, and camera interaction helpers."""

    _FISHEYE_MAX_FOV_DEGREES = 270.0

    def _rectilinear_max_fov_degrees(self, viewport_aspect: float) -> float:
        """Returns rectilinear max FOV matching fisheye max visual coverage for a viewport."""

        max_lens_radius = math.sqrt((viewport_aspect * viewport_aspect) + 1.0)
        fisheye_half_fov = math.radians(self._FISHEYE_MAX_FOV_DEGREES) * 0.5
        rectilinear_half_fov = math.atan(fisheye_half_fov / max_lens_radius)
        return math.degrees(rectilinear_half_fov * 2.0)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Stores start position for drag-based camera rotation."""

        if event is not None and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._last_mouse_pos = event.position().toPoint()
            self._pending_continuous_grab_warp_pos: QPoint | None = None
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.grabMouse()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Releases explicit mouse grab when drag buttons are no longer held."""

        if event is None:
            return

        super().mouseReleaseEvent(event)

        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            return

        if not (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton)):
            self.releaseMouse()
            self._pending_continuous_grab_warp_pos = None

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Rotates/pans camera while dragging with left or middle mouse button."""

        if event is None or not (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton)):
            return

        pos = event.position().toPoint()
        pending_warp_pos = getattr(self, "_pending_continuous_grab_warp_pos", None)
        if pending_warp_pos is not None:
            if (pos - pending_warp_pos).manhattanLength() <= 4:
                self._last_mouse_pos = pos
                self._pending_continuous_grab_warp_pos = None
                return

            # If we missed the synthetic post-warp event (fast motion/OS timing),
            # recover immediately instead of staying stuck in a pending state.
            self._last_mouse_pos = pos
            self._pending_continuous_grab_warp_pos = None
            return

        delta = pos - self._last_mouse_pos
        delta = self._normalize_continuous_grab_delta(
            delta, viewport_width=max(self.width(), 1), viewport_height=max(self.height(), 1)
        )
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

            self._camera.state.yaw_radians += yaw_delta
            self._camera.state.pitch_radians += pitch_delta
            if not self._projection_2d_wrap_enabled:
                self._clamp_2d_pan_for_drag()

            self.update()
            self._wrap_cursor_for_continuous_grab(pos)
            return
        else:
            half_fov_radians = math.radians(self._camera.state.fov_degrees) * 0.5
            aspect = viewport_width / viewport_height

            if self._fisheye_enabled:
                max_lens_radius = max(math.sqrt((aspect * aspect) + 1.0), 1e-6)
                yaw_radians_per_pixel = (2.0 * aspect * half_fov_radians) / (viewport_width * max_lens_radius)
                pitch_radians_per_pixel = (2.0 * half_fov_radians) / (viewport_height * max_lens_radius)
            else:
                tan_half_fov = math.tan(half_fov_radians)
                yaw_radians_per_pixel = (2.0 * aspect * tan_half_fov) / viewport_width
                pitch_radians_per_pixel = (2.0 * tan_half_fov) / viewport_height

            cursor_pitch = self._camera.state.pitch_radians + (
                (float(pos.y()) - (viewport_height * 0.5)) * pitch_radians_per_pixel
            )
            latitude_radians = abs(cursor_pitch)
            horizontal_pan_multiplier = min(2.5, 1.0 / max(math.cos(latitude_radians), 0.25))

            yaw_delta = float(delta.x()) * yaw_radians_per_pixel * horizontal_pan_multiplier
            pitch_delta = float(delta.y()) * pitch_radians_per_pixel

        self._camera.rotate_radians(yaw_delta, pitch_delta)
        self.update()
        self._wrap_cursor_for_continuous_grab(pos)

    @staticmethod
    def _normalize_continuous_grab_delta(delta: QPoint, viewport_width: int, viewport_height: int) -> QPoint:
        """Converts wrapped-edge deltas to shortest-path motion on each axis."""

        dx = int(delta.x())
        dy = int(delta.y())

        if viewport_width > 1:
            half_width = viewport_width // 2
            if dx > half_width:
                dx -= viewport_width
            elif dx < -half_width:
                dx += viewport_width

        if viewport_height > 1:
            half_height = viewport_height // 2
            if dy > half_height:
                dy -= viewport_height
            elif dy < -half_height:
                dy += viewport_height

        return QPoint(dx, dy)

    def _wrap_cursor_for_continuous_grab(self, pos: QPoint) -> None:
        """Wraps cursor across viewport edges to allow infinite drag panning."""

        viewport_width = max(self.width(), 1)
        viewport_height = max(self.height(), 1)
        if viewport_width < 3 or viewport_height < 3:
            return

        target_x = pos.x()
        target_y = pos.y()
        wrapped = False

        if pos.x() <= 0:
            target_x = viewport_width - 2
            wrapped = True
        elif pos.x() >= viewport_width - 1:
            target_x = 1
            wrapped = True

        if pos.y() <= 0:
            target_y = viewport_height - 2
            wrapped = True
        elif pos.y() >= viewport_height - 1:
            target_y = 1
            wrapped = True

        if not wrapped:
            return

        wrapped_pos = QPoint(target_x, target_y)
        self._last_mouse_pos = wrapped_pos
        self._pending_continuous_grab_warp_pos = wrapped_pos
        QCursor.setPos(self.mapToGlobal(wrapped_pos))

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
            if self._projection_2d_enabled:
                self._handle_2d_zoom_wheel(steps)
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
            self._set_projection_2d_mode(next_projection_2d_enabled)
            self.update()
            return
        if event.key() == Qt.Key.Key_Home:
            self._reset_view_to_original_state()
            self.update()
            return
        if event.key() == Qt.Key.Key_W:
            self._set_projection_2d_wrap_enabled(not self._projection_2d_wrap_enabled)
            self.update()
            return
        if event.key() == Qt.Key.Key_F2:
            self._set_metadata_overlay_visible(not self._metadata_overlay_visible)
            return
        if event.key() == Qt.Key.Key_T:
            self._toggle_standard_view()
            return
        if event.key() == Qt.Key.Key_F:
            next_fisheye_enabled = not self._fisheye_enabled
            viewport_aspect = max(self.width(), 1) / max(self.height(), 1)
            max_lens_radius = math.sqrt((viewport_aspect * viewport_aspect) + 1.0)
            current_half_fov = math.radians(self._camera.state.fov_degrees) * 0.5

            if next_fisheye_enabled:
                converted_half_fov = math.tan(current_half_fov) * max_lens_radius
                next_max_fov_degrees = self._FISHEYE_MAX_FOV_DEGREES
            else:
                converted_half_fov = math.atan(current_half_fov / max_lens_radius)
                next_max_fov_degrees = self._rectilinear_max_fov_degrees(viewport_aspect)

            converted_fov_degrees = math.degrees(converted_half_fov * 2.0)

            self._fisheye_enabled = next_fisheye_enabled
            self._renderer.set_fisheye_enabled(self._fisheye_enabled)
            self._camera.state.fov_degrees = converted_fov_degrees
            if self._projection_2d_enabled:
                self._camera.set_max_fov_degrees(self._MAX_FOV_DEGREES_2D)
                self._camera.state.fov_degrees = min(self._camera.state.fov_degrees, self._FIT_FOV_DEGREES_2D)
            else:
                self._camera.set_max_fov_degrees(next_max_fov_degrees)
            self.update()
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

    def _handle_2d_zoom_wheel(self, steps: float) -> None:
        """Applies continuous 2D zoom using one factor across resize and optical ranges."""

        if abs(steps) < 1e-8:
            return

        fit_fov = self._FIT_FOV_DEGREES_2D
        epsilon = 1e-6

        # Keep one logical "notch" unit independent of input hardware granularity.
        wheel_notches = steps / 6.0
        zoom_per_notch = 1.12
        remaining_factor = zoom_per_notch**wheel_notches

        if remaining_factor > 1.0 + epsilon:
            if self._camera.state.fov_degrees > fit_fov + epsilon:
                current_scale = self._current_2d_optical_zoom_scale()
                fit_scale = 1.0
                factor_to_fit = fit_scale / max(current_scale, epsilon)
                optical_factor = min(remaining_factor, factor_to_fit)
                achieved = self._apply_2d_optical_zoom_factor(optical_factor)
                remaining_factor /= max(achieved, epsilon)

                if self._camera.state.fov_degrees <= fit_fov + epsilon:
                    self._camera.state.fov_degrees = fit_fov

            if remaining_factor > 1.0 + epsilon:
                achieved_window = self._resize_window_for_2d_zoom(remaining_factor)
                remaining_factor /= max(achieved_window, epsilon)

            if remaining_factor > 1.0 + epsilon:
                self._apply_2d_optical_zoom_factor(remaining_factor)
            return

        if remaining_factor < 1.0 - epsilon:
            if self._camera.state.fov_degrees < fit_fov - epsilon:
                achieved = self._apply_2d_optical_zoom_factor(remaining_factor)
                remaining_factor /= max(achieved, epsilon)

                if self._camera.state.fov_degrees >= fit_fov - epsilon:
                    self._camera.state.fov_degrees = fit_fov

            if remaining_factor < 1.0 - epsilon:
                achieved_window = self._resize_window_for_2d_zoom(remaining_factor)
                remaining_factor /= max(achieved_window, epsilon)

            if remaining_factor < 1.0 - epsilon:
                self._apply_2d_optical_zoom_factor(remaining_factor)

    def _current_2d_optical_zoom_scale(self) -> float:
        """Returns current 2D optical zoom scale relative to fit-FOV state."""

        fit_half_fov = math.radians(self._FIT_FOV_DEGREES_2D) * 0.5
        fit_inv_zoom = max(math.tan(fit_half_fov), 1e-6)

        current_half_fov = math.radians(self._camera.state.fov_degrees) * 0.5
        current_inv_zoom = max(math.tan(current_half_fov), 1e-6)
        return fit_inv_zoom / current_inv_zoom

    def _apply_2d_optical_zoom_factor(self, factor: float) -> float:
        """Applies multiplicative optical zoom factor and returns achieved factor."""

        safe_factor = max(float(factor), 1e-6)
        current_scale = max(self._current_2d_optical_zoom_scale(), 1e-6)
        target_scale = max(current_scale * safe_factor, 1e-6)

        fit_half_fov = math.radians(self._FIT_FOV_DEGREES_2D) * 0.5
        fit_inv_zoom = max(math.tan(fit_half_fov), 1e-6)

        target_inv_zoom = fit_inv_zoom / target_scale
        target_half_fov = math.atan(max(target_inv_zoom, 1e-6))
        target_fov = math.degrees(target_half_fov * 2.0)

        current_fov = self._camera.state.fov_degrees
        self._camera.adjust_fov(target_fov - current_fov)

        achieved_scale = max(self._current_2d_optical_zoom_scale(), 1e-6)
        return achieved_scale / current_scale

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
