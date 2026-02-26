from __future__ import annotations

import math
import os
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QThreadPool
from PyQt6.QtGui import QShowEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QLabel, QMainWindow, QWidget

from hdri_viewer.color.ocio_manager import OcioManager
from hdri_viewer.preferences import AppPreferences, PreferredViewTransform, load_preferences, save_preferences
from hdri_viewer.viewer.camera import CameraController
from hdri_viewer.viewer.renderer import PanoramaRenderer

from .input_controls import InputControlsMixin
from .loading import _ImageLoadSignals
from .loading_controls import LoadingControlsMixin
from .menu_controls import MenuControlsMixin
from .overlay_controls import OverlayControlsMixin
from .progress_bar import _LoadingProgressBar
from .types import FileInfo


class HdriViewerWidget(
    MenuControlsMixin,
    InputControlsMixin,
    LoadingControlsMixin,
    OverlayControlsMixin,
    QOpenGLWidget,
):
    """OpenGL widget that manages user interaction, rendering, and async loading."""

    _MIN_WINDOW_EDGE_PX = 170
    _FIT_FOV_DEGREES_2D = 90.0
    _MAX_FOV_DEGREES_2D = 170.0

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        base_dir = Path(__file__).resolve().parents[2]
        self._renderer = PanoramaRenderer(shaders_dir=base_dir / "viewer" / "shaders")
        self._camera = CameraController()
        self._fisheye_enabled = False
        initial_aspect = max(self.width(), 1) / max(self.height(), 1)
        self._camera.set_max_fov_degrees(self._rectilinear_max_fov_degrees(initial_aspect))
        self._ocio_manager = OcioManager(
            resources_dir=base_dir / "resources",
            custom_config_dir=base_dir / "resources" / "ocio_configs",
        )

        self._thread_pool = QThreadPool.globalInstance()
        default_threaded_loading = "1"
        self._threaded_loading_enabled = os.environ.get("IMGVWR_THREADED_LOAD", default_threaded_loading) == "1"
        self._gl_initialized = False
        self._initial_open_scheduled = False
        self._loading = False
        self._last_mouse_pos = QPoint()
        self._image_path: Path | None = None
        self._pending_initial_path: Path | None = None
        self._exposure_stops = 0.0
        self._gamma = 1.0
        self._file_info = FileInfo()
        self._preferred_view_by_display: dict[str, str] = {}
        self._preferences = load_preferences()
        self._active_loader_signals: _ImageLoadSignals | None = None
        self._load_progress_value = 0.0
        self._awaiting_first_present = False
        self._projection_2d_enabled = False
        self._projection_2d_wrap_enabled = False
        self._sync_loading_in_progress = False
        self._metadata_overlay_visible = False
        self._reset_state_yaw_radians = 0.0
        self._reset_state_pitch_radians = 0.0
        self._reset_state_fov_degrees = self._camera.state.fov_degrees
        self._reset_state_projection_2d_enabled = False
        self._reset_state_window_size: tuple[int, int] | None = None

        self._overlay_label = QLabel("", self)
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._overlay_label.setStyleSheet("color: white; background: transparent;")

        self._loading_overlay = QWidget(self)
        self._loading_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._loading_overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._loading_overlay.setStyleSheet("background: transparent;")

        self._loading_status_label = QLabel("", self._loading_overlay)
        self._loading_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_status_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._loading_status_label.setStyleSheet("color: white; background: transparent;")

        self._loading_progress_bar = _LoadingProgressBar(self._loading_overlay)
        self._loading_overlay.setVisible(False)

        self._metadata_overlay_label = QLabel("", self)
        self._metadata_overlay_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._metadata_overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._metadata_overlay_label.setWordWrap(True)
        self._metadata_overlay_label.setStyleSheet(
            "color: white;"
            "background-color: rgba(0, 0, 0, 160);"
            "border: 1px solid rgba(255, 255, 255, 90);"
            "border-radius: 4px;"
            "padding: 6px;"
        )
        self._metadata_overlay_label.setVisible(False)

        self._update_overlay_geometries()
        self._set_overlay_text("Right-click to open a file")

    @property
    def current_path(self) -> Path | None:
        """Returns currently loaded image path, if any."""

        return self._image_path

    def initializeGL(self) -> None:
        """Initializes renderer and OCIO resources after GL context is ready."""

        self._renderer.initialize()
        self._gl_initialized = True
        self._renderer.set_exposure(self._exposure_stops)
        self._renderer.set_gamma(self._gamma)
        self._renderer.set_projection_2d_enabled(self._projection_2d_enabled)
        self._renderer.set_projection_2d_wrap_enabled(self._projection_2d_wrap_enabled)
        self._renderer.set_fisheye_enabled(self._fisheye_enabled)
        self._ocio_manager.reload()
        self._restore_preferred_view_transform(self._pending_initial_path)
        self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
        self._schedule_initial_open_if_ready()

    def set_initial_path(self, path: Path | None) -> None:
        """Stores startup path to open after OpenGL/OCIO initialization."""

        self._pending_initial_path = path
        self._schedule_initial_open_if_ready()

    def showEvent(self, event: QShowEvent | None) -> None:
        """Schedules startup image opening once widget is shown and GL is ready."""

        super().showEvent(event)
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._schedule_initial_open_if_ready()

    def resizeGL(self, width: int, height: int) -> None:
        """Updates renderer viewport."""

        self._renderer.set_viewport(width, height)
        if self._projection_2d_enabled:
            self._camera.set_max_fov_degrees(self._MAX_FOV_DEGREES_2D)
            self._clamp_2d_pan_to_image_bounds()
            return

        if not self._fisheye_enabled:
            viewport_aspect = max(width, 1) / max(height, 1)
            self._camera.set_max_fov_degrees(self._rectilinear_max_fov_degrees(viewport_aspect))

    def _set_projection_2d_mode(self, enabled: bool) -> None:
        """Updates 2D projection state and its camera constraints."""

        self._projection_2d_enabled = enabled
        self._renderer.set_projection_2d_enabled(enabled)

        if enabled:
            self._camera.set_max_fov_degrees(self._MAX_FOV_DEGREES_2D)
            self._camera.state.fov_degrees = min(self._camera.state.fov_degrees, self._FIT_FOV_DEGREES_2D)
            self._normalize_2d_pan_offsets_modulo()
            if not self._projection_2d_wrap_enabled:
                self._clamp_2d_pan_to_image_bounds()
            return

        if self._fisheye_enabled:
            self._camera.set_max_fov_degrees(self._FISHEYE_MAX_FOV_DEGREES)
            return

        viewport_aspect = max(self.width(), 1) / max(self.height(), 1)
        self._camera.set_max_fov_degrees(self._rectilinear_max_fov_degrees(viewport_aspect))

    def _normalize_2d_pan_offsets_modulo(self) -> None:
        """Normalizes 2D horizontal pan offset to modulo 1.0 in signed [-0.5, 0.5) range."""

        pan_u = self._camera.state.yaw_radians / (2.0 * math.pi)
        pan_u = ((pan_u + 0.5) % 1.0) - 0.5
        self._camera.state.yaw_radians = pan_u * (2.0 * math.pi)

    def _set_projection_2d_wrap_enabled(self, enabled: bool) -> None:
        """Sets 2D wrap mode: tiled repeat on both axes or no wrapping."""

        self._projection_2d_wrap_enabled = enabled
        self._renderer.set_projection_2d_wrap_enabled(enabled)
        if not enabled:
            self._clamp_2d_pan_to_image_bounds()

    @staticmethod
    def _fit_aspect_within_bounds(max_width: int, max_height: int, aspect_ratio: float) -> tuple[int, int]:
        """Fits a size preserving aspect ratio into bounds."""

        bounded_width = max(1, int(max_width))
        bounded_height = max(1, int(max_height))
        aspect = max(float(aspect_ratio), 1e-6)

        width = min(float(bounded_width), float(bounded_height) * aspect)
        height = width / aspect
        if height > float(bounded_height):
            height = float(bounded_height)
            width = height * aspect

        return max(1, int(round(width))), max(1, int(round(height)))

    @classmethod
    def _minimum_2d_window_size_for_aspect(cls, aspect_ratio: float) -> tuple[int, int]:
        """Computes minimum 2D window size preserving image aspect and minimum edge."""

        aspect = max(float(aspect_ratio), 1e-6)
        min_edge = cls._MIN_WINDOW_EDGE_PX

        width = float(min_edge)
        height = width / aspect
        if height < float(min_edge):
            height = float(min_edge)
            width = height * aspect

        return int(round(width)), int(round(height))

    def _resize_window_for_2d_zoom(self, zoom_multiplier: float) -> float:
        """Resizes top-level window for 2D zoom and returns achieved zoom factor."""

        if zoom_multiplier <= 0.0:
            return 1.0

        window = self.window()
        if window is None or not isinstance(window, QMainWindow):
            return 1.0
        if window.isFullScreen() or window.isMaximized():
            return 1.0

        screen = window.screen()
        if screen is None:
            handle = window.windowHandle()
            if handle is not None:
                screen = handle.screen()
        if screen is None:
            return 1.0

        before_width = max(window.width(), 1)

        available = screen.availableGeometry()
        frame_left, frame_top, frame_right, frame_bottom = self._window_frame_margins(window)
        max_window_width = max(1, int(available.width()) - frame_left - frame_right)
        max_window_height = max(1, int(available.height()) - frame_top - frame_bottom)

        image_aspect = max(self._renderer.image_aspect, 1e-6)
        min_width, min_height = self._minimum_2d_window_size_for_aspect(image_aspect)
        max_width, max_height = self._fit_aspect_within_bounds(max_window_width, max_window_height, image_aspect)

        current_width = max(window.width(), 1)
        current_height = max(window.height(), 1)
        current_aspect = current_width / current_height
        if abs(current_aspect - image_aspect) > 1e-3:
            current_width, current_height = self._fit_aspect_within_bounds(current_width, current_height, image_aspect)

        target_width = int(round(current_width * zoom_multiplier))
        target_height = int(round(current_height * zoom_multiplier))
        target_width, target_height = self._fit_aspect_within_bounds(target_width, target_height, image_aspect)

        clamped_width = min(max(target_width, min_width), max_width)
        clamped_height = min(max(target_height, min_height), max_height)
        clamped_width, clamped_height = self._fit_aspect_within_bounds(clamped_width, clamped_height, image_aspect)

        resized = self._resize_window_centered(clamped_width, clamped_height)
        if not resized:
            return 1.0

        after_width = max(window.width(), 1)
        achieved_factor = after_width / before_width
        if zoom_multiplier >= 1.0:
            return min(max(achieved_factor, 1.0), zoom_multiplier)
        return max(min(achieved_factor, 1.0), zoom_multiplier)

    def _clamp_2d_pan_to_image_bounds(self) -> None:
        """Keeps 2D view inside image bounds (no black background) when wrapping is disabled."""

        self._clamp_2d_pan_to_image_bounds_for_viewport(max(self.width(), 1), max(self.height(), 1))

    def _clamp_2d_pan_to_image_bounds_for_viewport(self, viewport_width: int, viewport_height: int) -> None:
        """Clamps 2D pan for an explicit viewport size to avoid post-resize correction jumps."""

        if not self._projection_2d_enabled or self._projection_2d_wrap_enabled:
            return
        if not self._renderer.has_texture:
            return

        viewport_width = max(int(viewport_width), 1)
        viewport_height = max(int(viewport_height), 1)
        viewport_aspect = viewport_width / viewport_height
        image_aspect = max(self._renderer.image_aspect, 1e-6)

        inv_zoom = max(math.tan(math.radians(self._camera.state.fov_degrees) * 0.5), 0.02)
        scale_x = inv_zoom * (viewport_aspect / image_aspect)
        scale_y = inv_zoom

        half_span_u = 0.5 * scale_x
        half_span_v = 0.5 * scale_y

        pan_u = self._camera.state.yaw_radians / (2.0 * math.pi)
        pan_v = -self._camera.state.pitch_radians / math.pi

        if half_span_u < 0.5:
            min_pan_u = -0.5 + half_span_u
            max_pan_u = 0.5 - half_span_u
            pan_u = min(max(pan_u, min_pan_u), max_pan_u)
        else:
            pan_u = 0.0

        if half_span_v < 0.5:
            min_pan_v = -0.5 + half_span_v
            max_pan_v = 0.5 - half_span_v
            pan_v = min(max(pan_v, min_pan_v), max_pan_v)
        else:
            pan_v = 0.0

        self._camera.state.yaw_radians = pan_u * (2.0 * math.pi)
        self._camera.state.pitch_radians = -pan_v * math.pi

    def _clamp_2d_pan_for_drag(self) -> None:
        """Clamps 2D drag pan so image corners can reach viewport center."""

        if not self._projection_2d_enabled or self._projection_2d_wrap_enabled:
            return
        if not self._renderer.has_texture:
            return

        pan_u = self._camera.state.yaw_radians / (2.0 * math.pi)
        pan_v = -self._camera.state.pitch_radians / math.pi
        pan_u = min(max(pan_u, -0.9), 0.9)
        pan_v = min(max(pan_v, -0.9), 0.9)

        self._camera.state.yaw_radians = pan_u * (2.0 * math.pi)
        self._camera.state.pitch_radians = -pan_v * math.pi

    @staticmethod
    def _window_frame_margins(window: QMainWindow) -> tuple[int, int, int, int]:
        """Returns current frame margins as left, top, right, bottom pixel values."""

        geometry = window.geometry()
        frame = window.frameGeometry()

        frame_left = max(0, geometry.x() - frame.x())
        frame_top = max(0, geometry.y() - frame.y())
        frame_right = max(0, frame.right() - geometry.right())
        frame_bottom = max(0, frame.bottom() - geometry.bottom())
        return frame_left, frame_top, frame_right, frame_bottom

    def _resize_window_centered(self, target_width: int, target_height: int) -> bool:
        """Resizes the top-level window while keeping its center position stable."""

        window = self.window()
        if window is None or not isinstance(window, QMainWindow):
            return False
        if window.isFullScreen() or window.isMaximized():
            return False

        screen = window.screen()
        if screen is None:
            handle = window.windowHandle()
            if handle is not None:
                screen = handle.screen()

        width = max(1, int(target_width))
        height = max(1, int(target_height))

        frame_left, frame_top, frame_right, frame_bottom = self._window_frame_margins(window)
        if screen is not None:
            available = screen.availableGeometry()
            max_client_width = max(1, int(available.width()) - frame_left - frame_right)
            max_client_height = max(1, int(available.height()) - frame_top - frame_bottom)
            width = min(width, max_client_width)
            height = min(height, max_client_height)

        if width == window.width() and height == window.height():
            return False

        old_frame = window.frameGeometry()
        center = old_frame.center()

        target_frame_width = width + frame_left + frame_right
        target_frame_height = height + frame_top + frame_bottom
        target_frame_x = center.x() - (target_frame_width // 2)
        target_frame_y = center.y() - (target_frame_height // 2)

        if screen is not None:
            available = screen.availableGeometry()
            min_frame_x = available.left()
            min_frame_y = available.top()
            max_frame_x = available.right() - target_frame_width + 1
            max_frame_y = available.bottom() - target_frame_height + 1
            if max_frame_x < min_frame_x:
                max_frame_x = min_frame_x
            if max_frame_y < min_frame_y:
                max_frame_y = min_frame_y
            target_frame_x = min(max(target_frame_x, min_frame_x), max_frame_x)
            target_frame_y = min(max(target_frame_y, min_frame_y), max_frame_y)

        # Clamp pan in the same pass as zoom-resize so we avoid a second visual jump.
        self._clamp_2d_pan_to_image_bounds_for_viewport(width, height)

        # Single top-level geometry update to minimize OS-level jitter.
        client_x = target_frame_x + frame_left
        client_y = target_frame_y + frame_top
        window.setGeometry(client_x, client_y, width, height)

        return True

    def _fit_window_to_image_on_first_open(self, image_width: int, image_height: int) -> None:
        """Fits window to first opened 2D image, constrained by screen bounds and min size."""

        if not self._projection_2d_enabled:
            return

        safe_width = max(int(image_width), 1)
        safe_height = max(int(image_height), 1)
        image_aspect = safe_width / safe_height

        requested_width = safe_width
        requested_height = safe_height

        window = self.window()
        if window is None or not isinstance(window, QMainWindow):
            return

        screen = window.screen()
        if screen is None:
            handle = window.windowHandle()
            if handle is not None:
                screen = handle.screen()
        if screen is None:
            return

        available = screen.availableGeometry()
        frame_left, frame_top, frame_right, frame_bottom = self._window_frame_margins(window)
        max_width = max(1, int(available.width()) - frame_left - frame_right)
        max_height = max(1, int(available.height()) - frame_top - frame_bottom)
        min_width, min_height = self._minimum_2d_window_size_for_aspect(image_aspect)
        fit_width, fit_height = self._fit_aspect_within_bounds(max_width, max_height, image_aspect)

        target_width = requested_width
        target_height = requested_height
        if target_width > max_width or target_height > max_height:
            target_width = fit_width
            target_height = fit_height
        else:
            target_width, target_height = self._fit_aspect_within_bounds(target_width, target_height, image_aspect)

        target_width = min(max(target_width, min_width), fit_width)
        target_height = min(max(target_height, min_height), fit_height)
        target_width, target_height = self._fit_aspect_within_bounds(target_width, target_height, image_aspect)

        self._camera.state.fov_degrees = self._FIT_FOV_DEGREES_2D
        self._resize_window_centered(target_width, target_height)

    def _capture_view_reset_state(self) -> None:
        """Captures the current pan/zoom state used by Home reset."""

        self._reset_state_yaw_radians = self._camera.state.yaw_radians
        self._reset_state_pitch_radians = self._camera.state.pitch_radians
        self._reset_state_fov_degrees = self._camera.state.fov_degrees
        self._reset_state_projection_2d_enabled = self._projection_2d_enabled

        window = self.window()
        if window is not None and isinstance(window, QMainWindow):
            self._reset_state_window_size = (max(window.width(), 1), max(window.height(), 1))
        else:
            self._reset_state_window_size = None

    def _reset_view_to_original_state(self) -> None:
        """Restores pan/zoom state captured for the currently opened image."""

        self._camera.state.yaw_radians = self._reset_state_yaw_radians
        self._camera.state.pitch_radians = self._reset_state_pitch_radians

        if self._projection_2d_enabled:
            self._camera.state.fov_degrees = min(self._reset_state_fov_degrees, self._FIT_FOV_DEGREES_2D)
            if self._reset_state_projection_2d_enabled and self._reset_state_window_size is not None:
                target_width, target_height = self._reset_state_window_size
                self._resize_window_centered(target_width, target_height)
        else:
            self._camera.state.fov_degrees = self._reset_state_fov_degrees

    def paintGL(self) -> None:
        """Delegates frame rendering to the OpenGL renderer."""

        self._renderer.render(self._camera.state)
        if self._awaiting_first_present and self._renderer.has_texture:
            self._awaiting_first_present = False
            self._set_loading_overlay("", False)
            self._set_overlay_text("")

    def _restore_preferred_view_transform(self, path: Path | None = None) -> None:
        """Applies persisted display/view preference for the given file type."""

        if self._projection_2d_enabled:
            self._apply_standard_view_default()
            return

        preferred = self._preferred_transform_for_path(path)
        if preferred is None:
            self._apply_standard_view_default()
            return

        self._ocio_manager.set_active_view(preferred.display, preferred.view)
        active = self._ocio_manager.active_view
        if active.display != preferred.display or active.view != preferred.view:
            self._apply_standard_view_default()
            return

        if active.view.lower() != "standard":
            self._preferred_view_by_display[active.display] = active.view

    def _persist_active_view_transform(self, display: str, view: str) -> None:
        """Persists the active display/view under the current image file type key."""

        if self._projection_2d_enabled:
            return

        file_type_key = self._current_file_type_key()
        preferred_by_filetype = dict(self._preferences.preferred_view_transform_by_filetype or {})
        preferred_by_filetype[file_type_key] = PreferredViewTransform(display=display, view=view)
        self._preferences = AppPreferences(preferred_view_transform_by_filetype=preferred_by_filetype)
        try:
            save_preferences(self._preferences)
        except OSError:
            return

    def _preferred_transform_for_path(self, path: Path | None) -> PreferredViewTransform | None:
        """Returns preferred transform for a path extension."""

        preferred_by_filetype = self._preferences.preferred_view_transform_by_filetype or {}
        file_type_key = self._file_type_key(path)
        return preferred_by_filetype.get(file_type_key)

    def _apply_standard_view_default(self) -> None:
        """Selects Standard view when available, preferring current display."""

        active_display = self._ocio_manager.active_view.display

        active_display_views = self._views_for_display(active_display)
        standard_view = self._find_case_insensitive(active_display_views, "Standard")
        if standard_view is not None:
            self._ocio_manager.set_active_view(active_display, standard_view)
            return

        for display in self._available_displays():
            display_views = self._views_for_display(display)
            standard_view = self._find_case_insensitive(display_views, "Standard")
            if standard_view is not None:
                self._ocio_manager.set_active_view(display, standard_view)
                return

    def _current_file_type_key(self) -> str:
        """Returns current file-type key used for preference persistence."""

        if self._image_path is not None:
            return self._file_type_key(self._image_path)
        if self._pending_initial_path is not None:
            return self._file_type_key(self._pending_initial_path)
        return "*"

    @staticmethod
    def _file_type_key(path: Path | None) -> str:
        """Converts image path to normalized extension key."""

        if path is None:
            return "*"
        suffix = path.suffix.strip().lower()
        if suffix:
            return suffix
        return "*"
