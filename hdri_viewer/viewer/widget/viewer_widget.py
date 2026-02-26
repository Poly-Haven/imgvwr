from __future__ import annotations

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
        self._file_info = FileInfo()
        self._preferred_view_by_display: dict[str, str] = {}
        self._preferences = load_preferences()
        self._active_loader_signals: _ImageLoadSignals | None = None
        self._load_progress_value = 0.0
        self._awaiting_first_present = False
        self._projection_2d_enabled = False

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
        self._renderer.set_projection_2d_enabled(self._projection_2d_enabled)
        self._renderer.set_fisheye_enabled(self._fisheye_enabled)
        self._ocio_manager.reload()
        self._restore_preferred_view_transform()
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
        if not self._fisheye_enabled:
            viewport_aspect = max(width, 1) / max(height, 1)
            self._camera.set_max_fov_degrees(self._rectilinear_max_fov_degrees(viewport_aspect))

    def paintGL(self) -> None:
        """Delegates frame rendering to the OpenGL renderer."""

        self._renderer.render(self._camera.state)
        if self._awaiting_first_present and self._renderer.has_texture:
            self._awaiting_first_present = False
            self._set_loading_overlay("", False)
            self._set_overlay_text("")

    def _restore_preferred_view_transform(self) -> None:
        """Applies persisted display/view preference when available in current config."""

        preferred = self._preferences.preferred_view_transform
        if preferred is None:
            return

        self._ocio_manager.set_active_view(preferred.display, preferred.view)
        active = self._ocio_manager.active_view
        if active.display != preferred.display or active.view != preferred.view:
            return

        if active.view.lower() != "standard":
            self._preferred_view_by_display[active.display] = active.view

    def _persist_active_view_transform(self, display: str, view: str) -> None:
        """Persists the currently selected display/view preference to disk."""

        self._preferences = AppPreferences(preferred_view_transform=PreferredViewTransform(display=display, view=view))
        try:
            save_preferences(self._preferences)
        except OSError:
            return
