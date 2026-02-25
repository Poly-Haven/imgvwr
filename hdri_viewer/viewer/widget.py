from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QPoint, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import (
    QAction,
    QContextMenuEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QMouseEvent,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMenu

from hdri_viewer.color.ocio_manager import DisplayView, OcioManager
from hdri_viewer.io.image_loader import ImageData, is_supported_image_path, load_image
from hdri_viewer.viewer.camera import CameraController
from hdri_viewer.viewer.renderer import PanoramaRenderer


class _ImageLoadSignals(QObject):
    """Qt signal bridge used by background image loading tasks."""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)


class _ImageLoadTask(QRunnable):
    """Worker task that performs OpenImageIO loading in a thread-pool thread."""

    def __init__(self, path: Path, signals: _ImageLoadSignals) -> None:
        super().__init__()
        self._path = path
        self._signals = signals

    def run(self) -> None:
        """Loads image and emits completion or failure signal."""

        try:
            image = load_image(self._path)
        except Exception as error:  # pragma: no cover - tested via signal path
            self._signals.failed.emit(str(error))
            return
        self._signals.loaded.emit(image)


@dataclass(slots=True)
class FileInfo:
    """Simple metadata record for context menu display."""

    width: int = 0
    height: int = 0
    channels: int = 0
    dtype_name: str = "-"


class HdriViewerWidget(QOpenGLWidget):
    """OpenGL widget that manages user interaction, rendering, and async loading."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        base_dir = Path(__file__).resolve().parents[1]
        self._renderer = PanoramaRenderer(shaders_dir=base_dir / "viewer" / "shaders")
        self._camera = CameraController()
        self._ocio_manager = OcioManager(
            resources_dir=base_dir / "resources",
            custom_config_dir=base_dir / "resources" / "ocio_configs",
        )

        self._thread_pool = QThreadPool.globalInstance()
        default_threaded_loading = "0" if os.name == "nt" else "1"
        self._threaded_loading_enabled = os.environ.get("IMGVWR_THREADED_LOAD", default_threaded_loading) == "1"
        self._gl_initialized = False
        self._initial_open_scheduled = False
        self._loading = False
        self._last_mouse_pos = QPoint()
        self._image_path: Path | None = None
        self._pending_initial_path: Path | None = None
        self._exposure_stops = 2.0
        self._file_info = FileInfo()
        self._active_loader_signals: _ImageLoadSignals | None = None

        self._overlay_label = QLabel("", self)
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._overlay_label.setStyleSheet("color: white; background: transparent;")
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
        self._ocio_manager.reload()
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

    def paintGL(self) -> None:
        """Delegates frame rendering to the OpenGL renderer."""

        self._renderer.render(self._camera.state)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keeps overlay label in sync with widget geometry."""

        super().resizeEvent(event)
        self._overlay_label.setGeometry(self.rect())

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        """Shows right-click context menu with file and color transform actions."""

        menu = QMenu(self)

        open_action = QAction("Open file…", self)
        open_action.triggered.connect(self._open_file_dialog)
        menu.addAction(open_action)

        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.reload_current)
        reload_action.setEnabled(self._image_path is not None)
        menu.addAction(reload_action)

        view_menu = menu.addMenu("View transform")
        if view_menu is not None:
            for display_view in self._ocio_manager.display_views:
                label = f"{display_view.display} / {display_view.view}"
                action = QAction(label, self)
                action.setCheckable(True)
                action.setChecked(display_view == self._ocio_manager.active_view)
                action.triggered.connect(self._make_view_setter(display_view))
                view_menu.addAction(action)

        menu.addSeparator()
        info_label = (
            f"{self._file_info.width}x{self._file_info.height}, "
            f"channels: {self._file_info.channels}, dtype: {self._file_info.dtype_name}"
        )
        info_action = QAction(
            info_label,
            self,
        )
        info_action.setEnabled(False)
        menu.addAction(info_action)

        if event is not None:
            menu.exec(event.globalPos())

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Stores start position for drag-based camera rotation."""

        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Rotates camera while dragging with left mouse button."""

        if event is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()
        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos

        viewport_width = max(self.width(), 1)
        viewport_height = max(self.height(), 1)
        tan_half_fov = math.tan(math.radians(self._camera.state.fov_degrees) * 0.5)
        aspect = viewport_width / viewport_height

        yaw_radians_per_pixel = (2.0 * aspect * tan_half_fov) / viewport_width
        pitch_radians_per_pixel = (2.0 * tan_half_fov) / viewport_height

        yaw_delta = float(delta.x()) * yaw_radians_per_pixel
        pitch_delta = float(delta.y()) * pitch_radians_per_pixel

        self._camera.rotate_radians(yaw_delta, pitch_delta)
        self.update()

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
        """Handles exposure hotkeys using comma/period."""

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

        super().keyPressEvent(event)

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

    def open_path(self, path: Path) -> None:
        """Starts background image loading and updates UI state."""

        if self._loading:
            return

        self._loading = True
        self._set_overlay_text("Loading…")
        self.update()

        if not self._threaded_loading_enabled:
            self._load_path_sync(path)
            return

        signals = _ImageLoadSignals()
        signals.loaded.connect(self._on_image_loaded)
        signals.failed.connect(self._on_image_load_failed)
        self._active_loader_signals = signals

        task = _ImageLoadTask(path, signals)
        if self._thread_pool is not None:
            self._thread_pool.start(task)
        else:
            self._on_image_load_failed("Thread pool unavailable.")

    def _load_path_sync(self, path: Path) -> None:
        """Loads image on UI thread as a safe fallback for unstable runtimes."""

        try:
            image = load_image(path)
        except Exception as error:
            self._on_image_load_failed(str(error))
            return
        self._on_image_loaded(image)

    def reload_current(self) -> None:
        """Reloads current image and OCIO config."""

        self._ocio_manager.reload()
        self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
        if self._image_path is not None:
            self.open_path(self._image_path)
        else:
            self.update()

    def _open_file_dialog(self) -> None:
        """Opens file picker for supported image types."""

        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.exr *.hdr *.jpg *.jpeg)",
        )
        if selected_path:
            self.open_path(Path(selected_path))

    def _make_view_setter(self, display_view: DisplayView) -> Callable[[], None]:
        """Creates closure that applies a selected display/view transform."""

        def _setter() -> None:
            self._ocio_manager.set_active_view(display_view.display, display_view.view)
            self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
            self.update()

        return _setter

    def _on_image_loaded(self, payload: object) -> None:
        """Handles loaded image payload and uploads texture on UI thread."""

        image = payload
        if not isinstance(image, ImageData):
            self._on_image_load_failed("Unexpected loader payload type.")
            return

        self.makeCurrent()
        self._renderer.set_image(image)
        self.doneCurrent()

        self._image_path = image.source_path
        self._file_info = FileInfo(
            width=image.width,
            height=image.height,
            channels=image.channels,
            dtype_name=image.dtype_name,
        )

        self._loading = False
        self._active_loader_signals = None
        self._set_overlay_text("")
        parent_window = self.window()
        if parent_window is not None:
            parent_window.setWindowTitle(f"imgvwr - {image.source_path.name}")
        self.update()

    def _on_image_load_failed(self, message: str) -> None:
        """Shows loader error as overlay text."""

        self._loading = False
        self._active_loader_signals = None
        self._set_overlay_text(f"Load failed: {message}")
        self.update()

    def _schedule_initial_open_if_ready(self) -> None:
        """Queues initial file opening after GL initialization and first show."""

        if not self._gl_initialized or self._initial_open_scheduled:
            return
        if self._pending_initial_path is None:
            return

        pending_path = self._pending_initial_path
        self._pending_initial_path = None
        self._initial_open_scheduled = True
        QTimer.singleShot(0, lambda: self.open_path(pending_path))

    def _set_overlay_text(self, text: str) -> None:
        """Updates the minimal centered overlay text."""

        self._overlay_label.setText(text)
        self._overlay_label.setVisible(bool(text))


class ViewerWindow(QMainWindow):
    """Main application window wrapper for the OpenGL viewer widget."""

    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("imgvwr")
        self.resize(1280, 720)

        self._widget = HdriViewerWidget(self)
        self.setCentralWidget(self._widget)
        self._widget.set_initial_path(initial_path)
