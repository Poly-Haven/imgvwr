from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QPoint, QRectF, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMenu, QWidget

from hdri_viewer.color.ocio_manager import DisplayView, OcioManager
from hdri_viewer.io.image_loader import ImageData, is_supported_image_path, load_image
from hdri_viewer.viewer.camera import CameraController
from hdri_viewer.viewer.renderer import PanoramaRenderer


class _ImageLoadSignals(QObject):
    """Qt signal bridge used by background image loading tasks."""

    loaded = pyqtSignal(object)
    progress = pyqtSignal(float)
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
            image = load_image(self._path, progress_callback=self._emit_progress)
        except Exception as error:  # pragma: no cover - tested via signal path
            self._signals.failed.emit(str(error))
            return
        self._signals.loaded.emit(image)

    def _emit_progress(self, value: float) -> None:
        """Relays loading progress from loader thread to UI signal."""

        self._signals.progress.emit(value)


class _LoadingProgressBar(QWidget):
    """Minimal determinate loading bar with rounded fill."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._progress = 0.0

    def start(self) -> None:
        """Resets bar to start state."""

        self.set_progress(0.0)

    def stop(self) -> None:
        """Resets bar after loading is complete."""

        self._progress = 0.0
        self.update()

    def set_progress(self, value: float) -> None:
        """Sets progress value in the inclusive range [0, 1]."""

        bounded = max(0.0, min(1.0, value))
        if abs(self._progress - bounded) < 1e-6:
            return
        self._progress = bounded
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """Renders rounded border and moving rounded white capsule."""

        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        outer = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if outer.width() <= 0.0 or outer.height() <= 0.0:
            painter.end()
            return

        radius = outer.height() * 0.5
        painter.setPen(QColor("white"))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(outer, radius, radius)

        inner = outer.adjusted(2, 2, -2, -2)
        if inner.width() <= 0.0 or inner.height() <= 0.0:
            painter.end()
            return

        chunk_width = inner.width() * self._progress
        clipped_chunk = QRectF(inner.left(), inner.top(), chunk_width, inner.height())

        if clipped_chunk.width() > 0.0 and clipped_chunk.height() > 0.0:
            chunk_radius = clipped_chunk.height() * 0.5
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("white"))
            painter.drawRoundedRect(clipped_chunk, chunk_radius, chunk_radius)

        painter.end()


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
        if self._awaiting_first_present and self._renderer.has_texture:
            self._awaiting_first_present = False
            self._set_loading_overlay("", False)
            self._set_overlay_text("")

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keeps overlay label in sync with widget geometry."""

        super().resizeEvent(event)
        self._update_overlay_geometries()

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

            yaw_radians_per_pixel = (2.0 * aspect * tan_half_fov) / viewport_width
            pitch_radians_per_pixel = (2.0 * tan_half_fov) / viewport_height

            yaw_delta = float(delta.x()) * yaw_radians_per_pixel
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

    def open_path(self, path: Path) -> None:
        """Starts background image loading and updates UI state."""

        if self._loading:
            return

        self._loading = True
        self._load_progress_value = 0.0
        self._awaiting_first_present = False
        self._set_overlay_text("")
        self._set_loading_overlay("Opening file…", True)
        self._loading_progress_bar.set_progress(0.0)
        self.update()

        if not self._threaded_loading_enabled:
            self._load_path_sync(path)
            return

        signals = _ImageLoadSignals()
        signals.loaded.connect(self._on_image_loaded)
        signals.progress.connect(self._on_image_load_progress)
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
            image = load_image(path, progress_callback=self._on_image_load_progress)
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

        upload_progress = max(self._load_progress_value, 0.995)
        self._set_loading_overlay("Uploading to GPU… 99%", True)
        self._loading_progress_bar.set_progress(upload_progress)

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
        self._load_progress_value = upload_progress
        self._loading_progress_bar.set_progress(1.0)
        self._set_loading_overlay("Rendering frame… 100%", True)
        self._awaiting_first_present = True
        parent_window = self.window()
        if parent_window is not None:
            parent_window.setWindowTitle(f"imgvwr - {image.source_path.name}")
        self.update()

    def _on_image_load_failed(self, message: str) -> None:
        """Shows loader error as overlay text."""

        self._loading = False
        self._active_loader_signals = None
        self._load_progress_value = 0.0
        self._set_loading_overlay("", False)
        self._set_overlay_text(f"Load failed: {message}")
        self.update()

    def _on_image_load_progress(self, progress_value: float) -> None:
        """Updates loading overlay according to actual load progress."""

        bounded_progress = max(0.0, min(1.0, progress_value))
        if bounded_progress < self._load_progress_value:
            return

        self._load_progress_value = bounded_progress
        progress_percent = int(round(self._load_progress_value * 100.0))
        self._set_loading_overlay(self._build_loading_status(progress_percent), True)
        self._loading_progress_bar.set_progress(self._load_progress_value)

    def _build_loading_status(self, progress_percent: int) -> str:
        """Returns stage-specific status text for current loading progress."""

        if progress_percent < 5:
            return f"Opening file… {progress_percent}%"
        if progress_percent < 90:
            return f"Reading image data… {progress_percent}%"
        if progress_percent < 95:
            return f"Converting channels… {progress_percent}%"
        if progress_percent < 99:
            return f"Preparing pixel buffer… {progress_percent}%"
        return f"Finalizing image… {progress_percent}%"

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

    def _set_loading_overlay(self, status_text: str, visible: bool) -> None:
        """Shows or hides centered loading status and progress bar."""

        was_visible = self._loading_overlay.isVisible()
        self._loading_status_label.setText(status_text)
        self._loading_overlay.setVisible(visible)
        if visible and not was_visible:
            self._loading_progress_bar.start()
        elif not visible and was_visible:
            self._loading_progress_bar.stop()

    def _update_overlay_geometries(self) -> None:
        """Updates overlay widget positions relative to the current viewport."""

        self._overlay_label.setGeometry(self.rect())

        overlay_width = 260
        overlay_height = 46
        x = max((self.width() - overlay_width) // 2, 0)
        y = max((self.height() - overlay_height) // 2, 0)
        self._loading_overlay.setGeometry(x, y, overlay_width, overlay_height)
        self._loading_status_label.setGeometry(0, 0, overlay_width, 20)
        self._loading_progress_bar.setGeometry(0, 28, overlay_width, 12)


class ViewerWindow(QMainWindow):
    """Main application window wrapper for the OpenGL viewer widget."""

    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("imgvwr")
        self.resize(1280, 720)

        self._widget = HdriViewerWidget(self)
        self.setCentralWidget(self._widget)
        self._widget.set_initial_path(initial_path)
