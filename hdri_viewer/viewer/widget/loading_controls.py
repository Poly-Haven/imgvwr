from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QTimer

from hdri_viewer.io.image_loader import ImageData, load_image

from .loading import _ImageLoadSignals, _ImageLoadTask
from .types import FileInfo


class LoadingControlsMixin:
    """Image loading, reload, and loading-state lifecycle helpers."""

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
        self._restore_preferred_view_transform(image.source_path)
        self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
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
            parent_window.setWindowTitle(image.source_path.name)
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
