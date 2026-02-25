from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from hdri_viewer.io.image_loader import load_image


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
