from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMainWindow

from hdri_viewer.app_icon import application_icon
from .viewer_widget import HdriViewerWidget


class ViewerWindow(QMainWindow):
    """Main application window wrapper for the OpenGL viewer widget."""

    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("panoviewer")
        icon = application_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(170, 170)
        self.resize(1280, 720)

        self._widget = HdriViewerWidget(self)
        self.setCentralWidget(self._widget)
        self._widget.set_initial_path(initial_path)
