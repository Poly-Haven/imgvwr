from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPaintEvent, QPainter
from PyQt6.QtWidgets import QWidget


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
