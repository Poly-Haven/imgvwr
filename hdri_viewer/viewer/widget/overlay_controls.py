from __future__ import annotations

from PyQt6.QtGui import QResizeEvent


class OverlayControlsMixin:
    """Overlay/status label and loading progress layout helpers."""

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keeps overlay label in sync with widget geometry."""

        super().resizeEvent(event)
        self._update_overlay_geometries()

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
