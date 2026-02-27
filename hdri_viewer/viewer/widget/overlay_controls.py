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

    def _set_metadata_overlay_visible(self, visible: bool) -> None:
        """Shows or hides the top-left metadata box."""

        self._metadata_overlay_visible = visible
        self._refresh_metadata_overlay()

    def _refresh_metadata_overlay(self) -> None:
        """Updates metadata box text and visibility."""

        if not self._metadata_overlay_visible:
            self._metadata_overlay_label.setVisible(False)
            return

        toolbar_visible = getattr(self, "_is_toolbar_overlay_visible", None)
        if callable(toolbar_visible) and toolbar_visible():
            self._metadata_overlay_label.setVisible(False)
            return

        metadata_text = self._format_metadata_overlay_text()
        self._metadata_overlay_label.setText(metadata_text)
        self._metadata_overlay_label.adjustSize()
        self._metadata_overlay_label.setVisible(bool(metadata_text))
        self._update_overlay_geometries()

    def _format_metadata_overlay_text(self) -> str:
        """Formats currently available image metadata for on-screen display."""

        path = self.current_path
        display_path = "-" if path is None else str(path)

        resolution_text = f"{self._file_info.width} x {self._file_info.height}"
        encoded_text = "yes" if self._file_info.input_is_encoded_srgb else "no"
        return "\n".join(
            [
                "Image Metadata",
                f"Path: {display_path}",
                f"Name: {self._file_info.source_name}",
                f"Resolution: {resolution_text}",
                f"Channels: {self._file_info.channels}",
                f"Data type: {self._file_info.dtype_name}",
                f"Compression: {self._file_info.compression_name}",
                f"Encoded sRGB input: {encoded_text}",
            ]
        )

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

        if self._metadata_overlay_label.isVisible():
            margin = 10
            label_width = min(self._metadata_overlay_label.sizeHint().width(), max(self.width() - (margin * 2), 0))
            label_height = self._metadata_overlay_label.sizeHint().height()
            self._metadata_overlay_label.setGeometry(margin, margin, max(label_width, 0), max(label_height, 0))

        update_toolbar_geometries = getattr(self, "_update_toolbar_geometries", None)
        if callable(update_toolbar_geometries):
            update_toolbar_geometries()
