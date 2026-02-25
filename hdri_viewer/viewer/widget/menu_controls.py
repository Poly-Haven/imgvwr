from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtGui import QAction, QContextMenuEvent
from PyQt6.QtWidgets import QFileDialog, QMenu


class MenuControlsMixin:
    """Context-menu and view-transform selection helpers."""

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
            active_display = self._ocio_manager.active_view.display
            active_view = self._ocio_manager.active_view.view

            display_menu = view_menu.addMenu("Display")
            if display_menu is not None:
                for display in self._available_displays():
                    action = QAction(display, self)
                    action.setCheckable(True)
                    action.setChecked(display == active_display)
                    action.triggered.connect(self._make_display_setter(display))
                    display_menu.addAction(action)

            display_view_menu = view_menu.addMenu("View")
            if display_view_menu is not None:
                for view in self._views_for_display(active_display):
                    action = QAction(view, self)
                    action.setCheckable(True)
                    action.setChecked(view == active_view)
                    action.triggered.connect(self._make_display_view_setter(active_display, view))
                    display_view_menu.addAction(action)

            view_menu.addSeparator()
            active_label = QAction(f"Active: {active_display} / {active_view}", self)
            active_label.setEnabled(False)
            view_menu.addAction(active_label)

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

    def _available_displays(self) -> list[str]:
        """Returns unique displays preserving config order."""

        displays: list[str] = []
        for display_view in self._ocio_manager.display_views:
            if display_view.display not in displays:
                displays.append(display_view.display)
        return displays

    def _views_for_display(self, display: str) -> list[str]:
        """Returns available views for a specific display preserving config order."""

        views: list[str] = []
        for display_view in self._ocio_manager.display_views:
            if display_view.display == display and display_view.view not in views:
                views.append(display_view.view)
        return views

    @staticmethod
    def _find_case_insensitive(items: list[str], target: str) -> str | None:
        """Returns first case-insensitive match from a list."""

        target_lower = target.lower()
        for item in items:
            if item.lower() == target_lower:
                return item
        return None

    def _apply_display_view(self, display: str, view: str, *, remember_non_standard: bool = True) -> None:
        """Applies display/view transform and refreshes rendering state."""

        self._ocio_manager.set_active_view(display, view)
        active = self._ocio_manager.active_view
        if remember_non_standard and active.view.lower() != "standard":
            self._preferred_view_by_display[active.display] = active.view
        self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
        self.update()

    def _set_display(self, display: str) -> None:
        """Switches display while preserving current view when available."""

        views = self._views_for_display(display)
        if not views:
            return

        current_view = self._ocio_manager.active_view.view
        standard_view = self._find_case_insensitive(views, "Standard")
        if current_view in views:
            target_view = current_view
        elif standard_view is not None:
            target_view = standard_view
        else:
            target_view = views[0]

        self._apply_display_view(display, target_view)

    def _toggle_standard_view(self) -> None:
        """Toggles active display between Standard and preferred alternate view."""

        active = self._ocio_manager.active_view
        display = active.display
        views = self._views_for_display(display)
        if not views:
            return

        standard_view = self._find_case_insensitive(views, "Standard")
        if standard_view is None:
            return

        if active.view.lower() == "standard":
            preferred_view = self._preferred_view_by_display.get(display)
            if preferred_view not in views or (preferred_view is not None and preferred_view.lower() == "standard"):
                preferred_view = self._find_case_insensitive(views, "Filmic")
            if preferred_view is None:
                preferred_view = next((item for item in views if item.lower() != "standard"), standard_view)
            self._apply_display_view(display, preferred_view)
            return

        self._preferred_view_by_display[display] = active.view
        self._apply_display_view(display, standard_view, remember_non_standard=False)

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

    def _make_display_setter(self, display: str) -> Callable[[], None]:
        """Creates closure that applies a selected display."""

        def _setter() -> None:
            self._set_display(display)

        return _setter

    def _make_display_view_setter(self, display: str, view: str) -> Callable[[], None]:
        """Creates closure that applies a selected view for current display."""

        def _setter() -> None:
            self._apply_display_view(display, view)

        return _setter
