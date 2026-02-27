from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtGui import QContextMenuEvent, QCursor
from PyQt6.QtWidgets import QFileDialog, QStyle

from .toolbar_icons import toolbar_icon
from .toolbar_overlay import ToolbarButtonSpec, ToolbarColumnSpec, ToolbarOverlayWidget


@dataclass(slots=True)
class _ToolbarEntry:
    """Represents an actionable toolbar row or expandable submenu node."""

    key: str
    text: str
    icon_name: str
    fallback_icon: QStyle.StandardPixmap
    enabled: bool = True
    action: Callable[[], None] | None = None
    children_factory: Callable[[], list["_ToolbarEntry"]] | None = None


class ToolbarControlsMixin:
    """Left-edge toolbar overlay behavior replacing right-click context menu."""

    _LEFT_EDGE_ACTIVATION_PX = 28
    _SUBMENU_HOVER_SLOP_PX = 2
    _MENU_CLOSE_DELAY_MS = 100

    def _init_toolbar_overlay(self) -> None:
        """Creates toolbar overlay and initializes root column state."""

        self._toolbar_overlay = ToolbarOverlayWidget(self)
        self._toolbar_overlay.setVisible(False)
        self._toolbar_overlay.column_button_pressed.connect(self._on_toolbar_button_pressed)
        self._toolbar_overlay.column_button_hovered.connect(self._on_toolbar_button_hovered)
        self._toolbar_columns_entries: list[list[_ToolbarEntry]] = []
        self._toolbar_column_titles: list[str] = []
        self._toolbar_submenu_openers: list[str | None] = []
        self._toolbar_overlay_close_deadline_ms: int | None = None
        self._toolbar_submenu_close_deadline_by_index: dict[int, int] = {}
        self._reset_toolbar_columns()
        self._toolbar_hover_close_timer = QTimer(self)
        self._toolbar_hover_close_timer.setInterval(60)
        self._toolbar_hover_close_timer.timeout.connect(self._enforce_toolbar_close_state)

    def _update_toolbar_overlay(self) -> None:
        """Refreshes toolbar content to reflect dynamic enabled/check states."""

        self._reset_toolbar_columns()

    def _reset_toolbar_columns(self) -> None:
        """Resets toolbar to the single root column state."""

        self._toolbar_columns_entries = [self._build_root_toolbar_entries()]
        self._toolbar_column_titles = [""]
        self._toolbar_submenu_openers = [None]
        self._toolbar_submenu_close_deadline_by_index.clear()
        self._render_toolbar_columns()

    def _render_toolbar_columns(self) -> None:
        """Applies current toolbar model to overlay widgets."""

        self._toolbar_overlay.set_columns(self._to_column_specs(self._toolbar_columns_entries))
        self._update_toolbar_geometries()

    def _build_root_toolbar_entries(self) -> list[_ToolbarEntry]:
        """Builds the left-most toolbar root entries."""

        info_text = (
            f"{self._file_info.width}x{self._file_info.height}, "
            f"ch: {self._file_info.channels}, {self._file_info.dtype_name}"
        )

        return [
            _ToolbarEntry(
                key="open_file",
                text="Open file…",
                icon_name="mdi6.folder-open-outline",
                fallback_icon=QStyle.StandardPixmap.SP_DialogOpenButton,
                action=self._open_file_dialog,
            ),
            _ToolbarEntry(
                key="reload",
                text="Reload",
                icon_name="mdi6.refresh",
                fallback_icon=QStyle.StandardPixmap.SP_BrowserReload,
                enabled=self._image_path is not None,
                action=self.reload_current,
            ),
            _ToolbarEntry(
                key="view_transform",
                text="View transform",
                icon_name="mdi6.tune-variant",
                fallback_icon=QStyle.StandardPixmap.SP_FileDialogDetailedView,
                children_factory=self._build_view_transform_entries,
            ),
            _ToolbarEntry(
                key="file_info",
                text=info_text,
                icon_name="mdi6.information-outline",
                fallback_icon=QStyle.StandardPixmap.SP_MessageBoxInformation,
                enabled=False,
            ),
        ]

    def _build_view_transform_entries(self) -> list[_ToolbarEntry]:
        """Builds the second-level view-transform entries."""

        active_display = self._ocio_manager.active_view.display
        active_view = self._ocio_manager.active_view.view
        active_display_text = f"Display: {active_display}"
        active_view_text = f"View: {active_view}"

        entries: list[_ToolbarEntry] = [
            _ToolbarEntry(
                key="display_submenu",
                text=active_display_text,
                icon_name="mdi6.monitor",
                fallback_icon=QStyle.StandardPixmap.SP_DesktopIcon,
                children_factory=self._build_display_entries,
            )
        ]

        for view in self._views_for_display(active_display):
            entries.append(
                _ToolbarEntry(
                    key=f"view::{view}",
                    text=view,
                    icon_name="mdi6.eye-outline",
                    fallback_icon=QStyle.StandardPixmap.SP_FileDialogContentsView,
                    action=self._make_display_view_setter(active_display, view),
                )
            )

        entries.insert(
            1,
            _ToolbarEntry(
                key="active_view_label",
                text=active_view_text,
                icon_name="mdi6.eye-outline",
                fallback_icon=QStyle.StandardPixmap.SP_FileDialogContentsView,
                enabled=False,
            ),
        )
        return entries

    def _build_display_entries(self) -> list[_ToolbarEntry]:
        """Builds available display actions for display submenu."""

        entries: list[_ToolbarEntry] = []
        for display in self._available_displays():
            entries.append(
                _ToolbarEntry(
                    key=f"display::{display}",
                    text=display,
                    icon_name="mdi6.monitor",
                    fallback_icon=QStyle.StandardPixmap.SP_DesktopIcon,
                    action=self._make_display_setter(display),
                )
            )
        return entries

    def _to_column_specs(self, columns_entries: list[list[_ToolbarEntry]]) -> list[ToolbarColumnSpec]:
        """Converts internal toolbar entries to UI column specs."""

        specs: list[ToolbarColumnSpec] = []
        for index, entries in enumerate(columns_entries):
            title = self._toolbar_column_titles[index] if index < len(self._toolbar_column_titles) else ""
            buttons = [
                ToolbarButtonSpec(
                    key=entry.key,
                    text=entry.text,
                    icon=toolbar_icon(entry.icon_name, entry.fallback_icon),
                    enabled=entry.enabled,
                    has_children=entry.children_factory is not None,
                )
                for entry in entries
            ]
            specs.append(ToolbarColumnSpec(title=title, buttons=buttons))
        return specs

    def _on_toolbar_button_pressed(self, column_index: int, key: str) -> None:
        """Dispatches toolbar button click from a column/button key."""

        if column_index >= len(self._toolbar_columns_entries):
            return

        entries = self._toolbar_columns_entries[column_index]
        entry = next((item for item in entries if item.key == key), None)
        if entry is None or not entry.enabled:
            return

        changed = self._collapse_columns_from(column_index + 1)

        if entry.children_factory is not None:
            changed = self._open_submenu_column(column_index, entry) or changed
        elif entry.action is not None:
            entry.action()

        if changed:
            self._render_toolbar_columns()
        self.update()

    def _on_toolbar_button_hovered(self, column_index: int, key: str) -> None:
        """Expands submenu columns on hover for entries that have children."""

        if column_index >= len(self._toolbar_columns_entries):
            return

        entries = self._toolbar_columns_entries[column_index]
        entry = next((item for item in entries if item.key == key), None)
        if entry is None or not entry.enabled:
            return

        if entry.children_factory is None:
            return

        if self._open_submenu_column(column_index, entry):
            self._render_toolbar_columns()

    def _open_submenu_column(self, parent_index: int, parent_entry: _ToolbarEntry) -> bool:
        """Opens submenu for a parent entry, replacing deeper columns."""

        if parent_entry.children_factory is None:
            return False

        existing_child_index = parent_index + 1
        if existing_child_index < len(self._toolbar_submenu_openers):
            if self._toolbar_submenu_openers[existing_child_index] == parent_entry.key:
                return False

        child_entries = parent_entry.children_factory()
        if not child_entries:
            return self._collapse_columns_from(parent_index + 1)

        changed = self._collapse_columns_from(parent_index + 1)
        self._toolbar_columns_entries.append(child_entries)
        self._toolbar_column_titles.append(parent_entry.text)
        self._toolbar_submenu_openers.append(parent_entry.key)
        return True or changed

    def _collapse_columns_from(self, start_index: int) -> bool:
        """Removes columns from start index to the end."""

        if start_index <= 0:
            start_index = 1

        if start_index >= len(self._toolbar_columns_entries):
            return False

        self._toolbar_columns_entries = self._toolbar_columns_entries[:start_index]
        self._toolbar_column_titles = self._toolbar_column_titles[:start_index]
        self._toolbar_submenu_openers = self._toolbar_submenu_openers[:start_index]
        self._toolbar_submenu_close_deadline_by_index = {
            index: deadline
            for index, deadline in self._toolbar_submenu_close_deadline_by_index.items()
            if index < start_index
        }
        return True

    def _update_toolbar_visibility_from_local_pos(self, local_pos: QPoint) -> None:
        """Shows/hides toolbar based on proximity to left edge while cursor moves."""

        global_pos = self.mapToGlobal(local_pos)
        self._sync_toolbar_with_cursor(global_pos)

    def _sync_toolbar_with_cursor(self, global_pos: QPoint) -> None:
        """Single source of truth for toolbar visibility/submenu state from cursor position."""

        window = self.window()
        if window is None:
            self._hide_toolbar_overlay()
            return

        frame = window.frameGeometry()
        within_vertical_bounds = frame.top() <= global_pos.y() <= frame.bottom()
        near_left_edge = within_vertical_bounds and global_pos.x() <= frame.left() + self._LEFT_EDGE_ACTIVATION_PX

        if near_left_edge:
            self._show_toolbar_overlay()
            self._toolbar_overlay_close_deadline_ms = None

        if not self._toolbar_overlay.isVisible():
            return

        inside_toolbar = self._is_global_pos_inside_toolbar(global_pos)
        if near_left_edge or inside_toolbar:
            self._toolbar_overlay_close_deadline_ms = None
        else:
            now_ms = self._now_ms()
            if self._toolbar_overlay_close_deadline_ms is None:
                self._toolbar_overlay_close_deadline_ms = now_ms + self._MENU_CLOSE_DELAY_MS
            elif now_ms >= self._toolbar_overlay_close_deadline_ms:
                self._hide_toolbar_overlay()
                return

        self._enforce_submenu_hover_state(global_pos)

    def _update_toolbar_geometries(self) -> None:
        """Updates toolbar overlay bounds to the left side of the viewport."""

        max_width = min(max(int(self.width() * 0.8), 208), 720)
        content_width = max(self._toolbar_overlay.content_width_hint(), 192)
        overlay_width = min(max_width, content_width)
        self._toolbar_overlay.setGeometry(0, 0, overlay_width, max(self.height(), 1))

    def _show_toolbar_overlay(self) -> None:
        """Shows toolbar and starts leave-detection timer."""

        self._update_toolbar_geometries()
        if not self._toolbar_overlay.isVisible():
            self._toolbar_overlay.setVisible(True)
            refresh_metadata_overlay = getattr(self, "_refresh_metadata_overlay", None)
            if callable(refresh_metadata_overlay):
                refresh_metadata_overlay()
        if not self._toolbar_hover_close_timer.isActive():
            self._toolbar_hover_close_timer.start()

    def _hide_toolbar_overlay(self) -> None:
        """Hides toolbar and collapses any open submenu columns."""

        was_visible = self._toolbar_overlay.isVisible()
        if self._toolbar_overlay.isVisible():
            self._toolbar_overlay.setVisible(False)
        self._toolbar_overlay_close_deadline_ms = None
        self._reset_toolbar_columns()
        if was_visible:
            refresh_metadata_overlay = getattr(self, "_refresh_metadata_overlay", None)
            if callable(refresh_metadata_overlay):
                refresh_metadata_overlay()
        if self._toolbar_hover_close_timer.isActive():
            self._toolbar_hover_close_timer.stop()

    def _enforce_toolbar_close_state(self) -> None:
        """Closes toolbar reliably when cursor leaves both edge and toolbar region."""

        if not self._toolbar_overlay.isVisible() and self._toolbar_hover_close_timer.isActive():
            self._toolbar_hover_close_timer.stop()
            return

        self._sync_toolbar_with_cursor(QCursor.pos())

    def _enforce_submenu_hover_state(self, global_pos: QPoint) -> None:
        """Keeps submenu visible only while hovering opener button or submenu column."""

        if len(self._toolbar_columns_entries) <= 1:
            return

        for column_index in range(1, len(self._toolbar_columns_entries)):
            parent_index = column_index - 1
            opener_key = self._toolbar_submenu_openers[column_index]
            if opener_key is None:
                continue

            submenu_hover_valid = False

            # Keep ancestor submenus open while cursor is over this submenu OR any of its descendants.
            for descendant_index in range(column_index, len(self._toolbar_columns_entries)):
                if self._toolbar_overlay.column_contains_global_pos(
                    descendant_index,
                    global_pos,
                    slop_px=self._SUBMENU_HOVER_SLOP_PX,
                ):
                    submenu_hover_valid = True
                    break

            if not submenu_hover_valid and self._toolbar_overlay.column_contains_global_pos(
                parent_index, global_pos, slop_px=self._SUBMENU_HOVER_SLOP_PX
            ):
                if self._toolbar_overlay.button_contains_global_pos(
                    parent_index,
                    opener_key,
                    global_pos,
                    slop_px=self._SUBMENU_HOVER_SLOP_PX,
                ):
                    submenu_hover_valid = True

            if submenu_hover_valid:
                self._toolbar_submenu_close_deadline_by_index.pop(column_index, None)
                continue

            now_ms = self._now_ms()
            deadline = self._toolbar_submenu_close_deadline_by_index.get(column_index)
            if deadline is None:
                self._toolbar_submenu_close_deadline_by_index[column_index] = now_ms + self._MENU_CLOSE_DELAY_MS
                continue

            if now_ms < deadline:
                continue

            if self._collapse_columns_from(column_index):
                self._render_toolbar_columns()
            return

    @staticmethod
    def _now_ms() -> int:
        """Returns monotonic clock in milliseconds for close-delay checks."""

        return int(time.monotonic() * 1000.0)

    def _is_global_pos_inside_toolbar(self, global_pos: QPoint) -> bool:
        """Returns whether a global cursor position lies within toolbar geometry."""

        if not self._toolbar_overlay.isVisible():
            return False

        top_left = self._toolbar_overlay.mapToGlobal(QPoint(0, 0))
        local_pos = global_pos - top_left
        return self._toolbar_overlay.rect().contains(local_pos)

    def _is_pointer_over_toolbar(self) -> bool:
        """Returns whether current pointer position is inside visible toolbar overlay."""

        return self._is_global_pos_inside_toolbar(QCursor.pos())

    def _should_block_viewer_wheel_input(self) -> bool:
        """Returns whether wheel input should be ignored due to toolbar hover."""

        return self._is_pointer_over_toolbar()

    def _is_toolbar_overlay_visible(self) -> bool:
        """Returns whether the toolbar overlay is currently visible."""

        return self._toolbar_overlay.isVisible()

    def _open_file_dialog(self) -> None:
        """Opens file picker and attempts to load the selected file."""

        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "All files (*)",
        )
        if selected_path:
            self.open_path(Path(selected_path))

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        """Disables right-click context menu in favor of left-edge toolbar."""

        if event is None:
            return
        event.ignore()
