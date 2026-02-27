from __future__ import annotations

from typing import Callable


class MenuControlsMixin:
    """View-transform and OCIO display/view selection helpers."""

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
        self._persist_active_view_transform(active.display, active.view)
        if remember_non_standard and active.view.lower() != "standard":
            self._preferred_view_by_display[active.display] = active.view
        self._renderer.update_ocio_shader(self._ocio_manager.build_gpu_shader())
        update_toolbar_overlay = getattr(self, "_update_toolbar_overlay", None)
        if callable(update_toolbar_overlay):
            update_toolbar_overlay()
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
