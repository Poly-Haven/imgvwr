from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(slots=True)
class ToolbarButtonSpec:
    """Describes one toolbar button in a sidebar column."""

    key: str
    text: str
    icon: object
    enabled: bool = True
    has_children: bool = False


@dataclass(slots=True)
class ToolbarColumnSpec:
    """Describes one sidebar column including heading and buttons."""

    title: str
    buttons: list[ToolbarButtonSpec]


class _ToolbarButton(QToolButton):
    """Tool button emitting a hover signal with its logical key."""

    hovered = pyqtSignal(str)

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = key
        self.setMouseTracking(True)

    def enterEvent(self, event: QEvent | None) -> None:
        """Emits hovered key when cursor enters button."""

        self.hovered.emit(self._key)
        super().enterEvent(event)


class _ToolbarColumn(QFrame):
    """Single vertically scrollable sidebar column."""

    button_pressed = pyqtSignal(str)
    button_hovered = pyqtSignal(str)

    def __init__(self, spec: ToolbarColumnSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 170);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._buttons_by_key: dict[str, _ToolbarButton] = {}

        if spec.title:
            title_label = QLabel(spec.title, self)
            title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            title_label.setStyleSheet("color: white;" "font-weight: 600;" "background: transparent;" "padding: 8px;")
            layout.addWidget(title_label)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")

        content = QWidget(scroll_area)
        content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        for button_spec in spec.buttons:
            button = _ToolbarButton(button_spec.key, content)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setAutoRaise(True)
            button.setEnabled(button_spec.enabled)
            button.setIcon(button_spec.icon)
            label = f"{button_spec.text} ›" if button_spec.has_children else button_spec.text
            button.setText(label)
            button.setStyleSheet(
                "QToolButton {"
                "color: white;"
                "text-align: left;"
                "padding: 6px 8px;"
                "background: transparent;"
                "border: none;"
                "}"
                "QToolButton:hover { color: rgba(255, 255, 255, 220); }"
                "QToolButton:pressed { color: rgba(255, 255, 255, 180); }"
                "QToolButton:disabled { color: rgba(220, 220, 220, 120); }"
            )
            button.clicked.connect(lambda checked=False, key=button_spec.key: self.button_pressed.emit(key))
            button.hovered.connect(self.button_hovered.emit)
            content_layout.addWidget(button)
            self._buttons_by_key[button_spec.key] = button

        content_layout.addStretch(1)
        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)

    def global_rect(self) -> QRect:
        """Returns column geometry in global coordinates."""

        top_left = self.mapToGlobal(QPoint(0, 0))
        return QRect(top_left, self.size())

    def button_global_rect(self, key: str) -> QRect | None:
        """Returns global geometry for a button key, when present."""

        button = self._buttons_by_key.get(key)
        if button is None:
            return None
        top_left = button.mapToGlobal(QPoint(0, 0))
        return QRect(top_left, button.size())

    def preferred_width(self) -> int:
        """Returns natural width for this column based on its content."""

        self.layout().activate()
        return max(self.sizeHint().width(), 140)


class ToolbarOverlayWidget(QFrame):
    """Left-edge overlay that hosts one or more scrollable sidebar columns."""

    column_button_pressed = pyqtSignal(int, str)
    column_button_hovered = pyqtSignal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setMouseTracking(True)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._columns_root = QWidget(self)
        self._columns_root.setStyleSheet("background: transparent;")
        self._columns_layout = QHBoxLayout(self._columns_root)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        self._columns_layout.setSpacing(0)

        root_layout.addWidget(self._columns_root)

        self._columns: list[_ToolbarColumn] = []

    def set_columns(self, columns: list[ToolbarColumnSpec]) -> None:
        """Rebuilds all columns from the given specs."""

        while self._columns_layout.count() > 0:
            item = self._columns_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(False)
                widget.setParent(None)
                widget.deleteLater()

        self._columns.clear()
        total_width = 0
        max_height = 0
        for column_index, column_spec in enumerate(columns):
            column = _ToolbarColumn(column_spec, self._columns_root)
            column_width = min(column.preferred_width(), 520)
            column.setFixedWidth(column_width)
            total_width += column_width
            max_height = max(max_height, column.sizeHint().height())
            column.button_pressed.connect(lambda key, idx=column_index: self.column_button_pressed.emit(idx, key))
            column.button_hovered.connect(lambda key, idx=column_index: self.column_button_hovered.emit(idx, key))
            self._columns_layout.insertWidget(column_index, column)
            self._columns.append(column)

        if columns:
            self._columns_root.setFixedSize(total_width, max(max_height, self.height()))
        else:
            self._columns_root.setFixedSize(0, max(self.height(), 1))

        self._columns_layout.activate()
        self._columns_root.updateGeometry()

    def content_width_hint(self) -> int:
        """Returns preferred overlay content width based on visible columns."""

        if not self._columns:
            return 0
        return sum(column.maximumWidth() for column in self._columns)

    def column_contains_global_pos(self, column_index: int, global_pos: QPoint, slop_px: int = 0) -> bool:
        """Returns whether cursor global position lies within given column."""

        if column_index < 0 or column_index >= len(self._columns):
            return False
        rect = self._columns[column_index].global_rect().adjusted(-slop_px, -slop_px, slop_px, slop_px)
        return rect.contains(global_pos)

    def button_contains_global_pos(
        self, column_index: int, key: str, global_pos: QPoint, slop_px: int = 0
    ) -> bool:
        """Returns whether cursor global position lies within given button key."""

        if column_index < 0 or column_index >= len(self._columns):
            return False
        rect = self._columns[column_index].button_global_rect(key)
        if rect is None:
            return False
        return rect.adjusted(-slop_px, -slop_px, slop_px, slop_px).contains(global_pos)
