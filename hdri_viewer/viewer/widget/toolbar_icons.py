from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QStyle

try:
    import qtawesome as qta
except ImportError:  # pragma: no cover - optional dependency fallback
    qta = None


def toolbar_icon(name: str, fallback: QStyle.StandardPixmap) -> QIcon:
    """Returns toolbar icon from QtAwesome with a Qt fallback."""

    if qta is not None:
        try:
            return qta.icon(name, color="#d9d9d9")
        except Exception:
            pass

    app = QApplication.instance()
    if app is None:
        return QIcon()
    return app.style().standardIcon(fallback)
