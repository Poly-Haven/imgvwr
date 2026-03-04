from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from hdri_viewer.app_icon import application_icon
from hdri_viewer.viewer.widget import ViewerWindow


def _env_flag(name: str, default: str) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raw_value = default
    return raw_value == "1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses CLI arguments for optional initial image path."""

    parser = argparse.ArgumentParser(description="panoviewer - GPU HDRI viewer")
    parser.add_argument("path", nargs="?", help="Optional image path to open")
    return parser.parse_args(argv)


def _preload_native_modules() -> None:
    """Preloads native extension modules on the main thread.

    This avoids late import of OCIO/OIIO while Qt/OpenGL initialization is in progress,
    which can cause unstable startup behavior on some Windows environments.
    """

    ocio_preload_enabled = _env_flag("PANOVIEWER_PRELOAD_OCIO", "0")

    ocio_enabled = not _env_flag("PANOVIEWER_DISABLE_OCIO", "0")
    if ocio_preload_enabled and ocio_enabled:
        try:
            import PyOpenColorIO  # noqa: F401
        except ImportError:
            pass

    try:
        from PIL import Image  # noqa: F401
        from PIL import ImageCms  # noqa: F401
    except ImportError:
        pass

    use_subprocess_loader = _env_flag("PANOVIEWER_USE_SUBPROCESS_LOADER", "1")
    if not (os.name == "nt" and use_subprocess_loader):
        try:
            import OpenImageIO  # noqa: F401
        except ImportError:
            pass


def main() -> int:
    """Application entrypoint."""

    args = _parse_args(sys.argv[1:])
    initial_path = Path(args.path) if args.path else None

    _preload_native_modules()

    app = QApplication(sys.argv)
    icon = application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = ViewerWindow(initial_path=initial_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
