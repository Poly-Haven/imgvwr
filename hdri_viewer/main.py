from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from hdri_viewer.viewer.widget import ViewerWindow


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses CLI arguments for optional initial image path."""

    parser = argparse.ArgumentParser(description="imgvwr - GPU HDRI viewer")
    parser.add_argument("path", nargs="?", help="Optional image path to open")
    return parser.parse_args(argv)


def _preload_native_modules() -> None:
    """Preloads native extension modules on the main thread.

    This avoids late import of OCIO/OIIO while Qt/OpenGL initialization is in progress,
    which can cause unstable startup behavior on some Windows environments.
    """

    ocio_enabled = os.environ.get("IMGVWR_DISABLE_OCIO", "0") != "1"
    if os.name == "nt" and os.environ.get("IMGVWR_ENABLE_OCIO", "0") != "1":
        ocio_enabled = False

    if ocio_enabled:
        try:
            import PyOpenColorIO  # noqa: F401
        except ImportError:
            pass

    use_subprocess_loader = os.environ.get("IMGVWR_USE_SUBPROCESS_LOADER", "1") == "1"
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
    window = ViewerWindow(initial_path=initial_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
