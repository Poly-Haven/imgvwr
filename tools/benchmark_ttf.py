from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from hdri_viewer.main import _preload_native_modules
from hdri_viewer.viewer.widget import ViewerWindow


def measure_time_to_first_frame(path: Path, timeout_seconds: float) -> float:
    _preload_native_modules()
    app = QApplication(sys.argv)
    t0 = time.perf_counter()
    window = ViewerWindow(initial_path=path)
    window.show()

    done = {"value": None}

    def tick() -> None:
        widget = window._widget
        is_ready = (
            widget.current_path is not None
            and widget._renderer.has_texture
            and not widget._loading
            and not widget._awaiting_first_present
        )
        if is_ready:
            done["value"] = time.perf_counter() - t0
            app.quit()
            return

        if (time.perf_counter() - t0) >= timeout_seconds:
            raise TimeoutError(f"Timed out after {timeout_seconds:.1f}s waiting for first frame.")

    poll_timer = QTimer()
    poll_timer.timeout.connect(tick)
    poll_timer.start(10)

    app.exec()

    if done["value"] is None:
        raise RuntimeError("Failed to measure time to first frame.")

    return float(done["value"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure panoviewer time-to-first-frame for a single image file.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    seconds = measure_time_to_first_frame(args.image, timeout_seconds=args.timeout)
    print(f"TTF_SECONDS={seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
