from __future__ import annotations

from pathlib import Path

from hdri_viewer.color.ocio_manager import DisplayView, OcioManager


def test_choose_default_view_prefers_filmic_terms() -> None:
    views = [
        DisplayView("sRGB", "Raw"),
        DisplayView("Main", "ACES 1.0 SDR-video"),
    ]
    chosen = OcioManager._choose_default_view(views)
    assert chosen == views[1]


def test_custom_config_path_has_priority(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    custom = resources / "ocio_configs"
    resources.mkdir(parents=True)
    custom.mkdir(parents=True)

    bundled = resources / "config.ocio"
    bundled.write_text("bundled", encoding="utf-8")

    custom_cfg = custom / "user.ocio"
    custom_cfg.write_text("custom", encoding="utf-8")

    manager = OcioManager(resources_dir=resources, custom_config_dir=custom)
    resolved = manager._resolve_config_path()
    assert resolved == custom_cfg


def test_fallback_to_bundled_config_path(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    custom = resources / "ocio_configs"
    resources.mkdir(parents=True)
    custom.mkdir(parents=True)

    bundled = resources / "config.ocio"
    bundled.write_text("bundled", encoding="utf-8")

    manager = OcioManager(resources_dir=resources, custom_config_dir=custom)
    resolved = manager._resolve_config_path()
    assert resolved == bundled
