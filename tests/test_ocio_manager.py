from __future__ import annotations

from pathlib import Path

from hdri_viewer.color.ocio_manager import DisplayView, OcioManager


def test_choose_default_view_prefers_standard_srgb() -> None:
    views = [
        DisplayView("Rec.1886", "Filmic"),
        DisplayView("sRGB", "Standard"),
        DisplayView("sRGB", "Raw"),
    ]
    chosen = OcioManager._choose_default_view(views)
    assert chosen == views[1]


def test_choose_default_view_falls_back_to_standard_then_srgb() -> None:
    standard_only = [
        DisplayView("Display P3", "AgX"),
        DisplayView("Rec.1886", "Standard"),
    ]
    chosen_standard = OcioManager._choose_default_view(standard_only)
    assert chosen_standard == standard_only[1]

    srgb_only = [
        DisplayView("Display P3", "AgX"),
        DisplayView("sRGB", "Filmic"),
    ]
    chosen_srgb = OcioManager._choose_default_view(srgb_only)
    assert chosen_srgb == srgb_only[1]


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
