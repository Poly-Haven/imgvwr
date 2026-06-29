//! Versioned, JSON-backed user preferences (see plans/rewrite.md §13).
//!
//! Stored at `%APPDATA%\imgvwr\preferences.json`. Saves are atomic (write to a
//! temp file, then rename). Any load/parse failure falls back to defaults.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

const PREFS_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PreferredView {
    pub display: String,
    pub view: String,
}

/// Saved outer position and inner size of the main window (physical pixels).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AppPreferences {
    /// Schema version, for forward-compatible migrations.
    #[serde(default)]
    pub version: u32,
    /// Keyed by lowercased file extension including the dot (e.g. ".exr").
    #[serde(default)]
    pub preferred_view_by_filetype: HashMap<String, PreferredView>,
    /// Last window position/size, restored on launch.
    #[serde(default)]
    pub window: Option<WindowGeometry>,
    /// Monitor to open on by default (winit monitor name). `None` = remember the
    /// last-used position (restore `window`).
    #[serde(default)]
    pub startup_monitor: Option<String>,
    /// Window corner radius in physical pixels (0 = square corners).
    #[serde(default = "default_corner_radius")]
    pub corner_radius: u32,
    /// Background colour (sRGB 0–255) shown behind transparent images.
    #[serde(default = "default_background_color")]
    pub background_color: [u8; 3],
    /// The view transform the T key toggles to (from Standard), and the default
    /// applied to HDRIs on load. A view name matched case-insensitively against
    /// the active OCIO display's views (e.g. "Filmic", "AgX", "ACES 2.0").
    #[serde(default = "default_view_transform")]
    pub default_view_transform: String,
    /// Auto-pick a starting exposure for HDR panoramas on load.
    #[serde(default = "default_true")]
    pub auto_exposure: bool,
    /// Auto-pick a starting exposure for RAW photos on load. Off by default: a
    /// RAW's scene-linear develop already respects the actual photo exposure
    /// (white = 1.0), so exposure 0 is faithful. Opt in to brighten dark shots.
    #[serde(default)]
    pub raw_auto_exposure: bool,
    /// Colour (sRGB 0–255) of guide lines.
    #[serde(default = "default_guide_color")]
    pub guide_color: [u8; 3],
    /// Clipping-overlay margin: a channel counts as clipped when its original
    /// value is within this fraction of the format max (1.0). E.g. 0.005 ≈ within
    /// ~1 code of 255 at 8-bit. Configured in Settings; see the C-key overlay.
    #[serde(default = "default_clip_margin")]
    pub clip_margin: f32,
    /// Store 32-bit-float images as 16-bit half on the GPU, roughly halving their
    /// VRAM use (a small precision trade-off). Off by default; see Settings.
    #[serde(default)]
    pub half_float_textures: bool,
    /// Internal (not user-facing): unix-seconds of the last successful update
    /// check, so the daily check throttles. `0` = never checked.
    #[serde(default)]
    pub last_update_check: i64,
    /// Internal: the latest release tag the update check found (e.g. "v1.2.0"),
    /// cached so the Settings dialog can show an "update available" link within
    /// the daily window without re-hitting the network. Empty = none/unknown.
    #[serde(default)]
    pub latest_known_version: String,
}

fn default_corner_radius() -> u32 {
    6
}

fn default_background_color() -> [u8; 3] {
    // 25% grey — a neutral mid-dark backdrop (was near-black [5, 5, 5]).
    [64, 64, 64]
}

fn default_guide_color() -> [u8; 3] {
    [255, 80, 80]
}

fn default_clip_margin() -> f32 {
    0.005 // ≈ within ~1 code of 255 at 8-bit
}

fn default_view_transform() -> String {
    "Filmic".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for AppPreferences {
    fn default() -> Self {
        Self {
            version: PREFS_VERSION,
            preferred_view_by_filetype: HashMap::new(),
            window: None,
            startup_monitor: None,
            corner_radius: default_corner_radius(),
            background_color: default_background_color(),
            default_view_transform: default_view_transform(),
            auto_exposure: true,
            raw_auto_exposure: false,
            guide_color: default_guide_color(),
            clip_margin: default_clip_margin(),
            half_float_textures: false,
            last_update_check: 0,
            latest_known_version: String::new(),
        }
    }
}

impl AppPreferences {
    /// Load preferences, returning defaults on any error or version mismatch.
    pub fn load() -> Self {
        let Some(path) = prefs_path() else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match serde_json::from_str::<AppPreferences>(&text) {
            Ok(prefs) => Self::migrate(prefs),
            Err(e) => {
                log::warn!("failed to parse preferences ({e}); using defaults");
                Self::default()
            }
        }
    }

    /// Apply forward-compatible migrations. For v1, unknown versions fall back to
    /// defaults (newer fields still parse via `#[serde(default)]`).
    fn migrate(prefs: AppPreferences) -> Self {
        if prefs.version == PREFS_VERSION {
            prefs
        } else {
            log::info!(
                "preferences version {} != {PREFS_VERSION}; using defaults",
                prefs.version
            );
            Self::default()
        }
    }

    /// Atomically persist the preferences. Errors are logged, never fatal.
    pub fn save(&self) {
        let Some(path) = prefs_path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                log::warn!("could not create prefs dir: {e}");
                return;
            }
        }
        let json = match serde_json::to_string_pretty(self) {
            Ok(j) => j,
            Err(e) => {
                log::warn!("could not serialise preferences: {e}");
                return;
            }
        };
        let tmp = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, json) {
            log::warn!("could not write preferences: {e}");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            log::warn!("could not finalise preferences: {e}");
        }
    }

    pub fn preferred_view(&self, ext: &str) -> Option<&PreferredView> {
        self.preferred_view_by_filetype.get(&normalise_ext(ext))
    }

    pub fn set_preferred_view(&mut self, ext: &str, view: PreferredView) {
        self.preferred_view_by_filetype
            .insert(normalise_ext(ext), view);
    }
}

/// Lower-case the extension and ensure a leading dot.
pub fn normalise_ext(ext: &str) -> String {
    let lower = ext.trim_start_matches('.').to_ascii_lowercase();
    format!(".{lower}")
}

fn prefs_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("imgvwr").join("preferences.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_current_version() {
        assert_eq!(AppPreferences::default().version, PREFS_VERSION);
    }

    #[test]
    fn round_trips_through_json() {
        let mut prefs = AppPreferences::default();
        prefs.set_preferred_view(
            "EXR",
            PreferredView {
                display: "sRGB".into(),
                view: "AgX".into(),
            },
        );
        let json = serde_json::to_string(&prefs).unwrap();
        let back: AppPreferences = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.preferred_view(".exr"),
            Some(&PreferredView {
                display: "sRGB".into(),
                view: "AgX".into()
            })
        );
    }

    #[test]
    fn ext_is_normalised() {
        let mut prefs = AppPreferences::default();
        prefs.set_preferred_view(
            ".HDR",
            PreferredView {
                display: "d".into(),
                view: "v".into(),
            },
        );
        assert!(prefs.preferred_view("hdr").is_some());
        assert!(prefs.preferred_view(".hdr").is_some());
    }

    #[test]
    fn unknown_version_falls_back_to_default() {
        let json = r#"{"version": 999, "preferred_view_by_filetype": {".exr": {"display":"d","view":"v"}}}"#;
        let parsed: AppPreferences = serde_json::from_str(json).unwrap();
        let migrated = AppPreferences::migrate(parsed);
        assert_eq!(migrated.version, PREFS_VERSION);
        assert!(migrated.preferred_view_by_filetype.is_empty());
    }

    #[test]
    fn missing_fields_use_defaults() {
        // An empty object should parse via serde(default).
        let parsed: AppPreferences = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.version, 0); // default for u32 is 0; migrate() handles it
        assert!(parsed.preferred_view_by_filetype.is_empty());
    }
}
