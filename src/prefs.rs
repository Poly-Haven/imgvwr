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
}

impl Default for AppPreferences {
    fn default() -> Self {
        Self {
            version: PREFS_VERSION,
            preferred_view_by_filetype: HashMap::new(),
            window: None,
            startup_monitor: None,
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
