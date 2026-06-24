//! egui-based overlay UI: the left-edge toolbar (and, from Commit 9, the loading
//! / error / hint overlays). Rendered on top of the OpenGL scene each frame.

mod toolbar;

pub use toolbar::build_toolbar;

/// Immutable per-frame inputs handed to the UI (gathered from `App` before the
/// mutable egui borrow, to avoid borrow conflicts).
pub struct UiInputs {
    pub toolbar_visible: bool,
    pub has_image: bool,
    pub file_info: String,
    /// All `(display, view)` pairs from the active OCIO config.
    pub display_views: Vec<(String, String)>,
    pub active: Option<(String, String)>,
    pub ocio_available: bool,
}

impl UiInputs {
    pub fn displays(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (d, _) in &self.display_views {
            if !out.contains(d) {
                out.push(d.clone());
            }
        }
        out
    }

    pub fn views_for(&self, display: &str) -> Vec<String> {
        self.display_views
            .iter()
            .filter(|(d, _)| d == display)
            .map(|(_, v)| v.clone())
            .collect()
    }
}

/// Transient UI state that persists across frames.
#[derive(Default)]
pub struct UiState {
    pub show_view_submenu: bool,
    pub show_display_submenu: bool,
    /// Display currently being browsed in the submenu (defaults to active).
    pub browse_display: Option<String>,
    /// Updated after each egui pass: is the pointer over toolbar chrome?
    pub pointer_over_panel: bool,
}

/// Actions emitted by the UI, processed by `App` after the egui pass.
pub enum UiAction {
    OpenFile,
    Reload,
    SetView { display: String, view: String },
}
