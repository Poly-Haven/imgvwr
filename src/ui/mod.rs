//! egui-based overlay UI: the left-edge toolbar (and, from Commit 9, the loading
//! / error / hint overlays). Rendered on top of the OpenGL scene each frame.

mod overlay;
mod toolbar;

/// Build the whole overlay UI for a frame: toolbar + overlays.
pub fn build(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    toolbar::build_toolbar(ctx, inputs, state, actions);
    overlay::build_overlays(ctx, inputs, actions);
}

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

    // Overlays (Commit 9).
    pub loading: bool,
    pub loading_name: Option<String>,
    pub error: Option<String>,
    pub show_hint: bool,
    pub show_metadata: bool,
    pub metadata: Vec<(String, String)>,
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
    DismissError,
}
