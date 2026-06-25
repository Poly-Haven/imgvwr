//! egui-based overlay UI: the left-edge toolbar (and, from Commit 9, the loading
//! / error / hint overlays). Rendered on top of the OpenGL scene each frame.

mod colors;
mod overlay;
mod toolbar;

/// Give a clickable widget's response the pointing-hand cursor on hover (egui
/// only does this for hyperlinks by default).
pub(crate) fn clickable(resp: egui::Response) -> egui::Response {
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Build the whole overlay UI for a frame: toolbar + overlays.
pub fn build(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    toolbar::build_toolbar(ctx, inputs, state, actions);
    overlay::build_overlays(ctx, inputs, state, actions);
    // Borderless edge resize cursor, applied last so it overrides widget cursors
    // at the very border. Routing it through egui (rather than winit directly)
    // means egui-winit resets it when the pointer leaves the edge.
    if let Some(c) = inputs.resize_cursor {
        ctx.set_cursor_icon(c);
    }
}

/// Immutable per-frame inputs handed to the UI (gathered from `App` before the
/// mutable egui borrow, to avoid borrow conflicts).
pub struct UiInputs {
    pub toolbar_visible: bool,
    pub has_image: bool,
    /// All `(display, view)` pairs from the active OCIO config.
    pub display_views: Vec<(String, String)>,
    pub active: Option<(String, String)>,
    pub ocio_available: bool,
    /// Current tone adjustments, shown in the sidebar.
    pub exposure: f32,
    pub gamma: f32,

    // Overlays (Commit 9).
    pub loading: bool,
    /// Upload progress (0..1) during the GPU phase; `None` = indeterminate.
    pub progress: Option<f32>,
    pub loading_name: Option<String>,
    pub error: Option<String>,
    pub show_hint: bool,
    pub show_metadata: bool,
    pub metadata: Vec<(String, String)>,
    pub show_help: bool,
    /// Transient bottom-right toast: `(text, alpha)`, drawn while alpha > 0.
    pub toast: Option<(String, f32)>,
    /// Per comparator slot (1..=9 → index 0..=8): `Some(tooltip_label)` when the
    /// slot holds an image (label disambiguated by path when names collide).
    pub slot_labels: [Option<String>; 9],
    /// The slot whose image is currently displayed (for flag highlighting).
    pub active_slot: Option<usize>,

    // Borderless custom titlebar.
    /// Eased 0..1 opacity of the auto-hiding titlebar (0 = fully hidden).
    pub titlebar_alpha: f32,
    /// Filename shown in the titlebar (empty when no image is loaded).
    pub title: String,
    /// App icon texture shown at the left of the titlebar.
    pub icon: Option<egui::TextureHandle>,
    /// Available monitors as `(winit name, friendly label)` for the settings
    /// startup-display picker.
    pub monitors: Vec<(String, String)>,
    /// The configured startup monitor name (`None` = remember last position).
    pub startup_display: Option<String>,
    /// Window corner radius in pixels (0 = square), for the settings spinner.
    pub corner_radius: u32,
    /// Background colour (sRGB 0–255) behind transparent images.
    pub background_color: [u8; 3],
    /// Whether the window is maximized (drives the maximize/restore glyph).
    pub is_maximized: bool,
    /// Resize cursor for a borderless edge under the pointer (set via egui so it
    /// resets correctly when the pointer leaves the edge).
    pub resize_cursor: Option<egui::CursorIcon>,
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
    /// Updated after each egui pass: is the pointer over the metadata box?
    pub pointer_over_metadata: bool,
    /// Whether the H help dialog is open.
    pub show_help: bool,
    /// Whether the settings dialog is open.
    pub show_settings: bool,
    /// Whether the "set as default viewer" confirmation is showing.
    pub confirm_default: bool,
}

/// Actions emitted by the UI, processed by `App` after the egui pass.
pub enum UiAction {
    OpenFile,
    Reload,
    SetView {
        display: String,
        view: String,
    },
    DismissError,
    CloseHelp,
    /// Recall the comparator slot at this index (0..=8).
    RecallSlot(usize),
    /// Register imgvwr as the default app for supported file types.
    SetDefaultApp,
    /// Set the monitor to open on by default (`None` = remember last position).
    SetStartupDisplay(Option<String>),
    /// Set the window corner radius (pixels) and re-apply it live.
    SetCornerRadius(u32),
    /// Set the background colour (sRGB 0–255) behind transparent images.
    SetBackgroundColor([u8; 3]),
    // Borderless titlebar controls.
    /// Start an OS window move (titlebar drag).
    DragWindow,
    Minimize,
    ToggleMaximize,
    ToggleFullscreen,
    Close,
}
