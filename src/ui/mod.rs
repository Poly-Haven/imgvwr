//! egui-based overlay UI: the left-edge toolbar (and, from Commit 9, the loading
//! / error / hint overlays). Rendered on top of the OpenGL scene each frame.

mod colors;
mod overlay;

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
    /// Bottom panel slide-in progress, 0 (hidden below the edge) … 1 (fully in).
    pub bottom_slide: f32,
    pub has_image: bool,
    /// All `(display, view)` pairs from the active OCIO config.
    pub display_views: Vec<(String, String)>,
    pub active: Option<(String, String)>,
    pub ocio_available: bool,
    /// Current tone adjustments (eased target values), shown in the bottom panel.
    pub exposure: f32,
    pub gamma: f32,
    /// Clarity (local-contrast) strength and radius for the bottom-panel sliders.
    pub clarity_amount: f32,
    pub clarity_radius: f32,
    /// Image↔screen mapping for the pixel rulers (2D only; `None` hides them).
    pub ruler: Option<RulerInfo>,
    /// Left-ruler reveal progress, 0 (hidden off the left) … 1 (fully in). The
    /// bottom ruler rides the bottom panel's `bottom_slide` (drawn merged with it).
    pub left_ruler_slide: f32,

    // Overlays (Commit 9).
    pub loading: bool,
    /// Upload progress (0..1) during the GPU phase; `None` = indeterminate.
    pub progress: Option<f32>,
    pub loading_name: Option<String>,
    pub error: Option<String>,
    pub show_hint: bool,
    /// Metadata box slide-in progress, 0 (hidden off the right) … 1 (fully in).
    pub metadata_slide: f32,
    pub metadata: Vec<(String, String)>,
    /// Original channel count (1/3/4), for the metadata-box channel boxes.
    pub channel_count: u8,
    /// Currently isolated channel (0=R 1=G 2=B 3=A), `None` = all.
    pub isolate_channel: Option<u8>,
    /// Histogram of the displayed image (after exposure and the view transform),
    /// measured on the GPU. `None` until the first measurement lands, or when the
    /// driver has no compute support — the graph is then simply omitted.
    pub histogram: Option<std::sync::Arc<crate::renderer::Histogram>>,
    /// Vertical scale for that graph (the L / Sq / Log selector).
    pub histogram_scale: crate::prefs::HistogramScale,
    /// Display black/white points as `(black, white)` in the histogram's own
    /// 0..1 axis — the positions of the two handles under the graph.
    pub levels: (f32, f32),
    /// Active guides as `[coord (0..1), orientation]` (orientation ≥ 0.5 =
    /// horizontal). Listed in the metadata box with a remove button each.
    pub guides: Vec<[f32; 2]>,
    /// Loaded image dimensions (px), for the guide list's pixel readout.
    pub image_size: (u32, u32),
    pub show_help: bool,
    /// Transient bottom-right toast: `(text, alpha)`, drawn while alpha > 0.
    pub toast: Option<(String, f32)>,
    /// Per comparator slot (1..=9 → index 0..=8): `Some(tooltip_label)` when the
    /// slot holds an image (label disambiguated by path when names collide).
    pub slot_labels: [Option<String>; 9],
    /// The slot whose image is currently displayed (for flag highlighting).
    pub active_slot: Option<usize>,
    /// The slot currently being shown as a difference (Alt+N); highlighted too.
    pub diff_slot: Option<usize>,

    // Borderless custom titlebar.
    /// Titlebar slide-in progress, 0 (hidden above the edge) … 1 (fully in).
    pub titlebar_slide: f32,
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
    /// Whether HDR panoramas get an auto-exposure pick on load.
    pub auto_exposure: bool,
    /// Whether RAW photos get an auto-exposure pick on load (off by default).
    pub raw_auto_exposure: bool,
    /// Clipping-overlay margin (normalised fraction of the format max), for the
    /// settings slider.
    pub clip_margin: f32,
    /// Whether 32-bit-float images upload as 16-bit half (the VRAM-saving toggle).
    pub half_float_textures: bool,
    /// Whether the titlebar stays permanently revealed instead of auto-hiding.
    pub pin_titlebar: bool,
    /// A newer release than the running build, if known: `(version_label, url)`.
    /// Shown as a lime-green "update available" link in the Settings dialog.
    pub available_update: Option<(String, String)>,
    /// The configured Default View Transform (T-key target), for the settings
    /// dropdown.
    pub default_view_transform: String,
    /// Guide-line colour (sRGB 0–255), for the settings picker.
    pub guide_color: [u8; 3],
    /// Whether the window is maximized (drives the maximize/restore glyph).
    pub is_maximized: bool,
    /// Whether the window is fullscreen (so the help dialog hides its "Show more"
    /// → fullscreen button when there's nothing more fullscreen could reveal).
    pub is_fullscreen: bool,
    /// Resize cursor for a borderless edge under the pointer (set via egui so it
    /// resets correctly when the pointer leaves the edge).
    pub resize_cursor: Option<egui::CursorIcon>,
    /// Bottom-right navigation minimap (border + current-view box). `None` when
    /// hidden; the thumbnail itself is drawn by the GL renderer, not egui.
    pub minimap: Option<MinimapInfo>,
    /// Right-drag colour-pick tooltip data, one frame behind the cursor. `None`
    /// when not colour-picking or the cursor is off the image.
    pub color_pick: Option<ColorPickInfo>,
}

/// Live pixel-inspection readout for the right-drag colour-pick tooltip:
/// displayed pixel coords, panorama degrees, the raw ("Linear") value straight
/// from the decoded image, and the on-screen ("Display") value after the full
/// exposure / OCIO / gamma / clarity pipeline.
#[derive(Clone, Copy)]
pub struct ColorPickInfo {
    pub x: i64,
    pub y: i64,
    /// `(longitude, latitude)` degrees — panorama mode only.
    pub degrees: Option<(f32, f32)>,
    pub linear: [f32; 4],
    pub display: [f32; 4],
}

/// The bottom-right minimap overlay: where to draw the border and the
/// current-view outline, and at what fade opacity. The low-LOD image thumbnail is
/// drawn by the GL renderer (so it is tone-mapped / tiled like the main view);
/// egui only strokes the border and the view-region box on top. All geometry is
/// in egui points.
pub struct MinimapInfo {
    /// The minimap panel rectangle (points).
    pub rect: egui::Rect,
    /// Fade opacity, 0..1.
    pub alpha: f32,
    /// The current view region outline as one or more polylines (points). 2D is a
    /// rectangle (one per wrapped tile); panorama is the rectilinearly un-projected
    /// screen border, split into segments at the longitude wrap.
    pub view_segments: Vec<Vec<egui::Pos2>>,
    /// The current view region as a filled triangle list (every 3 points = one
    /// triangle), drawn under the outline as a faint shade. A projected mesh, so it
    /// fills correctly regardless of concavity (panorama poles) or wrapping.
    pub view_fill: Vec<[egui::Pos2; 3]>,
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

/// The 2D image↔screen mapping the rulers need. Screen UV (`v_uv`, GL y-up) maps
/// to image uv as `uv = 0.5 + pan + (v_uv - 0.5) * s` (y negated); image pixel =
/// `uv * size`. Resolution-independent, so the UI applies it against the egui
/// `screen_rect` (points). When `pano` is `Some`, the rulers show longitude /
/// latitude degrees via that projection instead of pixels, and the 2D fields are
/// unused (guide grab / ruler-spawn are 2D only).
#[derive(Clone, Copy)]
pub struct RulerInfo {
    pub sx: f32,
    pub sy: f32,
    pub pan_u: f32,
    pub pan_v: f32,
    pub img_w: f32,
    pub img_h: f32,
    pub pano: Option<PanoProj>,
}

/// Panorama screen→sphere projection params (mirrors the fragment shader's pano
/// branch), so the rulers can read out the longitude/latitude under each screen
/// position in degrees.
#[derive(Clone, Copy)]
pub struct PanoProj {
    pub yaw: f32,
    pub pitch: f32,
    pub tan_half_fov: f32,
    pub aspect: f32,
}

/// Transient UI state that persists across frames.
#[derive(Default)]
pub struct UiState {
    /// Updated after each egui pass: is the pointer over the bottom panel?
    pub pointer_over_panel: bool,
    /// Updated after each egui pass: is the pointer over the metadata box?
    pub pointer_over_metadata: bool,
    /// Updated after each egui pass: is the pointer over the left ruler?
    pub pointer_over_left_ruler: bool,
    /// Updated after each egui pass: is the metadata box's View menu (or one of
    /// its sub-menus) open? Keeps the box revealed while navigating the menu.
    pub view_menu_open: bool,
    /// Guide line currently under the pointer (or being dragged), so the renderer
    /// can draw it in the hover colour. Refreshed each egui pass.
    pub hovered_guide: Option<usize>,
    /// Index of the guide currently being dragged OUT of a ruler, captured at
    /// drag-start so the rest of the gesture targets that exact guide (robust to
    /// other guides being added/at the cap mid-drag). `None` = no ruler spawn-drag.
    pub guide_spawn: Option<usize>,
    /// Which levels handle is being dragged under the histogram (`false` = the
    /// black point, `true` = the white point), captured at drag-start so the rest
    /// of the gesture stays on that handle even once the pointer passes the other
    /// one. Also keeps the metadata box revealed for the duration of the drag.
    pub levels_drag: Option<bool>,
    /// Whether the H help dialog is open.
    pub show_help: bool,
    /// Whether the settings dialog is open.
    pub show_settings: bool,
    /// Whether the "set as default viewer" confirmation is showing.
    pub confirm_default: bool,
    /// Whether the Delete-key "delete this file?" confirmation is showing.
    pub confirm_delete: bool,
}

/// Actions emitted by the UI, processed by `App` after the egui pass.
pub enum UiAction {
    OpenFile,
    SetView {
        display: String,
        view: String,
    },
    /// Set the Default View Transform preference (the T-key / HDRI-load target).
    SetDefaultView(String),
    DismissError,
    CloseHelp,
    /// Confirmed on the Delete-key dialog: delete the current image file.
    DeleteCurrentFile,
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
    /// Toggle the HDR-panorama auto-exposure-on-load setting.
    SetAutoExposure(bool),
    /// Toggle the RAW-photo auto-exposure-on-load setting.
    SetRawAutoExposure(bool),
    /// Set the clipping-overlay margin (normalised fraction of the format max).
    SetClipMargin(f32),
    /// Toggle storing 32-bit-float images as 16-bit half on the GPU (less VRAM).
    SetHalfFloat(bool),
    /// Toggle keeping the titlebar permanently revealed.
    SetPinTitlebar(bool),
    /// Set the guide-line colour (sRGB 0–255).
    SetGuideColor([u8; 3]),
    /// Isolate a single channel as greyscale (`None` = show all channels).
    SetChannelIsolate(Option<u8>),
    /// Set the histogram's vertical scale (the L / Sq / Log selector).
    SetHistogramScale(crate::prefs::HistogramScale),
    /// Set the display black/white points (dragged on the histogram's handles).
    SetLevels {
        black: f32,
        white: f32,
    },
    /// Set the exposure target (EV) from the bottom-panel slider / buttons.
    SetExposure(f32),
    /// Set the gamma target from the bottom-panel slider / buttons.
    SetGamma(f32),
    /// Set the Clarity strength (0 = off) from the bottom-panel slider.
    SetClarity(f32),
    /// Set the Clarity blur radius (viewport px) from the bottom-panel slider.
    SetClarityRadius(f32),
    /// Add a guide line dragged out of a ruler: image uv coord, horizontal?
    AddGuide {
        coord: f32,
        horizontal: bool,
    },
    /// Move an existing guide (drag): set guide `index`'s coord (image uv 0..1).
    MoveGuide {
        index: usize,
        coord: f32,
    },
    /// Remove the guide at this index (metadata list ×, right-click, drag-off).
    RemoveGuide(usize),
    /// Reset all image adjustments (the bottom-panel Reset button = Ctrl+R).
    ResetAdjustments,
    /// Open the settings dialog (titlebar gear button).
    OpenSettings,
    // Borderless titlebar controls.
    /// Start an OS window move (titlebar drag).
    DragWindow,
    Minimize,
    ToggleMaximize,
    ToggleFullscreen,
    Close,
}
