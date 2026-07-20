//! Application struct and winit event loop.
//!
//! Brings up the window, the glutin OpenGL 4.3 core context, and a glow
//! context; loads images on a background thread; handles mouse/keyboard input;
//! and renders each frame.
//!
//! ## Headless framebuffer capture
//!
//! Because `imgvwr.exe` is a dev binary (no Start-menu registration), the
//! computer-use screenshot tool cannot see its window contents. To verify
//! rendering headlessly we read the GL back buffer with `glReadPixels` and write
//! a PNG. Driven by env vars:
//!   * `IMGVWR_CAPTURE`           – output PNG path (enables capture)
//!   * `IMGVWR_CAPTURE_DELAY_MS`  – minimum wait before capturing.
//!
//! Interactive input cannot be exercised headlessly (the dev window is invisible
//! to the input tool), so a set of `IMGVWR_DEBUG_*` env overrides apply a camera
//! / exposure state right after load so the visual outcome can be captured:
//!   * `IMGVWR_DEBUG_EXPOSURE` / `_GAMMA`
//!   * `IMGVWR_DEBUG_YAW` / `_PITCH` / `_FOV` (panorama, degrees)
//!   * `IMGVWR_DEBUG_ZOOM` (2D)
//!   * `IMGVWR_DEBUG_PROJECTION` = `pano` | `flat`
//!   * `IMGVWR_DEBUG_WRAP` = `1`
//!   * `IMGVWR_DEBUG_SLOT` = `1` (pin the loaded image into comparator slot 1)
//!   * `IMGVWR_DEBUG_CURSOR` = `x,y` (physical px; also needed by `_COLOR_PICK`)
//!   * `IMGVWR_DEBUG_COLOR_PICK` = `1` (force the colour-pick tooltip on, as if
//!     right-drag-held at `_CURSOR`)
//!   * `IMGVWR_DEBUG_GUIDE_CMD` = comma-separated `g`|`shift`|`ctrl` (replays a
//!     sequence of G / Shift+G / Ctrl+G presses; applied after `_PROJECTION`)
//!   * `IMGVWR_DEBUG_CLIPBOARD_COPY` = `1` (as if Ctrl+C were pressed)
//!   * `IMGVWR_DEBUG_DELETE_CONFIRM` = `1` (show the Delete-key confirm dialog)
//!   * `IMGVWR_DEBUG_PIN_TITLEBAR` = `1` (force the titlebar permanently shown,
//!     in-memory only — never writes the real preferences file)

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use glam::Vec2;
use glow::HasContext as _;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext,
    Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::monitor::MonitorHandle;
use winit::window::{CursorGrabMode, Fullscreen, ResizeDirection, Window, WindowId, WindowLevel};

use crate::camera::{Camera, CameraController};
use crate::image_loader::{
    can_be_panorama, equirect_content_scores, is_equirectangular, is_supported, load_image,
    probe_dimensions, supported_extensions, ImageData,
};
use crate::ocio::OcioManager;
use crate::prefs::{AppPreferences, PreferredView};
use crate::renderer::{RenderParams, Renderer};
use crate::ui::{self, UiAction, UiInputs, UiState};
use crate::UserEvent;

/// Metadata for the currently-loaded image (toolbar + overlay).
#[derive(Clone, Default)]
struct FileInfo {
    name: String,
    width: u32,
    height: u32,
    channels: u8,
    dtype: String,
    compression: String,
    panorama: bool,
    /// Camera EXIF (RAW files only); `None` for everything else.
    camera: Option<crate::image_loader::CameraMeta>,
}

/// Hard cap on how long a capture run waits for a load before grabbing anyway.
const CAPTURE_LOAD_CAP: Duration = Duration::from_secs(120);

/// Max interval between two left-clicks to count as a double-click.
const DOUBLE_CLICK: Duration = Duration::from_millis(350);

/// Toast: full opacity for `TOAST_HOLD`, then fades over `TOAST_FADE`.
const TOAST_HOLD: Duration = Duration::from_millis(1400);
const TOAST_FADE: Duration = Duration::from_millis(600);

/// Cursor travel (device px) during a double-click that suppresses the
/// fullscreen toggle (so a double-click-and-drag is treated as a drag).
const DBLCLICK_DRAG_TOL: f32 = 6.0;

/// Navigation minimap geometry (egui points): longest side and edge margin.
const MINIMAP_MAX: f32 = 200.0;
const MINIMAP_MARGIN: f32 = 16.0;
/// Samples per window edge / grid cells per axis for the panorama minimap view
/// region. Shared by the outline and the fill so the filled mesh boundary lands
/// exactly on the outline (no triangular gap between a coarse fill and finer line).
const PANO_VIEW_SAMPLES: usize = 40;
/// Auto-shown minimap (2D pan/zoom): stays fully visible for `MINIMAP_HOLD` after
/// the last view change, then fades over `MINIMAP_FADE`.
const MINIMAP_HOLD: Duration = Duration::from_millis(1800);
const MINIMAP_FADE: f32 = 0.6;

/// Fullscreen only: hide the mouse cursor after this long with no real movement.
const CURSOR_IDLE_HIDE: Duration = Duration::from_millis(2500);

/// Hold the loading bar back this long after a load starts, so it doesn't flash
/// on-screen for a small image that decodes/uploads in a frame or two. A load
/// slower than this crosses the threshold and reveals the bar as normal.
const LOADING_BAR_DELAY: Duration = Duration::from_millis(200);

/// A held adjustment key's repeats coalesce into one undo entry while presses
/// keep arriving within this window (longer than the OS key-repeat interval, so
/// the ramp stays one gesture; the entry commits this long after release).
const ADJUST_COALESCE: Duration = Duration::from_millis(350);

/// Minimum logical (point) size the window is grown to so the Settings dialog's
/// fixed-width controls fit without wrapping (the dialog content is ~440 pt wide
/// plus frame, scrollbar and breathing room).
const SETTINGS_MIN_LOGICAL: (f32, f32) = (540.0, 720.0);

/// Max decoded images kept in memory for instant navigation (current + a
/// previous + a next, plus one spare so back-and-forth stays cached). Each entry
/// can be several GB for 24k+ images, so this is deliberately small.
const IMAGE_CACHE_CAP: usize = 4;

/// The window auto-sizes to frame the image but never grows past this fraction
/// of the monitor in either dimension (the rest is breathing room / taskbar).
const FILL_FRACTION: f32 = 0.9;

/// Target average linear value for HDR-panorama auto-exposure on load.
const AUTO_EXPOSURE_TARGET: f32 = 0.732;

/// Time constant (seconds) for easing exposure / gamma toward their targets.
const TONE_EASE_TAU: f32 = 0.045;

/// Duration (seconds) of the auto-hiding panels' slide in/out.
const SLIDE_SECS: f32 = 0.15;

/// Move `current` linearly toward `target` so a full 0→1 traversal takes
/// `SLIDE_SECS`. Used to slide the auto-hiding panels in/out.
fn approach(current: f32, target: f32, dt: f32) -> f32 {
    let step = dt / SLIDE_SECS;
    if target > current {
        (current + step).min(target)
    } else {
        (current - step).max(target)
    }
}

/// Smallest window dimension (physical px); matches `with_min_inner_size`.
const MIN_DIM: u32 = 170;

/// Time constant (seconds) for easing the window geometry toward its follow
/// target. Snappy so the window settles quickly (then the loop returns to Wait).
const WINDOW_EASE_TAU: f32 = 0.0225;

/// Result of a background decode, tagged with its generation id.
struct LoadResult {
    gen: u64,
    result: Result<ImageData, String>,
}

/// Result of a background preload (arrow-key look-ahead).
struct PreloadResult {
    gen: u64,
    result: Result<ImageData, String>,
}

/// A transient bottom-right HUD message.
struct Toast {
    text: String,
    born: Instant,
}

/// Background backdrop presets, cycled by the `B` key. `UserSetting` uses the
/// configured `background_color`; the others override it for the session only
/// (not persisted, not part of undo).
#[derive(Clone, Copy, PartialEq, Eq)]
enum BgPreset {
    UserSetting,
    Black,
    Checker,
    White,
}

impl BgPreset {
    fn next(self) -> Self {
        match self {
            BgPreset::UserSetting => BgPreset::Black,
            BgPreset::Black => BgPreset::Checker,
            BgPreset::Checker => BgPreset::White,
            BgPreset::White => BgPreset::UserSetting,
        }
    }

    fn label(self) -> &'static str {
        match self {
            BgPreset::UserSetting => "Default",
            BgPreset::Black => "Black",
            BgPreset::Checker => "Checkerboard",
            BgPreset::White => "White",
        }
    }

    /// The solid clear colour (sRGB 0–1) and whether to draw the checkerboard.
    /// `user` is the configured `background_color` for the `UserSetting` preset.
    fn resolve(self, user: [u8; 3]) -> ([f32; 3], bool) {
        match self {
            BgPreset::UserSetting => (srgb_u8_to_f32(user), false),
            BgPreset::Black => ([0.0, 0.0, 0.0], false),
            BgPreset::Checker => (srgb_u8_to_f32(user), true),
            BgPreset::White => ([1.0, 1.0, 1.0], false),
        }
    }
}

/// An image whose GPU upload is in progress; the view state is applied once the
/// incremental upload completes (`finalize_adopt`).
struct PendingAdopt {
    data: Arc<ImageData>,
    for_compare: bool,
    /// Pre-swap 2D `(zoom, height)` for native-scale matching on a slot recall.
    old_scale: Option<(f32, f32)>,
}

/// Colour-pick data computable without a GPU readback (see
/// `App::color_pick_partial`); combined with the readback's "Display" value in
/// `App::render` to build a frame's [`crate::ui::ColorPickInfo`].
struct ColorPickPartial {
    x: i64,
    y: i64,
    degrees: Option<(f32, f32)>,
    linear: [f32; 4],
}

/// Playback state for an animated GIF. The frames themselves live in the
/// `current_image`'s [`ImageData::animation`]; this just tracks which frame is
/// showing, when to advance, and whether the user paused (Space).
struct AnimState {
    /// Index of the frame currently uploaded to the texture.
    frame: usize,
    /// When the current frame should give way to the next.
    next_at: Instant,
    paused: bool,
}

/// Coarse load status (the error string is surfaced in the UI from Commit 9).
enum LoadState {
    Idle,
    Loading,
    Loaded,
    Failed(String),
}

/// Live graphics state, created once the event loop is `Resumed`.
struct Gfx {
    gl: Arc<glow::Context>,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    window: Window,
    renderer: Renderer,
    egui: egui_glow::EguiGlow,
}

/// Headless framebuffer-capture request (see module docs).
struct Capture {
    path: PathBuf,
    delay: Duration,
    start: Instant,
    done: bool,
}

enum RenderOutcome {
    Idle,
    Drew,
    Captured,
}

/// Max entries kept on the undo (and redo) stack.
const UNDO_LIMIT: usize = 256;

/// A snapshot of the undoable editing state: guides, image adjustments and toggle
/// modes. Navigation / positioning (pan, zoom, look, projection) and the per-image
/// rotation are deliberately excluded.
#[derive(Clone, PartialEq)]
struct UndoState {
    guides: Vec<[f32; 2]>,
    guides_visible: bool,
    exposure_target: f32,
    gamma_target: f32,
    clarity_amount: f32,
    clarity_radius: f32,
    isolate_channel: Option<u8>,
    sharpness: bool,
    clip_overlay: bool,
    wrap_2d: bool,
    nearest_filter: bool,
    nearest_auto: bool,
    image_stretch: Vec2,
}

impl UndoState {
    /// The "no edits" baseline a freshly-loaded image starts from.
    fn fresh() -> Self {
        Self {
            guides: Vec::new(),
            guides_visible: true,
            exposure_target: 0.0,
            gamma_target: 1.0,
            clarity_amount: 0.0,
            clarity_radius: 64.0,
            isolate_channel: None,
            sharpness: false,
            clip_overlay: false,
            wrap_2d: false,
            nearest_filter: false,
            nearest_auto: true,
            image_stretch: Vec2::ONE,
        }
    }
}

/// Everything that changes the *displayed* pixel values, and so the histogram.
/// Compared verbatim each frame to decide whether to re-measure.
///
/// The levels adjustment is deliberately absent: the graph is what the levels
/// handles are aimed at, so it must not shift underneath them as they move.
/// Clarity is absent too — it is a screen-space pass whose result depends on the
/// viewport, not on the image (the histogram pass skips it, as the minimap does).
#[derive(Clone, Copy, PartialEq)]
struct HistogramKey {
    epoch: u64,
    exposure: f32,
    gamma: f32,
}

/// The navigation minimap panel rectangle in physical pixels (top-left origin),
/// plus the window scale factor so the same rect can be expressed in egui points.
#[derive(Clone, Copy)]
struct MinimapMetrics {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    scale: f32,
}

pub struct App {
    proxy: EventLoopProxy<UserEvent>,
    initial_path: Option<PathBuf>,
    gfx: Option<Gfx>,
    capture: Option<Capture>,
    /// Ctrl+C: copy the composited render to the clipboard on the next frame
    /// (needs a GPU readback of the just-drawn scene, so it can't happen inline
    /// in the key handler — see `render`).
    clipboard_copy_pending: bool,

    // Background loading.
    load_tx: Sender<LoadResult>,
    load_rx: Receiver<LoadResult>,
    load_gen: u64,
    load_state: LoadState,
    /// When the current load began, for the load-to-ready timing log (§17.1).
    load_start: Instant,

    // View state.
    camera: CameraController,
    /// Rendered (eased) exposure / gamma. The `*_target` values are what the
    /// user dialed; the rendered values chase them so adjustments animate.
    exposure: f32,
    gamma: f32,
    exposure_target: f32,
    gamma_target: f32,
    wrap_2d: bool,
    /// Manual nearest-neighbour filtering choice (the value the I key last set),
    /// used only while `nearest_auto` is false.
    nearest_filter: bool,
    /// When true (the default), the sampling filter is chosen automatically: a 2D
    /// image magnified past 200% samples nearest (crisp pixels), everything else
    /// bilinear. The I key turns this off and pins `nearest_filter`.
    nearest_auto: bool,
    /// Isolated channel shown as greyscale (0=R 1=G 2=B 3=A); `None` = all.
    isolate_channel: Option<u8>,
    /// Clarity (local-contrast) strength (0 = off) and blur radius (viewport px).
    clarity_amount: f32,
    clarity_radius: f32,
    /// Per-axis image squash/stretch (Alt+middle-drag); 1,1 = none.
    image_stretch: Vec2,
    /// Active Alt+middle-drag squash/stretch gesture.
    stretching: bool,
    /// Sharpness checker (S): show the original-resolution high-pass.
    sharpness: bool,
    /// Clipping overlay (C): animated diagonal stripes over regions whose
    /// *original* (pre-adjustment) per-channel value is within the configured
    /// margin of the format max. In [`UndoState`] like the other review toggles.
    clip_overlay: bool,
    /// Monotonic clock start, feeds the clipping overlay's stripe animation (a
    /// continuous wall-clock time uniform; independent of per-image load timing).
    app_epoch: Instant,
    /// The clipping overlay's max-mip mask needs (re)building — set when the image
    /// or the clip margin changes, consumed in `render` while the overlay is on.
    clip_mask_dirty: bool,
    /// Latest display histogram for the F2 box, measured on the GPU off the live
    /// display pipeline (there is no CPU-side display transform to reproduce it
    /// with). `None` until the first measurement lands, or on a driver with no
    /// compute support.
    histogram: Option<Arc<crate::renderer::Histogram>>,
    /// The display state `histogram` describes, so a change can be spotted.
    histogram_key: Option<HistogramKey>,
    /// Bumped by anything *other than* the eased tone values that changes the
    /// displayed pixels: a new image, a new animation frame, a different OCIO
    /// view. Cheaper than re-deriving all of that into [`HistogramKey`].
    histogram_epoch: u64,
    /// While a held adjustment key (exposure / gamma / clarity) is auto-repeating,
    /// this is pushed a little past each repeat. `undo_gesture_active` treats the
    /// window as a gesture so the whole ramp coalesces into ONE undo entry instead
    /// of flooding the stack with a step per repeat.
    adjust_repeat_until: Option<Instant>,
    /// Settings-dialog open state last frame, to detect open/close transitions.
    settings_was_open: bool,
    /// Window inner size to restore when the Settings dialog closes (the dialog
    /// grows a too-small window so its controls fit; see `sync_settings_window`).
    settings_restore_size: Option<PhysicalSize<u32>>,
    /// Guide lines: `[image_coord (0..1), 0=vertical / 1=horizontal]`
    /// (max [`crate::renderer::MAX_GUIDES`]).
    guides: Vec<[f32; 2]>,
    /// Whether existing guides are shown/hittable (G toggles this; adding one
    /// always turns it back on). The guides themselves are untouched — this only
    /// gates rendering, the F2 list, and hit-testing (see `guide_at_cursor`).
    guides_visible: bool,
    show_metadata: bool,
    /// Display rotation of the current 2D image in 90° clockwise quarter-turns
    /// (0–3), applied in the shader. Persisted per image path for the session in
    /// [`image_rotations`](Self::image_rotations); not reset by R / Ctrl+R.
    rotation: u8,
    /// Per-image-path display rotation, remembered for the session so reopening a
    /// rotated image (or stepping back to it) restores the rotation.
    image_rotations: HashMap<PathBuf, u8>,
    /// Debug A/B override: when set (via `IMGVWR_DEBUG_NO_LANCZOS` in debug
    /// builds), forces bilinear minification even for 8-bit images so the Lanczos
    /// path can be compared. Always false in release.
    debug_no_lanczos: bool,
    /// Background backdrop preset (B key); session-only, defaults to the
    /// configured `background_color`.
    bg_preset: BgPreset,
    /// Undo / redo of editing state (guides, adjustments, toggle modes). The
    /// baseline is the last-committed snapshot; a change away from it (outside a
    /// gesture) pushes the baseline onto `undo_stack`. Cleared on each image load.
    undo_stack: Vec<UndoState>,
    redo_stack: Vec<UndoState>,
    undo_baseline: UndoState,
    /// Navigation minimap toggled on with M (persists until toggled off). When
    /// off, the minimap can still appear automatically on 2D pan/zoom.
    minimap_on: bool,
    /// Deadline for the auto-shown minimap (set on each 2D view change): full
    /// opacity until then, fading over `MINIMAP_FADE` after. `None` = no auto-show.
    minimap_auto_until: Option<Instant>,
    /// Last view signature `(mode, a, b, c)` seen by `render`, to detect a view
    /// change that should auto-show the minimap. `mode` is the projection (so a
    /// P-toggle isn't read as a pan); the floats are `(pan.x, pan.y, zoom)` in 2D
    /// or `(yaw, pitch, fov)` in panorama.
    minimap_prev_view: Option<(u8, f32, f32, f32)>,
    /// True while the left button is dragging inside the minimap (navigating).
    minimap_drag: bool,
    /// Headless-test override (debug only): force the minimap fade alpha to a
    /// fixed value, so the cross-fade composite can be verified in a capture.
    #[cfg(debug_assertions)]
    debug_minimap_alpha: Option<f32>,

    // Colour management.
    ocio: OcioManager,
    prefs: AppPreferences,

    // UI.
    ui_state: UiState,
    bottom_visible: bool,
    bottom_hide_deadline: Option<Instant>,
    /// The left ruler reveals near the left edge only (independent of the bottom
    /// panel), and stays up while hovered so a guide can be dragged off it.
    left_ruler_visible: bool,
    left_ruler_hide_deadline: Option<Instant>,
    file_info: FileInfo,
    loaded_path: Option<PathBuf>,
    /// File name of the in-flight / last-attempted load (for overlays).
    pending_name: Option<String>,
    /// Headless-test override: force the bottom panel visible (IMGVWR_DEBUG_BOTTOM).
    force_bottom: bool,
    /// Headless-test override: force an overlay ("loading"/"error"/"hint").
    force_overlay: Option<String>,

    // Input state.
    modifiers: ModifiersState,
    dragging: bool,
    /// Index of the guide currently being dragged with the left button (grabbed
    /// directly on the image). `None` = not dragging a guide.
    guide_drag: Option<usize>,
    cursor_pos: PhysicalPosition<f64>,
    /// Cursor position when a pan/look drag began, restored on release (a
    /// confined/locked grab otherwise drops the cursor at the window centre).
    drag_start_cursor: PhysicalPosition<f64>,
    last_left_press: Option<Instant>,
    fullscreen: bool,
    /// Keep the window above all others (toggled with A).
    always_on_top: bool,
    /// True once the user manually resizes the window: the window then stops
    /// auto-following the zoom so their chosen size sticks (cleared on the next
    /// image load, which re-frames the window).
    manual_window: bool,
    /// While `Some` and in the future, incoming `Resized` events are treated as
    /// our own programmatic resizes (not a manual drag). Set right after every
    /// `request_inner_size`.
    suppress_manual_until: Option<Instant>,
    /// A double-click is pending; the fullscreen toggle fires on release unless
    /// the cursor moved more than `DBLCLICK_DRAG_TOL` (i.e. it was a drag).
    pending_dblclick: bool,
    dblclick_motion: f32,
    /// Guide captured at a right-press (if the cursor was over one), deleted on
    /// release only if the press stayed under `DBLCLICK_DRAG_TOL` — past that it
    /// becomes a colour-pick drag instead (see `color_picking`) and the guide
    /// survives.
    right_press_guide: Option<usize>,
    /// Screen position (physical px) at a right-press; `None` when the right
    /// button isn't held. Compared against the current cursor each move to detect
    /// the colour-pick drag threshold.
    right_press_pos: Option<PhysicalPosition<f64>>,
    /// True once a right-press has dragged past the threshold: the colour-pick
    /// tooltip is showing and the auto-hiding toolbars are suppressed so they
    /// can't cover the pixel under inspection.
    color_picking: bool,
    /// The colour-pick tooltip's data, one frame behind the cursor — the display
    /// value needs a GPU readback of the previous frame's finished scene (see
    /// `render`). `None` whenever `color_picking` is false or the cursor is off
    /// the image.
    color_pick_last: Option<crate::ui::ColorPickInfo>,

    // Borderless window interaction.
    /// True while the cursor is inside the window, driving the titlebar reveal.
    cursor_in_window: bool,
    /// True once the user has physically moved the mouse (`DeviceEvent::MouseMotion`)
    /// since the last window geometry change. Cleared on `Resized`/`Moved`: a
    /// window-follow resize slides an edge under a stationary cursor (winit still
    /// reports `CursorMoved`), and that must NOT pop the edge panels / minimap —
    /// only a real mouse move reveals them. An already-open panel stays open.
    cursor_moved_by_user: bool,
    /// Wall-clock of the last real mouse move (`DeviceEvent::MouseMotion`), or the
    /// last cursor-enter / fullscreen entry. Drives the fullscreen idle cursor
    /// auto-hide: navigation emits no motion, so the cursor stays hidden until the
    /// user actually moves it.
    last_cursor_motion: Option<Instant>,
    /// Whether the fullscreen idle auto-hide currently has the OS cursor hidden.
    cursor_idle_hidden: bool,
    /// Set on leaving fullscreen; the next (window-restore) Resized re-frames the
    /// window to the *current* image and resets the 2D fit zoom (after navigating
    /// in fullscreen the restored window size is stale).
    refit_windowed_pending: bool,
    /// Slide-in progress 0..1 for the auto-hiding panels (0 = tucked off the
    /// window edge, 1 = fully in). Ramped over `SLIDE_SECS` toward each panel's
    /// visibility so they slide rather than pop.
    titlebar_slide: f32,
    metadata_slide: f32,
    bottom_slide: f32,
    left_ruler_slide: f32,
    /// Active Alt+right-drag resize: the edge(s) being dragged. Resized manually
    /// (not via the OS loop, which is left-button only) so it ends on release.
    alt_resize: Option<ResizeDirection>,
    /// Window outer rect `(x, y, w, h)` at the start of an Alt-resize.
    alt_resize_origin: (i32, i32, u32, u32),
    /// Cursor *screen* position at the start of an Alt-resize. The resize tracks
    /// the cursor's screen movement (DPI/ballistics-accurate), not raw device
    /// motion (which doesn't match the visible cursor 1:1).
    alt_resize_press: (f64, f64),
    /// A left-press landed in a window-move zone (Alt anywhere, or a 2D-fit
    /// body): it becomes an OS move on the first motion, or a click on release.
    window_drag_armed: bool,
    /// Pointer travel since `window_drag_armed`, to tell a click from a drag.
    window_drag_motion: f32,
    /// Set by the titlebar Close button; honoured in `about_to_wait`.
    should_exit: bool,
    /// The app icon as an egui texture, shown in the titlebar (loaded once).
    titlebar_icon: Option<egui::TextureHandle>,
    /// Target outer (position, size) the window geometry eases toward, so a
    /// window-follow resize animates smoothly instead of snapping. `None` =
    /// settled.
    window_anim_target: Option<(PhysicalPosition<i32>, PhysicalSize<u32>)>,

    // Folder navigation, look-ahead preload, and view lock.
    /// L lock: carry zoom/pan/exposure to the next image and skip auto-resize.
    locked: bool,
    /// Last arrow-key direction (+1 next, -1 prev), used to pick the preload.
    nav_dir: i32,
    /// Set once the user starts arrow-navigating, so neighbours get preloaded.
    preload_armed: bool,
    preload_tx: Sender<PreloadResult>,
    preload_rx: Receiver<PreloadResult>,
    preload_gen: u64,
    /// Recently-seen decoded images (current + previous + next look-ahead), most-
    /// recent first, so back-and-forth navigation is instant. Capped at
    /// [`IMAGE_CACHE_CAP`]; for very large images this holds several GB. Shared
    /// (`Arc`) so comparator slots can pin an image without copying it.
    image_cache: Vec<Arc<ImageData>>,
    /// Net offset (from the displayed image) of an in-flight folder-navigation
    /// load: `Some(+1)` while loading the next image, `Some(-1)` the previous,
    /// `None` when settled. While a nav load is in flight, pressing the same arrow
    /// again is ignored (no queueing) and pressing the opposite arrow cancels it
    /// and stays on the current image (so RL == stay, RRRRL == RL).
    nav_pending: Option<i32>,

    /// Transient bottom-right status toast.
    toast: Option<Toast>,

    // Update check (Settings dialog).
    update_tx: Sender<Option<crate::update::LatestRelease>>,
    update_rx: Receiver<Option<crate::update::LatestRelease>>,
    /// True while a background update check is in flight (so we don't spawn more).
    update_checking: bool,
    /// A newer release than the running build, if one is known (from the daily
    /// check or its cached result): `(version_label, release_url)`. Shown as a
    /// lime-green link in the Settings dialog.
    available_update: Option<(String, String)>,

    // Image comparator.
    /// The currently-displayed decoded image (shared so a slot can pin it).
    current_image: Option<Arc<ImageData>>,
    /// GIF playback state, `Some` only while the current image is an animated GIF
    /// (the frames live in `current_image`'s `ImageData::animation`).
    anim: Option<AnimState>,
    /// Comparator slots (Ctrl+1..=9 → index 0..=8); each pins a decoded image.
    slots: [Option<Arc<ImageData>>; 9],
    /// The image shown before the last slot recall, for the A/B toggle-back.
    compare_prev: Option<Arc<ImageData>>,
    /// Slot whose image is currently displayed (drives the active flag).
    active_slot: Option<usize>,
    /// Slot being shown as a difference against the current image (Alt+N).
    diff_slot: Option<usize>,

    // F2 metadata box hover-reveal (near the top-right corner).
    metadata_hover: bool,
    metadata_hide_deadline: Option<Instant>,
    /// Keeps the metadata box revealed for a grace period while/after its View
    /// menu is open, so moving into the (popup) menu doesn't dismiss the box and
    /// close the menu out from under the cursor.
    metadata_menu_grace: Option<Instant>,

    /// In-progress adoption waiting on the incremental GPU upload.
    pending: Option<PendingAdopt>,
    /// Upload fraction (0..1) while `pending`, for the progress bar.
    upload_progress: f32,
    /// File-read progress for the in-flight decode, shared with the load thread,
    /// so the loading bar is determinate while reading (e.g. off a network drive)
    /// for the formats that stream through a counting reader.
    decode_progress: Arc<crate::image_loader::ReadProgress>,

    /// Timestamp of the previous rendered frame, for frame-rate-independent
    /// easing (`None` until the first frame).
    last_frame: Option<Instant>,
    /// Timestamp of the last window-ease step (the ease is driven by the Resized
    /// event chain, vsync-gated, so it times itself rather than using the render
    /// dt). `None` between animations.
    last_window_ease: Option<Instant>,
    /// True while the camera is easing toward its target, so `about_to_wait`
    /// keeps scheduling frames (and stops, returning to `Wait`, once settled).
    animating: bool,
}

impl App {
    pub fn new(initial_path: Option<PathBuf>, proxy: EventLoopProxy<UserEvent>) -> Self {
        let capture = std::env::var_os("IMGVWR_CAPTURE").map(|path| {
            let delay = std::env::var("IMGVWR_CAPTURE_DELAY_MS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_millis)
                .unwrap_or_default();
            Capture {
                path: PathBuf::from(path),
                delay,
                start: Instant::now(),
                done: false,
            }
        });

        let (load_tx, load_rx) = std::sync::mpsc::channel();
        let (preload_tx, preload_rx) = std::sync::mpsc::channel();
        let (update_tx, update_rx) = std::sync::mpsc::channel();
        let ocio = OcioManager::new(resolve_resources_dir());
        let prefs = AppPreferences::load();

        Self {
            proxy,
            initial_path,
            gfx: None,
            capture,
            clipboard_copy_pending: false,
            load_tx,
            load_rx,
            load_gen: 0,
            load_state: LoadState::Idle,
            load_start: Instant::now(),
            camera: CameraController::for_image(false),
            exposure: 0.0,
            gamma: 1.0,
            exposure_target: 0.0,
            gamma_target: 1.0,
            wrap_2d: false,
            nearest_filter: false,
            nearest_auto: true,
            isolate_channel: None,
            clarity_amount: 0.0,
            clarity_radius: 64.0,
            image_stretch: Vec2::ONE,
            stretching: false,
            sharpness: false,
            clip_overlay: false,
            app_epoch: Instant::now(),
            clip_mask_dirty: true,
            histogram: None,
            histogram_key: None,
            histogram_epoch: 0,
            adjust_repeat_until: None,
            settings_was_open: false,
            settings_restore_size: None,
            guides: Vec::new(),
            guides_visible: true,
            show_metadata: false,
            rotation: 0,
            image_rotations: HashMap::new(),
            debug_no_lanczos: false,
            bg_preset: BgPreset::UserSetting,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            undo_baseline: UndoState::fresh(),
            minimap_on: false,
            minimap_auto_until: None,
            minimap_prev_view: None,
            minimap_drag: false,
            #[cfg(debug_assertions)]
            debug_minimap_alpha: None,
            ocio,
            prefs,
            ui_state: UiState::default(),
            bottom_visible: false,
            bottom_hide_deadline: None,
            left_ruler_visible: false,
            left_ruler_hide_deadline: None,
            file_info: FileInfo::default(),
            loaded_path: None,
            pending_name: None,
            // The IMGVWR_DEBUG_* overrides force internal state for headless
            // testing; they are dev-only and ignored in release builds.
            force_bottom: cfg!(debug_assertions)
                && std::env::var_os("IMGVWR_DEBUG_BOTTOM").is_some(),
            force_overlay: if cfg!(debug_assertions) {
                std::env::var("IMGVWR_DEBUG_OVERLAY").ok()
            } else {
                None
            },
            modifiers: ModifiersState::empty(),
            dragging: false,
            guide_drag: None,
            // Start far from the left edge so the toolbar stays hidden until the
            // cursor actually moves there (and so headless captures are clean).
            cursor_pos: PhysicalPosition::new(1.0e6, 1.0e6),
            drag_start_cursor: PhysicalPosition::new(0.0, 0.0),
            last_left_press: None,
            fullscreen: false,
            always_on_top: false,
            manual_window: false,
            suppress_manual_until: None,
            pending_dblclick: false,
            dblclick_motion: 0.0,
            right_press_guide: None,
            right_press_pos: None,
            color_picking: false,
            color_pick_last: None,
            locked: false,
            nav_dir: 1,
            preload_armed: false,
            preload_tx,
            preload_rx,
            preload_gen: 0,
            image_cache: Vec::new(),
            nav_pending: None,
            toast: None,
            update_tx,
            update_rx,
            update_checking: false,
            available_update: None,
            current_image: None,
            anim: None,
            slots: std::array::from_fn(|_| None),
            compare_prev: None,
            active_slot: None,
            diff_slot: None,
            metadata_hover: false,
            metadata_hide_deadline: None,
            metadata_menu_grace: None,
            pending: None,
            upload_progress: 0.0,
            decode_progress: Arc::new(crate::image_loader::ReadProgress::default()),
            last_frame: None,
            last_window_ease: None,
            animating: false,
            cursor_in_window: false,
            cursor_moved_by_user: false,
            last_cursor_motion: None,
            cursor_idle_hidden: false,
            refit_windowed_pending: false,
            titlebar_slide: 0.0,
            metadata_slide: 0.0,
            bottom_slide: 0.0,
            left_ruler_slide: 0.0,
            alt_resize: None,
            alt_resize_origin: (0, 0, 0, 0),
            alt_resize_press: (0.0, 0.0),
            window_drag_armed: false,
            window_drag_motion: 0.0,
            should_exit: false,
            titlebar_icon: None,
            window_anim_target: None,
        }
    }

    /// Build the OCIO program for the active display/view and upload it.
    fn rebuild_ocio(&mut self) {
        let shader = self.ocio.build_gpu_shader();
        if let Some(gfx) = &mut self.gfx {
            gfx.renderer.set_ocio_shader(&shader);
        }
        // A different view transform maps the same pixels to different display
        // values, so whatever the histogram last measured no longer applies.
        self.invalidate_histogram();
        self.request_redraw();
    }

    /// Mark the display histogram stale, so the next frame re-measures it (only
    /// if the F2 box is actually on screen — nothing is computed while it is
    /// hidden). The previous graph stays visible until the new one lands, which
    /// keeps it from flickering during an exposure ramp.
    fn invalidate_histogram(&mut self) {
        self.histogram_epoch = self.histogram_epoch.wrapping_add(1);
    }

    /// T: toggle between the active view and the display's Standard/Raw view.
    fn toggle_view_transform(&mut self) {
        let Some(active) = self.ocio.active().cloned() else {
            return;
        };
        let views = self.ocio.views_for(&active.display);
        let standard = views
            .iter()
            .find(|v| v.eq_ignore_ascii_case("standard"))
            .or_else(|| views.iter().find(|v| v.eq_ignore_ascii_case("raw")))
            .cloned();
        let Some(standard) = standard else {
            log::info!("T: no Standard/Raw view available for {}", active.display);
            return;
        };

        let target = if active.view.eq_ignore_ascii_case(&standard) {
            // Standard → the configured Default View Transform (Settings), if it
            // exists on this display and isn't Standard itself; else the first
            // non-Standard view.
            let default = self.prefs.default_view_transform.clone();
            views
                .iter()
                .find(|v| v.eq_ignore_ascii_case(&default) && !v.eq_ignore_ascii_case(&standard))
                .or_else(|| views.iter().find(|v| !v.eq_ignore_ascii_case(&standard)))
                .cloned()
        } else {
            Some(standard)
        };

        if let Some(view) = target {
            if self.ocio.set_active(&active.display, &view) {
                log::info!("view transform -> {}/{}", active.display, view);
                self.rebuild_ocio();
                self.persist_view_if_panorama();
                self.show_toast(view.clone());
            }
        }
    }

    /// Pick the OCIO view for a freshly-loaded image (§13). A saved per-extension
    /// view (panoramas) wins; otherwise HDRIs (2:1 EXR/HDR) default to Filmic and
    /// everything else to Standard.
    fn select_view_for_load(&mut self, panorama: bool, path: &Path) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let applied = if panorama {
            self.prefs
                .preferred_view(&ext)
                .cloned()
                .map(|pv| self.ocio.set_active(&pv.display, &pv.view))
                .unwrap_or(false)
        } else {
            false
        };
        if !applied {
            let is_hdri = panorama && matches!(ext.as_str(), "exr" | "hdr" | "pic");
            let default_view = self.prefs.default_view_transform.clone();
            if !(is_hdri && self.select_view_named(&default_view)) {
                self.select_standard_view();
            }
        }
        self.rebuild_ocio();
    }

    /// Select a view by name (case-insensitive), preferring the current display.
    /// Returns true if a matching view was found and applied.
    fn select_view_named(&mut self, name: &str) -> bool {
        let current_display = self.ocio.active().map(|dv| dv.display.clone());
        let pairs: Vec<(String, String)> = self
            .ocio
            .display_views()
            .iter()
            .map(|dv| (dv.display.clone(), dv.view.clone()))
            .collect();

        if let Some(display) = &current_display {
            if let Some((d, v)) = pairs
                .iter()
                .find(|(d, v)| d == display && v.eq_ignore_ascii_case(name))
            {
                self.ocio.set_active(d, v);
                return true;
            }
        }
        if let Some((d, v)) = pairs.iter().find(|(_, v)| v.eq_ignore_ascii_case(name)) {
            self.ocio.set_active(d, v);
            return true;
        }
        false
    }

    /// Select a "Standard" view (preferring the current display), falling back
    /// to "Raw" if there is no Standard.
    fn select_standard_view(&mut self) {
        if !self.select_view_named("standard") {
            self.select_view_named("raw");
        }
    }

    /// Persist the active view for the current image's extension (panoramas only).
    fn persist_view_if_panorama(&mut self) {
        if !self.file_info.panorama {
            return;
        }
        let (Some(path), Some(active)) = (&self.loaded_path, self.ocio.active()) else {
            return;
        };
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            return;
        };
        self.prefs.set_preferred_view(
            ext,
            PreferredView {
                display: active.display.clone(),
                view: active.view.clone(),
            },
        );
        self.prefs.save();
        log::info!(
            "saved preferred view {}/{} for .{}",
            active.display,
            active.view,
            ext.to_ascii_lowercase()
        );
    }

    fn create_gfx(&mut self, event_loop: &ActiveEventLoop) -> Result<Gfx> {
        // No winit window icon: it takes a single bitmap and Windows scales it
        // badly for both the title bar and taskbar. We instead set crisp
        // native-size icons from the multi-resolution .ico after creation
        // (`set_window_icons`), backed by the icon embedded in the .exe.
        let mut window_attributes = Window::default_attributes()
            .with_title("imgvwr")
            // Borderless: the image *is* the window. Move/resize/close are
            // provided by the custom titlebar, body-drag, and edge hit-zones.
            // (DWM drop-shadow / Aero-snap polish is a follow-up.)
            .with_decorations(false)
            // Created hidden to avoid a startup flash: an immediately-visible
            // window would show the OS default (a standard-size white window on the
            // primary monitor) for a few frames before our position/size attributes
            // and the first GL clear land. `resumed` renders the first frame and
            // then reveals it already framed, placed, and cleared to the backdrop.
            .with_visible(false)
            .with_min_inner_size(LogicalSize::new(MIN_DIM as f64, MIN_DIM as f64));
        // Probe the initial image's dimensions from its header (cheap, no decode)
        // so the window opens already framing it — eliminating the size/position
        // jump that a post-decode resize would cause. RAW and equirectangular
        // images aren't pre-sized (no cheap probe / panoramas keep the window).
        // Pick the monitor to open on (by winit name): an explicit "Open on
        // display: X" wins; otherwise "Remember last used" reopens on the monitor
        // we were on at last exit (`last_monitor`). Fall back to the primary if the
        // chosen monitor isn't currently connected. A fresh launch always auto-sizes
        // and centres on that monitor (the previous size/position isn't restored).
        let wanted_monitor = self
            .prefs
            .startup_monitor
            .clone()
            .or_else(|| self.prefs.last_monitor.clone());
        let monitor = wanted_monitor
            .as_deref()
            .and_then(|name| {
                event_loop
                    .available_monitors()
                    .find(|m| m.name().as_deref() == Some(name))
            })
            .or_else(|| event_loop.primary_monitor());
        let scale = monitor.as_ref().map(|m| m.scale_factor()).unwrap_or(1.0);
        let probed = self
            .initial_path
            .as_ref()
            .and_then(|p| probe_dimensions(p))
            .filter(|(w, h)| !is_equirectangular(*w, *h));
        // Set both size and position at creation so the window never visibly
        // jumps into place.
        let (size, position) = startup_geometry(probed, monitor.as_ref(), scale);
        window_attributes = window_attributes.with_inner_size(size);
        if let Some(pos) = position {
            window_attributes = window_attributes.with_position(pos);
        }

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_depth_size(0)
            .with_stencil_size(0);

        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));
        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                // The scene is a fullscreen textured quad plus egui's own
                // (already-antialiased) triangles, so MSAA buys nothing and only
                // costs fill rate — pick the config with the fewest samples
                // (0 = no MSAA) rather than the most.
                configs
                    .reduce(|a, b| {
                        if b.num_samples() < a.num_samples() {
                            b
                        } else {
                            a
                        }
                    })
                    .expect("at least one GL config")
            })
            .map_err(|e| anyhow::anyhow!("failed to build GL display: {e}"))?;
        let window = window.context("winit did not return a window")?;

        let gl_display = gl_config.display();
        let raw_window_handle = window.window_handle()?.as_raw();

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 3))))
            .with_profile(GlProfile::Core)
            .build(Some(raw_window_handle));

        let not_current = unsafe { gl_display.create_context(&gl_config, &context_attributes) }
            .context("failed to create OpenGL 4.3 core context")?;

        let surface_attrs = window
            .build_surface_attributes(Default::default())
            .context("failed to build surface attributes")?;
        let gl_surface = unsafe { gl_display.create_window_surface(&gl_config, &surface_attrs) }
            .context("failed to create window surface")?;

        let gl_context = not_current
            .make_current(&gl_surface)
            .context("failed to make GL context current")?;

        if self.capture.is_none() {
            if let Err(e) = gl_surface
                .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
            {
                log::warn!("could not set vsync swap interval: {e}");
            }
        }

        // `gl` is only mutated by the debug-only KHR_debug callback install.
        #[cfg_attr(not(debug_assertions), allow(unused_mut))]
        let mut gl = unsafe {
            glow::Context::from_loader_function_cstr(|s| gl_display.get_proc_address(s).cast())
        };

        let (version, renderer_name) = unsafe {
            (
                gl.get_parameter_string(glow::VERSION),
                gl.get_parameter_string(glow::RENDERER),
            )
        };
        log::info!("OpenGL {version} on {renderer_name}");

        #[cfg(debug_assertions)]
        install_debug_callback(&mut gl);

        let gl = Arc::new(gl);
        let mut renderer = Renderer::new(gl.clone()).context("failed to create renderer")?;
        // Apply the float-precision preference before the first upload.
        renderer.set_half_float(self.prefs.half_float_textures);
        let egui = egui_glow::EguiGlow::new(event_loop, gl.clone(), None, None, false);
        install_ui_font(&egui.egui_ctx);
        // SVG loader for the Bootstrap titlebar icons (egui::include_image!).
        egui_extras::install_image_loaders(&egui.egui_ctx);
        self.titlebar_icon = load_titlebar_icon(&egui.egui_ctx);

        // Crisp multi-resolution title-bar + taskbar icon. Position was set at
        // creation (restored or centred), so no post-creation move here.
        set_window_icons(&window);
        // Round the borderless window's corners per the saved preference.
        apply_window_corners(&window, self.prefs.corner_radius);
        // Stop DWM drawing the legacy non-client frame (it flashes the old-style
        // caption/border on focus change for an undecorated window).
        disable_dwm_decorations(&window);
        // Suppress the classic GDI non-client frame paint on focus change /
        // restore (winit keeps WS_CAPTION for Aero-snap and repaints it via
        // DefWindowProc on WM_NCACTIVATE — the old-style titlebar flash).
        suppress_nonclient_frame(&window);

        Ok(Gfx {
            gl,
            gl_surface,
            gl_context,
            window,
            renderer,
            egui,
        })
    }

    /// Load the path given on the command line, if any. With no argument the
    /// window opens showing the hint; images arrive via drag-drop or the toolbar.
    fn load_initial_image(&mut self) {
        match self.initial_path.clone() {
            Some(p) => self.load_path(p),
            None => log::info!("no initial image; showing hint"),
        }
    }

    /// Size the window to frame a freshly-loaded `width`×`height` 2D image,
    /// scaled down uniformly to fit within [`FILL_FRACTION`] of the monitor (never
    /// upscaled past native), keeping the window centred on its current centre and
    /// on-screen. A fresh load re-frames even after a manual resize (so navigation
    /// recomputes the size as usual). Skipped while maximized or fullscreen.
    fn resize_window_to_image(&mut self, width: u32, height: u32) {
        if self.fullscreen || width == 0 || height == 0 {
            return;
        }
        let mon = match &self.gfx {
            Some(gfx) if gfx.window.is_maximized() => return,
            Some(gfx) => gfx.window.current_monitor(),
            None => return,
        };
        let (w, h) = fit_to_monitor(width as f32, height as f32, mon.as_ref());
        // A new image re-frames the window, overriding any earlier manual size.
        self.manual_window = false;
        self.resize_window_centered(PhysicalSize::new(w, h));
    }

    /// When the 2D zoom *target* changes, grow/shrink the window once so it keeps
    /// framing the image at the target on-screen scale — until it would exceed
    /// [`FILL_FRACTION`] of the monitor, after which the window caps there and the
    /// image overflows (panned into). The target zoom is re-derived from the
    /// (capped) window height so the dialled-in pixel scale is kept, and the
    /// rendered zoom then *eases* toward it within the (already-resized) window —
    /// the window is never resized per animation frame. No-op in panorama mode or
    /// when maximized/fullscreen. A zoom re-asserts the follow even after a manual
    /// resize (the manual size persists only until the next zoom), so the window
    /// re-hugs the image and no black canvas is left around it.
    /// Returns whether the window ended up *uncapped* (the image fits and the pan
    /// was re-centred), so the caller can skip zoom-toward-cursor (which would
    /// otherwise re-introduce an offset on top of the recentre).
    fn follow_zoom_with_window(&mut self) -> bool {
        if self.fullscreen {
            return false;
        }
        // With 2D wrap on the image tiles infinitely, so framing the window to it
        // is meaningless — a zoom should just scale the image in place. Returning
        // false (as if "capped") makes the caller zoom toward the cursor instead.
        if self.wrap_2d && matches!(self.camera.camera, Camera::Flat { .. }) {
            return false;
        }
        let Some(zoom) = self.camera.target_zoom() else {
            return false;
        };
        let (img_w, img_h) = self.display_dims();
        if img_w == 0 || img_h == 0 {
            return false;
        }
        let mon = match &self.gfx {
            Some(gfx) if gfx.window.is_maximized() => return false,
            Some(gfx) => gfx.window.current_monitor(),
            None => return false,
        };
        // A zoom overrides any earlier manual resize.
        self.manual_window = false;
        // Frame against the in-flight follow target's height when an ease is
        // already running, not the current (lagging) window height: each notch
        // then compounds the zoom by the same factor regardless of how fast the
        // user scrolls. Anchoring to the lagging size makes a fast burst zoom
        // far less per notch than the same number of slow clicks.
        let vh = self
            .window_anim_target
            .map(|(_, sz)| sz.height as f32)
            .unwrap_or_else(|| self.viewport().1);
        // Target on-screen scale: device pixels per image pixel (100% == 1.0).
        let scale = zoom * vh / img_h as f32;
        // Frame the image at `scale`, scaled down uniformly (keeping the image's
        // aspect) so it never exceeds FILL_FRACTION of the monitor in any axis.
        let (win_w, win_h) =
            fit_to_monitor(img_w as f32 * scale, img_h as f32 * scale, mon.as_ref());
        // Re-target the zoom to preserve the on-screen scale after the
        // window-height change: uncapped this lands on ~1.0 (image fills the
        // window); capped, the target zoom > 1 so the image overflows the window
        // uniformly and can be panned into. The rendered zoom eases to this.
        let new_zoom = scale * img_h as f32 / win_h as f32;
        self.camera.set_zoom(new_zoom);
        // Once the window can fit the whole image again (uncapped), re-centre it
        // so a leftover pan from the zoomed-in view doesn't leave black canvas.
        let uncapped = new_zoom <= 1.0 + 1e-3;
        if uncapped {
            self.camera.set_pan_target(Vec2::ZERO);
        }
        self.resize_window_centered(PhysicalSize::new(win_w, win_h));
        uncapped
    }

    /// Grow/shrink the window by `factor` about its centre (capped to
    /// [`FILL_FRACTION`] of the monitor, min [`MIN_DIM`]), animated. Used by
    /// Alt+scroll in panorama mode (panoramas have no 2D zoom to drive the
    /// window-follow). No-op when maximized/fullscreen.
    fn resize_window_by_factor(&mut self, factor: f32) {
        if self.fullscreen {
            return;
        }
        let mon = match &self.gfx {
            Some(gfx) if gfx.window.is_maximized() => return,
            Some(gfx) => gfx.window.current_monitor(),
            None => return,
        };
        let (vw, vh) = self.viewport();
        let (w, h) = fit_to_monitor(vw * factor, vh * factor, mon.as_ref());
        self.manual_window = false;
        self.resize_window_centered(PhysicalSize::new(w, h));
    }

    /// Resize the window to `target` inner size, keeping it centred on its current
    /// centre and clamped on-screen. Redundant resizes (within 2 px) are skipped
    /// so per-notch zooming doesn't thrash once the window is capped. Records the
    /// resize so the resulting `Resized` event isn't mistaken for a manual drag.
    /// Grow the window to comfortably fit the Settings dialog when it opens (so the
    /// fixed-width controls never wrap inside a small image window), then restore
    /// the previous size when it closes. No-op in fullscreen / when maximized / when
    /// the window is already large enough.
    fn sync_settings_window(&mut self) {
        let open = self.ui_state.show_settings;
        if open == self.settings_was_open {
            return;
        }
        self.settings_was_open = open;
        if open {
            // Surface a cached update link now and (daily) kick off a fresh check.
            self.maybe_check_for_update();
        }
        if self.fullscreen {
            return;
        }
        let Some((scale, cur, maximized)) = self.gfx.as_ref().map(|g| {
            (
                g.window.scale_factor() as f32,
                g.window.inner_size(),
                g.window.is_maximized(),
            )
        }) else {
            return;
        };
        if maximized {
            return;
        }
        if open {
            let want_w = (SETTINGS_MIN_LOGICAL.0 * scale).ceil() as u32;
            let want_h = (SETTINGS_MIN_LOGICAL.1 * scale).ceil() as u32;
            let target = PhysicalSize::new(cur.width.max(want_w), cur.height.max(want_h));
            if target.width != cur.width || target.height != cur.height {
                self.settings_restore_size = Some(cur);
                self.resize_window_centered(target);
            }
        } else if let Some(prev) = self.settings_restore_size.take() {
            self.resize_window_centered(prev);
        }
    }

    /// On Settings open: show any cached "update available" link immediately, and
    /// — at most once per day — spawn a background check for a newer GitHub
    /// release. The result updates the cache (and the link) when it returns.
    fn maybe_check_for_update(&mut self) {
        self.refresh_update_from_cache();
        if self.update_checking {
            return;
        }
        let now = crate::update::unix_now();
        let last = self.prefs.last_update_check;
        // Throttle to once per day; a backwards clock jump (now < last) re-checks.
        if last > 0 && now >= last && now - last < 86_400 {
            return;
        }
        self.update_checking = true;
        let tx = self.update_tx.clone();
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new()
            .name("update-check".into())
            .spawn(move || {
                // Guard like the load threads: a panic must still send a result so
                // poll_update clears `update_checking` (else checks stall for the
                // session). fetch_latest_release has no panic path today, but the
                // sibling spawns wrap their work too — keep it consistent.
                let result =
                    std::panic::catch_unwind(AssertUnwindSafe(crate::update::fetch_latest_release))
                        .unwrap_or(None);
                let _ = tx.send(result);
                let _ = proxy.send_event(UserEvent::UpdateChecked);
            });
        if spawned.is_err() {
            self.update_checking = false;
        }
    }

    /// Drain finished update checks: cache a successful result (so it persists and
    /// throttles the next check) and refresh the Settings "update available" link.
    fn poll_update(&mut self) {
        let mut got = false;
        while let Ok(msg) = self.update_rx.try_recv() {
            got = true;
            self.update_checking = false;
            // Only a *successful* check advances the daily throttle; a failure
            // (offline / rate-limited) leaves it so the next open retries.
            if let Some(rel) = msg {
                log::info!("update check: latest release is {}", rel.tag);
                self.prefs.last_update_check = crate::update::unix_now();
                self.prefs.latest_known_version = rel.tag;
                self.prefs.save();
            } else {
                log::debug!("update check: no result (offline or no releases)");
            }
        }
        if got {
            self.refresh_update_from_cache();
            self.request_redraw();
        }
    }

    /// Set `available_update` from the cached latest-known release tag (no
    /// network): `Some((label, url))` when it is newer than the running build.
    fn refresh_update_from_cache(&mut self) {
        let cached = self.prefs.latest_known_version.clone();
        self.available_update = if !cached.is_empty()
            && crate::update::is_newer(&cached, crate::update::current_version())
        {
            Some((cached.clone(), crate::update::release_url(&cached)))
        } else {
            None
        };
    }

    fn resize_window_centered(&mut self, target: PhysicalSize<u32>) {
        // Compute the target OUTER rect (centred on the current centre, clamped
        // on-screen) and hand it to the geometry easing (`ease_window`) rather
        // than snapping there immediately.
        let computed = self.gfx.as_ref().and_then(|gfx| {
            let cur = gfx.window.inner_size();
            let close = (cur.width as i32 - target.width as i32).abs() <= 2
                && (cur.height as i32 - target.height as i32).abs() <= 2;
            if close {
                return None;
            }
            let op = gfx.window.outer_position().ok()?;
            let outer = gfx.window.outer_size();
            // Carry the decoration delta so the centre is in outer-frame coords.
            let dx = outer.width.saturating_sub(cur.width);
            let dy = outer.height.saturating_sub(cur.height);
            let new_outer_w = target.width + dx;
            let new_outer_h = target.height + dy;
            let cx = op.x + outer.width as i32 / 2;
            let cy = op.y + outer.height as i32 / 2;
            let mut x = cx - new_outer_w as i32 / 2;
            let mut y = cy - new_outer_h as i32 / 2;
            if let Some(m) = gfx.window.current_monitor() {
                let (mp, ms) = (m.position(), m.size());
                x = x.clamp(mp.x, mp.x + (ms.width as i32 - new_outer_w as i32).max(0));
                y = y.clamp(mp.y, mp.y + (ms.height as i32 - new_outer_h as i32).max(0));
            }
            Some((
                PhysicalPosition::new(x, y),
                PhysicalSize::new(new_outer_w, new_outer_h),
            ))
        });
        if computed.is_some() {
            self.window_anim_target = computed;
            // Just retarget; `about_to_wait` advances the ease one step per loop
            // iteration (after OS input is processed, so a fast scroll burst
            // isn't starved). A redundant per-notch reseed here is what made
            // fast zooming lurch. The redraw kicks the loop into its easing path.
            self.request_redraw();
        }
    }

    /// Advance the window's outer rect one step toward `window_anim_target`,
    /// returning true while still moving. Called once per loop iteration from
    /// `about_to_wait` (NOT from the `Resized` handler): each step posts one
    /// `SetWindowPos`, whose `Resized` renders the new size synchronously (one
    /// vsync-gated present), then the loop yields — processing any queued scroll
    /// input — before the next step. Driving it from `Resized` instead would
    /// drain the whole chain before yielding and starve input (fast-scroll
    /// shudder). The move + resize go through a single atomic `SetWindowPos`.
    /// Self-timed (its own `dt`) since it's not on the render clock.
    fn ease_window(&mut self) -> bool {
        if self.fullscreen {
            self.window_anim_target = None;
            self.last_window_ease = None;
            return false;
        }
        let Some((tpos, tsize)) = self.window_anim_target else {
            self.last_window_ease = None;
            return false;
        };
        let now = Instant::now();
        let dt = self
            .last_window_ease
            .map(|t| now.saturating_duration_since(t).as_secs_f32())
            .unwrap_or(0.016)
            .clamp(0.001, 0.1);
        self.last_window_ease = Some(now);
        let result = self.gfx.as_ref().map(|gfx| {
            let cur = gfx.window.outer_size();
            let curpos = gfx
                .window
                .outer_position()
                .unwrap_or(PhysicalPosition::new(tpos.x, tpos.y));
            let k = 1.0 - (-dt / WINDOW_EASE_TAU).exp();
            let lu = |a: u32, b: u32| {
                ((a as f32 + (b as f32 - a as f32) * k).round() as i32).max(MIN_DIM as i32) as u32
            };
            let (nw, nh) = (lu(cur.width, tsize.width), lu(cur.height, tsize.height));
            let settled = (nw as i32 - tsize.width as i32).abs() <= 1
                && (nh as i32 - tsize.height as i32).abs() <= 1;
            let (w, h) = if settled {
                (tsize.width, tsize.height)
            } else {
                (nw, nh)
            };
            // Ease only the SIZE; derive the position from the fixed target
            // centre so the centre doesn't wobble (easing x/y independently of
            // w/h drifts the centre ±1px and reads as a shiver).
            let cx = tpos.x + tsize.width as i32 / 2;
            let cy = tpos.y + tsize.height as i32 / 2;
            let x = cx - w as i32 / 2;
            let y = cy - h as i32 / 2;
            // Only post when the rect actually changes, so a settled step that
            // lands on the current size doesn't emit a redundant SetWindowPos.
            let changed = w != cur.width || h != cur.height || x != curpos.x || y != curpos.y;
            if changed {
                set_window_outer_rect(&gfx.window, x, y, w, h);
            }
            (settled, changed)
        });
        match result {
            Some((settled, changed)) => {
                if changed {
                    self.suppress_manual_until = Some(Instant::now() + Duration::from_millis(120));
                }
                if settled {
                    self.window_anim_target = None;
                    self.last_window_ease = None;
                }
                !settled
            }
            None => {
                self.window_anim_target = None;
                self.last_window_ease = None;
                false
            }
        }
    }

    fn load_path(&mut self, path: PathBuf) {
        self.load_gen += 1;
        let gen = self.load_gen;
        self.load_start = Instant::now();
        self.pending_name = path.file_name().map(|s| s.to_string_lossy().into_owned());

        // Cache hit (preloaded next, or a previously-viewed image): adopt it
        // immediately, skipping the loading state and the decode thread.
        if let Some(data) = self.cache_take(&path) {
            log::info!("adopting cached {}", path.display());
            self.begin_adopt(data, false, None);
            return;
        }

        self.load_state = LoadState::Loading;
        log::info!("loading (gen {gen}) {}", path.display());

        // Fresh progress for this load; the main loop polls it for the bar.
        self.decode_progress = Arc::new(crate::image_loader::ReadProgress::default());
        let progress = self.decode_progress.clone();
        let tx = self.load_tx.clone();
        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name(format!("image-load-{gen}"))
            .spawn(move || {
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                    load_image(&path, &progress)
                })) {
                    Ok(Ok(data)) => Ok(data),
                    Ok(Err(e)) => Err(format!("{e:#}")),
                    Err(_) => Err("decoder panicked".to_string()),
                };
                let _ = tx.send(LoadResult { gen, result });
                let _ = proxy.send_event(UserEvent::LoadFinished(gen));
            })
            .expect("spawn image-load thread");
    }

    fn poll_loads(&mut self) {
        let mut adopted = false;
        while let Ok(msg) = self.load_rx.try_recv() {
            if msg.gen != self.load_gen {
                log::debug!("discarding stale load result (gen {})", msg.gen);
                continue;
            }
            match msg.result {
                Ok(data) => {
                    self.begin_adopt(Arc::new(data), false, None);
                    adopted = true;
                }
                Err(e) => {
                    log::error!("load failed (gen {}): {e}", msg.gen);
                    self.load_state = LoadState::Failed(e);
                    self.nav_pending = None;
                    adopted = true;
                }
            }
        }
        if adopted {
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        }
    }

    /// Begin adopting a decoded image (from a background load, cache hit, or slot
    /// recall): start the incremental GPU upload and defer the view-state swap to
    /// `finalize_adopt` so the old image stays visible (with a progress bar)
    /// until the new one is ready.
    fn begin_adopt(
        &mut self,
        data: Arc<ImageData>,
        for_compare: bool,
        old_scale: Option<(f32, f32)>,
    ) {
        // Changing the displayed image (folder navigation or a comparator-slot
        // recall) ends any active slot-difference view.
        if self.diff_slot.take().is_some() {
            if let Some(gfx) = &mut self.gfx {
                gfx.renderer.set_diff_image(None);
            }
        }
        if !for_compare {
            log::info!(
                "decoded in {:.2}s, uploading {}",
                self.load_start.elapsed().as_secs_f32(),
                data.path.display()
            );
        }
        // The progress/loading HUD reads `pending_name`; set it to the image
        // actually being adopted (a comparator recall doesn't go through
        // load_path, so it would otherwise show a stale name).
        self.pending_name = data
            .path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned());
        // Begin the GPU upload. If the allocation fails (out of VRAM, or dims past
        // a driver limit), free memory and retry before giving up — so a too-big
        // image reports an error in the centre of the screen instead of silently
        // showing a blank frame.
        let started = self
            .gfx
            .as_mut()
            .is_some_and(|gfx| gfx.renderer.start_upload(data.as_ref()))
            || self.recover_vram_and_retry(data.as_ref());
        if !started {
            log::error!(
                "could not allocate GPU memory for {} ({}x{}, {})",
                data.path.display(),
                data.width,
                data.height,
                data.dtype_name
            );
            self.load_state = LoadState::Failed(
                "Not enough GPU memory to open this image.\n\nTry enabling \
                 \u{201c}Store 32-bit float as 16-bit\u{201d} in Settings, or close \
                 other open images first."
                    .to_string(),
            );
            self.pending = None;
            self.nav_pending = None;
            self.request_redraw();
            return;
        }
        // For a recall there is no decode phase, so time the upload itself.
        if for_compare {
            self.load_start = Instant::now();
        }
        self.upload_progress = 0.0;
        self.pending = Some(PendingAdopt {
            data,
            for_compare,
            old_scale,
        });
        self.request_redraw();
    }

    /// An image upload couldn't be allocated (out of VRAM). Free GPU memory in
    /// stages — cheapest first, keeping the displayed image as long as possible —
    /// retrying the upload after each, and return whether it eventually started.
    /// Note: the comparator slots and the look-ahead cache live in *system* RAM,
    /// not VRAM, so only freeing GPU textures (aux, then the resident image) can
    /// actually relieve VRAM pressure; the cache is dropped too to relieve overall
    /// memory pressure.
    fn recover_vram_and_retry(&mut self, data: &ImageData) -> bool {
        log::warn!("image upload allocation failed; freeing memory and retrying");
        // Drop the RAM look-ahead cache (decoded images kept for instant
        // back/forward). Oldest are at the end, but we clear the lot — it only
        // holds neighbours, and the current image stays in `current_image`.
        self.image_cache.clear();
        // We're about to free the clip-overlay mask (an aux GPU texture); mark it
        // dirty so the overlay rebuilds it on the next frame instead of silently
        // dropping to the per-texel fallback (the mask is normally only re-marked
        // dirty by finalize_adopt, which won't run if every retry still fails).
        self.clip_mask_dirty = true;
        let Some(gfx) = self.gfx.as_mut() else {
            return false;
        };
        // Free auxiliary GPU textures (slot-diff image + clip-overlay mask) and
        // retry; they rebuild lazily when next needed.
        let freed = gfx.renderer.free_aux_gpu_memory();
        if freed > 0 && gfx.renderer.start_upload(data) {
            log::info!("upload recovered after freeing {freed} auxiliary texture(s)");
            return true;
        }
        // Last resort: drop the resident image texture itself and retry. The old
        // image disappears during the upload, but that beats a blank failure.
        if gfx.renderer.free_image_texture() && gfx.renderer.start_upload(data) {
            log::info!("upload recovered after freeing the resident image texture");
            return true;
        }
        false
    }

    /// Advance the in-progress GPU upload by one budget; finalize when complete.
    fn pump_upload(&mut self) {
        let Some(data) = self.pending.as_ref().map(|p| p.data.clone()) else {
            return;
        };
        let progress = match &mut self.gfx {
            Some(gfx) => gfx.renderer.pump_upload(data.as_ref()),
            None => return,
        };
        self.upload_progress = progress;
        if progress >= 1.0 {
            if let Some(pending) = self.pending.take() {
                self.finalize_adopt(pending);
            }
        }
        self.request_redraw();
    }

    /// Apply the view state for a freshly-uploaded image (the texture is already
    /// installed). `for_compare` preserves the current view (like the lock) and
    /// skips folder preload + window auto-resize.
    fn finalize_adopt(&mut self, pending: PendingAdopt) {
        let PendingAdopt {
            data,
            for_compare,
            old_scale,
        } = pending;
        let equirect = data.is_equirectangular();
        // Diagnostic (debug builds only): the raw content scores behind the
        // panorama-vs-2D verdict, for tuning / explaining a misclassification.
        if log::log_enabled!(log::Level::Debug)
            && can_be_panorama(&data.path)
            && is_equirectangular(data.width, data.height)
        {
            if let Some(s) = equirect_content_scores(data.width, data.height, &data.pixels) {
                log::debug!(
                    "pano-detect {}x{}: pole_top={:.4} pole_bottom={:.4} wrap={:.4} -> {}",
                    data.width,
                    data.height,
                    s.pole_top,
                    s.pole_bottom,
                    s.wrap,
                    if equirect { "panorama" } else { "2D" },
                );
            }
        }
        log::info!(
            "loaded {}x{} ({} ch, {}) {} from {}",
            data.width,
            data.height,
            data.channels,
            data.dtype_name,
            if equirect { "[panorama]" } else { "[2D]" },
            data.path.display()
        );
        log::info!(
            "load-to-ready: {:.2}s for {}",
            self.load_start.elapsed().as_secs_f32(),
            data.path.display()
        );

        // A panorama opens in panorama mode by default — but if the user had
        // flipped the *current* panorama to 2D, keep showing panoramas in 2D
        // across the load (don't yank them back into the sphere).
        let prev_pano_in_2d =
            self.loaded_path.is_some() && self.file_info.panorama && !self.camera.is_panorama();
        let want_pano = equirect && !prev_pano_in_2d;

        // Keep the current zoom/pan/exposure when the projection mode matches —
        // for the L lock, and always for a comparator recall (to compare the
        // same region). A 2D <-> panorama change resets to the per-image default.
        let keep_view = (self.locked || for_compare)
            && self.loaded_path.is_some()
            && self.camera.is_panorama() == want_pano;

        self.file_info = FileInfo {
            name: data
                .path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            width: data.width,
            height: data.height,
            channels: data.channels,
            dtype: data.dtype_name.clone(),
            compression: data.compression.clone(),
            panorama: equirect,
            camera: data.camera.clone(),
        };
        self.loaded_path = Some(data.path.clone());
        // Restore this image's remembered display rotation (default upright). A
        // session-only property — kept across navigation / reopen, not persisted.
        self.rotation = self.image_rotations.get(&data.path).copied().unwrap_or(0);
        if !keep_view {
            self.camera = CameraController::for_image(want_pano);
            self.exposure = 0.0;
            self.gamma = 1.0;
            self.wrap_2d = false;
            // Auto-expose: pick the starting exposure so the average linear value
            // lands on AUTO_EXPOSURE_TARGET. (No-op for 8-bit images, where
            // average_linear_luminance returns None.) RAW photos default to NO
            // auto-exposure — their scene-linear develop already respects the
            // actual photo exposure (white = 1.0), so exposure 0 is faithful;
            // an opt-in pref enables it. Non-RAW HDR panoramas auto-expose as
            // before.
            let auto_expose = if crate::image_loader::is_raw(&data.path) {
                self.prefs.raw_auto_exposure
            } else {
                equirect && self.prefs.auto_exposure
            };
            if auto_expose {
                if let Some(luma) = data.average_linear_luminance() {
                    if luma > 1e-6 {
                        self.exposure = (AUTO_EXPOSURE_TARGET / luma).log2().clamp(-16.0, 16.0);
                        log::info!(
                            "auto-exposure: avg luma {luma:.4} -> EV {:+.2}",
                            self.exposure
                        );
                    }
                }
            }
            // A fresh image applies its default tone instantly (no animation).
            self.exposure_target = self.exposure;
            self.gamma_target = self.gamma;
            self.isolate_channel = None;
            self.image_stretch = Vec2::ONE;
            self.guides.clear();
            self.guides_visible = true;
            self.ui_state.guide_spawn = None;
        }
        self.load_state = LoadState::Loaded;
        self.update_window_title();
        // Choose the OCIO view: panoramas restore the saved view for their
        // extension; HDRIs default to Filmic, everything else to Standard.
        self.select_view_for_load(equirect, &data.path);
        // Frame the window to the image whenever it's shown flat (a real 2D image,
        // or a panorama kept in 2D) so it fills the window with no black canvas.
        // Panoramas shown in the sphere keep the window; locked/compared views
        // keep the current size for side-by-side compare.
        if !want_pano && !self.locked && !for_compare {
            let (dw, dh) = self.frame_dims();
            self.resize_window_to_image(dw, dh);
            // In fullscreen there's no window to re-hug (resize is a no-op there),
            // so fit to the screen — but show a sub-screen image at native 1:1
            // rather than magnifying it. No-op when windowed.
            self.apply_fullscreen_fit();
        }
        // Don't let the new image's camera reset (e.g. a fresh panorama's yaw/
        // pitch/fov) read as a "view moved" and auto-pop the minimap — the minimap
        // should only auto-show on a real pan/zoom/look gesture, not on navigation.
        self.minimap_prev_view = None;
        self.apply_debug_overrides();

        // Look-ahead: once arrow-navigating, decode the next neighbour in the
        // background so the following arrow press is instant. Not for a recall.
        if self.preload_armed && !for_compare {
            if let Some(next) = sibling_path(&data.path, self.nav_dir) {
                self.start_preload(next);
            }
        }

        self.current_image = Some(data.clone());
        // The clip-overlay mask is tied to this image; rebuild it lazily if/when
        // the overlay is on.
        self.clip_mask_dirty = true;
        // Same for the display histogram — but drop the old one outright rather
        // than leaving it up, so the F2 box never shows the previous image's
        // graph beside the new image's metadata.
        self.invalidate_histogram();
        self.histogram = None;
        // Animated GIF: begin playback from frame 0 (already uploaded as the
        // static image). Frame 0 shows for its own delay before frame 1. A static
        // image / single-frame GIF clears any prior playback.
        self.anim = data.animation.as_ref().and_then(|a| {
            a.frames.first().map(|f0| AnimState {
                frame: 0,
                next_at: Instant::now() + f0.delay,
                paused: false,
            })
        });
        // Dev-only: pin a specific GIF frame (paused) so each frame's pixels can be
        // verified deterministically in a headless capture (animation timing is
        // process-relative and jittery otherwise).
        #[cfg(debug_assertions)]
        if let Ok(spec) = std::env::var("IMGVWR_DEBUG_GIF_FRAME") {
            if let (Ok(k), Some(img)) = (spec.parse::<usize>(), self.current_image.clone()) {
                if let Some(frames) = img.animation.as_ref().map(|a| &a.frames) {
                    let k = k % frames.len().max(1);
                    if let Some(anim) = self.anim.as_mut() {
                        anim.frame = k;
                        anim.paused = true;
                    }
                    if let Some(gfx) = self.gfx.as_mut() {
                        gfx.renderer.update_animation_frame(
                            img.width as i32,
                            img.height as i32,
                            &frames[k].pixels,
                        );
                    }
                }
            }
        }
        // Dev-only: IMGVWR_DEBUG_SLOT pins the loaded image into slot 1 so the
        // comparator flag can be verified headlessly.
        #[cfg(debug_assertions)]
        if std::env::var_os("IMGVWR_DEBUG_SLOT").is_some() && self.slots[0].is_none() {
            self.slots[0] = self.current_image.clone();
        }
        // Dev-only: verify the diff path headlessly. IMGVWR_DEBUG_DIFF self-diffs
        // (renders 0); IMGVWR_DEBUG_DIFF_FILE=<path> diffs against another file (so
        // identical regions can be confirmed 0 even when minified). Both go through
        // the same precompute as the real Alt+N path.
        #[cfg(debug_assertions)]
        if std::env::var_os("IMGVWR_DEBUG_DIFF").is_some()
            || std::env::var_os("IMGVWR_DEBUG_DIFF_FILE").is_some()
        {
            let target: Option<Arc<ImageData>> =
                if let Some(p) = std::env::var_os("IMGVWR_DEBUG_DIFF_FILE") {
                    let prog = Arc::new(crate::image_loader::ReadProgress::default());
                    load_image(Path::new(&p), &prog).ok().map(Arc::new)
                } else {
                    self.slots[0].clone().or_else(|| self.current_image.clone())
                };
            if let (Some(slot), Some(cur)) = (target, self.current_image.clone()) {
                if let Some(diff) = abs_diff_image(&cur, &slot) {
                    if let Some(gfx) = &mut self.gfx {
                        gfx.renderer.set_diff_image(Some(&diff));
                    }
                    self.diff_slot = Some(0);
                }
            }
        }
        self.recompute_active_slot();
        // Retain the decoded image so navigating back to it is instant.
        self.cache_insert(data);
        // Match on-screen pixel scale for a comparator swap (native resolution).
        self.preserve_native_scale(old_scale);
        // The new image starts settled — freeze the easing target at the camera
        // we just configured (incl. any debug override) so it doesn't animate in.
        self.camera.settle();
        // The navigation (if any) has arrived: we're settled on this image.
        self.nav_pending = None;
        // Undo never crosses an image load: start a fresh stack at this state.
        self.reset_undo();
        self.request_redraw();
    }

    /// Take the cached decoded image for `path`, removing it from the cache.
    fn cache_take(&mut self, path: &Path) -> Option<Arc<ImageData>> {
        self.image_cache
            .iter()
            .position(|d| d.path == path)
            .map(|i| self.image_cache.remove(i))
    }

    /// Insert `data` at the front (most-recent), de-duplicating by path and
    /// evicting the oldest beyond [`IMAGE_CACHE_CAP`].
    fn cache_insert(&mut self, data: Arc<ImageData>) {
        self.image_cache.retain(|d| d.path != data.path);
        self.image_cache.insert(0, data);
        self.image_cache.truncate(IMAGE_CACHE_CAP);
    }

    /// The active flag follows whichever slot (if any) holds the current image.
    /// Matched by path, not `Arc` identity: navigating away and back yields a
    /// different decoded instance of the same file, which should still highlight.
    fn recompute_active_slot(&mut self) {
        self.active_slot = match &self.current_image {
            Some(cur) => self
                .slots
                .iter()
                .position(|s| s.as_ref().is_some_and(|d| d.path == cur.path)),
            None => None,
        };
    }

    /// Ctrl+N: pin the current image into comparator slot `n` (1..=9).
    fn save_slot(&mut self, n: usize) {
        let Some(cur) = self.current_image.clone() else {
            return;
        };
        self.slots[n - 1] = Some(cur);
        // Overwriting the slot currently being diffed against would otherwise leave
        // a stale precomputed diff (vs the old slot content) on screen — turn it
        // off so the comparison can't silently go wrong.
        if self.diff_slot == Some(n - 1) {
            self.diff_slot = None;
            if let Some(gfx) = &mut self.gfx {
                gfx.renderer.set_diff_image(None);
            }
        }
        self.recompute_active_slot();
        self.show_toast(format!("Saved slot {n}"));
        self.request_redraw();
    }

    /// Alt+N: toggle showing the absolute difference between the current image
    /// and comparator slot `n`. The diff is PRECOMPUTED at base resolution (so the
    /// GPU mip chain shows the average of the per-pixel differences — identical
    /// regions read 0 at every zoom) and uploaded as the renderer's diff texture;
    /// exposure/clarity/etc then act on the displayed difference.
    fn toggle_slot_diff(&mut self, n: usize) {
        let idx = n - 1;
        if self.diff_slot == Some(idx) {
            self.diff_slot = None;
            if let Some(gfx) = &mut self.gfx {
                gfx.renderer.set_diff_image(None);
            }
            self.show_toast("Diff off".to_string());
            self.request_redraw();
            return;
        }
        let Some(slot) = self.slots[idx].clone() else {
            self.show_toast(format!("Slot {n} empty"));
            return;
        };
        let Some(current) = self.current_image.clone() else {
            return;
        };
        // Reject oversize images BEFORE the (O(w·h), GB-scale-allocating) CPU diff
        // — the diff texture is current-sized, so if it can't be uploaded the whole
        // precompute would be wasted work / a multi-second freeze for nothing.
        let max_size = self
            .gfx
            .as_ref()
            .map(|g| g.renderer.max_texture_size())
            .unwrap_or(0);
        if current.width.max(current.height) as i32 > max_size {
            self.show_toast("Image too large to diff".to_string());
            return;
        }
        let Some(diff) = abs_diff_image(&current, &slot) else {
            self.show_toast("Can't diff (different pixel types)".to_string());
            return;
        };
        let ok = self
            .gfx
            .as_mut()
            .is_some_and(|gfx| gfx.renderer.set_diff_image(Some(&diff)));
        if ok {
            self.diff_slot = Some(idx);
            self.show_toast(format!("Diff vs slot {n}"));
        } else {
            self.diff_slot = None;
            self.show_toast("Image too large to diff".to_string());
        }
        self.request_redraw();
    }

    /// N: recall comparator slot `n`. Pressing it again while already viewing
    /// that slot toggles back to the previously-shown image (A/B compare).
    /// Recall comparator slot `n` (1-based). `refit_window` chooses the framing:
    /// the keyboard shortcut (`true`) re-frames the window to the image and shows
    /// it fit — exactly like folder navigation / opening a file — while a flag
    /// click (`false`) keeps the current window size and the native-scale A/B
    /// framing, so the flags don't reflow out from under the cursor.
    ///
    /// `for_compare` (the inverse of `refit_window`) drives keep-window +
    /// preserve-view; `old_scale` drives native-scale matching. Both are off for
    /// the re-fitting path so it behaves like a fresh load.
    fn recall_slot(&mut self, n: usize, refit_window: bool) {
        let idx = n - 1;
        let Some(target) = self.slots[idx].clone() else {
            return;
        };
        let for_compare = !refit_window;
        let old_scale = if refit_window {
            None
        } else {
            self.flat_scale_ref()
        };
        // Recalling a slot supersedes any in-flight folder-navigation load: bump
        // the load generation so the decode result is discarded when it lands
        // (else poll_loads would adopt it and clobber this recall), and clear the
        // nav debounce offset so a stale `nav_pending` doesn't perturb later arrows.
        self.load_gen += 1;
        self.nav_pending = None;
        if self.active_slot == Some(idx) {
            // Toggle back to the previously-viewed image (swap so a third press
            // returns to the slot).
            if let Some(prev) = self.compare_prev.take() {
                self.compare_prev = self.current_image.clone();
                self.begin_adopt(prev, for_compare, old_scale);
            }
        } else {
            self.compare_prev = self.current_image.clone();
            self.begin_adopt(target, for_compare, old_scale);
        }
    }

    /// `(zoom, displayed_image_height)` of the current 2D view, for native-scale
    /// matching. Uses the rotation-aware displayed height so two comparator slots
    /// with different rotations still match at their native pixel scale.
    fn flat_scale_ref(&self) -> Option<(f32, f32)> {
        match self.camera.camera {
            Camera::Flat { zoom, .. } => Some((zoom, self.display_dims().1.max(1) as f32)),
            Camera::Pano { .. } => None,
        }
    }

    /// After a comparator swap between 2D images of different resolutions, adjust
    /// the (fit-relative) zoom so the on-screen pixel scale is unchanged — each
    /// image is shown at its native resolution rather than scaled to match.
    fn preserve_native_scale(&mut self, old: Option<(f32, f32)>) {
        if let (Some((old_zoom, old_h)), Camera::Flat { .. }) = (old, self.camera.camera) {
            let new_h = self.display_dims().1.max(1) as f32;
            // Instant (no easing) — a comparator A/B should snap, not animate.
            self.camera.set_zoom_now(old_zoom * new_h / old_h);
        }
    }

    /// Decode `path` in the background and stash it in the image cache.
    fn start_preload(&mut self, path: PathBuf) {
        // Skip if it is already cached or is the current image.
        if self.image_cache.iter().any(|d| d.path == path)
            || self.loaded_path.as_deref() == Some(path.as_path())
        {
            return;
        }
        self.preload_gen += 1;
        let gen = self.preload_gen;
        let tx = self.preload_tx.clone();
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("image-preload-{gen}"))
            .spawn(move || {
                // Preload is a silent look-ahead, so its read progress is unused.
                let progress = Arc::new(crate::image_loader::ReadProgress::default());
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                    load_image(&path, &progress)
                })) {
                    Ok(Ok(data)) => Ok(data),
                    Ok(Err(e)) => Err(format!("{e:#}")),
                    Err(_) => Err("decoder panicked".to_string()),
                };
                let _ = tx.send(PreloadResult { gen, result });
                let _ = proxy.send_event(UserEvent::PreloadFinished(gen));
            });
        if let Err(e) = spawned {
            log::debug!("could not spawn preload thread: {e}");
        }
    }

    fn poll_preloads(&mut self) {
        while let Ok(msg) = self.preload_rx.try_recv() {
            if msg.gen != self.preload_gen {
                continue;
            }
            match msg.result {
                Ok(data) => {
                    log::debug!("preloaded {}", data.path.display());
                    self.cache_insert(Arc::new(data));
                }
                Err(e) => log::debug!("preload failed: {e}"),
            }
        }
    }

    /// Arrow-key navigation: load the alphabetical sibling `dir` steps away.
    fn navigate(&mut self, dir: i32) {
        let Some(current) = self.loaded_path.clone() else {
            return;
        };
        // A folder-navigation load is already in flight.
        if let Some(off) = self.nav_pending {
            if dir == off.signum() {
                // Same direction again: ignore it (don't queue a second step), so
                // spamming → just steps once and waits for the load.
                return;
            }
            // Opposite direction: the user reversed before the load finished, so
            // they'd end up back here anyway — cancel it and stay put (RL == stay,
            // RRRRL == RL).
            self.cancel_nav_load();
            return;
        }
        let Some(target) = sibling_path(&current, dir) else {
            // Only one supported image in the folder (or the file/folder vanished):
            // there's nowhere to navigate to.
            self.show_toast("No more images in this folder".to_string());
            return;
        };
        self.nav_dir = dir;
        self.preload_armed = true;
        self.nav_pending = Some(dir);
        self.load_path(target);
    }

    /// Delete-key confirmed: remove the current file from disk and step to the
    /// next image in the folder. The sibling is looked up *before* deleting —
    /// `sibling_path` scans the live directory listing, which won't contain the
    /// current file any more once it's gone, so computing it after would always
    /// report "no more images". Bypasses `navigate()`'s own lookup for the same
    /// reason; the nav-preload bookkeeping it also does is a minor optimisation
    /// not worth replicating for a one-shot delete.
    fn delete_current_file(&mut self) {
        let Some(path) = self.loaded_path.clone() else {
            return;
        };
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let next = sibling_path(&path, 1);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                log::info!("deleted {}", path.display());
                match next {
                    Some(target) => {
                        self.show_toast(format!("Deleted {name}"));
                        self.load_path(target);
                    }
                    None => {
                        self.show_toast(format!("Deleted {name} (no more images in folder)"));
                    }
                }
            }
            Err(e) => {
                log::error!("failed to delete {}: {e}", path.display());
                self.show_toast(format!("Could not delete {name}: {e}"));
            }
        }
        self.request_redraw();
    }

    /// Cancel an in-flight folder-navigation load and stay on the current image.
    /// Handles both phases: an in-progress GPU upload is abandoned (keeping the
    /// current texture), and a background decode is discarded by bumping the load
    /// generation so its result is ignored when it arrives. The decode thread
    /// itself can't be interrupted, but its output is dropped.
    fn cancel_nav_load(&mut self) {
        self.nav_pending = None;
        if let Some(pending) = self.pending.take() {
            if let Some(gfx) = &mut self.gfx {
                gfx.renderer.cancel_upload();
            }
            self.upload_progress = 0.0;
            // The decode already finished — keep it cached so revisiting that
            // image is still instant (the upload, not the decode, was abandoned).
            self.cache_insert(pending.data);
        }
        // Discard any in-flight / queued decode result for the abandoned target.
        self.load_gen += 1;
        // We're staying on the (already-loaded) current image.
        self.load_state = LoadState::Loaded;
        self.pending_name = self
            .loaded_path
            .as_deref()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned());
        log::info!("navigation cancelled; staying on current image");
        self.request_redraw();
    }

    /// Home: ease pan+zoom back to the fit view (2D) or default FOV (panorama).
    /// Tone adjustments are intentionally left alone (Ctrl+R resets those).
    fn reset_view_full(&mut self) {
        if self.camera.is_panorama() {
            // Snap the look back to centre (easing a spun-around yaw would whirl
            // back), ease the FOV to the default.
            self.camera.snap_look(0.0, 0.0);
            self.camera.set_fov(crate::camera::DEFAULT_PANO_FOV_DEG);
        } else if self.fullscreen {
            // No window to re-hug in fullscreen: fit to the screen, but show a
            // sub-screen image at native 1:1 instead of magnifying it.
            self.apply_fullscreen_fit();
        } else {
            // Fit the image: the window hugs it, so the fit zoom is 1.0.
            self.camera.set_zoom(1.0);
            self.camera.set_pan_target(Vec2::ZERO);
        }
        // Reset the geometric squash/stretch too.
        self.image_stretch = Vec2::ONE;
        // Re-frame the window to the image's default (framed) size for both 2D
        // and panorama (2D uses the rotation-aware displayed dims; panoramas frame
        // to their 2:1 aspect, capped to 90%). Rotation itself is NOT reset.
        let (dw, dh) = self.frame_dims();
        self.resize_window_to_image(dw, dh);
        self.request_redraw();
    }

    /// Backspace: centre the image and fit the window to it at the CURRENT zoom
    /// (unlike Home, which also resets the zoom). The on-screen pixel scale is
    /// preserved — only the pan recentres and the window resizes to hug the image.
    fn center_and_fit_window(&mut self) {
        if self.camera.is_panorama() {
            // No 2D zoom to preserve; re-frame the window to the 2:1 image.
            let (dw, dh) = self.frame_dims();
            self.resize_window_to_image(dw, dh);
        } else {
            // Centre, then frame the window to the image at the current scale.
            // follow_zoom_with_window keeps the on-screen scale (re-deriving the
            // zoom value for the resized window) rather than changing zoom.
            self.camera.set_pan_target(Vec2::ZERO);
            self.follow_zoom_with_window();
        }
        self.request_redraw();
    }

    /// Rotate the displayed image 90° (`dir` = +1 clockwise, −1 counter-clockwise),
    /// remembered per image path for the session (restored when the image is
    /// reopened or stepped back to). The rotation is a display property: it is NOT
    /// reset by R / Home / Ctrl+R. In 2D the window re-frames to the rotated image
    /// and recentres; panorama keeps its sphere framing (rotation is inert there
    /// until the image is viewed in 2D).
    fn rotate_image(&mut self, dir: i32) {
        if self.file_info.width == 0 {
            return;
        }
        self.rotation = (self.rotation as i32 + dir).rem_euclid(4) as u8;
        if let Some(path) = &self.loaded_path {
            self.image_rotations.insert(path.clone(), self.rotation);
        }
        if !self.camera.is_panorama() {
            self.camera.center_flat_now();
            let (dw, dh) = self.frame_dims();
            self.resize_window_to_image(dw, dh);
        }
        self.show_toast("Rotated 90°".to_string());
        self.request_redraw();
    }

    // ---- toast HUD -------------------------------------------------------

    fn show_toast(&mut self, text: String) {
        self.toast = Some(Toast {
            text,
            born: Instant::now(),
        });
        self.request_redraw();
    }

    /// `(text, alpha)` for the active toast, or `None` once it has expired.
    fn toast_render(&self) -> Option<(String, f32)> {
        let toast = self.toast.as_ref()?;
        let e = toast.born.elapsed();
        if e >= TOAST_HOLD + TOAST_FADE {
            return None;
        }
        let alpha = if e <= TOAST_HOLD {
            1.0
        } else {
            1.0 - (e - TOAST_HOLD).as_secs_f32() / TOAST_FADE.as_secs_f32()
        };
        Some((toast.text.clone(), alpha.clamp(0.0, 1.0)))
    }

    fn toast_active(&self) -> bool {
        // A small grace past the fade so one final frame draws the cleared
        // (alpha-0) state instead of leaving a faint toast until the next input.
        self.toast.as_ref().is_some_and(|t| {
            t.born.elapsed() < TOAST_HOLD + TOAST_FADE + Duration::from_millis(100)
        })
    }

    fn show_exposure_toast(&mut self) {
        let text = fmt_ev(self.exposure_target);
        self.show_toast(text);
    }

    fn show_gamma_toast(&mut self) {
        let text = format!("Gamma {:.1}", self.gamma_target);
        self.show_toast(text);
    }

    /// Reset all image-processing adjustments (Ctrl+R): exposure, gamma, clarity,
    /// channel isolation. (Geometric squash/stretch resets with R/Home instead.)
    fn reset_image_processing(&mut self) {
        self.exposure_target = 0.0;
        self.gamma_target = 1.0;
        self.clarity_amount = 0.0;
        self.clarity_radius = 64.0;
        self.isolate_channel = None;
        self.sharpness = false;
        self.clip_overlay = false;
        self.guides.clear();
        self.guides_visible = true;
        // Drop any in-flight guide gesture so its release can't touch a
        // since-cleared guide (the gesture's release, if any, becomes a no-op).
        self.guide_drag = None;
        self.ui_state.guide_spawn = None;
        self.diff_slot = None;
        if let Some(gfx) = &mut self.gfx {
            gfx.renderer.set_diff_image(None);
        }
        self.show_toast("Adjustments reset".to_string());
        self.request_redraw();
    }

    // ---- undo / redo -----------------------------------------------------

    /// Snapshot the current undoable editing state.
    fn undo_snapshot(&self) -> UndoState {
        UndoState {
            guides: self.guides.clone(),
            guides_visible: self.guides_visible,
            exposure_target: self.exposure_target,
            gamma_target: self.gamma_target,
            clarity_amount: self.clarity_amount,
            clarity_radius: self.clarity_radius,
            isolate_channel: self.isolate_channel,
            sharpness: self.sharpness,
            clip_overlay: self.clip_overlay,
            wrap_2d: self.wrap_2d,
            nearest_filter: self.nearest_filter,
            nearest_auto: self.nearest_auto,
            image_stretch: self.image_stretch,
        }
    }

    /// True while a continuous edit gesture is in progress — its many per-frame
    /// changes coalesce into one undo entry, committed when the gesture ends.
    fn undo_gesture_active(&self) -> bool {
        self.guide_drag.is_some()
            || self.ui_state.guide_spawn.is_some()
            || self.stretching
            || self.adjust_repeat_until.is_some_and(|t| Instant::now() < t)
    }

    /// Called once per frame after input: if the editing state changed away from
    /// the baseline (and no gesture is mid-flight), push the old baseline onto the
    /// undo stack so the change can be undone. A new change invalidates redo.
    /// `egui_busy` is egui's "using the pointer" flag, so a slider drag (which
    /// emits a value every frame) coalesces into one entry on release.
    fn commit_undo_if_changed(&mut self, egui_busy: bool) {
        if egui_busy || self.undo_gesture_active() {
            return;
        }
        // The held-adjustment coalesce window has closed (else undo_gesture_active
        // would be true); drop the timer so `about_to_wait` stops waking for it.
        self.adjust_repeat_until = None;
        let cur = self.undo_snapshot();
        if cur != self.undo_baseline {
            let old = std::mem::replace(&mut self.undo_baseline, cur);
            push_capped(&mut self.undo_stack, old, UNDO_LIMIT);
            self.redo_stack.clear();
        }
    }

    /// Reset undo/redo to the freshly-loaded image's state (undo never crosses an
    /// image load). Call at the end of a load.
    fn reset_undo(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.undo_baseline = self.undo_snapshot();
    }

    /// Apply a snapshot to the live state (instant — tone is snapped, not eased).
    fn restore_undo_state(&mut self, s: UndoState) {
        self.guides = s.guides;
        self.guides_visible = s.guides_visible;
        self.exposure_target = s.exposure_target;
        self.exposure = s.exposure_target;
        self.gamma_target = s.gamma_target;
        self.gamma = s.gamma_target;
        self.clarity_amount = s.clarity_amount;
        self.clarity_radius = s.clarity_radius;
        self.isolate_channel = s.isolate_channel;
        self.sharpness = s.sharpness;
        self.clip_overlay = s.clip_overlay;
        self.wrap_2d = s.wrap_2d;
        self.nearest_filter = s.nearest_filter;
        self.nearest_auto = s.nearest_auto;
        self.image_stretch = s.image_stretch;
        // Cancel any in-flight guide gesture (grab or ruler-spawn) so its release
        // can't move/keep/delete a guide against the just-replaced vector, and clear
        // the hover highlight.
        self.guide_drag = None;
        self.ui_state.guide_spawn = None;
        self.ui_state.hovered_guide = None;
        self.request_redraw();
    }

    fn undo(&mut self) {
        let Some(prev) = self.undo_stack.pop() else {
            self.show_toast("Nothing to undo".to_string());
            return;
        };
        let cur = self.undo_snapshot();
        push_capped(&mut self.redo_stack, cur, UNDO_LIMIT);
        self.undo_baseline = prev.clone();
        self.restore_undo_state(prev);
        self.show_toast("Undo".to_string());
    }

    fn redo(&mut self) {
        let Some(next) = self.redo_stack.pop() else {
            self.show_toast("Nothing to redo".to_string());
            return;
        };
        let cur = self.undo_snapshot();
        push_capped(&mut self.undo_stack, cur, UNDO_LIMIT);
        self.undo_baseline = next.clone();
        self.restore_undo_state(next);
        self.show_toast("Redo".to_string());
    }

    /// Add a guide line (image coord 0..1; `horizontal` = a constant-image-y
    /// line). Capped at [`crate::renderer::MAX_GUIDES`]. Guides clear with the
    /// image or Ctrl+R.
    fn add_guide(&mut self, coord: f32, horizontal: bool) {
        if self.guides.len() < crate::renderer::MAX_GUIDES {
            self.guides
                .push([coord.clamp(0.0, 1.0), if horizontal { 1.0 } else { 0.0 }]);
            self.guides_visible = true;
            self.show_toast("Guide added".to_string());
            self.request_redraw();
        }
    }

    /// Remove guide `idx`, reconciling the in-flight drag/hover indices so a
    /// removal during an LMB guide-drag (e.g. a concurrent right-click delete)
    /// keeps the drag targeting the same line instead of silently shifting onto
    /// the guide that slid into the freed slot.
    fn remove_guide(&mut self, idx: usize) {
        if idx >= self.guides.len() {
            return;
        }
        self.guides.remove(idx);
        match self.guide_drag {
            Some(d) if d == idx => self.guide_drag = None,
            Some(d) if d > idx => self.guide_drag = Some(d - 1),
            _ => {}
        }
        self.ui_state.hovered_guide = None;
        self.request_redraw();
    }

    /// Is there already a guide at `pos` (image fraction) with this orientation?
    fn has_guide(&self, pos: f32, horizontal: bool) -> bool {
        let orient = if horizontal { 1.0 } else { 0.0 };
        self.guides
            .iter()
            .any(|g| (g[0] - pos).abs() < 1e-3 && (g[1] - orient).abs() < 0.5)
    }

    /// Completed subdivision levels (denoms 2,4,8,16,32 → max 5) on one axis: the
    /// largest N such that every odd-numerator position at denom 2,4,…,2^N is
    /// present. Ignores anything else on the axis (a manually-added guide off the
    /// grid, or a level left incomplete by a manual removal) — those just don't
    /// count toward a level, so `add_next_guide` / `remove_guides_step` only ever
    /// act on a level they can positively identify as complete.
    fn completed_guide_levels(&self, horizontal: bool) -> u32 {
        let mut levels = 0u32;
        let mut denom = 2u32;
        while denom <= 32 {
            let full = (1..denom)
                .step_by(2)
                .all(|n| self.has_guide(n as f32 / denom as f32, horizontal));
            if full {
                levels += 1;
                denom *= 2;
            } else {
                break;
            }
        }
        levels
    }

    /// Shift+G subdivides ONE axis per press so the grid converges toward square
    /// cells. The very first guide is always horizontal; after that it subdivides
    /// whichever axis brings the cell aspect closest to 1:1 — i.e. the longer cell
    /// edge first. For a 32:1 image that means splitting the width (vertical
    /// guides) down to 1/32 before the 2nd horizontal guide; for a 2:1 HDRI it's
    /// H, V, V (→ 8 squares), then alternating H/V. Each press adds every
    /// odd-numerator position at that axis's coarsest not-yet-complete level (½, ¼
    /// … 1/32), capped at [`crate::renderer::MAX_GUIDES`]. In panorama mode,
    /// completing the first vertical level (the 180° guide) also drops its
    /// antipodal partner at 0° — a lone guide only draws a half-meridian on the
    /// unwrapped image, so bisecting the sphere with a full great circle needs
    /// both.
    fn add_next_guide(&mut self) {
        let (h, v) = (
            self.completed_guide_levels(true),
            self.completed_guide_levels(false),
        );
        let aspect = self.display_aspect();
        let Some(horizontal) = next_guide_horizontal(h, v, self.guides.is_empty(), aspect) else {
            return; // both axes full
        };
        let levels = if horizontal { h } else { v };
        let denom = 2u32 << levels; // 2, 4, 8, 16, 32
        let mut added = 0;
        for n in (1..denom).step_by(2) {
            if self.guides.len() >= crate::renderer::MAX_GUIDES {
                break;
            }
            let pos = n as f32 / denom as f32;
            if !self.has_guide(pos, horizontal) {
                self.guides.push([pos, if horizontal { 1.0 } else { 0.0 }]);
                added += 1;
            }
        }
        if !horizontal
            && levels == 0
            && self.camera.is_panorama()
            && self.guides.len() < crate::renderer::MAX_GUIDES
            && !self.has_guide(0.0, false)
        {
            self.guides.push([0.0, 0.0]);
            added += 1;
        }
        if added > 0 {
            self.guides_visible = true;
            let axis = if horizontal { "horizontal" } else { "vertical" };
            let s = if added == 1 { "" } else { "s" };
            self.show_toast(format!("Added {added} {axis} guide{s}"));
            self.request_redraw();
        }
    }

    /// Ctrl+G undoes one `add_next_guide` step: removes the finest completed
    /// level from whichever axis is currently more subdivided (peeling from the
    /// top, so repeated presses walk back the same progression `add_next_guide`
    /// would have walked forward). Robust to guides having been added, moved, or
    /// removed by hand in between — it only ever acts on a level it can positively
    /// identify as complete (see `completed_guide_levels`); once no such level
    /// remains, it falls back to clearing whatever's left in one press.
    fn remove_guides_step(&mut self) {
        if self.guides.is_empty() {
            return;
        }
        let (h, v) = (
            self.completed_guide_levels(true),
            self.completed_guide_levels(false),
        );
        if h == 0 && v == 0 {
            let n = self.guides.len();
            self.guides.clear();
            self.guide_drag = None;
            self.ui_state.guide_spawn = None;
            self.ui_state.hovered_guide = None;
            let s = if n == 1 { "" } else { "s" };
            self.show_toast(format!("Cleared {n} guide{s}"));
            self.request_redraw();
            return;
        }
        // Peel the more-subdivided axis first; a tie favours whichever axis
        // currently holds more guides overall (the more heavily hand-extended one).
        let horizontal = match h.cmp(&v) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => {
                let count = |horiz: bool| {
                    self.guides
                        .iter()
                        .filter(|g| (g[1] >= 0.5) == horiz)
                        .count()
                };
                count(true) >= count(false)
            }
        };
        let levels = if horizontal { h } else { v };
        let denom = 2u32 << (levels - 1); // the finest completed level's denom
        let before = self.guides.len();
        self.guides.retain(|g| {
            if (g[1] >= 0.5) != horizontal {
                return true;
            }
            !(1..denom)
                .step_by(2)
                .any(|n| (g[0] - n as f32 / denom as f32).abs() < 1e-3)
        });
        // The 180°/0° bisector pair (see `add_next_guide`) was added together;
        // remove it together too.
        if !horizontal && denom == 2 && self.camera.is_panorama() {
            self.guides.retain(|g| !(g[1] < 0.5 && g[0].abs() < 1e-3));
        }
        let removed = before - self.guides.len();
        if removed > 0 {
            self.guide_drag = None;
            self.ui_state.guide_spawn = None;
            self.ui_state.hovered_guide = None;
            let axis = if horizontal { "horizontal" } else { "vertical" };
            let s = if removed == 1 { "" } else { "s" };
            self.show_toast(format!("Removed {removed} {axis} guide{s}"));
            self.request_redraw();
        }
    }

    /// G (no modifiers): show/hide the existing guides without touching them —
    /// unless there are none yet, in which case it behaves like the first
    /// Shift+G press and adds the first one.
    fn toggle_guides_visibility(&mut self) {
        if self.guides.is_empty() {
            self.add_next_guide();
            return;
        }
        self.guides_visible = !self.guides_visible;
        self.ui_state.hovered_guide = None;
        self.show_toast(
            if self.guides_visible {
                "Guides shown"
            } else {
                "Guides hidden"
            }
            .to_string(),
        );
        self.request_redraw();
    }

    /// Ctrl+drag guide-coordinate snapping: nearest whole degree in panorama,
    /// nearest 10 displayed px in 2D. `coord` is the guide's stored uv fraction
    /// along its constant axis (`horizontal` selects which). Mirrors
    /// `ui::overlay`'s ruler-spawn version (same formulas, App-side data instead
    /// of `RulerInfo`).
    fn snap_guide_coord(&self, coord: f32, horizontal: bool) -> f32 {
        if self.camera.is_panorama() {
            if horizontal {
                let lat = (0.5 - coord) * 180.0;
                0.5 - lat.round() / 180.0
            } else {
                (coord * 360.0).round() / 360.0
            }
        } else {
            let (dw, dh) = self.display_dims();
            let dim = if horizontal { dh } else { dw } as f32;
            if dim <= 0.0 {
                return coord;
            }
            (coord * dim / 10.0).round() * 10.0 / dim
        }
        .clamp(0.0, 1.0)
    }

    /// Adjust the Clarity strength (0 = off). Chunky steps; the range goes well
    /// past photographic levels so issues can be cranked into the extreme.
    fn adjust_clarity(&mut self, delta: f32) {
        self.clarity_amount = (self.clarity_amount + delta).clamp(0.0, 10.0);
        let msg = if self.clarity_amount <= 0.0 {
            "Clarity off".to_string()
        } else {
            format!("Clarity {:.2}", self.clarity_amount)
        };
        self.show_toast(msg);
        self.request_redraw();
    }

    /// Adjust the Clarity unsharp-mask radius (viewport pixels).
    fn adjust_clarity_radius(&mut self, delta: f32) {
        self.clarity_radius = (self.clarity_radius + delta).clamp(8.0, 256.0);
        self.show_toast(format!("Clarity radius {:.0} px", self.clarity_radius));
        self.request_redraw();
    }

    /// Ease the rendered exposure / gamma toward their targets (so keyboard and
    /// slider adjustments animate like zoom). Returns true while still moving.
    fn animate_tone(&mut self, dt: f32) -> bool {
        let k = 1.0 - (-dt / TONE_EASE_TAU).exp();
        let mut moving = false;
        let de = self.exposure_target - self.exposure;
        if de.abs() > 1e-4 {
            self.exposure += de * k;
            moving = true;
        } else {
            self.exposure = self.exposure_target;
        }
        let dg = self.gamma_target - self.gamma;
        if dg.abs() > 1e-4 {
            self.gamma += dg * k;
            moving = true;
        } else {
            self.gamma = self.gamma_target;
        }
        moving
    }

    fn show_zoom_toast(&mut self) {
        // Report the target (where the ease is heading), not the mid-animation
        // value, so the toast shows the level the user dialled in.
        let text = match self.camera.target() {
            Camera::Pano { fov_deg, .. } => format!("FOV {}°", fov_deg.round() as i32),
            Camera::Flat { .. } => match self.flat_zoom_percent() {
                Some(p) => format!("{}%", p.round() as i32),
                None => return,
            },
        };
        self.show_toast(text);
    }

    /// Current (live, not eased target) 2D on-screen scale — device pixels per
    /// image pixel, where 1.0 == 100% (1:1). `None` in panorama mode (no 2D zoom).
    fn flat_scale_now(&self) -> Option<f32> {
        let img_h = self.display_dims().1;
        if img_h == 0 {
            return None;
        }
        if let Camera::Flat { zoom, .. } = self.camera.camera {
            let (_, vh) = self.viewport();
            Some(zoom * vh / img_h as f32)
        } else {
            None
        }
    }

    /// Centre-of-view on-screen scale for a panorama — device pixels per *image*
    /// (equirectangular texel) pixel at the look direction. `None` in 2D mode.
    ///
    /// The equirect image is isotropic at `H/π` texels per radian (height `H` spans
    /// 180°). The rectilinear projection's angular resolution at screen centre is
    /// `vh / (2·tan(½fov))` screen px per radian (the shader builds the centre ray
    /// as `ndc.y · tan_half_fov`, so `dθ/dpx = 2·tan_half_fov/vh` there; horizontal
    /// matches once aspect is applied). Their ratio is the texel→screen scale, so
    /// the 200% nearest switch lands when one texel covers two screen pixels — same
    /// rule as 2D, just FOV/dimension-aware.
    fn pano_scale_now(&self) -> Option<f32> {
        if !self.camera.is_panorama() {
            return None;
        }
        let ih = self.file_info.height;
        if ih == 0 {
            return None;
        }
        let (_, vh) = self.viewport();
        let tan_half = self.camera.camera.tan_half_fov().max(1e-6);
        Some(pano_center_scale(vh, ih as f32, tan_half))
    }

    /// Whether to sample nearest-neighbour this frame. With `nearest_auto` (the
    /// default) it's chosen automatically — an image magnified past 200% at the
    /// view centre reads nearest (crisp pixels), less-magnified views bilinear;
    /// this now covers panoramas too (via [`pano_scale_now`], FOV/dimension-aware).
    /// Once the I key pins a manual choice (`nearest_auto` off) that value is used
    /// verbatim. Feeds both the sampler and the Lanczos gate (`is_u8 && !nearest`),
    /// so a manual nearest also turns Lanczos off, exactly as the auto path does.
    fn effective_nearest(&self) -> bool {
        let scale = self.flat_scale_now().or_else(|| self.pano_scale_now());
        pick_nearest(self.nearest_auto, self.nearest_filter, scale)
    }

    /// Target 2D zoom as a percentage where 100% == 1 image px : 1 monitor px.
    fn flat_zoom_percent(&self) -> Option<f32> {
        // Rotation-aware displayed height — the basis set_exact_zoom uses, so the
        // toast reads the % the user actually dialled in.
        let img_h = self.display_dims().1;
        if img_h == 0 {
            return None;
        }
        if let Camera::Flat { zoom, .. } = self.camera.target() {
            let (_, vh) = self.viewport();
            Some(zoom * vh / img_h as f32 * 100.0)
        } else {
            None
        }
    }

    /// Numpad exact zoom. Each digit doubles the previous: plain `d` zooms in to
    /// `2^(d-1) ×` (1=100%, 2=200%, 3=400%…); Ctrl zooms out to `1/2^(d-1)`
    /// (Ctrl+2=50%, Ctrl+3=25%…). Exact 1:1 device pixels for 2D; an FOV
    /// approximation for panoramas.
    fn set_exact_zoom(&mut self, digit: u32, ctrl: bool) {
        if digit == 0 {
            return;
        }
        let mult = 2f32.powi(digit as i32 - 1);
        let pct = if ctrl { 100.0 / mult } else { 100.0 * mult };
        let (_, vh) = self.viewport();
        let img_h = self.display_dims().1.max(1) as f32;
        let zoom = (pct / 100.0) * (img_h / vh);
        let z0 = self.camera.target_zoom();
        match self.camera.camera {
            Camera::Flat { .. } => self.camera.set_zoom(zoom),
            Camera::Pano { .. } => {
                let fov = (1.0 / zoom.max(1e-4)).atan().to_degrees() * 2.0;
                self.camera.set_fov(fov);
            }
        }
        self.show_zoom_toast();
        let uncapped = self.follow_zoom_with_window();
        if let Some(z0) = z0 {
            if !uncapped {
                self.zoom_toward_cursor(z0);
            }
        }
        self.request_redraw();
    }

    // ---- input -----------------------------------------------------------

    fn ctrl(&self) -> bool {
        self.modifiers.control_key()
    }

    fn viewport(&self) -> (f32, f32) {
        self.gfx
            .as_ref()
            .map(|g| {
                let s = g.window.inner_size();
                (s.width.max(1) as f32, s.height.max(1) as f32)
            })
            .unwrap_or((1.0, 1.0))
    }

    fn request_redraw(&self) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }

    fn start_drag(&mut self) {
        self.dragging = true;
        // Cut any in-flight zoom/pan animation short so the drag takes over now.
        self.freeze_animations();
        self.drag_start_cursor = self.cursor_pos;
        if let Some(gfx) = &self.gfx {
            // Confine (fallback to lock) so look-around/pan is unbounded; hide
            // the cursor for the duration of the gesture.
            let grabbed = gfx
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| gfx.window.set_cursor_grab(CursorGrabMode::Locked));
            if let Err(e) = grabbed {
                log::debug!("cursor grab failed: {e}");
            }
            gfx.window.set_cursor_visible(false);
        }
        // The drag now owns cursor visibility; clear the idle-hide flag so its
        // per-frame check doesn't fight (end_drag re-shows the cursor).
        self.cursor_idle_hidden = false;
    }

    fn end_drag(&mut self) {
        if !self.dragging {
            return;
        }
        self.dragging = false;
        if let Some(gfx) = &self.gfx {
            let _ = gfx.window.set_cursor_grab(CursorGrabMode::None);
            // Put the cursor back where the drag began (a confined/locked grab
            // leaves it at the window centre otherwise).
            let _ = gfx.window.set_cursor_position(self.drag_start_cursor);
            gfx.window.set_cursor_visible(true);
        }
    }

    /// Apply a relative drag delta (raw device motion, pixels).
    fn on_drag_motion(&mut self, dx: f32, dy: f32) {
        // Track total travel so a double-click-and-drag suppresses fullscreen.
        if self.pending_dblclick {
            self.dblclick_motion += (dx * dx + dy * dy).sqrt();
        }
        // Axis lock while dragging: Shift = horizontal only, Ctrl = vertical only.
        let dx = if self.modifiers.control_key() {
            0.0
        } else {
            dx
        };
        let dy = if self.modifiers.shift_key() { 0.0 } else { dy };
        let (vw, vh) = self.viewport();
        match self.camera.camera {
            Camera::Pano { pitch_rad, .. } => {
                // Pixels-per-radian from the current vertical FOV, +20% overall.
                let rad_per_px = (2.0 * self.camera.camera.half_fov_radians()) / vh * 1.2;
                // Gentle latitude compensation for equirect horizontal stretch
                // (capped low so it never feels like runaway speed near the poles).
                let h_mult = (1.0 / pitch_rad.abs().cos().max(0.25)).min(1.5);
                let dyaw = -dx * rad_per_px * h_mult;
                let dpitch = -dy * rad_per_px;
                self.camera.rotate(dyaw, dpitch);
            }
            Camera::Flat { .. } => {
                let inv_zoom = self.camera.camera.tan_half_fov();
                // Rotation-aware (displayed) aspect, matching the shader / viewport_uv,
                // so a drag follows the cursor 1:1 at every rotation.
                let image_aspect = self.display_aspect();
                // Match the shader's screen→image scale (which divides by the
                // squash/stretch) so the image follows the cursor 1:1 regardless
                // of squash — otherwise a narrow squash slows panning to a crawl.
                let sx = inv_zoom * (vw / vh) / image_aspect / self.image_stretch.x;
                let sy = inv_zoom / self.image_stretch.y;
                // Grab feel: content follows the cursor. Panning is unbounded —
                // the image may be moved freely past the viewport edge.
                let du = -(dx / vw) * sx;
                let dv = -(dy / vh) * sy;
                self.camera.pan(Vec2::new(du, dv));
            }
        }
        self.request_redraw();
    }

    /// On leaving wrap mode, fold the (possibly large) pan back into the
    /// canonical range so the same on-screen region maps onto the real image
    /// rather than a now-clipped tiled clone.
    fn normalize_pan_to_canonical(&mut self) {
        // Normalise both the rendered and target pan, so turning off wrap after a
        // long wrapped pan doesn't ease the image off-screen toward a far target.
        self.camera.normalize_flat_pan();
    }

    fn on_wheel(&mut self, delta: MouseScrollDelta) {
        let steps = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => (p.y / 20.0) as f32,
        };
        if steps == 0.0 {
            return;
        }

        // Shift = horizontal pan, Ctrl = vertical pan (both 2D and panorama).
        let shift = self.modifiers.shift_key();
        if shift || self.ctrl() {
            self.wheel_pan(steps, shift);
            self.request_redraw();
            return;
        }

        // Alt + scroll always grows/shrinks the window, in both modes — even when
        // zoomed into a 2D image past the window-fill cap. A uniform window resize
        // keeps the visible image region the same (the screen→image scale depends
        // on the window's aspect, not its size), so the image scales with the
        // window: it reads as zooming the whole view out/in while the portion of
        // the image you're looking at stays put.
        if self.modifiers.alt_key() {
            self.resize_window_by_factor(1.21_f32.powf(steps));
            self.request_redraw();
            return;
        }

        let z0 = self.camera.target_zoom();
        match self.camera.camera {
            Camera::Pano { fov_deg, .. } => {
                // Progressive feel: step scaled by current FOV (2× strength).
                let step = (fov_deg / 90.0) * 10.0;
                self.camera.adjust_fov(-steps * step);
            }
            Camera::Flat { .. } => {
                // 2× per-notch strength (1.1² ≈ 1.21).
                self.camera.adjust_zoom(1.21_f32.powf(steps));
            }
        }
        self.show_zoom_toast();
        let uncapped = self.follow_zoom_with_window();
        if let Some(z0) = z0 {
            if !uncapped {
                self.zoom_toward_cursor(z0);
            }
        }
        self.request_redraw();
    }

    /// Shift the 2D pan target so the image point under the cursor stays put as
    /// the zoom changes from `z0` to the current target — i.e. zoom toward the
    /// cursor rather than the view centre. No-op in panorama mode. (In the
    /// uncapped window-follow regime the zoom barely changes, so this is ~0 and
    /// zooming just grows the window; it matters once the window is capped.)
    fn zoom_toward_cursor(&mut self, z0: f32) {
        let Some(z1) = self.camera.target_zoom() else {
            return;
        };
        if (z1 - z0).abs() < 1e-6 || z0 <= 0.0 || z1 <= 0.0 {
            return;
        }
        let (vw, vh) = self.viewport();
        // Rotation-aware aspect so zoom stays pinned to the cursor at every rotation.
        let aspect = self.display_aspect();
        // UV offset of the cursor from the view centre, per unit inverse-zoom.
        let off = Vec2::new(
            (self.cursor_pos.x as f32 - vw * 0.5) / (aspect * vh),
            (self.cursor_pos.y as f32 - vh * 0.5) / vh,
        );
        self.camera.pan_target(off * (1.0 / z0 - 1.0 / z1));
    }

    /// Wheel pan: `horizontal` pans left/right (yaw in panorama), otherwise
    /// up/down (pitch). Scroll-up pans right / up.
    fn wheel_pan(&mut self, steps: f32, horizontal: bool) {
        let (vw, vh) = self.viewport();
        match self.camera.camera {
            Camera::Pano { .. } => {
                let step = self.camera.camera.half_fov_radians() * 0.25 * steps;
                if horizontal {
                    self.camera.rotate(step, 0.0);
                } else {
                    self.camera.rotate(0.0, -step);
                }
            }
            Camera::Flat { .. } => {
                let inv_zoom = self.camera.camera.tan_half_fov();
                let image_aspect = self.display_aspect(); // rotation-aware
                let sx = inv_zoom * (vw / vh) / image_aspect;
                let sy = inv_zoom;
                let k = 0.15 * steps;
                // Eased (wheel) pan — eases to target, unlike a direct drag.
                if horizontal {
                    self.camera.pan_target(Vec2::new(k * sx, 0.0));
                } else {
                    self.camera.pan_target(Vec2::new(0.0, -k * sy));
                }
            }
        }
    }

    fn on_key(&mut self, event_loop: &ActiveEventLoop, key: &Key, is_char: Option<&str>) {
        let ctrl = self.ctrl();
        match (key, is_char) {
            (_, Some(",")) => {
                if ctrl {
                    self.gamma_target = (self.gamma_target - 0.1).max(0.1);
                    self.show_gamma_toast();
                } else {
                    self.exposure_target -= 0.5;
                    self.show_exposure_toast();
                }
            }
            (_, Some(".")) => {
                if ctrl {
                    self.gamma_target = (self.gamma_target + 0.1).min(4.0);
                    self.show_gamma_toast();
                } else {
                    self.exposure_target += 0.5;
                    self.show_exposure_toast();
                }
            }
            // Clarity (local contrast): [ ] radius, ; ' strength.
            (_, Some("[")) => self.adjust_clarity_radius(-16.0),
            (_, Some("]")) => self.adjust_clarity_radius(16.0),
            (_, Some(";")) => self.adjust_clarity(-0.5),
            (_, Some("'")) => self.adjust_clarity(0.5),
            // Ctrl+Z undo, Ctrl+Shift+Z / Ctrl+Y redo (editing state only).
            (_, Some("z")) | (_, Some("Z")) if ctrl => {
                if self.modifiers.shift_key() {
                    self.redo();
                } else {
                    self.undo();
                }
            }
            (_, Some("y")) | (_, Some("Y")) if ctrl => self.redo(),
            // Ctrl+R: reset all image-processing adjustments.
            (_, Some("r")) | (_, Some("R")) if ctrl => self.reset_image_processing(),
            // R (no Ctrl): reset the view + window, same as Home.
            (_, Some("r")) | (_, Some("R")) => self.reset_view_full(),
            (_, Some("a")) | (_, Some("A")) => {
                self.always_on_top = !self.always_on_top;
                if let Some(gfx) = &self.gfx {
                    gfx.window.set_window_level(if self.always_on_top {
                        WindowLevel::AlwaysOnTop
                    } else {
                        WindowLevel::Normal
                    });
                }
                self.update_window_title();
                self.show_toast(
                    if self.always_on_top {
                        "Always on top"
                    } else {
                        "Always on top off"
                    }
                    .to_string(),
                );
            }
            (_, Some("p")) | (_, Some("P")) => {
                let want = !self.camera.is_panorama();
                let max_or_fs =
                    self.fullscreen || self.gfx.as_ref().is_some_and(|g| g.window.is_maximized());
                self.camera.set_mode(want);
                // pano → 2D in a normal window: carrying the look direction across
                // pans the image partly off the window, leaving black canvas —
                // which fights "the image is the window". Centre it and re-frame
                // the window to the image instead. Fullscreen / maximized keeps
                // the look so a region under inspection stays put.
                if !want && !max_or_fs {
                    self.camera.center_flat_now();
                    let (dw, dh) = self.frame_dims();
                    self.resize_window_to_image(dw, dh);
                }
                log::info!(
                    "projection -> {}",
                    if self.camera.is_panorama() {
                        "panorama"
                    } else {
                        "2D"
                    }
                );
            }
            (_, Some("w")) | (_, Some("W")) => {
                self.wrap_2d = !self.wrap_2d;
                if !self.wrap_2d {
                    self.normalize_pan_to_canonical();
                }
                log::info!("2D wrap -> {}", self.wrap_2d);
            }
            // B: cycle the background backdrop (configured / black / checkerboard /
            // white). Session-only; not persisted, not part of undo.
            (_, Some("b")) | (_, Some("B")) => {
                self.bg_preset = self.bg_preset.next();
                self.show_toast(format!("Background: {}", self.bg_preset.label()));
            }
            (_, Some("t")) | (_, Some("T")) => self.toggle_view_transform(),
            (_, Some("o")) | (_, Some("O")) => self.open_file_dialog(),
            (_, Some("l")) | (_, Some("L")) => {
                self.locked = !self.locked;
                self.show_toast(if self.locked { "Lock on" } else { "Lock off" }.to_string());
                log::info!("view lock -> {}", self.locked);
            }
            (_, Some("h")) | (_, Some("H")) => {
                self.ui_state.show_help = !self.ui_state.show_help;
            }
            (_, Some("f")) | (_, Some("F")) => self.toggle_fullscreen(),
            (_, Some("i")) | (_, Some("I")) => {
                // Flip from whatever's on screen now (auto or manual) and pin it:
                // the manual choice persists, disabling the >200% auto-switch.
                self.nearest_filter = !self.effective_nearest();
                self.nearest_auto = false;
                self.show_toast(
                    if self.nearest_filter {
                        "Nearest"
                    } else {
                        "Bilinear"
                    }
                    .to_string(),
                );
            }
            (_, Some("s")) | (_, Some("S")) => {
                self.sharpness = !self.sharpness;
                self.show_toast(
                    if self.sharpness {
                        "Sharpness check"
                    } else {
                        "Sharpness off"
                    }
                    .to_string(),
                );
                self.request_redraw();
            }
            // Ctrl+C: copy the composited window render (adjustments, guides,
            // minimap, everything currently on screen) to the clipboard.
            (_, Some("c")) | (_, Some("C")) if ctrl => {
                if self.current_image.is_some() {
                    self.clipboard_copy_pending = true;
                    self.request_redraw();
                }
            }
            (_, Some("c")) | (_, Some("C")) => {
                self.clip_overlay = !self.clip_overlay;
                self.show_toast(
                    if self.clip_overlay {
                        "Clipping overlay on"
                    } else {
                        "Clipping overlay off"
                    }
                    .to_string(),
                );
                self.request_redraw();
            }
            // Ctrl+G: remove one guide subdivision level (undoes a Shift+G step).
            (_, Some("g")) | (_, Some("G")) if ctrl => self.remove_guides_step(),
            // Shift+G: add the next guide subdivision level (old plain-G behaviour).
            (_, Some("g")) | (_, Some("G")) if self.modifiers.shift_key() => {
                self.add_next_guide()
            }
            // G: show/hide the existing guides (adds the first one if there are
            // none yet).
            (_, Some("g")) | (_, Some("G")) => self.toggle_guides_visibility(),
            // M: toggle the navigation minimap (only visible while zoomed in past
            // fit; it also auto-appears on 2D pan/zoom).
            (_, Some("m")) | (_, Some("M")) => {
                self.minimap_on = !self.minimap_on;
                self.show_toast(
                    if self.minimap_on {
                        "Minimap on"
                    } else {
                        "Minimap off"
                    }
                    .to_string(),
                );
            }
            (_, Some("q")) | (_, Some("Q")) => self.escape_or_exit(event_loop),
            (Key::Named(NamedKey::F2), _) => {
                self.show_metadata = !self.show_metadata;
                log::info!("metadata overlay -> {}", self.show_metadata);
            }
            (Key::Named(NamedKey::Home), _) => self.reset_view_full(),
            // Delete: prompt to remove the current file from disk.
            (Key::Named(NamedKey::Delete), _) => {
                if self.loaded_path.is_some() {
                    self.ui_state.confirm_delete = true;
                }
            }
            // Backspace: centre + fit the window at the current zoom (keeps scale).
            (Key::Named(NamedKey::Backspace), _) => self.center_and_fit_window(),
            (Key::Named(NamedKey::F11), _) => self.toggle_fullscreen(),
            (Key::Named(NamedKey::ArrowRight), _) => self.navigate(1),
            (Key::Named(NamedKey::ArrowLeft), _) => self.navigate(-1),
            // Up / Down rotate the image 90° (CCW / CW), remembered per image.
            (Key::Named(NamedKey::ArrowUp), _) => self.rotate_image(-1),
            (Key::Named(NamedKey::ArrowDown), _) => self.rotate_image(1),
            // Space pauses / resumes GIF playback (no-op for static images).
            (Key::Named(NamedKey::Space), _) => self.toggle_animation_pause(),
            (Key::Named(NamedKey::Escape), _) => {
                if self.ui_state.confirm_delete {
                    self.ui_state.confirm_delete = false;
                } else if self.ui_state.show_help {
                    self.ui_state.show_help = false;
                } else {
                    self.escape_or_exit(event_loop);
                }
            }
            // Enter confirms the delete dialog, same as clicking its Delete button.
            (Key::Named(NamedKey::Enter), _) if self.ui_state.confirm_delete => {
                self.ui_state.confirm_delete = false;
                self.delete_current_file();
            }
            _ => return,
        }
        self.request_redraw();
    }

    fn escape_or_exit(&mut self, event_loop: &ActiveEventLoop) {
        if self.fullscreen {
            self.set_fullscreen(false);
        } else {
            event_loop.exit();
        }
    }

    fn toggle_fullscreen(&mut self) {
        self.set_fullscreen(!self.fullscreen);
    }

    fn set_fullscreen(&mut self, on: bool) {
        // Cancel any in-flight window-geometry ease first — it would keep
        // SetWindowPos-ing the window and fight the fullscreen transition.
        self.window_anim_target = None;
        self.fullscreen = on;
        if on {
            // Fit the 2D image to the screen, showing a sub-screen image at native
            // 1:1 rather than magnifying it. Compute against the target monitor's
            // size — for Borderless(None) that's exactly the fullscreen viewport,
            // which the async resize hasn't applied yet. Restart the idle timer.
            if !self.camera.is_panorama() {
                let mon = self
                    .gfx
                    .as_ref()
                    .and_then(|g| g.window.current_monitor())
                    .map(|m| m.size());
                if let Some(s) = mon {
                    let z = self.fit_no_upscale_zoom(s.width.max(1) as f32, s.height.max(1) as f32);
                    self.camera.fit_flat_now(z);
                }
            }
            self.last_cursor_motion = Some(Instant::now());
        } else {
            // Leaving fullscreen: re-frame the window to the *current* image and
            // reset the fit on the restore Resized (its size is stale after
            // navigating in fullscreen, and a fullscreen sub-screen fit zoom would
            // shrink the image in the smaller window). Reveal the cursor.
            self.refit_windowed_pending = true;
            self.show_cursor_now();
        }
        if let Some(gfx) = &self.gfx {
            gfx.window
                .set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
        }
    }

    /// Leave fullscreen for a titlebar drag, repositioning the restored window so
    /// the mouse cursor stays on the titlebar. Without this, `set_fullscreen(false)`
    /// restores the window to its pre-fullscreen position elsewhere on screen, and
    /// the subsequent OS move loop would grab it from there — leaving the window
    /// jumping away from the cursor that is still at the top of the screen.
    fn exit_fullscreen_under_cursor(&mut self) {
        // Global cursor position is independent of the window, but capture it
        // before the restore for clarity.
        let cursor = global_cursor_pos();
        self.set_fullscreen(false);
        // We place the window by hand below, so cancel the restore-time re-centre
        // (`refit_windowed_pending`, just armed by set_fullscreen) — otherwise the
        // next Resized would yank the window to the image-framed centre, away from
        // the cursor mid-drag.
        self.refit_windowed_pending = false;
        let Some((cx, cy)) = cursor else {
            return;
        };
        if let Some(gfx) = &self.gfx {
            // winit applies the fullscreen-exit synchronously on Windows, so the
            // window already has its restored windowed size here.
            let scale = gfx.window.scale_factor();
            let outer = gfx.window.outer_size();
            // Titlebar height in physical px (overlay's TITLEBAR_H is 30 logical pt).
            let tb = (30.0 * scale) as i32;
            // Centre the window horizontally under the cursor and put the cursor in
            // the middle of the titlebar, so the OS move loop grabs it right where
            // the mouse is rather than from the stale pre-fullscreen position.
            let mut x = cx - outer.width as i32 / 2;
            let mut y = cy - tb / 2;
            if let Some(m) = gfx.window.current_monitor() {
                let (mp, ms) = (m.position(), m.size());
                x = x.clamp(mp.x, mp.x + (ms.width as i32 - outer.width as i32).max(0));
                y = y.clamp(mp.y, mp.y + (ms.height as i32 - outer.height as i32).max(0));
            }
            gfx.window.set_outer_position(PhysicalPosition::new(x, y));
        }
    }

    /// 2D zoom that fits the image to the `(vw, vh)` viewport but never magnifies
    /// past native 1:1 (device px). On-screen scale = `zoom * vh / img_h`, so a
    /// fit-scale clamped to ≤ 1 converts to this zoom: a sub-viewport image lands
    /// at native scale, a larger one shrinks to fit.
    fn fit_no_upscale_zoom(&self, vw: f32, vh: f32) -> f32 {
        let (iw, ih) = self.display_dims();
        fit_zoom_no_upscale(vw, vh, iw.max(1) as f32, ih.max(1) as f32)
    }

    /// Centre the 2D image at the fullscreen "fit but don't upscale" zoom (small
    /// images show at 1:1). No-op outside fullscreen or in panorama.
    fn apply_fullscreen_fit(&mut self) {
        if !self.fullscreen || self.camera.is_panorama() {
            return;
        }
        let (vw, vh) = self.viewport();
        let z = self.fit_no_upscale_zoom(vw, vh);
        self.camera.fit_flat_now(z);
    }

    /// Reveal the OS cursor and reset the fullscreen idle-hide state.
    fn show_cursor_now(&mut self) {
        if self.cursor_idle_hidden {
            if let Some(gfx) = &self.gfx {
                gfx.window.set_cursor_visible(true);
            }
            self.cursor_idle_hidden = false;
        }
    }

    /// Fullscreen only: hide the cursor after `CURSOR_IDLE_HIDE` of no real motion,
    /// show it otherwise. Idempotent (only touches the OS state on a change). Driven
    /// from `about_to_wait` (per loop iteration, scheduled to wake at the deadline)
    /// and from the motion handler (instant show). Navigation emits no motion, so a
    /// hidden cursor stays hidden across image changes until the user moves it.
    /// Any in-progress pointer gesture (pan/look, squash-stretch, alt-resize,
    /// guide or minimap drag). The cursor must stay visible through these even if
    /// the user pauses mid-gesture, so they suppress the fullscreen idle-hide.
    fn in_gesture(&self) -> bool {
        self.dragging
            || self.stretching
            || self.alt_resize.is_some()
            || self.guide_drag.is_some()
            || self.minimap_drag
            || self.ui_state.guide_spawn.is_some()
    }

    fn update_cursor_idle_hide(&mut self) {
        let idle = self.fullscreen
            && self.cursor_in_window
            && !self.in_gesture()
            && self
                .last_cursor_motion
                .is_some_and(|t| t.elapsed() >= CURSOR_IDLE_HIDE);
        if idle == self.cursor_idle_hidden {
            return;
        }
        if let Some(gfx) = &self.gfx {
            gfx.window.set_cursor_visible(!idle);
        }
        self.cursor_idle_hidden = idle;
    }

    /// Cut any in-flight zoom/pan/window animation short at its current value so
    /// a new drag (move / pan / resize) takes over immediately, rather than the
    /// animation fighting it or having to finish first.
    fn freeze_animations(&mut self) {
        self.camera.settle();
        self.window_anim_target = None;
    }

    /// Available monitors as `(winit name, friendly label)` for the settings
    /// startup-display picker. Monitors without a name are skipped (can't be
    /// persisted by identity).
    fn monitor_list(&self) -> Vec<(String, String)> {
        let Some(gfx) = &self.gfx else {
            return Vec::new();
        };
        gfx.window
            .available_monitors()
            .enumerate()
            .filter_map(|(i, m)| {
                let name = m.name()?;
                let s = m.size();
                Some((
                    name,
                    format!("Display {} ({}×{})", i + 1, s.width, s.height),
                ))
            })
            .collect()
    }

    /// The image's displayed pixel dimensions — the source dimensions with width
    /// and height swapped for a 90°/270° rotation. All the 2D view maths (aspect,
    /// rulers, fit, minimap) work in displayed space, so they use this, not the raw
    /// `file_info` dimensions.
    fn display_dims(&self) -> (u32, u32) {
        let (w, h) = (self.file_info.width, self.file_info.height);
        if self.rotation % 2 == 1 {
            (h, w)
        } else {
            (w, h)
        }
    }

    /// The displayed aspect (width / height), accounting for rotation.
    fn display_aspect(&self) -> f32 {
        let (w, h) = self.display_dims();
        (w as f32 / h.max(1) as f32).max(1e-4)
    }

    /// Dimensions to frame the window to: the displayed (rotation-aware) image in
    /// 2D; the source 2:1 in panorama (rotation doesn't apply to the sphere).
    fn frame_dims(&self) -> (u32, u32) {
        if self.camera.is_panorama() {
            (self.file_info.width, self.file_info.height)
        } else {
            self.display_dims()
        }
    }

    /// Image uv under a screen-pixel position (physical px), mirroring the
    /// shader's 2D / panorama projection. Used for direct guide hit-testing.
    fn viewport_uv(&self, sx: f64, sy: f64) -> Option<(f32, f32)> {
        use std::f32::consts::{PI, TAU};
        let (vw, vh) = self.viewport();
        if vw <= 0.0 || vh <= 0.0 || self.file_info.width == 0 {
            return None;
        }
        let nx = sx as f32 / vw;
        let ny = sy as f32 / vh;
        let cam = &self.camera.camera;
        match cam {
            Camera::Flat { .. } => {
                let image_aspect = self.display_aspect();
                let inv_zoom = cam.tan_half_fov();
                let s_x = inv_zoom * (vw / vh) / image_aspect / self.image_stretch.x;
                let s_y = inv_zoom / self.image_stretch.y;
                let pan_u = cam.yaw() / TAU;
                let pan_v = -cam.pitch() / PI;
                let u = 0.5 + pan_u + (nx - 0.5) * s_x;
                let v = 0.5 + pan_v - ((1.0 - ny) - 0.5) * s_y;
                Some((u, v))
            }
            Camera::Pano { .. } => {
                let thf = cam.tan_half_fov();
                let ndc_x = (nx * 2.0 - 1.0) / self.image_stretch.x;
                let ndc_y = ((1.0 - ny) * 2.0 - 1.0) / self.image_stretch.y;
                let rx = ndc_x * (vw / vh) * thf;
                let ry = ndc_y * thf;
                let inv = 1.0 / (rx * rx + ry * ry + 1.0).sqrt();
                let (wx, wy, wz) = pano_rotate(cam.yaw(), cam.pitch(), rx * inv, ry * inv, inv);
                let lon = wz.atan2(wx);
                let lat = wy.clamp(-1.0, 1.0).asin();
                Some((1.0 - (lon / TAU + 0.5), 0.5 - lat / PI))
            }
        }
    }

    /// The guide nearest the cursor within 3 screen pixels, or `None`. Works in
    /// 2D and panorama (screen distance via the local uv derivative; longitude
    /// distance is circular in pano). Drives grab/delete and the hover highlight.
    fn guide_at_cursor(&self) -> Option<usize> {
        const GRAB_PX: f32 = 3.0;
        if self.guides.is_empty() || !self.cursor_in_window || !self.guides_visible {
            return None;
        }
        let (cx, cy) = (self.cursor_pos.x, self.cursor_pos.y);
        let (vw, vh) = self.viewport();
        if cx < 0.0 || cy < 0.0 || cx as f32 > vw || cy as f32 > vh {
            return None;
        }
        let uv0 = self.viewport_uv(cx, cy)?;
        let uvx = self.viewport_uv(cx + 1.0, cy)?; // +1px screen-x (for d uv/dx)
        let uvy = self.viewport_uv(cx, cy + 1.0)?; // +1px screen-y (for d uv/dy)
        let pano = self.camera.is_panorama();
        let mut best: Option<(usize, f32)> = None;
        // Seam-unwrap a per-pixel uv.x delta (longitude wraps in panorama).
        let unwrap = |mut d: f32| {
            if pano {
                if d > 0.5 {
                    d -= 1.0;
                } else if d < -0.5 {
                    d += 1.0;
                }
            }
            d
        };
        for (i, g) in self.guides.iter().enumerate() {
            // Screen-pixel distance, normalised by the FULL local gradient
            // (|d/dx| + |d/dy|) of the relevant uv axis — matching the shader's
            // fwidth so a tilted panorama guide is hit where it's actually drawn.
            let dist = if g[1] >= 0.5 {
                // Horizontal guide: constant uv.y.
                let grad = (uvx.1 - uv0.1).abs() + (uvy.1 - uv0.1).abs();
                if grad < 1e-9 {
                    f32::INFINITY
                } else {
                    (uv0.1 - g[0]).abs() / grad
                }
            } else {
                // Vertical guide: constant uv.x (circular distance in pano).
                let grad = unwrap(uvx.0 - uv0.0).abs() + unwrap(uvy.0 - uv0.0).abs();
                let mut delta = (uv0.0 - g[0]).abs();
                if pano {
                    delta = delta.min(1.0 - delta);
                }
                if grad < 1e-9 {
                    f32::INFINITY
                } else {
                    delta / grad
                }
            };
            if dist <= GRAB_PX && best.is_none_or(|(_, b)| dist < b) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Everything the colour-pick tooltip needs that doesn't require a GPU
    /// readback: displayed pixel coords, panorama degrees, and the raw ("Linear")
    /// value straight from the decoded image at the cursor. Combined with a
    /// readback of the just-rendered frame's "Display" value in `render` to build
    /// the frame's `color_pick_last`. `None` when not colour-picking, there's no
    /// image, or the cursor is off a non-wrapping 2D image.
    fn color_pick_partial(&self) -> Option<ColorPickPartial> {
        if !self.color_picking || !self.cursor_in_window {
            return None;
        }
        let img = self.current_image.as_ref()?;
        let (u, v) = self.viewport_uv(self.cursor_pos.x, self.cursor_pos.y)?;
        let pano = self.camera.is_panorama();
        let (u, v) = if pano {
            (u, v)
        } else if self.wrap_2d {
            (u.rem_euclid(1.0), v.rem_euclid(1.0))
        } else if (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v) {
            (u, v)
        } else {
            return None;
        };
        let (dw, dh) = self.display_dims();
        let disp_x = ((u * dw as f32).floor() as i64).clamp(0, dw as i64 - 1);
        let disp_y = ((v * dh as f32).floor() as i64).clamp(0, dh as i64 - 1);
        // Longitude 0..360° left→right, latitude +90° top … -90° bottom (0° at the
        // centre) — the same convention as the guide tooltip's `guide_degrees`.
        let degrees = pano.then_some((u * 360.0, (0.5 - v) * 180.0));
        // Displayed uv -> raw (un-rotated) buffer uv: identity in panorama
        // (rotation is inert there), the shader's `rotate_uv` permutation in 2D.
        let (ru, rv) = if pano {
            (u, v)
        } else {
            rotate_uv(u, v, self.rotation)
        };
        let rx = ((ru * img.width as f32).floor() as i64).clamp(0, img.width as i64 - 1) as u32;
        let ry = ((rv * img.height as f32).floor() as i64).clamp(0, img.height as i64 - 1) as u32;
        let anim_frame = self.anim.as_ref().map(|a| a.frame);
        let raw = img.raw_pixel_at(anim_frame, rx, ry)?;
        // Alpha is never sRGB-encoded (the shader passes `texel.a` straight
        // through), so only the colour channels go through the EOTF.
        let linear = if img.is_encoded_srgb {
            [
                srgb_to_linear_channel(raw[0]),
                srgb_to_linear_channel(raw[1]),
                srgb_to_linear_channel(raw[2]),
                raw[3],
            ]
        } else {
            raw
        };
        Some(ColorPickPartial {
            x: disp_x,
            y: disp_y,
            degrees,
            linear,
        })
    }

    // ---- Navigation minimap ----------------------------------------------

    /// The minimap panel rectangle in physical pixels (top-left origin, matching
    /// `cursor_pos` / `viewport`). Sized to the image aspect, bounded to
    /// [`MINIMAP_MAX`] points on the long side, anchored bottom-right with a
    /// [`MINIMAP_MARGIN`]-point gap. `None` when there's no image or the window is
    /// too small to seat it.
    fn minimap_metrics(&self) -> Option<MinimapMetrics> {
        let gfx = self.gfx.as_ref()?;
        gfx.renderer.image_aspect()?; // None when no image is loaded
                                      // The thumbnail is rendered rotated, so size the panel to the displayed
                                      // (rotation-aware) aspect, not the raw texture aspect.
        let aspect = self.display_aspect();
        if !aspect.is_finite() || aspect <= 0.0 {
            return None;
        }
        let scale = gfx.window.scale_factor() as f32;
        let (vw, vh) = self.viewport();
        let (w_pt, h_pt) = if aspect >= 1.0 {
            (MINIMAP_MAX, MINIMAP_MAX / aspect)
        } else {
            (MINIMAP_MAX * aspect, MINIMAP_MAX)
        };
        // Snap to whole physical pixels so the GL thumbnail rect and the egui
        // border (rect / scale) land on the same edges — otherwise a fractional
        // DPI scale leaves a 1px scene-coloured seam between them.
        let (w, h, margin) = (
            (w_pt * scale).round(),
            (h_pt * scale).round(),
            MINIMAP_MARGIN * scale,
        );
        let x = (vw - margin - w).round();
        let y = (vh - margin - h).round();
        if x < margin || y < margin {
            return None; // window too small to seat the minimap
        }
        Some(MinimapMetrics { x, y, w, h, scale })
    }

    /// Whether the view is zoomed in enough that part of the image is off-screen,
    /// so a minimap is actually useful. In panorama this is always true (you never
    /// see the whole sphere); in 2D it's false at contain-fit and below.
    fn minimap_gated_in(&self) -> bool {
        !self.image_fits_viewport()
    }

    /// Current minimap opacity (0 = hidden). Toggled on → fully opaque; otherwise
    /// the auto-show fade after a 2D pan/zoom. Always 0 unless zoomed in past fit
    /// and an image with room for the panel is present.
    fn minimap_alpha(&self) -> f32 {
        if self.minimap_metrics().is_none() || !self.minimap_gated_in() {
            return 0.0;
        }
        #[cfg(debug_assertions)]
        if let Some(a) = self.debug_minimap_alpha {
            return a;
        }
        if self.minimap_on {
            return 1.0;
        }
        match self.minimap_auto_until {
            Some(t) => {
                let now = Instant::now();
                if now <= t {
                    1.0
                } else {
                    (1.0 - now.duration_since(t).as_secs_f32() / MINIMAP_FADE).clamp(0.0, 1.0)
                }
            }
            None => 0.0,
        }
    }

    /// True when the cursor is over the visible minimap — routes a left-press to
    /// minimap navigation instead of a pan / guide-grab / window-move.
    fn minimap_hit(&self) -> bool {
        if self.minimap_alpha() <= 0.0 {
            return false;
        }
        match self.minimap_metrics() {
            Some(m) => {
                let (cx, cy) = (self.cursor_pos.x as f32, self.cursor_pos.y as f32);
                cx >= m.x && cx <= m.x + m.w && cy >= m.y && cy <= m.y + m.h
            }
            None => false,
        }
    }

    /// True while the auto-shown minimap is still holding or fading (drives the
    /// redraw scheduling so the fade animates even when nothing else moves).
    fn minimap_fading(&self) -> bool {
        if self.minimap_on || !self.minimap_gated_in() {
            return false;
        }
        match self.minimap_auto_until {
            Some(t) => Instant::now() <= t + Duration::from_secs_f32(MINIMAP_FADE),
            None => false,
        }
    }

    /// Half-extent (in image uv) of the on-screen view rectangle along each axis —
    /// mirrors the shader's 2D uv mapping (and `ruler_info`). Used to draw the 2D
    /// view box and to clamp minimap navigation so the view stays on-image.
    fn view_half_extent_uv(&self) -> (f32, f32) {
        let cam = &self.camera.camera;
        let (vw, vh) = self.viewport();
        let image_aspect = self.display_aspect();
        let inv = cam.tan_half_fov();
        let sx = inv * (vw / vh) / image_aspect / self.image_stretch.x;
        let sy = inv / self.image_stretch.y;
        (0.5 * sx, 0.5 * sy)
    }

    /// Snap the view to the image point under a minimap cursor position (physical
    /// px). 2D centres the pan there (eased, clamped on-image unless wrapping);
    /// panorama aims the look at that equirectangular point (instant).
    fn minimap_navigate(&mut self, cursor: PhysicalPosition<f64>) {
        let Some(m) = self.minimap_metrics() else {
            return;
        };
        let u = ((cursor.x as f32 - m.x) / m.w).clamp(0.0, 1.0);
        let v = ((cursor.y as f32 - m.y) / m.h).clamp(0.0, 1.0);
        match self.camera.camera {
            Camera::Flat { .. } => {
                let (mut cu, mut cv) = (u, v);
                if !self.wrap_2d {
                    // Keep the view rectangle inside the image (like a navigator),
                    // so clicking near an edge doesn't reveal letterbox.
                    let (hu, hv) = self.view_half_extent_uv();
                    cu = if hu * 2.0 >= 1.0 {
                        0.5
                    } else {
                        u.clamp(hu, 1.0 - hu)
                    };
                    cv = if hv * 2.0 >= 1.0 {
                        0.5
                    } else {
                        v.clamp(hv, 1.0 - hv)
                    };
                }
                self.camera.set_pan_target(Vec2::new(cu - 0.5, cv - 0.5));
            }
            Camera::Pano { .. } => {
                self.camera.look_at_uv(Vec2::new(u, v));
            }
        }
        self.request_redraw();
    }

    /// Build the UI-side minimap info (panel rect + view outline, in points) for
    /// this frame, or `None` when the minimap is hidden.
    fn minimap_info(&self) -> Option<crate::ui::MinimapInfo> {
        let alpha = self.minimap_alpha();
        if alpha <= 0.0 {
            return None;
        }
        let m = self.minimap_metrics()?;
        let rect = egui::Rect::from_min_size(
            egui::pos2(m.x / m.scale, m.y / m.scale),
            egui::vec2(m.w / m.scale, m.h / m.scale),
        );
        let to_pt = |u: f32, v: f32| {
            egui::pos2(
                rect.min.x + u * rect.width(),
                rect.min.y + v * rect.height(),
            )
        };
        let mut view_segments: Vec<Vec<egui::Pos2>> = Vec::new();
        let mut view_fill: Vec<[egui::Pos2; 3]> = Vec::new();
        match self.camera.camera {
            Camera::Flat { pan, .. } => {
                // The view rectangle in image uv. With wrap on it can run past
                // [0,1]; minimap_spans tiles each axis so the box appears on both
                // edges of the minimap (matching the wrapped image).
                let (hu, hv) = self.view_half_extent_uv();
                let (cu, cv) = (0.5 + pan.x, 0.5 + pan.y);
                let u_spans = self.minimap_spans(cu - hu, cu + hu);
                let v_spans = self.minimap_spans(cv - hv, cv + hv);
                for &(ua, ub) in &u_spans {
                    for &(va, vb) in &v_spans {
                        let (tl, tr) = (to_pt(ua, va), to_pt(ub, va));
                        let (br, bl) = (to_pt(ub, vb), to_pt(ua, vb));
                        view_fill.push([tl, tr, br]);
                        view_fill.push([tl, br, bl]);
                        view_segments.push(vec![tl, tr, br, bl, tl]);
                    }
                }
            }
            Camera::Pano { .. } => {
                view_segments = self.pano_view_segments(&to_pt);
                view_fill = self.pano_view_fill(&to_pt);
            }
        }
        Some(crate::ui::MinimapInfo {
            rect,
            alpha,
            view_segments,
            view_fill,
        })
    }

    /// Split an image-uv interval `[a, b]` (one view-box axis) into the minimap's
    /// local `[0,1]` segments. Without wrap it's the clamped overlap with the image
    /// (a single span). With 2D wrap on it tiles: the parts of the interval that
    /// fall in each unit cell map back into `[0,1]`, so a box that runs off one
    /// edge reappears on the other.
    fn minimap_spans(&self, a: f32, b: f32) -> Vec<(f32, f32)> {
        if a >= b {
            return Vec::new();
        }
        if !self.wrap_2d {
            let (lo, hi) = (a.max(0.0), b.min(1.0));
            return if hi > lo { vec![(lo, hi)] } else { Vec::new() };
        }
        if b - a >= 1.0 {
            return vec![(0.0, 1.0)]; // covers the whole axis
        }
        let mut out = Vec::new();
        let k0 = a.floor() as i32;
        let k1 = (b - 1e-6).floor() as i32;
        for k in k0..=k1 {
            let lo = (a.max(k as f32) - k as f32).clamp(0.0, 1.0);
            let hi = (b.min((k + 1) as f32) - k as f32).clamp(0.0, 1.0);
            if hi > lo {
                out.push((lo, hi));
            }
        }
        out
    }

    /// The panorama view region as a filled triangle mesh on the minimap: project a
    /// grid over the window interior to equirect uv and emit two triangles per cell.
    /// The grid resolution matches the outline ([`PANO_VIEW_SAMPLES`]) so the fill
    /// boundary lands on the outline (no triangular gap). Cells straddling the
    /// longitude seam are unwrapped and emitted at both `u` and `u-1`, so the fill
    /// reaches both minimap edges (the painter clip trims the overflow) instead of
    /// leaving a hole. This fills correctly however concave the region is (e.g. a
    /// steep-pitch view wrapping a pole), where a single boundary polygon would
    /// mis-tessellate.
    fn pano_view_fill(&self, to_pt: &impl Fn(f32, f32) -> egui::Pos2) -> Vec<[egui::Pos2; 3]> {
        let (vw, vh) = self.viewport();
        let (vw, vh) = (vw as f64, vh as f64);
        let n = PANO_VIEW_SAMPLES;
        let stride = n + 1;
        let mut grid: Vec<Option<(f32, f32)>> = Vec::with_capacity(stride * stride);
        for j in 0..=n {
            for i in 0..=n {
                let px = i as f64 / n as f64 * vw;
                let py = j as f64 / n as f64 * vh;
                grid.push(self.viewport_uv(px, py));
            }
        }
        let mut tris: Vec<[egui::Pos2; 3]> = Vec::new();
        let quad = |tris: &mut Vec<[egui::Pos2; 3]>,
                    a: (f32, f32),
                    b: (f32, f32),
                    c: (f32, f32),
                    d: (f32, f32)| {
            let (pa, pb) = (to_pt(a.0, a.1), to_pt(b.0, b.1));
            let (pc, pd) = (to_pt(c.0, c.1), to_pt(d.0, d.1));
            tris.push([pa, pb, pc]);
            tris.push([pa, pc, pd]);
        };
        for j in 0..n {
            for i in 0..n {
                let (Some(a), Some(b), Some(c), Some(d)) = (
                    grid[j * stride + i],
                    grid[j * stride + i + 1],
                    grid[(j + 1) * stride + i + 1],
                    grid[(j + 1) * stride + i],
                ) else {
                    continue;
                };
                let us = [a.0, b.0, c.0, d.0];
                let umin = us.iter().copied().fold(f32::INFINITY, f32::min);
                let umax = us.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if umax - umin > 0.5 {
                    // Straddles the seam: lift the low-u corners by +1 so the cell is
                    // contiguous near u≈1, then emit it AND a copy shifted by −1 so
                    // it fills the panel's right and left edges around the wrap.
                    let un = |p: (f32, f32)| (if p.0 < 0.5 { p.0 + 1.0 } else { p.0 }, p.1);
                    let (a, b, c, d) = (un(a), un(b), un(c), un(d));
                    quad(&mut tris, a, b, c, d);
                    let sh = |p: (f32, f32)| (p.0 - 1.0, p.1);
                    quad(&mut tris, sh(a), sh(b), sh(c), sh(d));
                } else {
                    quad(&mut tris, a, b, c, d);
                }
            }
        }
        tris
    }

    /// The panorama view region as polylines on the minimap: sample the window
    /// border, project each point to equirect uv (the shader's pano mapping, via
    /// [`viewport_uv`](Self::viewport_uv)), map to the panel, and split the path at
    /// the longitude wrap so a wrap-around view doesn't streak across the minimap.
    fn pano_view_segments(&self, to_pt: &impl Fn(f32, f32) -> egui::Pos2) -> Vec<Vec<egui::Pos2>> {
        let (vw, vh) = self.viewport();
        let (vw, vh) = (vw as f64, vh as f64);
        const K: usize = PANO_VIEW_SAMPLES; // samples per window edge (matches the fill)
        let mut border: Vec<(f64, f64)> = Vec::with_capacity(K * 4 + 1);
        for i in 0..K {
            border.push((i as f64 / K as f64 * vw, 0.0)); // top L→R
        }
        for i in 0..K {
            border.push((vw, i as f64 / K as f64 * vh)); // right T→B
        }
        for i in 0..K {
            border.push(((1.0 - i as f64 / K as f64) * vw, vh)); // bottom R→L
        }
        for i in 0..K {
            border.push((0.0, (1.0 - i as f64 / K as f64) * vh)); // left B→T
        }
        border.push((0.0, 0.0)); // close the loop

        let mut segs: Vec<Vec<egui::Pos2>> = Vec::new();
        let mut cur: Vec<egui::Pos2> = Vec::new();
        let mut prev_u: Option<f32> = None;
        let flush = |cur: &mut Vec<egui::Pos2>, segs: &mut Vec<Vec<egui::Pos2>>| {
            if cur.len() >= 2 {
                segs.push(std::mem::take(cur));
            } else {
                cur.clear();
            }
        };
        for (px, py) in border {
            match self.viewport_uv(px, py) {
                Some((u, v)) => {
                    if let Some(pu) = prev_u {
                        if (u - pu).abs() > 0.5 {
                            flush(&mut cur, &mut segs); // longitude wrap: break here
                        }
                    }
                    cur.push(to_pt(u, v));
                    prev_u = Some(u);
                }
                None => {
                    flush(&mut cur, &mut segs);
                    prev_u = None;
                }
            }
        }
        flush(&mut cur, &mut segs);
        segs
    }

    /// True when the window is too small to comfortably overlay the auto-hiding
    /// panels. One predictable threshold for ALL of them (bottom panel, metadata
    /// box, rulers) so a tiny window doesn't keep popping a panel over most of
    /// itself — the titlebar is the only auto-hiding chrome that still reveals.
    fn window_is_small(&self) -> bool {
        let (vw, vh) = self.viewport();
        vw < 550.0 || vh < 400.0
    }

    /// The titlebar reveals only while the cursor is within the window and near
    /// its top edge (so it doesn't cover the image while looking around lower
    /// down). It reveals in fullscreen too, so the window controls and Open /
    /// Settings buttons stay reachable; dragging it there exits fullscreen and
    /// then moves the window (see the `DragWindow` action).
    fn titlebar_should_show(&self) -> bool {
        if cfg!(debug_assertions) && self.force_overlay.as_deref() == Some("titlebar") {
            return true;
        }
        // An explicit, persistent user preference: always shown, nothing suppresses it.
        if self.prefs.pin_titlebar {
            return true;
        }
        // While colour-picking, no auto-hiding chrome may reveal — it would cover
        // the pixel the user is trying to inspect.
        if self.color_picking {
            return false;
        }
        if !self.cursor_in_window {
            return false;
        }
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor())
            .unwrap_or(1.0);
        let near_top = self.cursor_pos.y <= 56.0 * scale;
        // Reveal only on a real mouse move; keep it shown once up (slide > ~0) so a
        // window resize under a stationary cursor doesn't flash it.
        near_top && (self.cursor_moved_by_user || self.titlebar_slide > 0.01)
    }

    /// The image↔screen mapping the rulers need. In 2D it mirrors the shader's UV
    /// mapping so ticks land on real image pixels; in panorama it carries the
    /// sphere projection so the rulers can read out longitude/latitude degrees.
    fn ruler_info(&self) -> Option<crate::ui::RulerInfo> {
        // The mapping is needed whenever either ruler might show; each ruler's
        // own slide (bottom panel / left ruler) gates its actual rendering.
        if self.file_info.width == 0 || self.window_is_small() {
            return None;
        }
        let (vw, vh) = self.viewport();
        let cam = &self.camera.camera;
        let img_w = self.file_info.width as f32;
        let img_h = self.file_info.height as f32;
        if self.camera.is_panorama() {
            // Pano: degree rulers via the projection; the 2D fields go unused.
            return Some(crate::ui::RulerInfo {
                sx: 1.0,
                sy: 1.0,
                pan_u: 0.0,
                pan_v: 0.0,
                img_w,
                img_h,
                pano: Some(crate::ui::PanoProj {
                    yaw: cam.yaw(),
                    pitch: cam.pitch(),
                    tan_half_fov: cam.tan_half_fov(),
                    aspect: (vw / vh).max(1e-4),
                }),
            });
        }
        // 2D: rulers read out the DISPLAYED image (rotation-aware) pixels.
        let (dw, dh) = self.display_dims();
        let (img_w, img_h) = (dw as f32, dh as f32);
        let image_aspect = self.display_aspect();
        let inv_zoom = cam.tan_half_fov();
        Some(crate::ui::RulerInfo {
            sx: inv_zoom * (vw / vh) / image_aspect / self.image_stretch.x,
            sy: inv_zoom / self.image_stretch.y,
            pan_u: cam.yaw() / std::f32::consts::TAU,
            pan_v: -cam.pitch() / std::f32::consts::PI,
            img_w,
            img_h,
            pano: None,
        })
    }

    /// Whether the F2 metadata box should be revealed: toggled on (F2), or the
    /// debounced hover state (`metadata_hover`, set in `tick_metadata` from the
    /// corner hover / pointer-over / open-menu, on a large-enough window).
    fn metadata_should_show(&self) -> bool {
        let forced = cfg!(debug_assertions) && self.force_overlay.as_deref() == Some("metadata");
        self.show_metadata || (self.metadata_hover && !self.window_is_small()) || forced
    }

    /// True in 2D when the whole image is visible (zoom ≤ contain-fit), so it
    /// doesn't overflow the viewport — a body left-drag then moves the window
    /// rather than panning. Always false in panorama mode.
    fn image_fits_viewport(&self) -> bool {
        // Use the zoom *target*, not the in-flight eased value, so a press that
        // lands mid-animation is decided by the destination (consistently a
        // window-move once zoomed out, not a pan).
        let Camera::Flat { zoom, .. } = self.camera.target() else {
            return false;
        };
        let (vw, vh) = self.viewport();
        let fit = (vw / vh / self.display_aspect()).min(1.0);
        // Generous tolerance: in the window-follow's hugging regime the target
        // zoom lands at ~1.0 but drifts by up to ~0.1% from integer window
        // rounding, which a tight threshold would flip in and out of. The next
        // regime (capped / zoomed-in) jumps to ~1.21, so 5% is safely between.
        zoom <= fit * 1.05
    }

    /// The resize direction for the window edge/corner under the cursor, within a
    /// DPI-scaled hit band; `None` in the interior or when maximized/fullscreen.
    /// The top edge is intentionally excluded — the titlebar owns it (a proper
    /// top/border resize comes with the DWM follow-up).
    fn resize_edge_at_cursor(&self) -> Option<ResizeDirection> {
        let gfx = self.gfx.as_ref()?;
        if self.fullscreen || gfx.window.is_maximized() {
            return None;
        }
        let band = 6.0 * gfx.window.scale_factor();
        let size = gfx.window.inner_size();
        let (w, h) = (size.width as f64, size.height as f64);
        let (x, y) = (self.cursor_pos.x, self.cursor_pos.y);
        if x < 0.0 || y < 0.0 || x > w || y > h {
            return None;
        }
        let west = x <= band;
        let east = x >= w - band;
        let south = y >= h - band;
        Some(match (south, west, east) {
            (true, true, _) => ResizeDirection::SouthWest,
            (true, _, true) => ResizeDirection::SouthEast,
            (true, _, _) => ResizeDirection::South,
            (_, true, _) => ResizeDirection::West,
            (_, _, true) => ResizeDirection::East,
            _ => return None,
        })
    }

    /// Begin an OS resize drag if the cursor is on an edge/corner; returns
    /// whether one started.
    fn start_edge_resize(&mut self) -> bool {
        let Some(dir) = self.resize_edge_at_cursor() else {
            return false;
        };
        let started = self
            .gfx
            .as_ref()
            .is_some_and(|g| g.window.drag_resize_window(dir).is_ok());
        if started {
            self.freeze_animations();
        }
        started
    }

    /// Resize direction from which third of the window the cursor is in (for the
    /// Alt+right-drag resize): e.g. middle-right → East, bottom-right → SouthEast.
    /// The centre cell resizes nothing.
    fn resize_third_at_cursor(&self) -> Option<ResizeDirection> {
        let gfx = self.gfx.as_ref()?;
        if gfx.window.is_maximized() {
            return None;
        }
        let size = gfx.window.inner_size();
        let (w, h) = (size.width as f64, size.height as f64);
        let col =
            (self.cursor_pos.x >= w / 3.0) as i32 + (self.cursor_pos.x > 2.0 * w / 3.0) as i32;
        let row =
            (self.cursor_pos.y >= h / 3.0) as i32 + (self.cursor_pos.y > 2.0 * h / 3.0) as i32;
        // col/row: 0 = first third (top/left), 1 = middle, 2 = last third.
        Some(match (row, col) {
            (0, 0) => ResizeDirection::NorthWest,
            (0, 2) => ResizeDirection::NorthEast,
            (2, 0) => ResizeDirection::SouthWest,
            (2, 2) => ResizeDirection::SouthEast,
            (0, 1) => ResizeDirection::North,
            (2, 1) => ResizeDirection::South,
            (1, 0) => ResizeDirection::West,
            (1, 2) => ResizeDirection::East,
            _ => return None, // centre cell
        })
    }

    /// Arm an Alt+right-drag resize from the cursor's third (applied in
    /// `update_alt_resize` as the mouse moves, ended on release). Captures the
    /// origin window rect and cursor screen position. Returns whether armed.
    fn start_third_resize(&mut self) -> bool {
        let Some(dir) = self.resize_third_at_cursor() else {
            return false;
        };
        let Some(gfx) = &self.gfx else {
            return false;
        };
        let Ok(op) = gfx.window.outer_position() else {
            return false;
        };
        let s = gfx.window.outer_size();
        self.alt_resize_origin = (op.x, op.y, s.width, s.height);
        self.alt_resize_press = (
            op.x as f64 + self.cursor_pos.x,
            op.y as f64 + self.cursor_pos.y,
        );
        self.alt_resize = Some(dir);
        // Cut any in-flight zoom animation so it doesn't fight the manual resize.
        self.freeze_animations();
        true
    }

    /// Resize the window so the dragged edge(s) follow the cursor's *screen*
    /// movement since the Alt-resize began. Tracking the cursor position (rather
    /// than accumulating raw device motion) makes it 1:1 with the visible cursor
    /// regardless of pointer speed / DPI. Called from `CursorMoved`.
    fn update_alt_resize(&mut self) {
        let Some(dir) = self.alt_resize else {
            return;
        };
        let applied = {
            let Some(gfx) = &self.gfx else {
                return;
            };
            let Ok(op) = gfx.window.outer_position() else {
                return;
            };
            // Cursor screen position now, vs. at the start of the resize.
            let sx = op.x as f64 + self.cursor_pos.x;
            let sy = op.y as f64 + self.cursor_pos.y;
            let dx = (sx - self.alt_resize_press.0).round() as i32;
            let dy = (sy - self.alt_resize_press.1).round() as i32;
            let (ox, oy, ow, oh) = self.alt_resize_origin;
            use ResizeDirection::*;
            let west = matches!(dir, West | NorthWest | SouthWest);
            let east = matches!(dir, East | NorthEast | SouthEast);
            let north = matches!(dir, North | NorthWest | NorthEast);
            let south = matches!(dir, South | SouthWest | SouthEast);
            // Move the dragged edges from the origin rect by the cursor delta.
            let mut left = ox;
            let mut top = oy;
            let mut right = ox + ow as i32;
            let mut bottom = oy + oh as i32;
            if east {
                right = ox + ow as i32 + dx;
            }
            if west {
                left = ox + dx;
            }
            if south {
                bottom = oy + oh as i32 + dy;
            }
            if north {
                top = oy + dy;
            }
            // Clamp to the minimum, keeping the opposite (fixed) edge in place.
            let min = MIN_DIM as i32;
            if right - left < min {
                if west {
                    left = right - min;
                } else {
                    right = left + min;
                }
            }
            if bottom - top < min {
                if north {
                    top = bottom - min;
                } else {
                    bottom = top + min;
                }
            }
            set_window_outer_rect(
                &gfx.window,
                left,
                top,
                (right - left) as u32,
                (bottom - top) as u32,
            )
        };
        if applied {
            // A deliberate manual resize: keep this size until the next zoom.
            self.manual_window = true;
        }
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        // Don't begin a pan/look gesture when pressing inside the metadata box,
        // so its text stays selectable.
        if state == ElementState::Pressed && self.ui_state.pointer_over_metadata {
            return;
        }
        match (state, button) {
            (ElementState::Pressed, MouseButton::Left) => {
                // A press inside the visible minimap navigates the view and starts
                // a minimap drag — highest priority (it's a corner overlay drawn on
                // top), so it pre-empts guide-grab / pan / window-move.
                if self.minimap_hit() {
                    self.minimap_drag = true;
                    self.minimap_navigate(self.cursor_pos);
                    return;
                }
                // Grabbing a guide takes priority over pan/window-move: if the
                // press lands on a guide line (2D or pano), drag THAT guide and
                // start no pan. Alt-held presses still move the window (so Alt is
                // an escape hatch over a guide).
                if !self.modifiers.alt_key() {
                    if let Some(idx) = self.guide_at_cursor() {
                        self.guide_drag = Some(idx);
                        self.freeze_animations();
                        return;
                    }
                }
                let now = Instant::now();
                let double = self
                    .last_left_press
                    .is_some_and(|t| now.duration_since(t) < DOUBLE_CLICK);
                self.last_left_press = Some(now);
                // Move the window when Alt is held (anywhere) or, in 2D, when the
                // whole image is visible (so a body-drag relocates the window like
                // a titlebar); otherwise pan/look. The window move is deferred to
                // the first motion (see `device_event`) so a stationary click can
                // still toggle fullscreen on a double-click.
                if !self.fullscreen && (self.modifiers.alt_key() || self.image_fits_viewport()) {
                    self.window_drag_armed = true;
                    self.window_drag_motion = 0.0;
                } else {
                    self.start_drag();
                }
                if double {
                    self.pending_dblclick = true;
                    self.dblclick_motion = 0.0;
                }
            }
            (ElementState::Pressed, MouseButton::Right) => {
                // A right-press over the minimap is inert (the overlay captures the
                // pointer). Otherwise: capture the guide under the cursor (if any)
                // but don't delete it yet — a stationary click deletes it on
                // release, while dragging past the threshold instead starts the
                // colour-pick tooltip and the guide survives (see `CursorMoved`
                // and the `Released` arm below). Alt+right is the third-resize,
                // handled earlier in the router, so it never reaches here.
                if self.minimap_hit() {
                    return;
                }
                self.right_press_guide = self.guide_at_cursor();
                self.right_press_pos = Some(self.cursor_pos);
            }
            (ElementState::Pressed, MouseButton::Middle) => {
                // A middle-press over the minimap is inert (the overlay captures the
                // pointer).
                if self.minimap_hit() {
                    return;
                }
                // Alt + middle-drag squashes/stretches the image within the same
                // window (to inspect line straightness); otherwise pan/look.
                if self.modifiers.alt_key() {
                    self.stretching = true;
                } else {
                    self.start_drag();
                }
            }
            (ElementState::Released, MouseButton::Left) => {
                // End a minimap drag (it started no pan/guide gesture).
                if self.minimap_drag {
                    self.minimap_drag = false;
                    self.request_redraw();
                    return;
                }
                // Finish a guide grab: drop it if released off the image (its
                // coord went outside 0..1), else keep it where it landed.
                if let Some(idx) = self.guide_drag.take() {
                    if let Some(g) = self.guides.get(idx).copied() {
                        let horizontal = g[1] >= 0.5;
                        let off = self
                            .viewport_uv(self.cursor_pos.x, self.cursor_pos.y)
                            .is_none_or(|(u, v)| {
                                let c = if horizontal { v } else { u };
                                !(0.0..=1.0).contains(&c)
                            });
                        if off {
                            self.guides.remove(idx);
                        }
                    }
                    self.ui_state.hovered_guide = None;
                    self.request_redraw();
                    return;
                }
                let was_window_drag = self.window_drag_armed;
                self.window_drag_armed = false;
                if !was_window_drag {
                    self.end_drag();
                }
                // A stationary click (window-move armed but never moved, or a
                // pan/look that didn't travel) on a double-click toggles fullscreen.
                if self.pending_dblclick {
                    self.pending_dblclick = false;
                    let moved = if was_window_drag {
                        self.window_drag_motion
                    } else {
                        self.dblclick_motion
                    };
                    if moved < DBLCLICK_DRAG_TOL {
                        self.toggle_fullscreen();
                        self.request_redraw();
                    }
                }
            }
            (ElementState::Released, MouseButton::Middle) => {
                if self.stretching {
                    self.stretching = false;
                } else {
                    self.end_drag();
                }
            }
            (ElementState::Released, MouseButton::Right) => {
                // Past the drag threshold this was a colour-pick, not a click: end
                // it without touching the guide. Otherwise it's a plain click —
                // delete the guide captured at press time, exactly as before.
                if self.color_picking {
                    self.color_picking = false;
                } else if let Some(idx) = self.right_press_guide {
                    self.remove_guide(idx);
                }
                self.right_press_guide = None;
                self.right_press_pos = None;
                self.request_redraw();
            }
            _ => {}
        }
    }

    /// Apply `IMGVWR_DEBUG_*` overrides after load (headless verification only).
    fn apply_debug_overrides(&mut self) {
        // Dev-only: the IMGVWR_DEBUG_* overrides exist purely for headless
        // testing and must not affect release builds.
        if !cfg!(debug_assertions) {
            return;
        }
        let f = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok());
        if let Some(v) = f("IMGVWR_DEBUG_EXPOSURE") {
            self.exposure = v;
            self.exposure_target = v;
        }
        if let Some(v) = f("IMGVWR_DEBUG_GAMMA") {
            self.gamma = v;
            self.gamma_target = v;
        }
        if let Ok(v) = std::env::var("IMGVWR_DEBUG_ISOLATE") {
            if let Ok(c) = v.parse::<i32>() {
                self.isolate_channel = (c >= 0).then_some(c as u8);
            }
        }
        if let Some(v) = f("IMGVWR_DEBUG_CLARITY") {
            self.clarity_amount = v;
        }
        if let Some(v) = f("IMGVWR_DEBUG_CLARITY_RADIUS") {
            self.clarity_radius = v;
        }
        if let Some(v) = f("IMGVWR_DEBUG_STRETCH_X") {
            self.image_stretch.x = v;
        }
        if std::env::var_os("IMGVWR_DEBUG_SHARPNESS").is_some() {
            self.sharpness = true;
        }
        if std::env::var_os("IMGVWR_DEBUG_CLIP").is_some() {
            self.clip_overlay = true;
        }
        if let Some(v) = f("IMGVWR_DEBUG_CLIP_MARGIN") {
            self.prefs.clip_margin = v.clamp(0.0, 1.0);
            self.clip_mask_dirty = true;
        }
        // A/B the Lanczos minification against bilinear for the same 8-bit image.
        if std::env::var_os("IMGVWR_DEBUG_NO_LANCZOS").is_some() {
            self.debug_no_lanczos = true;
        }
        // Force a background preset (the B-key cycle) for headless capture.
        if let Ok(v) = std::env::var("IMGVWR_DEBUG_BG") {
            self.bg_preset = match v.as_str() {
                "black" => BgPreset::Black,
                "checker" => BgPreset::Checker,
                "white" => BgPreset::White,
                _ => BgPreset::UserSetting,
            };
        }
        if let Ok(v) = std::env::var("IMGVWR_DEBUG_ROTATION") {
            if let Ok(r) = v.parse::<i32>() {
                self.rotation = r.rem_euclid(4) as u8;
            }
        }
        // Force the minimap on for headless capture (pair with a zoomed-in
        // IMGVWR_DEBUG_ZOOM so it passes the contain-fit gate).
        if std::env::var_os("IMGVWR_DEBUG_MINIMAP").is_some() {
            self.minimap_on = true;
        }
        // A value in (0,1) also forces that fade alpha, to verify the cross-fade.
        // The forced-alpha field is debug-only, so this is cfg-gated (the whole fn
        // already no-ops in release, but it must still compile there).
        #[cfg(debug_assertions)]
        if let Some(a) = std::env::var("IMGVWR_DEBUG_MINIMAP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            if a > 0.0 && a < 1.0 {
                self.debug_minimap_alpha = Some(a);
            }
        }
        if std::env::var_os("IMGVWR_DEBUG_GUIDES").is_some() {
            self.guides = vec![[0.5, 1.0], [0.5, 0.0], [0.25, 0.0]];
        }
        if let Ok(spec) = std::env::var("IMGVWR_DEBUG_CURSOR") {
            if let Some((x, y)) = spec.split_once(',') {
                if let (Ok(x), Ok(y)) = (x.trim().parse::<f64>(), y.trim().parse::<f64>()) {
                    self.cursor_pos = PhysicalPosition::new(x, y);
                    self.cursor_in_window = true;
                }
            }
        }
        if std::env::var_os("IMGVWR_DEBUG_COLOR_PICK").is_some() {
            self.color_picking = true;
        }
        if std::env::var_os("IMGVWR_DEBUG_CLIPBOARD_COPY").is_some() {
            self.clipboard_copy_pending = true;
        }
        if std::env::var_os("IMGVWR_DEBUG_DELETE_CONFIRM").is_some() {
            self.ui_state.confirm_delete = true;
        }
        if std::env::var_os("IMGVWR_DEBUG_PIN_TITLEBAR").is_some() {
            // In-memory only (no `save()`), so this never touches the real prefs file.
            self.prefs.pin_titlebar = true;
        }
        if let Ok(p) = std::env::var("IMGVWR_DEBUG_PROJECTION") {
            self.camera.set_mode(p.eq_ignore_ascii_case("pano"));
        }
        if let Ok(spec) = std::env::var("IMGVWR_DEBUG_GUIDE_CMD") {
            for cmd in spec.split(',') {
                match cmd.trim() {
                    "g" => self.toggle_guides_visibility(),
                    "shift" => self.add_next_guide(),
                    "ctrl" => self.remove_guides_step(),
                    _ => {}
                }
            }
        }
        if std::env::var("IMGVWR_DEBUG_WRAP").is_ok() {
            self.wrap_2d = true;
        }
        if let Ok(spec) = std::env::var("IMGVWR_DEBUG_VIEW") {
            if let Some((display, view)) = spec.split_once('/') {
                if self.ocio.set_active(display, view) {
                    self.rebuild_ocio();
                    // Exercise the same persistence path as a toolbar selection.
                    self.persist_view_if_panorama();
                } else {
                    log::warn!("IMGVWR_DEBUG_VIEW: no such display/view '{spec}'");
                }
            }
        }
        match &mut self.camera.camera {
            Camera::Pano {
                yaw_rad,
                pitch_rad,
                fov_deg,
            } => {
                if let Some(v) = f("IMGVWR_DEBUG_YAW") {
                    *yaw_rad = v.to_radians();
                }
                if let Some(v) = f("IMGVWR_DEBUG_PITCH") {
                    *pitch_rad = v.to_radians();
                }
                if let Some(v) = f("IMGVWR_DEBUG_FOV") {
                    *fov_deg = v;
                }
            }
            Camera::Flat { zoom, pan } => {
                if let Some(v) = f("IMGVWR_DEBUG_ZOOM") {
                    *zoom = v;
                }
                if let Some(v) = f("IMGVWR_DEBUG_PAN_X") {
                    pan.x = v;
                }
                if let Some(v) = f("IMGVWR_DEBUG_PAN_Y") {
                    pan.y = v;
                }
            }
        }
        self.camera.settle();
    }

    // ---- UI --------------------------------------------------------------

    fn update_window_title(&self) {
        if let Some(gfx) = &self.gfx {
            let mut title = if self.file_info.name.is_empty() {
                "imgvwr".to_string()
            } else {
                format!("{} · imgvwr", self.file_info.name)
            };
            if self.always_on_top {
                title.push_str(" (Always on Top [A])");
            }
            gfx.window.set_title(&title);
        }
    }

    fn ui_inputs(&self) -> UiInputs {
        let display_views = self
            .ocio
            .display_views()
            .iter()
            .map(|dv| (dv.display.clone(), dv.view.clone()))
            .collect();
        let active = self
            .ocio
            .active()
            .map(|dv| (dv.display.clone(), dv.view.clone()));
        let has_image = self
            .gfx
            .as_ref()
            .map(|g| g.renderer.has_image())
            .unwrap_or(false);

        // Overlay state, with headless-test overrides applied.
        let forced = self.force_overlay.as_deref();
        let busy = self.is_busy() || forced == Some("loading");
        // Reveal the loading bar only once a load has taken a moment, so it doesn't
        // flicker on-screen for a small/fast image (the delay window shows neither
        // bar nor hint, just the backdrop). A forced overlay skips the delay.
        let loading =
            busy && (forced == Some("loading") || self.load_start.elapsed() >= LOADING_BAR_DELAY);
        // One continuous bar across both phases: the file read is the first
        // 0→80% and the GPU upload is the final 80→100%, so it fills straight
        // through instead of zipping to full then restarting. The decode phase is
        // determinate only when the format streams through the counting reader
        // (e.g. a big file off a network drive); otherwise it's indeterminate.
        let progress = if self.pending.is_some() {
            Some(0.8 + self.upload_progress.clamp(0.0, 1.0) * 0.2)
        } else if loading {
            self.decode_progress.fraction().map(|f| f * 0.8)
        } else {
            None
        };
        let error = match &self.load_state {
            LoadState::Failed(e) => Some(e.clone()),
            _ if forced == Some("error") => Some("Example decode error: unsupported format".into()),
            _ => None,
        };
        let show_hint = (!has_image && !busy && error.is_none()) || forced == Some("hint");

        UiInputs {
            bottom_slide: self.bottom_slide,
            has_image,
            display_views,
            active,
            ocio_available: !self.ocio.display_views().is_empty(),
            // The bottom-panel sliders show/control the dialed targets.
            exposure: self.exposure_target,
            gamma: self.gamma_target,
            clarity_amount: self.clarity_amount,
            clarity_radius: self.clarity_radius,
            ruler: self.ruler_info(),
            left_ruler_slide: self.left_ruler_slide,
            // Hidden guides (G) vanish from the F2 list and the tooltip along with
            // the render + hit-testing (`guide_at_cursor`) — one flag, one behaviour.
            guides: if self.guides_visible {
                self.guides.clone()
            } else {
                Vec::new()
            },
            // Displayed (rotation-aware) dims: guides are in displayed uv, so the
            // meta-box pixel readout (`V 425px`) reports displayed pixels.
            image_size: self.display_dims(),
            loading,
            progress,
            loading_name: self.pending_name.clone(),
            error,
            show_hint,
            metadata_slide: self.metadata_slide,
            metadata: self.metadata_lines(),
            channel_count: self.file_info.channels,
            isolate_channel: self.isolate_channel,
            // One frame behind (measured inside the gfx borrow, below), exactly
            // like the colour-pick readback.
            histogram: self.histogram.clone(),
            histogram_scale: self.prefs.histogram_scale,
            show_help: self.ui_state.show_help || forced == Some("help"),
            toast: self.toast_render(),
            slot_labels: self.slot_labels(),
            active_slot: self.active_slot,
            diff_slot: self.diff_slot,
            titlebar_slide: self.titlebar_slide,
            title: {
                // The borderless titlebar shows the filename; flag always-on-top
                // here too (the OS title is invisible on a borderless window).
                let mut t = self.file_info.name.clone();
                if self.always_on_top {
                    if t.is_empty() {
                        t.push_str("imgvwr");
                    }
                    t.push_str("  (Always on Top [A])");
                }
                t
            },
            icon: self.titlebar_icon.clone(),
            monitors: self.monitor_list(),
            startup_display: self.prefs.startup_monitor.clone(),
            corner_radius: self.prefs.corner_radius,
            auto_exposure: self.prefs.auto_exposure,
            raw_auto_exposure: self.prefs.raw_auto_exposure,
            clip_margin: self.prefs.clip_margin,
            half_float_textures: self.prefs.half_float_textures,
            pin_titlebar: self.prefs.pin_titlebar,
            available_update: self.available_update.clone(),
            default_view_transform: self.prefs.default_view_transform.clone(),
            guide_color: self.prefs.guide_color,
            background_color: self.prefs.background_color,
            is_maximized: self.gfx.as_ref().is_some_and(|g| g.window.is_maximized()),
            is_fullscreen: self.fullscreen,
            resize_cursor: if self.guide_drag.is_some() {
                Some(egui::CursorIcon::Grabbing)
            } else if self.dragging || self.window_drag_armed {
                None
            } else if let Some(d) = self.alt_resize.or_else(|| self.resize_edge_at_cursor()) {
                // The active Alt-resize direction (if dragging), else the edge
                // under the cursor — these win over a guide hover at the border.
                Some(match d {
                    ResizeDirection::East | ResizeDirection::West => {
                        egui::CursorIcon::ResizeHorizontal
                    }
                    ResizeDirection::North | ResizeDirection::South => {
                        egui::CursorIcon::ResizeVertical
                    }
                    ResizeDirection::NorthEast | ResizeDirection::SouthWest => {
                        egui::CursorIcon::ResizeNeSw
                    }
                    ResizeDirection::NorthWest | ResizeDirection::SouthEast => {
                        egui::CursorIcon::ResizeNwSe
                    }
                })
            } else if self.ui_state.hovered_guide.is_some() {
                // Hovering a guide line → the grab hand.
                Some(egui::CursorIcon::Grab)
            } else {
                None
            },
            minimap: self.minimap_info(),
            color_pick: self.color_pick_last,
        }
    }

    /// Per-slot hover labels: the filename, or — when two saved slots share a
    /// filename — the path portion that differs (relative to the group's common
    /// ancestor, e.g. `a/b/c.jpg` vs `x/b/c.jpg`).
    fn slot_labels(&self) -> [Option<String>; 9] {
        let saved: Vec<(usize, &Path)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|d| (i, d.path.as_path())))
            .collect();
        std::array::from_fn(|i| {
            let path = self.slots[i].as_ref()?.path.as_path();
            let group: Vec<&Path> = saved
                .iter()
                .filter(|(_, q)| q.file_name() == path.file_name())
                .map(|(_, q)| *q)
                .collect();
            Some(if group.len() > 1 {
                disambiguated_path(path, &group)
            } else {
                path.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
        })
    }

    /// Key/value lines for the F2 metadata HUD (§12.3).
    fn metadata_lines(&self) -> Vec<(String, String)> {
        let fi = &self.file_info;
        if fi.width == 0 {
            return Vec::new();
        }
        let mut lines = vec![
            ("File".into(), fi.name.clone()),
            ("Size".into(), format!("{}×{}", fi.width, fi.height)),
            ("Type".into(), fi.dtype.clone()),
            ("Compression".into(), fi.compression.clone()),
            (
                // Reflect the current view mode, not the image's intrinsic type
                // (a 2:1 panorama can be viewed in 2D via the P key).
                "Mode".into(),
                if self.camera.is_panorama() {
                    "Panorama".into()
                } else {
                    "2D".into()
                },
            ),
        ];

        // Camera EXIF (RAW files): append only the fields the file actually
        // carries, so a sparse camera doesn't show blank rows.
        if let Some(cam) = &fi.camera {
            let body = match (&cam.make, &cam.model) {
                (Some(make), Some(model)) => {
                    // Some models already include the maker; avoid "Nikon NIKON Z6".
                    if model.to_lowercase().contains(&make.to_lowercase()) {
                        Some(model.clone())
                    } else {
                        Some(format!("{make} {model}"))
                    }
                }
                (Some(s), None) | (None, Some(s)) => Some(s.clone()),
                (None, None) => None,
            };
            if let Some(body) = body {
                lines.push(("Camera".into(), body));
            }
            if let Some(lens) = &cam.lens {
                lines.push(("Lens".into(), lens.clone()));
            }
            if let Some(iso) = cam.iso {
                lines.push(("ISO".into(), format!("{iso:.0}")));
            }
            if let Some(shutter) = cam.shutter {
                lines.push((
                    "Shutter".into(),
                    crate::image_loader::CameraMeta::shutter_display(shutter),
                ));
            }
            if let Some(aperture) = cam.aperture {
                lines.push(("Aperture".into(), format!("f/{aperture:.1}")));
            }
            if let Some(focal) = cam.focal_len {
                lines.push(("Focal length".into(), format!("{focal:.0} mm")));
            }
        }

        lines
        // ("View" is a dropdown and "Channels" are boxes — both rendered
        // directly in the metadata HUD, not as plain key/value text.)
    }

    /// Reveal the bottom panel when the cursor is near the window's bottom edge
    /// (or hovering the panel itself); hide it shortly after the cursor leaves.
    fn tick_bottom_panel(&mut self) {
        if self.force_bottom {
            self.bottom_visible = true;
            return;
        }
        // While colour-picking, no auto-hiding chrome may reveal — it would cover
        // the pixel the user is trying to inspect.
        if self.color_picking {
            self.bottom_visible = false;
            self.bottom_hide_deadline = None;
            return;
        }
        let (scale, vh) = self
            .gfx
            .as_ref()
            .map(|g| {
                (
                    g.window.scale_factor() as f32,
                    g.window.inner_size().height as f64,
                )
            })
            .unwrap_or((1.0, 0.0));
        let near_bottom = self.cursor_in_window && self.cursor_pos.y >= vh - (44.0 * scale) as f64;
        // The near-edge reveal only fires on a real mouse move (or to keep an
        // already-shown panel up); a window-follow resize sliding the bottom edge
        // under a stationary cursor must not pop it. Hovering the panel itself
        // (egui) always keeps it up.
        let reveal_edge = near_bottom && (self.bottom_visible || self.cursor_moved_by_user);
        if (reveal_edge || self.ui_state.pointer_over_panel) && !self.window_is_small() {
            self.bottom_visible = true;
            self.bottom_hide_deadline = None;
        } else if self.bottom_visible {
            match self.bottom_hide_deadline {
                None => {
                    self.bottom_hide_deadline = Some(Instant::now() + Duration::from_millis(100));
                }
                Some(t) if Instant::now() >= t => {
                    self.bottom_visible = false;
                    self.bottom_hide_deadline = None;
                }
                Some(_) => {}
            }
        }
    }

    /// Reveal the left ruler near the left edge, and keep it up while the cursor
    /// is over it so a guide can be dragged off. 2D only. Deliberately NOT tied to
    /// the bottom panel: hovering the bottom edge shows only the bottom ruler.
    fn tick_left_ruler(&mut self) {
        if self.color_picking {
            self.left_ruler_visible = false;
            self.left_ruler_hide_deadline = None;
            return;
        }
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor() as f32)
            .unwrap_or(1.0);
        let near_left = self.cursor_in_window && self.cursor_pos.x <= (44.0 * scale) as f64;
        // Reveal on a real mouse move only (or keep it up once shown); a window
        // resize sliding the left edge under a stationary cursor must not pop it.
        let reveal_edge = near_left && (self.left_ruler_visible || self.cursor_moved_by_user);
        // Rulers show in both 2D (pixels) and panorama (degrees).
        let eligible = !self.window_is_small() && self.file_info.width != 0;
        let show = eligible && (reveal_edge || self.ui_state.pointer_over_left_ruler);
        if show {
            self.left_ruler_visible = true;
            self.left_ruler_hide_deadline = None;
        } else if self.left_ruler_visible {
            match self.left_ruler_hide_deadline {
                None => {
                    self.left_ruler_hide_deadline =
                        Some(Instant::now() + Duration::from_millis(100));
                }
                Some(t) if Instant::now() >= t => {
                    self.left_ruler_visible = false;
                    self.left_ruler_hide_deadline = None;
                }
                Some(_) => {}
            }
        }
    }

    /// Temporarily reveal the F2 metadata box when the cursor is near the
    /// top-right corner (or hovering the box, or while its View menu is open).
    fn tick_metadata(&mut self) {
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor() as f32)
            .unwrap_or(1.0);
        let (vw, _) = self.viewport();
        // Keep the box up for a grace period while/after the View menu is open so
        // moving into the (popup) menu doesn't dismiss the box under the cursor.
        if self.ui_state.view_menu_open {
            self.metadata_menu_grace = Some(Instant::now() + Duration::from_millis(400));
        }
        let menu_sticky = self.ui_state.view_menu_open
            || self.metadata_menu_grace.is_some_and(|t| Instant::now() < t);
        // A small top-right corner triangle (~80px legs): reveal only when the
        // cursor is inside the diagonal from (w-80, 0) to (w, 80).
        let edge = (80.0 * scale) as f64;
        let near_corner = self.cursor_in_window
            && (vw as f64 - self.cursor_pos.x) + self.cursor_pos.y <= edge
            && !self.window_is_small()
            && !self.color_picking;
        // The corner reveal only fires on a real mouse move (or to keep the box
        // up once shown); a window-follow resize sliding the top-right corner
        // under a stationary cursor must not pop it. Hovering the box / an open
        // menu always keep it up.
        let reveal_edge = near_corner && (self.metadata_hover || self.cursor_moved_by_user);
        if reveal_edge || self.ui_state.pointer_over_metadata || menu_sticky {
            self.metadata_hover = true;
            self.metadata_hide_deadline = None;
        } else if self.metadata_hover {
            // Debounced collapse: a generous delay so micro-movements near the
            // reveal edge don't spam the box open/closed when the mouse is still.
            match self.metadata_hide_deadline {
                None => {
                    self.metadata_hide_deadline = Some(Instant::now() + Duration::from_millis(350));
                }
                Some(t) if Instant::now() >= t => {
                    self.metadata_hover = false;
                    self.metadata_hide_deadline = None;
                }
                Some(_) => {}
            }
        }
    }

    fn handle_ui_action(&mut self, action: UiAction) {
        match action {
            UiAction::OpenFile => self.open_file_dialog(),
            UiAction::SetView { display, view } => {
                if self.ocio.set_active(&display, &view) {
                    log::info!("view transform -> {display}/{view}");
                    self.rebuild_ocio();
                    self.persist_view_if_panorama();
                    self.show_toast(view.clone());
                }
            }
            UiAction::SetDefaultView(view) => {
                self.prefs.default_view_transform = view.clone();
                self.prefs.save();
                self.show_toast(format!("Default view: {view}"));
            }
            UiAction::DismissError => {
                self.load_state = LoadState::Idle;
                self.request_redraw();
            }
            UiAction::CloseHelp => {
                self.ui_state.show_help = false;
                self.request_redraw();
            }
            UiAction::DeleteCurrentFile => self.delete_current_file(),
            // A flag click keeps the window size so the flags don't reflow.
            UiAction::RecallSlot(i) => self.recall_slot(i + 1, false),
            UiAction::SetDefaultApp => match register_default_app() {
                Ok(n) => self.show_toast(format!("Default viewer for {n} file types")),
                Err(e) => {
                    log::error!("set-default failed: {e}");
                    self.show_toast("Could not set default".to_string());
                }
            },
            UiAction::SetStartupDisplay(name) => {
                self.prefs.startup_monitor = name;
                self.prefs.save();
                self.show_toast(
                    if self.prefs.startup_monitor.is_some() {
                        "Startup display saved"
                    } else {
                        "Will remember last position"
                    }
                    .to_string(),
                );
            }
            UiAction::SetCornerRadius(radius) => {
                self.prefs.corner_radius = radius;
                self.prefs.save();
                // Apply live (unless fullscreen/maximized, which stay square).
                if let Some(gfx) = &self.gfx {
                    let rounded = !self.fullscreen && !gfx.window.is_maximized();
                    apply_window_corners(&gfx.window, if rounded { radius } else { 0 });
                }
            }
            UiAction::SetBackgroundColor(color) => {
                self.prefs.background_color = color;
                self.prefs.save();
                // Show the just-picked colour live even if a B-key preset was active.
                self.bg_preset = BgPreset::UserSetting;
                self.request_redraw();
            }
            UiAction::SetChannelIsolate(channel) => {
                self.isolate_channel = channel;
                self.request_redraw();
            }
            UiAction::SetHistogramScale(scale) => {
                // Purely how the same counts are drawn — no re-measurement.
                self.prefs.histogram_scale = scale;
                self.prefs.save();
                self.request_redraw();
            }
            UiAction::SetExposure(v) => {
                self.exposure_target = v.clamp(-16.0, 16.0);
                self.request_redraw();
            }
            UiAction::SetGamma(v) => {
                self.gamma_target = v.clamp(0.1, 4.0);
                self.request_redraw();
            }
            UiAction::SetClarity(v) => {
                self.clarity_amount = v.clamp(0.0, 10.0);
                self.request_redraw();
            }
            UiAction::SetClarityRadius(v) => {
                self.clarity_radius = v.clamp(8.0, 256.0);
                self.request_redraw();
            }
            UiAction::AddGuide { coord, horizontal } => self.add_guide(coord, horizontal),
            UiAction::MoveGuide { index, coord } => {
                if let Some(g) = self.guides.get_mut(index) {
                    g[0] = coord.clamp(0.0, 1.0);
                    self.request_redraw();
                }
            }
            UiAction::RemoveGuide(i) => self.remove_guide(i),
            UiAction::ResetAdjustments => self.reset_image_processing(),
            UiAction::SetAutoExposure(on) => {
                self.prefs.auto_exposure = on;
                self.prefs.save();
                self.show_toast(
                    if on {
                        "Auto-exposure on"
                    } else {
                        "Auto-exposure off"
                    }
                    .to_string(),
                );
            }
            UiAction::SetRawAutoExposure(on) => {
                self.prefs.raw_auto_exposure = on;
                self.prefs.save();
                self.show_toast(
                    if on {
                        "RAW auto-exposure on"
                    } else {
                        "RAW auto-exposure off"
                    }
                    .to_string(),
                );
            }
            UiAction::SetGuideColor(c) => {
                self.prefs.guide_color = c;
                self.prefs.save();
                self.request_redraw();
            }
            UiAction::SetClipMargin(m) => {
                self.prefs.clip_margin = m.clamp(0.0, 0.05);
                self.prefs.save();
                self.clip_mask_dirty = true; // margin is baked into the mask
                self.request_redraw();
            }
            UiAction::SetHalfFloat(on) => {
                self.prefs.half_float_textures = on;
                self.prefs.save();
                // Applies to the next upload — `set_half_float` doesn't touch the
                // resident texture. (Re-uploading the current image in place was
                // removed: it ran through finalize_adopt, which wipes the per-image
                // undo history, and the new-precision upload could OOM and blank the
                // on-screen image — a destructive surprise from flipping a setting.)
                if let Some(gfx) = &mut self.gfx {
                    gfx.renderer.set_half_float(on);
                }
                self.show_toast(
                    if on {
                        "Float images stored as 16-bit (applies on next open)"
                    } else {
                        "Float images stored at full precision (on next open)"
                    }
                    .to_string(),
                );
            }
            UiAction::SetPinTitlebar(on) => {
                self.prefs.pin_titlebar = on;
                self.prefs.save();
                self.request_redraw();
            }
            UiAction::OpenSettings => {
                self.ui_state.show_settings = true;
                self.ui_state.confirm_default = false;
            }
            // Borderless titlebar controls.
            UiAction::DragWindow => {
                self.freeze_animations();
                // Dragging the titlebar in fullscreen leaves fullscreen first and
                // then moves the window — but the restored window must land under
                // the cursor (with the titlebar still grabbed), not back at its
                // stale pre-fullscreen position somewhere else on screen.
                if self.fullscreen {
                    self.exit_fullscreen_under_cursor();
                }
                if let Some(gfx) = &self.gfx {
                    let _ = gfx.window.drag_window();
                }
            }
            UiAction::Minimize => {
                if let Some(gfx) = &self.gfx {
                    gfx.window.set_minimized(true);
                }
            }
            UiAction::ToggleMaximize => {
                // Maximize is incompatible with fullscreen (the titlebar — hence
                // this button — is now reachable in fullscreen); leave fullscreen
                // first so the OS window and `self.fullscreen` don't disagree.
                if self.fullscreen {
                    self.set_fullscreen(false);
                }
                if let Some(gfx) = &self.gfx {
                    let max = gfx.window.is_maximized();
                    gfx.window.set_maximized(!max);
                }
            }
            UiAction::ToggleFullscreen => self.toggle_fullscreen(),
            UiAction::Close => self.should_exit = true,
        }
    }

    fn open_file_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter("Images", &supported_extensions());
        // Open in the folder of the image currently on screen, so browsing starts
        // where the user is looking (falls back to rfd's default when nothing is
        // loaded or the path has no parent).
        if let Some(dir) = self.loaded_path.as_deref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let file = dialog.pick_file();
        if let Some(path) = file {
            // A manual open ends any arrow-nav preload chain (saved comparator
            // slots persist; only the A/B scratch is dropped).
            self.preload_armed = false;
            self.nav_pending = None;
            self.image_cache.clear();
            self.compare_prev = None;
            self.load_path(path);
        }
    }

    // ---- render ----------------------------------------------------------

    fn is_loading(&self) -> bool {
        matches!(self.load_state, LoadState::Loading)
    }

    fn capture_active(&self) -> bool {
        self.capture.as_ref().is_some_and(|c| !c.done)
    }

    /// Busy = decoding on a thread or uploading to the GPU.
    fn is_busy(&self) -> bool {
        self.is_loading() || self.pending.is_some()
    }

    fn capture_ready(&self) -> bool {
        self.capture.as_ref().is_some_and(|c| {
            let elapsed = c.start.elapsed();
            !c.done
                && elapsed >= c.delay
                && (!self.is_busy() || elapsed >= c.delay + CAPTURE_LOAD_CAP)
        })
    }

    /// Toggle GIF play/pause (Space). No-op for a static image. On resume, the
    /// paused frame is held for its full delay rather than flipping immediately.
    fn toggle_animation_pause(&mut self) {
        if self.anim.is_none() {
            return;
        }
        // The resume delay is the paused frame's own duration; read it before the
        // mutable borrow of `self.anim`.
        let resume_delay = self.current_image.as_ref().and_then(|img| {
            let frame = self.anim.as_ref()?.frame;
            img.animation.as_ref()?.frames.get(frame).map(|f| f.delay)
        });
        let anim = self.anim.as_mut().unwrap();
        anim.paused = !anim.paused;
        let paused = anim.paused;
        if !paused {
            if let Some(d) = resume_delay {
                anim.next_at = Instant::now() + d;
            }
        }
        self.show_toast(if paused { "Paused" } else { "Playing" }.to_string());
    }

    /// Advance animated-GIF playback: when the current frame's delay has elapsed,
    /// step to the next frame (wrapping) and upload it to the texture. No-op when
    /// the current image isn't an animated GIF or playback is paused.
    fn advance_animation(&mut self, now: Instant) {
        match self.anim.as_ref() {
            Some(a) if !a.paused && now >= a.next_at => {}
            _ => return,
        }
        // The frames live in the current image; clone the Arc so the borrow of
        // `self.current_image` ends before we touch `self.gfx`. If the backing
        // frames are somehow gone (shouldn't happen — `anim` and `current_image`
        // are set together), clear `anim` so the scheduler stops waking for it
        // rather than spinning on a permanently-past `next_at`.
        let Some(img) = self.current_image.clone() else {
            self.anim = None;
            return;
        };
        let frames = match img.animation.as_ref() {
            Some(a) if a.frames.len() > 1 => &a.frames,
            _ => {
                self.anim = None;
                return;
            }
        };
        let n = frames.len();
        let anim = self.anim.as_mut().expect("anim present (checked above)");
        anim.frame = (anim.frame + 1) % n;
        let idx = anim.frame;
        anim.next_at = now + frames[idx].delay;
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.renderer.update_animation_frame(
                img.width as i32,
                img.height as i32,
                &frames[idx].pixels,
            );
        }
        // Different pixels are on screen now, so the histogram describes the
        // previous frame.
        self.invalidate_histogram();
    }

    fn render(&mut self) -> RenderOutcome {
        // Advance any in-progress incremental upload before drawing this frame.
        if self.pending.is_some() {
            self.pump_upload();
        }
        // Advance the zoom/pan easing toward the target (frame-rate independent;
        // dt is clamped so a long idle gap can't cause a jump). `animating`
        // drives the redraw scheduling in `about_to_wait`. The window-geometry
        // ease is NOT advanced here — it rides its own Resized-event chain so the
        // window resize and the content present stay one-to-one (see ease_window).
        let now = Instant::now();
        // Advance animated-GIF playback (uploads the next frame when its delay has
        // elapsed). No-op for static images and while paused.
        self.advance_animation(now);
        let dt = self
            .last_frame
            .replace(now)
            .map(|prev| now.saturating_duration_since(prev).as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1);
        let cam_moving = self.camera.animate(dt);
        let tone_moving = self.animate_tone(dt);
        // Auto-show the navigation minimap on any pan/zoom/look-around (both 2D
        // and panorama). Comparing the rendered camera each frame catches drags,
        // wheel, numpad, eased animation and minimap-drag navigation in one place.
        // The mode tag means a P-toggle (mode change) isn't mistaken for a move.
        let cur = match self.camera.camera {
            Camera::Flat { pan, zoom } => (1u8, pan.x, pan.y, zoom),
            Camera::Pano {
                yaw_rad,
                pitch_rad,
                fov_deg,
            } => (0u8, yaw_rad, pitch_rad, fov_deg),
        };
        if let Some(p) = self.minimap_prev_view {
            let moved = p.0 == cur.0
                && ((p.1 - cur.1).abs() > 1e-5
                    || (p.2 - cur.2).abs() > 1e-5
                    || (p.3 - cur.3).abs() > 1e-4);
            if moved {
                self.minimap_auto_until = Some(now + MINIMAP_HOLD);
            }
        }
        self.minimap_prev_view = Some(cur);
        // Dev-only: force the settings dialog open for headless verification.
        #[cfg(debug_assertions)]
        if self.force_overlay.as_deref() == Some("settings") {
            self.ui_state.show_settings = true;
        }
        self.tick_bottom_panel();
        self.tick_left_ruler();
        self.tick_metadata();

        // Slide the auto-hiding panels in/out from their edges over SLIDE_SECS,
        // toward each panel's current visibility target.
        let tb_t = self.titlebar_should_show() as i32 as f32;
        let md_t = self.metadata_should_show() as i32 as f32;
        let bp_t = self.bottom_visible as i32 as f32;
        let lr_t = self.left_ruler_visible as i32 as f32;
        self.titlebar_slide = approach(self.titlebar_slide, tb_t, dt);
        self.metadata_slide = approach(self.metadata_slide, md_t, dt);
        self.bottom_slide = approach(self.bottom_slide, bp_t, dt);
        self.left_ruler_slide = approach(self.left_ruler_slide, lr_t, dt);
        let slides_moving = self.titlebar_slide != tb_t
            || self.metadata_slide != md_t
            || self.bottom_slide != bp_t
            || self.left_ruler_slide != lr_t;

        // Keep scheduling timed frames while the camera, tone, or panels are
        // moving (the window geometry self-drives via Resized events, so it's not
        // included here — that's what keeps it from double-presenting).
        self.animating = cam_moving || tone_moving || slides_moving;

        // (Re)build the clipping-overlay max-mip mask if it's stale and the
        // overlay is on (lazy — nothing is built while the overlay is off).
        if self.clip_overlay && self.clip_mask_dirty {
            if let (Some(img), Some(gfx)) = (self.current_image.clone(), self.gfx.as_mut()) {
                gfx.renderer
                    .set_clip_mask(Some(&img), self.prefs.clip_margin);
            }
            self.clip_mask_dirty = false;
        }

        // Display histogram (F2 box): re-measure when the graph no longer matches
        // the tone state, and only while the box is actually on screen — exactly
        // the same laziness as the clip mask above. `wanted` carries the image's
        // raw (unrotated) dimensions, which is what sizes the sample grid; a
        // rotation permutes pixels but cannot change their values.
        let histogram_want = if self.metadata_slide > 0.001 {
            let key = HistogramKey {
                epoch: self.histogram_epoch,
                exposure: self.exposure,
                gamma: self.gamma,
            };
            self.current_image
                .as_ref()
                .filter(|_| self.histogram_key != Some(key))
                .map(|img| (key, img.width as i32, img.height as i32))
        } else {
            None
        };

        // Gather everything the frame needs before the mutable gfx/ui borrows.
        let inputs = self.ui_inputs();
        let cam = self.camera.camera;
        let mut guide_arr = [[0.0f32; 2]; crate::renderer::MAX_GUIDES];
        // Hidden guides (G) don't draw at all.
        let guide_n = if self.guides_visible {
            self.guides.len().min(crate::renderer::MAX_GUIDES)
        } else {
            0
        };
        guide_arr[..guide_n].copy_from_slice(&self.guides[..guide_n]);
        let (bg_color, bg_checker) = self.bg_preset.resolve(self.prefs.background_color);
        let base = RenderParams {
            viewport: (1, 1),
            exposure: self.exposure,
            gamma: self.gamma,
            projection_mode: cam.projection_mode(),
            yaw: cam.yaw(),
            pitch: cam.pitch(),
            half_fov_radians: cam.half_fov_radians(),
            tan_half_fov: cam.tan_half_fov(),
            wrap_2d: self.wrap_2d,
            nearest: self.effective_nearest(),
            background: bg_color,
            bg_checker,
            isolate_channel: self.isolate_channel.map(|c| c as i32).unwrap_or(-1),
            stretch: [self.image_stretch.x, self.image_stretch.y],
            sharpness: self.sharpness,
            diff: self.diff_slot.is_some(),
            guides: guide_arr,
            guide_count: guide_n as i32,
            guide_color: srgb_u8_to_f32(self.prefs.guide_color),
            // Hovered guide (from last frame's egui pass) gets the inverse-hue
            // colour so it stands out under the grab cursor.
            guide_hover: self
                .ui_state
                .hovered_guide
                .filter(|&i| i < guide_n)
                .map(|i| i as i32)
                .unwrap_or(-1),
            // While grabbing a guide — or dragging a fresh one out of a ruler — it's
            // drawn in a hue halfway between the base and the hover (inverse) colour
            // (a 90° rotation); a plain hover uses the full inverse hue.
            guide_hover_color: {
                let base = srgb_u8_to_f32(self.prefs.guide_color);
                if self.guide_drag.is_some() || self.ui_state.guide_spawn.is_some() {
                    shift_hue(base, 90.0)
                } else {
                    inverse_hue(base)
                }
            },
            clarity_amount: self.clarity_amount,
            clarity_radius: self.clarity_radius,
            global_alpha: 1.0,
            // 2D display rotation (ignored by the panorama branch in the shader).
            rotation: self.rotation as i32,
            // Debug A/B only: force Lanczos off to compare against bilinear.
            lanczos_off: self.debug_no_lanczos,
            // Only the histogram pass point-samples (see `update_histogram`).
            point_sample: false,
            // Clipping overlay (C): animated stripes over near-/at-max regions,
            // judged on the original (pre-adjustment) per-channel values.
            clip_overlay: self.clip_overlay,
            clip_margin: self.prefs.clip_margin,
            clip_time: self.app_epoch.elapsed().as_secs_f32() % 3600.0,
        };
        // Minimap thumbnail pass parameters (panel rect + fade), gathered before
        // the mutable gfx borrow. Drawn after the scene, below the egui overlay.
        let minimap_pass: Option<(MinimapMetrics, f32)> = {
            let a = self.minimap_alpha();
            (a > 0.0)
                .then(|| self.minimap_metrics().map(|m| (m, a)))
                .flatten()
        };
        let capture_ready = self.capture_ready();
        let clipboard_copy_ready = self.clipboard_copy_pending;
        // Everything the colour-pick tooltip needs except the "Display" value,
        // which requires a GPU readback of this frame's just-rendered scene
        // (done inside the block below, once `gfx.renderer.render` has run).
        let color_pick_partial = self.color_pick_partial();
        let cursor_pos = self.cursor_pos;

        let mut actions: Vec<UiAction> = Vec::new();
        let mut grabbed: Option<(i32, i32, Vec<u8>)> = None;
        let mut clipboard_grab: Option<(i32, i32, Vec<u8>)> = None;
        let mut new_color_pick: Option<crate::ui::ColorPickInfo> = None;
        // Histogram results, applied after the gfx borrow ends. The initial
        // values only matter on the early-return paths inside that block, hence
        // the allow (same shape as `egui_busy` above).
        #[allow(unused_assignments)]
        let mut new_histogram: Option<crate::renderer::Histogram> = None;
        let mut histogram_measured: Option<HistogramKey> = None;
        #[allow(unused_assignments)]
        let mut histogram_pending = false;
        // egui "is dragging a widget" this frame — coalesces slider drags into one
        // undo entry (set inside the gfx borrow below; only read on the path that
        // sets it, hence the allow for the otherwise-overwritten initial value).
        #[allow(unused_assignments)]
        let mut egui_busy = false;

        {
            let ui_state = &mut self.ui_state;
            let Some(gfx) = self.gfx.as_mut() else {
                return RenderOutcome::Idle;
            };
            let size = gfx.window.inner_size();
            if size.width == 0 || size.height == 0 {
                return RenderOutcome::Idle;
            }
            let (w, h) = (size.width as i32, size.height as i32);

            // Land a finished measurement, then start the next one if the graph
            // has gone stale. Done before the scene draw because the pass binds
            // its own framebuffer and viewport; `render` re-binds both anyway.
            new_histogram = gfx.renderer.poll_histogram();
            if let Some((key, iw, ih)) = histogram_want {
                if gfx.renderer.update_histogram(&base, iw, ih) {
                    histogram_measured = Some(key);
                }
            }
            histogram_pending = gfx.renderer.histogram_pending();

            let params = RenderParams {
                viewport: (w, h),
                ..base
            };
            gfx.renderer.render(&params);

            // Colour-pick "Display" value: a 1×1 readback of the pixel under the
            // cursor from the scene just rendered above — before the minimap /
            // egui overlay draw on top of it, so it's the pure image colour. No
            // CPU-side OCIO processor exists (the display transform is GPU-only),
            // so a readback is the only way to get the exact on-screen value.
            if let Some(partial) = &color_pick_partial {
                let fx = (cursor_pos.x.round() as i32).clamp(0, w - 1);
                let fy = (h - 1 - cursor_pos.y.round() as i32).clamp(0, h - 1);
                let mut buf = [0u8; 4];
                unsafe {
                    gfx.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gfx.gl.read_pixels(
                        fx,
                        fy,
                        1,
                        1,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut buf)),
                    );
                }
                new_color_pick = Some(crate::ui::ColorPickInfo {
                    x: partial.x,
                    y: partial.y,
                    degrees: partial.degrees,
                    linear: partial.linear,
                    // Alpha isn't part of the RGB display transform (the shader
                    // passes `texel.a` straight through) — the readback's own alpha
                    // channel isn't a substitute for it, since compositing the scene
                    // over the opaque background blends it into a physically
                    // meaningless value (e.g. src_a=0.5 over an opaque bg reads back
                    // ~0.75, not 0.5). Reuse the already-correct linear value.
                    display: [
                        buf[0] as f32 / 255.0,
                        buf[1] as f32 / 255.0,
                        buf[2] as f32 / 255.0,
                        partial.linear[3],
                    ],
                });
            }

            // Minimap thumbnail: a fit-the-whole-image 2D view drawn into the
            // bottom-right corner (GL origin bottom-left), composited over the
            // scene at the fade alpha. Below the egui border / view box.
            if let Some((m, alpha)) = minimap_pass {
                let (gl_x, gl_w, gl_h) =
                    (m.x.round() as i32, m.w.round() as i32, m.h.round() as i32);
                let gl_y = (h as f32 - (m.y + m.h)).round() as i32;
                let mm_params = RenderParams {
                    viewport: (gl_w, gl_h),
                    projection_mode: 1,
                    yaw: 0.0,
                    pitch: 0.0,
                    half_fov_radians: std::f32::consts::FRAC_PI_4,
                    tan_half_fov: 1.0,
                    wrap_2d: false,
                    nearest: false,
                    isolate_channel: -1,
                    stretch: [1.0, 1.0],
                    sharpness: false,
                    diff: false,
                    guide_count: 0,
                    guide_hover: -1,
                    clarity_amount: 0.0,
                    global_alpha: alpha,
                    // No clipping stripes on the navigation thumbnail.
                    clip_overlay: false,
                    ..base
                };
                gfx.renderer
                    .render_minimap(&mm_params, gl_x, gl_y, gl_w, gl_h);
            }

            // egui overlay on top of the scene.
            gfx.egui.run(&gfx.window, |ctx| {
                ui::build(ctx, &inputs, ui_state, &mut actions);
            });
            egui_busy = gfx.egui.egui_ctx.is_using_pointer();
            gfx.egui.paint(&gfx.window);

            if capture_ready {
                let mut buf = vec![0u8; (w * h * 4) as usize];
                unsafe {
                    gfx.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gfx.gl.read_pixels(
                        0,
                        0,
                        w,
                        h,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut buf)),
                    );
                }
                grabbed = Some((w, h, buf));
            }

            // Ctrl+C: same idea as the debug capture above, but into the
            // clipboard — the composited render (adjustments, guides, minimap,
            // egui overlay), read back after `gfx.egui.paint` for the same reason.
            if clipboard_copy_ready {
                let mut buf = vec![0u8; (w * h * 4) as usize];
                unsafe {
                    gfx.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gfx.gl.read_pixels(
                        0,
                        0,
                        w,
                        h,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut buf)),
                    );
                }
                clipboard_grab = Some((w, h, buf));
            }

            if let Err(e) = gfx.gl_surface.swap_buffers(&gfx.gl_context) {
                log::error!("swap_buffers failed: {e}");
            }
        }
        self.color_pick_last = new_color_pick;
        if let Some(key) = histogram_measured {
            self.histogram_key = Some(key);
        }
        if let Some(h) = new_histogram {
            if log::log_enabled!(log::Level::Debug) {
                log_histogram(&h);
            }
            self.histogram = Some(Arc::new(h));
        }
        // `about_to_wait` would otherwise park on `Wait` and the finished
        // measurement would sit in its buffer until the next unrelated event.
        if histogram_pending {
            self.request_redraw();
        }

        for action in actions {
            self.handle_ui_action(action);
        }

        // Grow / restore the window around the Settings dialog (after actions and
        // the egui pass have settled this frame's open/close state).
        self.sync_settings_window();

        // Record an undo entry if this frame's input changed the editing state
        // (after the UI actions and key handling above; coalesced during gestures).
        self.commit_undo_if_changed(egui_busy);

        if let Some((w, h, buf)) = clipboard_grab {
            self.clipboard_copy_pending = false;
            self.show_toast(
                match copy_rgba_to_clipboard(w as u32, h as u32, buf) {
                    Ok(()) => "Copied to clipboard".to_string(),
                    Err(e) => {
                        log::error!("clipboard copy failed: {e}");
                        "Clipboard copy failed".to_string()
                    }
                },
            );
        }

        if let Some((w, h, buf)) = grabbed {
            self.write_capture(w as u32, h as u32, buf);
            if let Some(c) = &mut self.capture {
                c.done = true;
            }
            return RenderOutcome::Captured;
        }
        RenderOutcome::Drew
    }

    fn write_capture(&self, width: u32, height: u32, buf: Vec<u8>) {
        let Some(capture) = &self.capture else { return };
        match image::RgbaImage::from_raw(width, height, buf) {
            Some(mut img) => {
                image::imageops::flip_vertical_in_place(&mut img);
                match img.save(&capture.path) {
                    Ok(()) => log::info!(
                        "captured {width}x{height} framebuffer -> {}",
                        capture.path.display()
                    ),
                    Err(e) => log::error!("failed to write capture PNG: {e}"),
                }
            }
            None => log::error!("capture buffer size mismatch ({width}x{height})"),
        }
    }
}

/// Copy an RGBA framebuffer readback (GL row order: bottom row first) to the
/// clipboard as a 32-bit `CF_DIB`. A DIB with a positive `biHeight` is bottom-up
/// too, so — unlike `write_capture`'s PNG — the rows need no flip, just an
/// RGBA → BGRA channel swizzle (CF_DIB has no real alpha channel).
#[cfg(windows)]
fn copy_rgba_to_clipboard(width: u32, height: u32, mut rgba: Vec<u8>) -> Result<(), String> {
    use windows_sys::Win32::Graphics::Gdi::{BI_RGB, BITMAPINFOHEADER};
    use windows_sys::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData,
    };
    use windows_sys::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock};
    use windows_sys::Win32::System::Ole::CF_DIB;

    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
    }

    let header = BITMAPINFOHEADER {
        biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: width as i32,
        biHeight: height as i32, // positive = bottom-up, matching the GL readback order
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB,
        biSizeImage: 0,
        biXPelsPerMeter: 0,
        biYPelsPerMeter: 0,
        biClrUsed: 0,
        biClrImportant: 0,
    };
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const header) as *const u8,
            std::mem::size_of::<BITMAPINFOHEADER>(),
        )
    };
    let total = header_bytes.len() + rgba.len();

    unsafe {
        let hmem = GlobalAlloc(GMEM_MOVEABLE, total);
        if hmem.is_null() {
            return Err("GlobalAlloc failed".to_string());
        }
        let ptr = GlobalLock(hmem) as *mut u8;
        if ptr.is_null() {
            return Err("GlobalLock failed".to_string());
        }
        std::ptr::copy_nonoverlapping(header_bytes.as_ptr(), ptr, header_bytes.len());
        std::ptr::copy_nonoverlapping(rgba.as_ptr(), ptr.add(header_bytes.len()), rgba.len());
        GlobalUnlock(hmem);

        if OpenClipboard(std::ptr::null_mut()) == 0 {
            windows_sys::Win32::Foundation::GlobalFree(hmem);
            return Err("OpenClipboard failed".to_string());
        }
        EmptyClipboard();
        // On success the clipboard now owns `hmem` (must not free it ourselves);
        // on failure it's still ours to free.
        let ok = !SetClipboardData(CF_DIB as u32, hmem).is_null();
        CloseClipboard();
        if !ok {
            windows_sys::Win32::Foundation::GlobalFree(hmem);
            return Err("SetClipboardData failed".to_string());
        }
    }
    Ok(())
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match self.create_gfx(event_loop) {
            Ok(gfx) => {
                self.gfx = Some(gfx);
                if let Some(c) = &mut self.capture {
                    c.start = Instant::now();
                }
                // Build the OCIO program (or gamma fallback) before first draw.
                self.rebuild_ocio();
                self.load_initial_image();
                // Render the first frame (backdrop + hint / loading) into the back
                // buffer, then reveal the hidden window — so it appears already in
                // its final place at the right size with the correct background,
                // rather than flashing the OS default first (see `with_visible`).
                if matches!(self.render(), RenderOutcome::Captured) {
                    // Headless capture finished on the first frame — no window to show.
                    event_loop.exit();
                } else if let Some(gfx) = &self.gfx {
                    gfx.window.set_visible(true);
                    gfx.window.request_redraw();
                }
            }
            Err(e) => {
                log::error!("failed to initialise graphics: {e:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Let egui see the event first; it tells us whether it consumed it.
        let mut egui_consumed = false;
        if let Some(gfx) = &mut self.gfx {
            let resp = gfx.egui.on_window_event(&gfx.window, &event);
            if resp.repaint {
                gfx.window.request_redraw();
            }
            egui_consumed = resp.consumed;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // The window geometry changed: a stationary cursor is now over a
                // different part of the window, so disarm hover reveals until the
                // user actually moves the mouse again (a real move re-arms via
                // DeviceEvent::MouseMotion). Window-follow during arrow-nav is the
                // motivating case.
                self.cursor_moved_by_user = false;
                if let Some(gfx) = &self.gfx {
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        gfx.gl_surface.resize(&gfx.gl_context, w, h);
                    }
                    // The window region is in window-local pixels, so re-round
                    // the corners at the new size. Fullscreen/maximized windows
                    // fill their area with square corners (radius 0 clears the
                    // region), so the old rounded region can't clip them.
                    let rounded = !self.fullscreen && !gfx.window.is_maximized();
                    let radius = if rounded { self.prefs.corner_radius } else { 0 };
                    apply_window_corners(&gfx.window, radius);
                }
                // The window has restored to a (now stale) windowed size after
                // leaving fullscreen: re-frame it to the *current* image and reset
                // the 2D fit zoom, so navigating in fullscreen then exiting lands on
                // a correctly-framed window. (Entry fits from the monitor size in
                // set_fullscreen, so it needs nothing here.)
                if self.refit_windowed_pending && !self.fullscreen {
                    self.refit_windowed_pending = false;
                    if !self.camera.is_panorama() {
                        let (dw, dh) = self.frame_dims();
                        self.resize_window_to_image(dw, dh);
                        self.camera.fit_flat_now(1.0);
                    }
                }
                // Redraw synchronously so the new surface size is presented this
                // frame (otherwise the previous frame shows stretched). During an
                // Alt-resize, skip the synchronous (vsync-blocked) redraw — its
                // per-frame resize would otherwise crawl in slow motion; the timed
                // loop redraws instead.
                if self.alt_resize.is_some() {
                    self.request_redraw();
                } else if matches!(self.render(), RenderOutcome::Captured) {
                    event_loop.exit();
                }
                // A resize we didn't initiate (outside the suppression window),
                // and not a maximize/fullscreen, is the user dragging the border:
                // stop auto-following on zoom so their chosen size sticks.
                let programmatic = self
                    .suppress_manual_until
                    .is_some_and(|t| Instant::now() < t);
                let special =
                    self.fullscreen || self.gfx.as_ref().is_some_and(|g| g.window.is_maximized());
                if !programmatic && !special && self.loaded_path.is_some() {
                    self.manual_window = true;
                }
                // The geometry ease is advanced in `about_to_wait`, NOT here:
                // posting the next resize from this handler self-perpetuates
                // before the loop yields, starving OS input (queued scroll
                // notches) until the window settles — the fast-scroll shudder.
                // Advancing from `about_to_wait` lets input be processed between
                // every step while this synchronous render still presents each
                // size exactly once (vsync-paced, no double-present).
            }
            WindowEvent::Moved(_) => {
                // Same as Resized: a programmatic reposition (window-follow recentre
                // on arrow-nav) slides the window under a stationary cursor; disarm
                // hover reveals until the next real mouse move.
                self.cursor_moved_by_user = false;
            }
            WindowEvent::RedrawRequested => {
                if matches!(self.render(), RenderOutcome::Captured) {
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(mods) => self.modifiers = mods.state(),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = position;
                self.cursor_in_window = true;
                // Dragging inside the minimap continuously snaps the view to the
                // pointed-at region (no pan / guide hover during the drag).
                if self.minimap_drag {
                    self.minimap_navigate(position);
                    return;
                }
                if self.alt_resize.is_some() {
                    self.update_alt_resize();
                }
                // A right-press past the drag threshold turns into a colour-pick
                // (not a click) — cancel the pending guide-delete once it does.
                if !self.color_picking {
                    if let Some(press) = self.right_press_pos {
                        let (dx, dy) = (position.x - press.x, position.y - press.y);
                        if ((dx * dx + dy * dy).sqrt() as f32) >= DBLCLICK_DRAG_TOL {
                            self.color_picking = true;
                            self.right_press_guide = None;
                        }
                    }
                }
                // Move a grabbed guide to follow the cursor along its constant-uv
                // axis (clamped on-image; the release decides keep vs discard).
                // Ctrl snaps it to 10 displayed px (2D) or a whole degree (pano).
                if let Some(idx) = self.guide_drag {
                    if let Some((u, v)) = self.viewport_uv(position.x, position.y) {
                        if let Some(&g) = self.guides.get(idx) {
                            let horizontal = g[1] >= 0.5;
                            let raw = if horizontal { v } else { u };
                            let coord = if self.modifiers.control_key() {
                                self.snap_guide_coord(raw, horizontal)
                            } else {
                                raw
                            };
                            if let Some(g) = self.guides.get_mut(idx) {
                                g[0] = coord.clamp(0.0, 1.0);
                            }
                        }
                    }
                }
                // Hover highlight: the dragged guide, else the one under the
                // cursor — but not while panning/stretching/resizing/colour-picking
                // (the cursor is grabbed/busy then, so a stray highlight just
                // flickers or fights the colour-pick tooltip).
                self.ui_state.hovered_guide =
                    self.guide_drag.or(self.ui_state.guide_spawn).or_else(|| {
                        if self.dragging
                            || self.stretching
                            || self.alt_resize.is_some()
                            || self.color_picking
                        {
                            None
                        } else {
                            self.guide_at_cursor()
                        }
                    });
                self.tick_bottom_panel();
                self.tick_left_ruler();
                self.tick_metadata();
                self.request_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                self.cursor_in_window = true;
                // Give the cursor a fresh idle period after it (re-)enters.
                self.last_cursor_motion = Some(Instant::now());
                self.request_redraw();
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_in_window = false;
                self.ui_state.hovered_guide = None;
                self.request_redraw();
            }
            // A press that starts a borderless resize is handled before egui so
            // it wins over the toolbar/titlebar overlapping the window border:
            // left-press on an edge/corner hit-zone, or Alt+right-press anywhere
            // (direction from the cursor's third of the window). Otherwise
            // egui-consumed events fall through to `_ => {}`.
            WindowEvent::MouseInput { state, button, .. } => {
                // End an in-progress Alt+right-drag resize when the RIGHT button
                // (the one driving it) is released. Gating on the button matters:
                // swallowing *any* release here would skip `on_mouse_button` for a
                // left release that ended a pan / guide- or minimap-drag, leaving
                // that gesture flag stuck on.
                if state == ElementState::Released
                    && button == MouseButton::Right
                    && self.alt_resize.is_some()
                {
                    self.alt_resize = None;
                } else {
                    let resized = state == ElementState::Pressed
                        && !self.fullscreen
                        && match button {
                            MouseButton::Left => self.start_edge_resize(),
                            MouseButton::Right if self.modifiers.alt_key() => {
                                self.start_third_resize()
                            }
                            _ => false,
                        };
                    // Presses only start a gesture when egui didn't take them (and
                    // it isn't a border resize). RELEASES are always processed, so
                    // an app-started pan/stretch is always ended even if the
                    // pointer happened to end over a guide strip (which egui would
                    // consume) — otherwise `dragging` stuck on and panning broke.
                    let process = match state {
                        ElementState::Released => true,
                        ElementState::Pressed => !resized && !egui_consumed,
                    };
                    if process {
                        self.on_mouse_button(state, button);
                    }
                }
            }
            // egui consumes the wheel when a panel (e.g. the bottom sliders) is
            // hovered, so it doesn't also zoom the image (§11.3).
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => self.on_wheel(delta),
            WindowEvent::DroppedFile(path) => {
                // A manually-dropped file ends any arrow-nav preload chain.
                self.preload_armed = false;
                self.nav_pending = None;
                self.image_cache.clear();
                self.compare_prev = None;
                self.load_path(path);
            }
            WindowEvent::KeyboardInput { event, .. }
                if !egui_consumed && event.state == ElementState::Pressed =>
            {
                // OS auto-repeat drives folder navigation (hold ← / → to flip
                // through images) and the continuous adjustment keys (exposure ,/. ;
                // gamma Ctrl+,/. ; clarity [ ] and ; ') so they ramp while held.
                // Toggles, guides, slots and one-shot actions still fire once per
                // physical press, so a held toggle can't flicker or run away.
                let nav_repeat = matches!(
                    &event.logical_key,
                    Key::Named(NamedKey::ArrowLeft) | Key::Named(NamedKey::ArrowRight)
                );
                let adjust_repeat = matches!(
                    &event.logical_key,
                    Key::Character(s) if matches!(s.as_str(), "," | "." | "[" | "]" | ";" | "'")
                );
                if event.repeat && !(nav_repeat || adjust_repeat) {
                    return;
                }
                // Keep held adjustments as one coalesced undo gesture (see
                // ADJUST_COALESCE / undo_gesture_active): refresh the window on every
                // press and repeat so the per-frame undo commit defers until release.
                if adjust_repeat {
                    self.adjust_repeat_until = Some(Instant::now() + ADJUST_COALESCE);
                }
                // Numpad digits = exact zoom; top-row digits = comparator slots
                // (Ctrl+N saves, N recalls).
                if let Some(digit) = numpad_digit(&event.physical_key) {
                    self.set_exact_zoom(digit, self.ctrl());
                } else if let Some(slot) = toprow_digit(&event.physical_key) {
                    if self.modifiers.alt_key() {
                        self.toggle_slot_diff(slot as usize);
                    } else if self.ctrl() {
                        self.save_slot(slot as usize);
                    } else {
                        // Keyboard recall re-frames the window to the image (like
                        // arrow-key navigation / opening a file).
                        self.recall_slot(slot as usize, true);
                    }
                } else {
                    let text = match &event.logical_key {
                        Key::Character(s) => Some(s.as_str().to_string()),
                        _ => None,
                    };
                    self.on_key(event_loop, &event.logical_key, text.as_deref());
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // A real, physical mouse move — arm the edge-panel / minimap reveals
            // (a window-follow resize that slides the window under a stationary
            // cursor produces CursorMoved but NOT this raw motion, so it stays
            // disarmed). Any nonzero delta counts.
            if delta.0 != 0.0 || delta.1 != 0.0 {
                self.cursor_moved_by_user = true;
                // Restart the fullscreen idle-hide timer and reveal the cursor if it
                // was hidden (instant, not waiting for the next frame).
                self.last_cursor_motion = Some(Instant::now());
                self.show_cursor_now();
            }
            // (Alt-resize is driven by CursorMoved, which tracks the visible
            // cursor 1:1 — not raw device motion.)
            if self.stretching {
                // Alt+middle-drag: right→wider, up→taller (multiplicative). No
                // real limit on the squash/stretch (a single-pixel-wide image is
                // allowed); the bounds are only a numerical guard against 0/∞.
                const SENS: f32 = 0.004;
                self.image_stretch.x =
                    (self.image_stretch.x * (1.0 + delta.0 as f32 * SENS)).clamp(1e-4, 1e4);
                self.image_stretch.y =
                    (self.image_stretch.y * (1.0 - delta.1 as f32 * SENS)).clamp(1e-4, 1e4);
                self.request_redraw();
            } else if self.dragging {
                self.on_drag_motion(delta.0 as f32, delta.1 as f32);
            } else if self.window_drag_armed {
                // Past the click threshold, hand off to the OS move loop (so
                // Aero Snap works); a smaller travel stays a click.
                self.window_drag_motion += (delta.0 * delta.0 + delta.1 * delta.1).sqrt() as f32;
                if self.window_drag_motion >= DBLCLICK_DRAG_TOL {
                    self.window_drag_armed = false;
                    self.pending_dblclick = false;
                    // Cut any in-flight animation so it doesn't fight the move.
                    self.freeze_animations();
                    if let Some(gfx) = &self.gfx {
                        let _ = gfx.window.drag_window();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.should_exit {
            event_loop.exit();
            return;
        }
        // Advance the window-follow geometry ease one step per loop iteration —
        // HERE, after this iteration's OS input was processed, so a fast scroll
        // burst keeps retargeting smoothly instead of being starved by a
        // self-perpetuating Resized chain. The posted resize's Resized event
        // renders the new size synchronously (one present, vsync-paced via the
        // blocking swap). No-op when no follow target is pending.
        let easing = self.alt_resize.is_none() && self.window_anim_target.is_some();
        if easing {
            self.ease_window();
        }
        // Fullscreen idle cursor auto-hide: hide once the no-motion timer expires.
        // Runs every loop iteration; the deadline is added to the wait scheduling
        // below so the loop wakes to perform the hide even when otherwise idle.
        self.update_cursor_idle_hide();
        let cursor_idle_deadline = (self.fullscreen
            && self.cursor_in_window
            && !self.in_gesture()
            && !self.cursor_idle_hidden)
            .then(|| self.last_cursor_motion.map(|t| t + CURSOR_IDLE_HIDE))
            .flatten();
        // Wake to flip to the next GIF frame (the advance + upload happens in
        // `render`). `None` while paused or for a static image.
        let anim_deadline = self.anim.as_ref().filter(|a| !a.paused).map(|a| a.next_at);
        if self.capture_active() {
            // Drive continuous frames while a capture is pending.
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        } else if self.is_busy() {
            // Drive frames while decoding (spinner) or uploading (pump + bar).
            event_loop.set_control_flow(ControlFlow::Poll);
            self.request_redraw();
        } else if easing {
            // Keep the loop spinning at vsync rate so the next ease step follows
            // promptly. No request_redraw: the Resized render above is the frame;
            // adding one would double-present (the old slow-mo regression).
            event_loop.set_control_flow(ControlFlow::Poll);
        } else if self.toast_active()
            || self.animating
            || self.minimap_fading()
            || self.clip_overlay
        {
            // Drive ~60 fps while the toast fades, the zoom/pan eases, the
            // auto-shown minimap holds/fades, or the clipping overlay's stripes
            // scroll. The transient ones settle quickly; the clip overlay holds
            // ~60 fps until toggled off (its stripe animation needs fresh frames).
            let next = Instant::now() + Duration::from_millis(16);
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            self.request_redraw();
        } else if let Some(deadline) = [
            self.bottom_hide_deadline,
            self.metadata_hide_deadline,
            cursor_idle_deadline,
            anim_deadline,
            // Wake once the held-adjustment coalesce window closes so the deferred
            // undo entry commits even after the tone ease has settled.
            self.adjust_repeat_until,
        ]
        .into_iter()
        .flatten()
        .min()
        {
            // Wake at the earliest pending deadline (panel hide, or the fullscreen
            // cursor idle-hide). The cursor hide runs in `update_cursor_idle_hide`
            // at the top of the next iteration; no redraw needed for it.
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            if Instant::now() >= deadline {
                self.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Remember which monitor the window is on so "Remember last used" (the
        // default startup display) reopens there. Only the *display* is persisted —
        // the size/position are not; a fresh launch auto-sizes and centres.
        if let Some(name) = self
            .gfx
            .as_ref()
            .and_then(|g| g.window.current_monitor())
            .and_then(|m| m.name())
        {
            self.prefs.last_monitor = Some(name);
        }
        self.prefs.save();
        // Release egui's GL resources (textures/buffers) before the painter is
        // dropped, else it warns "You forgot to call destroy() on the egui glow
        // painter" — e.g. when quitting while an image is still loading. The GL
        // context is still current from the last render.
        if let Some(gfx) = &mut self.gfx {
            gfx.egui.destroy();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::LoadFinished(_gen) => self.poll_loads(),
            UserEvent::PreloadFinished(_gen) => self.poll_preloads(),
            UserEvent::UpdateChecked => self.poll_update(),
        }
    }
}

/// 2D zoom that fits an `iw`×`ih` image into a `vw`×`vh` viewport but never
/// magnifies past native 1:1 (device px). On-screen scale = `zoom * vh / img_h`
/// (matching `set_exact_zoom`), so a fit-scale clamped to ≤ 1 maps to this zoom:
/// a sub-viewport image lands at native scale, a larger one shrinks to fit.
fn fit_zoom_no_upscale(vw: f32, vh: f32, iw: f32, ih: f32) -> f32 {
    let s_fit = (vw / iw.max(1.0)).min(vh / ih.max(1.0));
    s_fit.min(1.0) * (ih.max(1.0) / vh.max(1.0))
}

/// Centre-of-view on-screen scale (device px per equirect texel) for a panorama,
/// given the viewport height `vh`, image height `ih`, and `tan(½ vertical-FOV)`.
/// The equirect is `ih/π` texels per radian; the rectilinear projection is
/// `vh/(2·tan½fov)` screen px per radian at centre — their ratio. > 2.0 ⇒ one
/// texel covers two screen px (the nearest-neighbour switch threshold).
fn pano_center_scale(vh: f32, ih: f32, tan_half_fov: f32) -> f32 {
    std::f32::consts::PI * vh / (2.0 * tan_half_fov.max(1e-6) * ih.max(1.0))
}

/// Choose nearest-neighbour vs bilinear. In `auto` mode a 2D image magnified past
/// 200% (`scale > 2.0`, device px per image px) reads nearest for crisp pixels;
/// panoramas (`scale == None`) and anything ≤ 200% stay bilinear. With `auto` off
/// the pinned `manual` value (last set by the I key) is used verbatim.
fn pick_nearest(auto: bool, manual: bool, scale: Option<f32>) -> bool {
    if auto {
        scale.is_some_and(|s| s > 2.0)
    } else {
        manual
    }
}

/// Uniformly scale `iw`×`ih` down to fit within [`FILL_FRACTION`] of `monitor`
/// (never upscaling past native), returning the framed window inner size in
/// physical pixels (each dimension at least [`MIN_DIM`]).
fn fit_to_monitor(iw: f32, ih: f32, monitor: Option<&MonitorHandle>) -> (u32, u32) {
    let (cap_w, cap_h) = monitor
        .map(|m| {
            let s = m.size();
            (
                s.width as f32 * FILL_FRACTION,
                s.height as f32 * FILL_FRACTION,
            )
        })
        .unwrap_or((iw, ih));
    let fit = (cap_w / iw).min(cap_h / ih).min(1.0);
    let w = ((iw * fit) as u32).max(MIN_DIM);
    let h = ((ih * fit) as u32).max(MIN_DIM);
    (w, h)
}

/// The mouse cursor's position in global (virtual-screen) physical pixels via
/// Win32 `GetCursorPos`. `None` if the call fails. Used to place the window under
/// the cursor when leaving fullscreen for a titlebar drag.
fn global_cursor_pos() -> Option<(i32, i32)> {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;
    let mut pt = POINT { x: 0, y: 0 };
    // SAFETY: `GetCursorPos` only writes through the out-pointer, which is valid.
    if unsafe { GetCursorPos(&mut pt) } != 0 {
        Some((pt.x, pt.y))
    } else {
        None
    }
}

/// Compute the window's inner size and outer position at creation so it opens
/// already framing its image, centred on the target monitor. A fresh launch
/// always auto-sizes and centres — the previous window geometry is deliberately
/// not restored:
///
/// * With a probed initial image → frame it (capped to the monitor).
/// * No image → a default size.
///
/// Either way the window is centred on `monitor` (the configured startup display,
/// or the primary), clamped on-screen.
fn startup_geometry(
    probed: Option<(u32, u32)>,
    monitor: Option<&MonitorHandle>,
    scale: f64,
) -> (PhysicalSize<u32>, Option<PhysicalPosition<i32>>) {
    let size = if let Some((iw, ih)) = probed {
        let (w, h) = fit_to_monitor(iw as f32, ih as f32, monitor);
        PhysicalSize::new(w, h)
    } else {
        PhysicalSize::new((1280.0 * scale) as u32, (720.0 * scale) as u32)
    };

    // Centre on the chosen monitor, clamped on-screen.
    let position = monitor.map(|m| {
        let (mp, ms) = (m.position(), m.size());
        let x = (mp.x + (ms.width as i32 - size.width as i32) / 2).max(mp.x);
        let y = (mp.y + (ms.height as i32 - size.height as i32) / 2).max(mp.y);
        PhysicalPosition::new(x, y)
    });
    (size, position)
}

/// Replace egui's thin default proportional font with the native Segoe UI for
/// sharper, more familiar text. Falls back to the egui default if not found.
fn install_ui_font(ctx: &egui::Context) {
    // Candidate fonts, most-preferred first: a debug-only override (for font
    // A/B testing), the bundled font (ships with the app, so the UI looks
    // identical on every machine), then the OS Segoe UI, then egui's default.
    let mut candidates: Vec<PathBuf> = Vec::new();
    #[cfg(debug_assertions)]
    if let Some(p) = std::env::var_os("IMGVWR_UI_FONT") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(
        resolve_resources_dir()
            .join("fonts")
            .join("Inter-Regular.otf"),
    );
    candidates.push(PathBuf::from(r"C:\Windows\Fonts\segoeui.ttf"));

    let mut loaded = false;
    for path in &candidates {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("ui".to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
        if let Some(fam) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
            fam.insert(0, "ui".to_owned());
        }
        ctx.set_fonts(fonts);
        log::info!("UI font: {}", path.display());
        loaded = true;
        break;
    }
    if !loaded {
        log::info!("UI font: egui default (no candidate font found)");
    }

    // Debug-only global zoom, for crispness A/B testing.
    #[cfg(debug_assertions)]
    if let Some(z) = std::env::var("IMGVWR_UI_ZOOM")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        ctx.set_zoom_factor(z);
        log::info!("UI zoom: {z}");
    }
}

/// The alphabetical sibling image `dir` steps from `current` in its folder
/// (wrapping at the ends). `None` if the folder can't be read or has no
/// supported images.
/// A human-readable type name for an extension, shown in Explorer's Type column
/// via the per-extension ProgID (see [`register_default_app`]). Distinct per
/// format so a folder can be sorted by type; camera RAW formats share a common
/// "Camera RAW Image" prefix so they cluster together in a sort yet stay
/// separable from JPEG / PNG / EXR / …. Unknown extensions fall back to an
/// uppercase `<EXT> Image`.
#[cfg(windows)]
fn friendly_type_name(ext: &str) -> String {
    let named = match ext {
        "jpg" | "jpeg" => "JPEG Image",
        "png" => "PNG Image",
        "apng" => "Animated PNG Image",
        "bmp" => "Bitmap Image",
        "tif" | "tiff" => "TIFF Image",
        "webp" => "WebP Image",
        "gif" => "GIF Image",
        "ico" => "Icon Image",
        "tga" => "Targa Image",
        "pnm" => "Netpbm Image",
        "hdr" | "pic" => "Radiance HDR Image",
        "exr" => "OpenEXR Image",
        "nef" | "nrw" => "Camera RAW Image (Nikon)",
        "cr2" | "cr3" | "crw" => "Camera RAW Image (Canon)",
        "arw" | "sr2" | "srf" => "Camera RAW Image (Sony)",
        "dng" => "Camera RAW Image (DNG)",
        "raf" => "Camera RAW Image (Fujifilm)",
        "orf" => "Camera RAW Image (Olympus)",
        "rw2" => "Camera RAW Image (Panasonic)",
        "pef" => "Camera RAW Image (Pentax)",
        "rwl" => "Camera RAW Image (Leica)",
        "raw" => "Camera RAW Image",
        other => return format!("{} Image", other.to_uppercase()),
    };
    named.to_string()
}

/// Register imgvwr (per-user, no admin) as the handler for every supported
/// extension. Each extension gets its **own** ProgID (`imgvwr.<ext>`) with a
/// distinct [`friendly_type_name`], so Explorer's Type column shows a real,
/// per-format type (JPEG Image, OpenEXR Image, Camera RAW Image (Nikon), …) and a
/// folder can be sorted by type — a single shared ProgID would collapse them all
/// to one type. Also cleans up the legacy single `imgvwr.Image` ProgID that older
/// builds used. Returns the number of extensions associated.
///
/// Note: Windows protects an extension's *current* default with a hashed
/// UserChoice, so already-defaulted types (e.g. .jpg) may still need confirmation
/// in Settings → Default apps; unassociated types (most HDR/EXR/RAW) take effect
/// immediately.
#[cfg(windows)]
pub fn register_default_app() -> Result<usize, String> {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
    use winreg::RegKey;

    let exe = std::env::current_exe().map_err(|e| format!("exe path: {e}"))?;
    let exe = exe.to_string_lossy().into_owned();
    let icon_val = format!("{exe},0");
    let cmd_val = format!("\"{exe}\" \"%1\"");

    let classes = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags(r"Software\Classes", KEY_READ | KEY_WRITE)
        .map_err(|e| format!("open HKCU Classes: {e}"))?;

    // Older builds registered a single "imgvwr.Image" ProgID for every extension,
    // so Explorer's Type column read the same "imgvwr Image" for all of them and a
    // folder couldn't be sorted by type (RAW vs JPEG, etc). Drop it; the
    // per-extension ProgIDs below replace it with distinct, sortable type names.
    let _ = classes.delete_subkey_all("imgvwr.Image");

    let mut count = 0usize;
    for ext in crate::image_loader::supported_extensions() {
        let progid = format!("imgvwr.{ext}");

        // Per-extension ProgID: its own friendly type name plus the shared icon
        // and open command.
        let Ok((prog, _)) = classes.create_subkey(&progid) else {
            continue;
        };
        let _ = prog.set_value("", &friendly_type_name(ext));
        if let Ok((icon, _)) = prog.create_subkey("DefaultIcon") {
            let _ = icon.set_value("", &icon_val);
        }
        if let Ok((cmd, _)) = prog.create_subkey(r"shell\open\command") {
            let _ = cmd.set_value("", &cmd_val);
        }

        // Point the extension at this ProgID (classic default + OpenWithProgids),
        // and drop the stale legacy entry from its Open-with list.
        let Ok((key, _)) = classes.create_subkey(format!(".{ext}")) else {
            continue;
        };
        if let Ok((owp, _)) = key.create_subkey("OpenWithProgids") {
            let _ = owp.set_value(&progid, &"");
            let _ = owp.delete_value("imgvwr.Image");
        }
        let _ = key.set_value("", &progid);
        count += 1;
    }

    // Refresh shell file-association state (icons / defaults / type names).
    unsafe {
        windows_sys::Win32::UI::Shell::SHChangeNotify(
            windows_sys::Win32::UI::Shell::SHCNE_ASSOCCHANGED as i32,
            windows_sys::Win32::UI::Shell::SHCNF_IDLIST,
            std::ptr::null(),
            std::ptr::null(),
        );
    }
    log::info!("registered imgvwr as handler for {count} extensions");
    Ok(count)
}

#[cfg(not(windows))]
pub fn register_default_app() -> Result<usize, String> {
    Err("only supported on Windows".into())
}

/// The portion of `path` below the deepest directory common to all of `group`,
/// joined with `/` (e.g. `a/b/c.jpg` vs `x/b/c.jpg`). Used to disambiguate
/// comparator slots whose filenames collide.
fn disambiguated_path(path: &Path, group: &[&Path]) -> String {
    let mut anc = path.to_path_buf();
    while !group.iter().all(|q| q.starts_with(&anc)) {
        if !anc.pop() {
            break;
        }
    }
    let rel = path.strip_prefix(&anc).unwrap_or(path);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn sibling_path(current: &Path, dir: i32) -> Option<PathBuf> {
    let parent = current.parent()?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(parent)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_supported(p))
        .collect();
    if files.is_empty() {
        return None;
    }
    // A single supported image: there's no sibling to navigate to. Detected here
    // (not by comparing the returned path to `current`) so it's robust to path-form
    // differences — e.g. a relative CLI arg vs the absolute `read_dir` path — which
    // would otherwise make a `target == current` equality check miss.
    if files.len() == 1 {
        return None;
    }
    files.sort_by_key(|p| {
        p.file_name()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    });
    let idx = files.iter().position(|p| p == current).or_else(|| {
        files
            .iter()
            .position(|p| p.file_name() == current.file_name())
    })?;
    let n = files.len() as i32;
    let next = (idx as i32 + dir).rem_euclid(n) as usize;
    Some(files[next].clone())
}

/// Debug-log a landed display histogram as a coarse 16-bucket digest (percent of
/// samples per bucket) plus the over-range counts. The graph is a GPU
/// measurement of a GPU-only transform, so this is the only way to check the
/// numbers rather than the picture — same role the `equirect_content_scores`
/// debug line plays for panorama detection.
fn log_histogram(h: &crate::renderer::Histogram) {
    let bins = crate::renderer::HISTOGRAM_BINS;
    let per = bins / 16;
    for (ch, name) in ["R", "G", "B"].iter().enumerate() {
        let total: u64 = h.bins[ch].iter().map(|&n| n as u64).sum::<u64>() + h.over[ch] as u64;
        if total == 0 {
            continue;
        }
        let digest: Vec<String> = (0..16)
            .map(|b| {
                let sum: u64 = h.bins[ch][b * per..(b + 1) * per]
                    .iter()
                    .map(|&n| n as u64)
                    .sum();
                format!("{:.1}", sum as f64 * 100.0 / total as f64)
            })
            .collect();
        log::debug!(
            "histogram {name}: total {total}, over {} ({:.2}%), 16ths [{}]",
            h.over[ch],
            h.over[ch] as f64 * 100.0 / total as f64,
            digest.join(" ")
        );
    }
}

/// Format an exposure value as e.g. "+3 EV" or "+0.05 EV".
fn fmt_ev(ev: f32) -> String {
    if ev.fract().abs() < 1e-3 {
        format!("{:+} EV", ev as i32)
    } else {
        format!("{:+.2} EV", ev)
    }
}

/// Normalise an sRGB 0–255 colour to the 0–1 floats written as the framebuffer
/// clear colour. The default framebuffer isn't sRGB, and the image shader writes
/// display-encoded output, so the picked sRGB value is used directly (no
/// linearisation) and appears as chosen.
/// Per-pixel absolute difference of `a` and `b`, at `a`'s resolution (nearest-
/// sampling `b` when sizes differ). Precomputed so the GPU mip chain of the diff
/// shows the *average of the per-pixel differences* — identical regions stay 0 at
/// every zoom. (Differencing two separately mip-averaged textures instead bleeds
/// nearby differences into identical regions when minified.) Computed in the
/// source (encoded) space, matching how the shader then linearises/views it.
/// Returns `None` if the two images use different pixel types.
fn abs_diff_image(a: &ImageData, b: &ImageData) -> Option<ImageData> {
    use crate::image_loader::PixelBuffer;
    let (w, h) = (a.width as usize, a.height as usize);
    let (bw, bh) = ((b.width as usize).max(1), (b.height as usize).max(1));
    // Nearest map from a's grid into b's (identity when the sizes match).
    let map = |x: usize, n: usize, bn: usize| (x * bn / n.max(1)).min(bn - 1);
    let pixels = match (&a.pixels, &b.pixels) {
        (PixelBuffer::U8(av), PixelBuffer::U8(bv)) => {
            let mut out = vec![0u8; w * h * 4];
            for y in 0..h {
                let by = map(y, h, bh);
                for x in 0..w {
                    let bx = map(x, w, bw);
                    let (ai, bi) = ((y * w + x) * 4, (by * bw + bx) * 4);
                    for c in 0..4 {
                        out[ai + c] = (av[ai + c] as i16 - bv[bi + c] as i16).unsigned_abs() as u8;
                    }
                }
            }
            PixelBuffer::U8(out)
        }
        (PixelBuffer::F32(av), PixelBuffer::F32(bv)) => {
            let mut out = vec![0f32; w * h * 4];
            for y in 0..h {
                let by = map(y, h, bh);
                for x in 0..w {
                    let bx = map(x, w, bw);
                    let (ai, bi) = ((y * w + x) * 4, (by * bw + bx) * 4);
                    for c in 0..4 {
                        out[ai + c] = (av[ai + c] - bv[bi + c]).abs();
                    }
                }
            }
            PixelBuffer::F32(out)
        }
        _ => return None,
    };
    Some(ImageData {
        path: a.path.clone(),
        width: a.width,
        height: a.height,
        channels: a.channels,
        dtype_name: a.dtype_name.clone(),
        compression: "-".to_string(),
        pixels,
        is_encoded_srgb: a.is_encoded_srgb,
        animation: None,
        camera: None,
        clip_max: a.clip_max,
    })
}

/// Rotate a camera-space ray by the panorama yaw/pitch (must match the shader's
/// `rotation_yaw_pitch`), returning the world direction. Used to project the
/// cursor onto the sphere for guide hit-testing.
fn pano_rotate(yaw: f32, pitch: f32, rx: f32, ry: f32, rz: f32) -> (f32, f32, f32) {
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let py = cp * ry + sp * rz; // pitch about X …
    let pz = -sp * ry + cp * rz;
    (cy * rx - sy * pz, py, sy * rx + cy * pz) // … then yaw about Y
}

fn srgb_u8_to_f32(c: [u8; 3]) -> [f32; 3] {
    [
        c[0] as f32 / 255.0,
        c[1] as f32 / 255.0,
        c[2] as f32 / 255.0,
    ]
}

/// Permute a displayed image uv to the source-texture uv for a 90°-CW quarter-turn
/// display rotation `rot` (0-3) — mirrors the fragment shader's `rotate_uv` exactly
/// (2D only; rotation is inert in panorama, see `App::rotation`).
fn rotate_uv(u: f32, v: f32, rot: u8) -> (f32, f32) {
    match rot {
        1 => (v, 1.0 - u),
        2 => (1.0 - u, 1.0 - v),
        3 => (1.0 - v, u),
        _ => (u, v),
    }
}

/// The sRGB electro-optical transfer function for one channel — mirrors the
/// fragment shader's `srgb_to_linear`. Used to turn a colour-picked 8-bit source
/// pixel into a true linear-light value for the tooltip's "Linear" row.
fn srgb_to_linear_channel(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c < 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Rotate a colour's hue by `deg` degrees while keeping its saturation and value.
/// Operates in the gamma-encoded space the colour is picked in (it's a UI accent,
/// not a physical mix). Used for the guide hover (180°) and grab (90°) highlights.
fn shift_hue(rgb: [f32; 3], deg: f32) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    // Hue in degrees.
    let mut h = if d <= 1e-6 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    h = (h + deg).rem_euclid(360.0);
    let s = if max <= 1e-6 { 0.0 } else { d / max };
    let v = max;
    // HSV -> RGB.
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r1 + m, g1 + m, b1 + m]
}

/// The guide hover highlight: the inverse hue (180° rotation) of the base colour.
fn inverse_hue(rgb: [f32; 3]) -> [f32; 3] {
    shift_hue(rgb, 180.0)
}

/// Push `item` onto `stack`, dropping the oldest entry first if it's at `cap`, so
/// the stack never exceeds `cap` (the undo / redo history bound).
fn push_capped<T>(stack: &mut Vec<T>, item: T, cap: usize) {
    if cap == 0 {
        return;
    }
    if stack.len() >= cap {
        stack.remove(0);
    }
    stack.push(item);
}

/// Decide which axis the next G subdivision should split, given the completed
/// subdivision levels per axis (`h` horizontal, `v` vertical), whether there are
/// no guides yet, and the image `aspect` (W/H). Returns `Some(true)` for a
/// horizontal guide level, `Some(false)` for vertical, or `None` when both axes
/// are already at the 1/32 cap.
///
/// The grid converges toward square cells: cell aspect = `aspect · 2^h / 2^v`,
/// and each press picks the axis whose next level brings `|ln(cell aspect)|`
/// closest to 0 (subdividing the longer cell edge first). The very first guide is
/// always horizontal; ties go to horizontal, so once cells are square the axes
/// then alternate H/V.
fn next_guide_horizontal(h: u32, v: u32, guides_empty: bool, aspect: f32) -> Option<bool> {
    if guides_empty {
        return Some(true);
    }
    match (h >= 5, v >= 5) {
        (true, true) => None,
        (true, false) => Some(false),
        (false, true) => Some(true),
        (false, false) => {
            use std::f32::consts::LN_2;
            let ln_a = aspect.max(1e-4).ln();
            let after_h = (ln_a + (h as f32 + 1.0 - v as f32) * LN_2).abs();
            let after_v = (ln_a + (h as f32 - v as f32 - 1.0) * LN_2).abs();
            Some(after_h <= after_v)
        }
    }
}

/// Map a numpad key to a zoom digit 1..=9, else `None`.
fn numpad_digit(key: &PhysicalKey) -> Option<u32> {
    let PhysicalKey::Code(code) = key else {
        return None;
    };
    Some(match code {
        KeyCode::Numpad1 => 1,
        KeyCode::Numpad2 => 2,
        KeyCode::Numpad3 => 3,
        KeyCode::Numpad4 => 4,
        KeyCode::Numpad5 => 5,
        KeyCode::Numpad6 => 6,
        KeyCode::Numpad7 => 7,
        KeyCode::Numpad8 => 8,
        KeyCode::Numpad9 => 9,
        _ => return None,
    })
}

/// Map a top-row digit key to a comparator slot 1..=9, else `None`.
fn toprow_digit(key: &PhysicalKey) -> Option<u32> {
    let PhysicalKey::Code(code) = key else {
        return None;
    };
    Some(match code {
        KeyCode::Digit1 => 1,
        KeyCode::Digit2 => 2,
        KeyCode::Digit3 => 3,
        KeyCode::Digit4 => 4,
        KeyCode::Digit5 => 5,
        KeyCode::Digit6 => 6,
        KeyCode::Digit7 => 7,
        KeyCode::Digit8 => 8,
        KeyCode::Digit9 => 9,
        _ => return None,
    })
}

/// Atomically move and resize the window to the given OUTER rect (physical px)
/// in a single OS call, so the frame doesn't visibly jump in two steps the way
/// `set_outer_position` followed by `request_inner_size` does. Returns false if
/// the native handle couldn't be obtained.
#[cfg(windows)]
fn set_window_outer_rect(window: &Window, x: i32, y: i32, w: u32, h: u32) -> bool {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetWindowPos, SWP_NOACTIVATE, SWP_NOZORDER};

    let Ok(handle) = window.window_handle() else {
        return false;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return false;
    };
    let hwnd = win32.hwnd.get() as *mut core::ffi::c_void;
    // SAFETY: `hwnd` is a live top-level window owned by `window`. With
    // SWP_NOZORDER the insert-after handle is ignored, so null is fine.
    unsafe {
        SetWindowPos(
            hwnd,
            std::ptr::null_mut(),
            x,
            y,
            w as i32,
            h as i32,
            SWP_NOZORDER | SWP_NOACTIVATE,
        );
    }
    true
}

#[cfg(not(windows))]
fn set_window_outer_rect(window: &Window, x: i32, y: i32, w: u32, h: u32) -> bool {
    window.set_outer_position(PhysicalPosition::new(x, y));
    let _ = window.request_inner_size(PhysicalSize::new(w, h));
    true
}

/// Clip the window to a rounded rectangle of `radius` physical pixels (0 = square)
/// via a Win32 window region. Must be re-applied whenever the window size changes
/// (the region is in window-local pixels). DWM corner presets only offer fixed
/// sizes, so an explicit region is used to honour an arbitrary radius.
#[cfg(windows)]
fn apply_window_corners(window: &Window, radius: u32) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return;
    };
    let hwnd = win32.hwnd.get() as *mut core::ffi::c_void;
    let size = window.outer_size();
    let (w, h) = (size.width as i32, size.height as i32);
    if w <= 0 || h <= 0 {
        return;
    }
    // SAFETY: `hwnd` is a live top-level window owned by `window`. SetWindowRgn
    // takes ownership of the new region (freeing any previous one); a null region
    // clears it back to a plain rectangle.
    unsafe {
        let rgn = if radius > 0 {
            // Right/bottom are exclusive, so +1 to include the last column/row.
            CreateRoundRectRgn(0, 0, w + 1, h + 1, radius as i32 * 2, radius as i32 * 2)
        } else {
            std::ptr::null_mut()
        };
        // bRedraw = 0: the GL scene is presented via swap_buffers every frame, so
        // the new clip shows on the next swap without forcing an extra repaint
        // (which would fight the geometry animation).
        SetWindowRgn(hwnd, rgn, 0);
    }
}

#[cfg(not(windows))]
fn apply_window_corners(_window: &Window, _radius: u32) {}

/// Disable DWM non-client rendering for the borderless window so the legacy
/// caption/border doesn't flash on focus changes. (DWMNCRP_DISABLED.)
#[cfg(windows)]
fn disable_dwm_decorations(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_NCRENDERING_POLICY};

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return;
    };
    let hwnd = win32.hwnd.get() as *mut core::ffi::c_void;
    // SAFETY: `hwnd` is a live top-level window; we pass a 4-byte DWORD = 1
    // (DWMNCRP_DISABLED) for the non-client rendering policy.
    let policy: u32 = 1;
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_NCRENDERING_POLICY as u32,
            &policy as *const u32 as *const core::ffi::c_void,
            4,
        );
    }
}

#[cfg(not(windows))]
fn disable_dwm_decorations(_window: &Window) {}

/// Subclass the window proc to stop the classic GDI non-client frame from being
/// painted on focus change and restore-from-minimize.
///
/// winit deliberately keeps `WS_CAPTION | WS_SIZEBOX` even for `with_decorations
/// (false)` windows (it needs them for Aero snap) and implements "borderless"
/// only by overriding `WM_NCCALCSIZE` to expand the client area over the whole
/// window. The frame is therefore invisible *except* when `DefWindowProc`
/// repaints it directly: on `WM_NCACTIVATE` (to show the active/inactive caption)
/// and `WM_NCPAINT`. Those paints flash the old-style titlebar before our GL
/// content redraws over it. winit's own `WM_NCACTIVATE` handler forwards to
/// `DefWindowProc` with the real `lParam`, so it can't be fixed from the winit
/// side — we wrap winit's proc and:
///   * `WM_NCACTIVATE`: forward to winit with `lParam = -1`. winit still updates
///     its active-focus state from `wParam`, but the `-1` tells the inner
///     `DefWindowProc` not to repaint the non-client area (the Chromium trick).
///   * `WM_NCPAINT`: swallow (return 0) — the client covers the whole window, so
///     there is never anything legitimate to paint in the non-client area.
///
/// Everything else chains unchanged to winit's proc.
#[cfg(windows)]
fn suppress_nonclient_frame(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use std::sync::atomic::{AtomicIsize, Ordering};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallWindowProcW, SetWindowLongPtrW, GWLP_WNDPROC, WM_NCACTIVATE, WM_NCPAINT,
    };

    type WndProc = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;
    // The original winit window proc, saved so our wrapper can chain to it. The
    // app owns exactly one window for its lifetime, so a single static suffices.
    static WINIT_PROC: AtomicIsize = AtomicIsize::new(0);

    unsafe extern "system" fn wrapper(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        let winit_proc: WndProc =
            unsafe { std::mem::transmute(WINIT_PROC.load(Ordering::Relaxed)) };
        match msg {
            // Suppress the active/inactive caption repaint but let winit keep
            // tracking focus (it reads wParam, not lParam).
            WM_NCACTIVATE => unsafe { CallWindowProcW(Some(winit_proc), hwnd, msg, wp, -1) },
            // Nothing legitimate lives in the non-client area; never paint it.
            WM_NCPAINT => 0,
            _ => unsafe { CallWindowProcW(Some(winit_proc), hwnd, msg, wp, lp) },
        }
    }

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return;
    };
    let hwnd = win32.hwnd.get() as HWND;
    // Install once: swap in our wrapper and remember winit's proc. Guard against
    // a double-install (which would chain the wrapper to itself → recursion).
    if WINIT_PROC.load(Ordering::Relaxed) != 0 {
        return;
    }
    // SAFETY: `hwnd` is a live top-level window owned by `window`; `wrapper` has
    // the correct WNDPROC ABI and chains to the proc we displace.
    unsafe {
        let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wrapper as *const () as isize);
        WINIT_PROC.store(prev, Ordering::Relaxed);
    }
}

#[cfg(not(windows))]
fn suppress_nonclient_frame(_window: &Window) {}

/// Set the window's title-bar (small) and taskbar (big) icons from the bundled
/// multi-resolution `app_icon.ico`, picking the exact native pixel sizes so they
/// stay crisp instead of being scaled from a single bitmap (as winit would).
#[cfg(windows)]
fn set_window_icons(window: &Window) {
    use std::os::windows::ffi::OsStrExt;

    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, LoadImageW, SendMessageW, ICON_BIG, ICON_SMALL, IMAGE_ICON,
        LR_LOADFROMFILE, SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, WM_SETICON,
    };

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(h) = handle.as_raw() else {
        return;
    };
    let hwnd = h.hwnd.get() as *mut core::ffi::c_void;

    let ico = resolve_resources_dir().join("icons").join("app_icon.ico");
    let wide: Vec<u16> = ico
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    // SAFETY: `wide` is a valid NUL-terminated path; LoadImageW returns null on
    // failure (guarded), and the resulting HICON is owned by the window.
    unsafe {
        let load = |cx: i32, cy: i32| {
            LoadImageW(
                std::ptr::null_mut(),
                wide.as_ptr(),
                IMAGE_ICON,
                cx,
                cy,
                LR_LOADFROMFILE,
            )
        };
        let big = load(GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON));
        if !big.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_BIG as usize, big as isize);
        }
        let small = load(GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON));
        if !small.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL as usize, small as isize);
        }
    }
}

#[cfg(not(windows))]
fn set_window_icons(_window: &Window) {}

/// Load the bundled app icon as an egui texture (≈36 px, drawn smaller) for the
/// custom titlebar. `None` if it can't be read/decoded.
fn load_titlebar_icon(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let path = resolve_resources_dir().join("icons").join("app_icon.png");
    let img = image::ImageReader::open(&path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()?
        .resize_exact(36, 36, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    Some(ctx.load_texture("titlebar_icon", color, egui::TextureOptions::LINEAR))
}

/// Locate the bundled `resources/` directory: next to the exe (packaged), the
/// current working dir (dev), or the compile-time manifest dir (fallback).
fn resolve_resources_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("resources");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    let cwd = PathBuf::from("resources");
    if cwd.exists() {
        return cwd;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources")
}

#[cfg(debug_assertions)]
fn install_debug_callback(gl: &mut glow::Context) {
    if !gl.supports_debug() {
        return;
    }
    unsafe {
        gl.enable(glow::DEBUG_OUTPUT);
        gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
        gl.debug_message_callback(|_source, _gltype, _id, severity, message| match severity {
            glow::DEBUG_SEVERITY_HIGH => log::error!("GL: {message}"),
            glow::DEBUG_SEVERITY_MEDIUM => log::warn!("GL: {message}"),
            _ => log::debug!("GL: {message}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 2.0 / 255.0)
    }

    #[test]
    fn pick_nearest_auto_switches_above_200_percent() {
        // Auto: nearest strictly above 200%, bilinear at/below, bilinear in pano.
        assert!(!pick_nearest(true, false, Some(1.0)), "100% -> bilinear");
        assert!(
            !pick_nearest(true, false, Some(2.0)),
            "exactly 200% -> bilinear"
        );
        assert!(
            pick_nearest(true, false, Some(2.0001)),
            "just over 200% -> nearest"
        );
        assert!(pick_nearest(true, false, Some(8.0)), "800% -> nearest");
        assert!(
            !pick_nearest(true, true, None),
            "panorama -> bilinear (manual ignored in auto)"
        );
    }

    #[cfg(windows)]
    #[test]
    fn friendly_type_names_are_distinct_and_sortable() {
        assert_eq!(friendly_type_name("jpg"), "JPEG Image");
        assert_eq!(friendly_type_name("jpeg"), "JPEG Image");
        assert_eq!(friendly_type_name("exr"), "OpenEXR Image");
        // RAW clusters under a common prefix but stays per-brand distinct.
        assert!(friendly_type_name("nef").starts_with("Camera RAW Image"));
        assert!(friendly_type_name("cr2").starts_with("Camera RAW Image"));
        assert_ne!(friendly_type_name("nef"), friendly_type_name("cr2"));
        // RAW differs from JPEG, so a folder can be sorted to separate them.
        assert_ne!(friendly_type_name("nef"), friendly_type_name("jpg"));
        // Unknown extensions fall back to "<EXT> Image".
        assert_eq!(friendly_type_name("xyz"), "XYZ Image");
    }

    #[test]
    fn pano_center_scale_math() {
        // Narrower FOV (more zoomed in) ⇒ larger texel→screen scale; wider ⇒ smaller.
        let narrow = pano_center_scale(1000.0, 512.0, (30f32).to_radians().tan()); // 60° FOV
        let wide = pano_center_scale(1000.0, 512.0, (70f32).to_radians().tan()); // 140° FOV
        assert!(
            narrow > wide,
            "narrow FOV must magnify more: {narrow} vs {wide}"
        );
        // The 200% switch (scale == 2) is self-consistently at tan(½fov)=π·vh/(4·ih).
        let tan_at_2 = std::f32::consts::PI * 1000.0 / (4.0 * 512.0);
        let s = pano_center_scale(1000.0, 512.0, tan_at_2);
        assert!((s - 2.0).abs() < 1e-3, "expected scale 2, got {s}");
        // Zoom in a touch past it (smaller tan) ⇒ nearest; out ⇒ bilinear.
        assert!(pano_center_scale(1000.0, 512.0, tan_at_2 * 0.95) > 2.0);
        assert!(pano_center_scale(1000.0, 512.0, tan_at_2 * 1.05) < 2.0);
    }

    #[test]
    fn pick_nearest_manual_uses_pinned_value() {
        // Manual (auto off): the pinned value wins regardless of zoom / mode.
        assert!(pick_nearest(false, true, Some(1.0)));
        assert!(pick_nearest(false, true, None));
        assert!(!pick_nearest(false, false, Some(8.0)));
    }

    #[test]
    fn i_key_toggle_flips_effective_then_persists() {
        // Model the I-key handler: `manual = !effective; auto = false`, where
        // `effective = pick_nearest(auto, manual, scale)`. It must flip whatever is
        // currently on screen, lock it (auto off), and keep toggling thereafter.
        let press = |auto: bool, manual: bool, scale: Option<f32>| {
            let eff = pick_nearest(auto, manual, scale);
            (false, !eff) // (new auto, new manual)
        };

        // From auto-nearest (250%): I -> manual bilinear, and it stays bilinear even
        // if we later zoom back under 200%.
        let (auto, manual) = press(true, false, Some(2.5));
        assert!(
            !auto && !pick_nearest(auto, manual, Some(2.5)),
            "250%: nearest -> bilinear"
        );
        assert!(
            !pick_nearest(auto, manual, Some(1.0)),
            "persists under 200%"
        );

        // From auto-bilinear (150%): I -> manual nearest.
        let (auto, manual) = press(true, false, Some(1.5));
        assert!(
            !auto && pick_nearest(auto, manual, Some(1.5)),
            "150%: bilinear -> nearest"
        );

        // Pressing I again flips back (still manual).
        let (auto, manual) = press(auto, manual, Some(1.5));
        assert!(
            !auto && !pick_nearest(auto, manual, Some(1.5)),
            "second press flips back"
        );
    }

    #[test]
    fn fullscreen_fit_never_upscales() {
        // On-screen scale (device px per image px) for a zoom is zoom * vh / ih.
        let scale = |z: f32, ih: f32, vh: f32| z * vh / ih;
        let eps = 1e-4;
        // Sub-screen image -> native 1:1 (scale == 1).
        let z = fit_zoom_no_upscale(2560.0, 1440.0, 800.0, 600.0);
        assert!((scale(z, 600.0, 1440.0) - 1.0).abs() < eps, "small -> 1:1");
        // Larger-than-screen image -> shrinks to fit (scale == fit < 1).
        let z = fit_zoom_no_upscale(2560.0, 1440.0, 4000.0, 3000.0);
        let s = scale(z, 3000.0, 1440.0);
        assert!(
            s < 1.0 && (s - 0.48).abs() < eps,
            "large -> fit 0.48, got {s}"
        );
        // Wide image (wider than the screen, but shorter) -> fits to width.
        let z = fit_zoom_no_upscale(2560.0, 1440.0, 5000.0, 1000.0);
        let s = scale(z, 1000.0, 1440.0);
        assert!(
            (s - 2560.0 / 5000.0).abs() < eps,
            "wide -> fit width, got {s}"
        );
        // Exactly screen-sized -> 1:1.
        let z = fit_zoom_no_upscale(2560.0, 1440.0, 2560.0, 1440.0);
        assert!((scale(z, 1440.0, 1440.0) - 1.0).abs() < eps, "exact -> 1:1");
        // Much-smaller-than-screen -> still strict 1:1 (the camera clamp must not
        // magnify it up to the scroll zoom-out limit).
        let z = fit_zoom_no_upscale(2560.0, 1440.0, 50.0, 50.0);
        assert!((scale(z, 50.0, 1440.0) - 1.0).abs() < eps, "tiny -> 1:1");
    }

    #[test]
    fn abs_diff_zero_for_identical_correct_otherwise() {
        use crate::image_loader::PixelBuffer;
        let mk = |px: Vec<u8>| ImageData {
            path: std::path::PathBuf::from("t.png"),
            width: 2,
            height: 1,
            channels: 4,
            dtype_name: "uint8".into(),
            compression: "-".into(),
            pixels: PixelBuffer::U8(px),
            is_encoded_srgb: true,
            animation: None,
            camera: None,
            clip_max: crate::image_loader::CLIP_MAX_NORM,
        };
        let a = mk(vec![10, 20, 30, 255, 100, 100, 100, 255]);
        let b = mk(vec![10, 20, 30, 255, 40, 90, 160, 255]);
        let d = abs_diff_image(&a, &b).unwrap();
        let PixelBuffer::U8(v) = &d.pixels else {
            panic!()
        };
        // Pixel 0 is identical -> 0; pixel 1 -> |100-40|,|100-90|,|100-160|.
        assert_eq!(&v[0..3], &[0, 0, 0]);
        assert_eq!(&v[4..7], &[60, 10, 60]);
        // A self-diff is exactly zero everywhere (the property the GPU mip chain
        // preserves at every LOD, so identical regions read 0 at any zoom).
        let s = abs_diff_image(&a, &a).unwrap();
        let PixelBuffer::U8(sv) = &s.pixels else {
            panic!()
        };
        assert!(sv.iter().all(|&x| x == 0));
    }

    #[test]
    fn pano_rotate_matches_shader_rotation() {
        let close = |a: (f32, f32, f32), b: (f32, f32, f32)| {
            (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5 && (a.2 - b.2).abs() < 1e-5
        };
        // Identity at yaw=pitch=0.
        assert!(close(pano_rotate(0.0, 0.0, 0.2, 0.3, 0.9), (0.2, 0.3, 0.9)));
        // Yaw +90° sends forward (0,0,1) to (-1,0,0) (matches the shader's `my`).
        let h = std::f32::consts::FRAC_PI_2;
        assert!(close(pano_rotate(h, 0.0, 0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)));
        // Pitch +90° sends forward (0,0,1) to (0,1,0) (matches the shader's `mp`).
        assert!(close(pano_rotate(0.0, h, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0)));
    }

    #[test]
    fn inverse_hue_red_is_cyan() {
        // The default guide red (255,80,80) inverts to cyan (80,255,255): hue
        // rotated 180°, saturation and value unchanged.
        let red = srgb_u8_to_f32([255, 80, 80]);
        let got = inverse_hue(red);
        assert!(close(got, srgb_u8_to_f32([80, 255, 255])), "got {got:?}");
    }

    #[test]
    fn inverse_hue_is_an_involution_and_preserves_sv() {
        // Inverting twice returns the original; a grey (no hue) is unchanged.
        for c in [[200, 120, 40], [10, 200, 90], [128, 128, 128], [0, 0, 0]] {
            let orig = srgb_u8_to_f32(c);
            assert!(close(inverse_hue(inverse_hue(orig)), orig), "color {c:?}");
        }
        // Grey has no hue to rotate, so it must map to itself.
        let grey = srgb_u8_to_f32([90, 90, 90]);
        assert!(close(inverse_hue(grey), grey));
    }

    /// Simulate pressing G `n` times for an image of the given aspect, returning
    /// the axis sequence ('H'/'V', or '.' when both axes are full).
    fn guide_axis_sequence(aspect: f32, n: usize) -> String {
        let (mut h, mut v) = (0u32, 0u32);
        let mut empty = true;
        let mut out = String::new();
        for _ in 0..n {
            match next_guide_horizontal(h, v, empty, aspect) {
                Some(true) => {
                    out.push('H');
                    h += 1;
                }
                Some(false) => {
                    out.push('V');
                    v += 1;
                }
                None => out.push('.'),
            }
            empty = false;
        }
        out
    }

    #[test]
    fn guide_order_first_is_always_horizontal() {
        // Regardless of aspect, the very first guide is horizontal.
        for a in [0.25_f32, 0.5, 1.0, 2.0, 32.0] {
            assert_eq!(&guide_axis_sequence(a, 1), "H", "aspect {a}");
        }
    }

    #[test]
    fn guide_order_hdri_2to1() {
        // A 2:1 HDRI: H, then V V (→ 8 equal squares), then alternating H/V.
        assert_eq!(guide_axis_sequence(2.0, 10), "HVVHVHVHVH");
    }

    #[test]
    fn guide_order_very_wide_subdivides_long_edge_first() {
        // A 32:1 image: split the width (vertical guides) down to the 1/32 cap
        // before the 2nd horizontal guide.
        assert_eq!(guide_axis_sequence(32.0, 7), "HVVVVVH");
    }

    #[test]
    fn push_capped_drops_oldest_at_cap() {
        let mut s: Vec<i32> = Vec::new();
        for i in 0..300 {
            push_capped(&mut s, i, 256);
        }
        assert_eq!(s.len(), 256);
        // Oldest 44 (0..44) were dropped; the window is 44..300.
        assert_eq!(*s.first().unwrap(), 44);
        assert_eq!(*s.last().unwrap(), 299);
    }

    #[test]
    fn guide_order_caps_then_stops() {
        // Each axis caps at 5 levels (1/32); once both are full, no more are added.
        let seq = guide_axis_sequence(1.0, 14);
        assert_eq!(seq.matches('H').count(), 5);
        assert_eq!(seq.matches('V').count(), 5);
        assert!(seq.ends_with(".."), "both-full tail: {seq}");
    }
}
