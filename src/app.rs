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
use winit::window::{CursorGrabMode, Fullscreen, ResizeDirection, Window, WindowId};

use crate::camera::{Camera, CameraController};
use crate::image_loader::{
    is_equirectangular, is_supported, load_image, probe_dimensions, supported_extensions, ImageData,
};
use crate::ocio::OcioManager;
use crate::prefs::{AppPreferences, PreferredView, WindowGeometry};
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

/// Max decoded images kept in memory for instant navigation (current + a
/// previous + a next, plus one spare so back-and-forth stays cached). Each entry
/// can be several GB for 24k+ images, so this is deliberately small.
const IMAGE_CACHE_CAP: usize = 4;

/// The window auto-sizes to frame the image but never grows past this fraction
/// of the monitor in either dimension (the rest is breathing room / taskbar).
const FILL_FRACTION: f32 = 0.9;

/// Smallest window dimension (physical px); matches `with_min_inner_size`.
const MIN_DIM: u32 = 170;

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

/// An image whose GPU upload is in progress; the view state is applied once the
/// incremental upload completes (`finalize_adopt`).
struct PendingAdopt {
    data: Arc<ImageData>,
    for_compare: bool,
    /// Pre-swap 2D `(zoom, height)` for native-scale matching on a slot recall.
    old_scale: Option<(f32, f32)>,
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

pub struct App {
    proxy: EventLoopProxy<UserEvent>,
    initial_path: Option<PathBuf>,
    gfx: Option<Gfx>,
    capture: Option<Capture>,

    // Background loading.
    load_tx: Sender<LoadResult>,
    load_rx: Receiver<LoadResult>,
    load_gen: u64,
    load_state: LoadState,
    /// When the current load began, for the load-to-ready timing log (§17.1).
    load_start: Instant,

    // View state.
    camera: CameraController,
    exposure: f32,
    gamma: f32,
    wrap_2d: bool,
    /// Nearest-neighbour filtering instead of bilinear (I key).
    nearest_filter: bool,
    show_metadata: bool,

    // Colour management.
    ocio: OcioManager,
    /// Last non-Standard view, for the T toggle.
    last_view: Option<String>,
    prefs: AppPreferences,

    // UI.
    ui_state: UiState,
    toolbar_visible: bool,
    toolbar_hide_deadline: Option<Instant>,
    file_info: FileInfo,
    loaded_path: Option<PathBuf>,
    /// File name of the in-flight / last-attempted load (for overlays).
    pending_name: Option<String>,
    /// Headless-test override: force the toolbar visible (IMGVWR_DEBUG_TOOLBAR).
    force_toolbar: bool,
    /// Headless-test override: force an overlay ("loading"/"error"/"hint").
    force_overlay: Option<String>,

    // Input state.
    modifiers: ModifiersState,
    dragging: bool,
    cursor_pos: PhysicalPosition<f64>,
    /// Cursor position when a pan/look drag began, restored on release (a
    /// confined/locked grab otherwise drops the cursor at the window centre).
    drag_start_cursor: PhysicalPosition<f64>,
    last_left_press: Option<Instant>,
    fullscreen: bool,
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

    // Borderless window interaction.
    /// True while the cursor is inside the window, driving the titlebar fade.
    cursor_in_window: bool,
    /// Eased 0..1 opacity of the auto-hiding titlebar.
    titlebar_alpha: f32,
    /// A left-press landed in a window-move zone (Alt anywhere, or a 2D-fit
    /// body): it becomes an OS move on the first motion, or a click on release.
    window_drag_armed: bool,
    /// Pointer travel since `window_drag_armed`, to tell a click from a drag.
    window_drag_motion: f32,
    /// Set by the titlebar Close button; honoured in `about_to_wait`.
    should_exit: bool,

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

    /// Transient bottom-right status toast.
    toast: Option<Toast>,

    // Image comparator.
    /// The currently-displayed decoded image (shared so a slot can pin it).
    current_image: Option<Arc<ImageData>>,
    /// Comparator slots (Ctrl+1..=9 → index 0..=8); each pins a decoded image.
    slots: [Option<Arc<ImageData>>; 9],
    /// The image shown before the last slot recall, for the A/B toggle-back.
    compare_prev: Option<Arc<ImageData>>,
    /// Slot whose image is currently displayed (drives the active flag).
    active_slot: Option<usize>,

    // F2 metadata box hover-reveal (near the top-right corner).
    metadata_hover: bool,
    metadata_hide_deadline: Option<Instant>,

    /// In-progress adoption waiting on the incremental GPU upload.
    pending: Option<PendingAdopt>,
    /// Upload fraction (0..1) while `pending`, for the progress bar.
    upload_progress: f32,

    /// Timestamp of the previous rendered frame, for frame-rate-independent
    /// easing (`None` until the first frame).
    last_frame: Option<Instant>,
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
        let ocio = OcioManager::new(resolve_resources_dir());
        let prefs = AppPreferences::load();

        Self {
            proxy,
            initial_path,
            gfx: None,
            capture,
            load_tx,
            load_rx,
            load_gen: 0,
            load_state: LoadState::Idle,
            load_start: Instant::now(),
            camera: CameraController::for_image(false),
            exposure: 0.0,
            gamma: 1.0,
            wrap_2d: false,
            nearest_filter: false,
            show_metadata: false,
            ocio,
            last_view: None,
            prefs,
            ui_state: UiState::default(),
            toolbar_visible: false,
            toolbar_hide_deadline: None,
            file_info: FileInfo::default(),
            loaded_path: None,
            pending_name: None,
            // The IMGVWR_DEBUG_* overrides force internal state for headless
            // testing; they are dev-only and ignored in release builds.
            force_toolbar: cfg!(debug_assertions)
                && std::env::var_os("IMGVWR_DEBUG_TOOLBAR").is_some(),
            force_overlay: if cfg!(debug_assertions) {
                std::env::var("IMGVWR_DEBUG_OVERLAY").ok()
            } else {
                None
            },
            modifiers: ModifiersState::empty(),
            dragging: false,
            // Start far from the left edge so the toolbar stays hidden until the
            // cursor actually moves there (and so headless captures are clean).
            cursor_pos: PhysicalPosition::new(1.0e6, 1.0e6),
            drag_start_cursor: PhysicalPosition::new(0.0, 0.0),
            last_left_press: None,
            fullscreen: false,
            manual_window: false,
            suppress_manual_until: None,
            pending_dblclick: false,
            dblclick_motion: 0.0,
            locked: false,
            nav_dir: 1,
            preload_armed: false,
            preload_tx,
            preload_rx,
            preload_gen: 0,
            image_cache: Vec::new(),
            toast: None,
            current_image: None,
            slots: std::array::from_fn(|_| None),
            compare_prev: None,
            active_slot: None,
            metadata_hover: false,
            metadata_hide_deadline: None,
            pending: None,
            upload_progress: 0.0,
            last_frame: None,
            animating: false,
            cursor_in_window: false,
            titlebar_alpha: 0.0,
            window_drag_armed: false,
            window_drag_motion: 0.0,
            should_exit: false,
        }
    }

    /// Build the OCIO program for the active display/view and upload it.
    fn rebuild_ocio(&mut self) {
        let shader = self.ocio.build_gpu_shader();
        if let Some(gfx) = &mut self.gfx {
            gfx.renderer.set_ocio_shader(&shader);
        }
        self.request_redraw();
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
            self.last_view
                .clone()
                .filter(|v| !v.eq_ignore_ascii_case(&standard))
                .or_else(|| {
                    views
                        .iter()
                        .find(|v| !v.eq_ignore_ascii_case(&standard))
                        .cloned()
                })
        } else {
            self.last_view = Some(active.view.clone());
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
            if !(is_hdri && self.select_view_named("filmic")) {
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
            .with_min_inner_size(LogicalSize::new(MIN_DIM as f64, MIN_DIM as f64));
        // Probe the initial image's dimensions from its header (cheap, no decode)
        // so the window opens already framing it — eliminating the size/position
        // jump that a post-decode resize would cause. RAW and equirectangular
        // images aren't pre-sized (no cheap probe / panoramas keep the window).
        let monitor = event_loop.primary_monitor();
        let scale = monitor.as_ref().map(|m| m.scale_factor()).unwrap_or(1.0);
        let probed = self
            .initial_path
            .as_ref()
            .and_then(|p| probe_dimensions(p))
            .filter(|(w, h)| !is_equirectangular(*w, *h));
        // Set both size and position at creation so the window never visibly
        // jumps into place.
        let (size, position) = startup_geometry(self.prefs.window, probed, monitor.as_ref(), scale);
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
        let renderer = Renderer::new(gl.clone()).context("failed to create renderer")?;
        let egui = egui_glow::EguiGlow::new(event_loop, gl.clone(), None, None, false);
        install_ui_font(&egui.egui_ctx);

        // Crisp multi-resolution title-bar + taskbar icon. Position was set at
        // creation (restored or centred), so no post-creation move here.
        set_window_icons(&window);

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
    fn follow_zoom_with_window(&mut self) {
        if self.fullscreen {
            return;
        }
        let Some(zoom) = self.camera.target_zoom() else {
            return;
        };
        let (img_w, img_h) = (self.file_info.width, self.file_info.height);
        if img_w == 0 || img_h == 0 {
            return;
        }
        let mon = match &self.gfx {
            Some(gfx) if gfx.window.is_maximized() => return,
            Some(gfx) => gfx.window.current_monitor(),
            None => return,
        };
        // A zoom overrides any earlier manual resize.
        self.manual_window = false;
        let (_, vh) = self.viewport();
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
        self.camera.set_zoom(scale * img_h as f32 / win_h as f32);
        self.resize_window_centered(PhysicalSize::new(win_w, win_h));
    }

    /// Resize the window to `target` inner size, keeping it centred on its current
    /// centre and clamped on-screen. Redundant resizes (within 2 px) are skipped
    /// so per-notch zooming doesn't thrash once the window is capped. Records the
    /// resize so the resulting `Resized` event isn't mistaken for a manual drag.
    fn resize_window_centered(&mut self, target: PhysicalSize<u32>) {
        let mut did = false;
        if let Some(gfx) = &self.gfx {
            let cur = gfx.window.inner_size();
            let close = (cur.width as i32 - target.width as i32).abs() <= 2
                && (cur.height as i32 - target.height as i32).abs() <= 2;
            if !close {
                if let Ok(op) = gfx.window.outer_position() {
                    let outer = gfx.window.outer_size();
                    // Carry the decoration delta so the centre is computed in
                    // outer-frame coordinates.
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
                    // Move and resize in a single OS call so the frame doesn't
                    // visibly jump in two steps (set_outer_position then
                    // request_inner_size would). `target` is the inner size we
                    // want; SetWindowPos takes the outer rect.
                    did = set_window_outer_rect(&gfx.window, x, y, new_outer_w, new_outer_h);
                }
            }
        }
        if did {
            // Cover both the move and resize events the OS will deliver.
            self.suppress_manual_until = Some(Instant::now() + Duration::from_millis(250));
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

        let tx = self.load_tx.clone();
        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name(format!("image-load-{gen}"))
            .spawn(move || {
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| load_image(&path)))
                {
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
        if !for_compare {
            log::info!(
                "decoded in {:.2}s, uploading {}",
                self.load_start.elapsed().as_secs_f32(),
                data.path.display()
            );
        }
        if let Some(gfx) = &mut self.gfx {
            gfx.renderer.start_upload(data.as_ref());
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

        // Keep the current zoom/pan/exposure when the projection mode matches —
        // for the L lock, and always for a comparator recall (to compare the
        // same region). A 2D <-> panorama change resets to the per-image default.
        let keep_view = (self.locked || for_compare)
            && self.loaded_path.is_some()
            && self.camera.is_panorama() == equirect;

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
        };
        self.loaded_path = Some(data.path.clone());
        if !keep_view {
            self.camera = CameraController::for_image(equirect);
            self.exposure = 0.0;
            self.gamma = 1.0;
            self.wrap_2d = false;
        }
        self.load_state = LoadState::Loaded;
        self.update_window_title();
        // Choose the OCIO view: panoramas restore the saved view for their
        // extension; HDRIs default to Filmic, everything else to Standard.
        self.select_view_for_load(equirect, &data.path);
        // Frame the window to the loaded 2D image (panoramas keep the window;
        // locked/compared views keep the current size for side-by-side compare).
        // The window was pre-sized at creation for the initial image, so this is
        // a no-op resize there; for navigation it re-frames and re-centres.
        if !equirect && !self.locked && !for_compare {
            self.resize_window_to_image(data.width, data.height);
        }
        self.apply_debug_overrides();

        // Look-ahead: once arrow-navigating, decode the next neighbour in the
        // background so the following arrow press is instant. Not for a recall.
        if self.preload_armed && !for_compare {
            if let Some(next) = sibling_path(&data.path, self.nav_dir) {
                self.start_preload(next);
            }
        }

        self.current_image = Some(data.clone());
        // Dev-only: IMGVWR_DEBUG_SLOT pins the loaded image into slot 1 so the
        // comparator flag can be verified headlessly.
        #[cfg(debug_assertions)]
        if std::env::var_os("IMGVWR_DEBUG_SLOT").is_some() && self.slots[0].is_none() {
            self.slots[0] = self.current_image.clone();
        }
        self.recompute_active_slot();
        // Retain the decoded image so navigating back to it is instant.
        self.cache_insert(data);
        // Match on-screen pixel scale for a comparator swap (native resolution).
        self.preserve_native_scale(old_scale);
        // The new image starts settled — freeze the easing target at the camera
        // we just configured (incl. any debug override) so it doesn't animate in.
        self.camera.settle();
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
        self.recompute_active_slot();
        self.show_toast(format!("Saved slot {n}"));
        self.request_redraw();
    }

    /// N: recall comparator slot `n`. Pressing it again while already viewing
    /// that slot toggles back to the previously-shown image (A/B compare).
    fn recall_slot(&mut self, n: usize) {
        let idx = n - 1;
        let Some(target) = self.slots[idx].clone() else {
            return;
        };
        let old_scale = self.flat_scale_ref();
        if self.active_slot == Some(idx) {
            // Toggle back to the previously-viewed image (swap so a third press
            // returns to the slot).
            if let Some(prev) = self.compare_prev.take() {
                self.compare_prev = self.current_image.clone();
                self.begin_adopt(prev, true, old_scale);
            }
        } else {
            self.compare_prev = self.current_image.clone();
            self.begin_adopt(target, true, old_scale);
        }
    }

    /// `(zoom, image_height)` of the current 2D view, for native-scale matching.
    fn flat_scale_ref(&self) -> Option<(f32, f32)> {
        match self.camera.camera {
            Camera::Flat { zoom, .. } => Some((zoom, self.file_info.height.max(1) as f32)),
            Camera::Pano { .. } => None,
        }
    }

    /// After a comparator swap between 2D images of different resolutions, adjust
    /// the (fit-relative) zoom so the on-screen pixel scale is unchanged — each
    /// image is shown at its native resolution rather than scaled to match.
    fn preserve_native_scale(&mut self, old: Option<(f32, f32)>) {
        if let (Some((old_zoom, old_h)), Camera::Flat { .. }) = (old, self.camera.camera) {
            let new_h = self.file_info.height.max(1) as f32;
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
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| load_image(&path)))
                {
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
        let Some(target) = sibling_path(&current, dir) else {
            return;
        };
        self.nav_dir = dir;
        self.preload_armed = true;
        self.load_path(target);
    }

    /// Home: ease pan+zoom back to the fit view (2D) or default FOV (panorama).
    /// Tone adjustments are intentionally left alone (Ctrl+R resets those).
    fn reset_view_full(&mut self) {
        if self.camera.is_panorama() {
            // Snap the look back to centre (easing a spun-around yaw would whirl
            // back), ease the FOV to the default.
            self.camera.snap_look(0.0, 0.0);
            self.camera.set_fov(crate::camera::DEFAULT_PANO_FOV_DEG);
        } else {
            let (vw, vh) = self.viewport();
            let aspect = self
                .gfx
                .as_ref()
                .and_then(|g| g.renderer.image_aspect())
                .unwrap_or(1.0);
            // Contain-fit: largest zoom that still shows the whole image. Ease
            // zoom and pan toward it within the current window (Home doesn't
            // re-frame the window).
            let fit = (vw / vh / aspect.max(1e-4)).min(1.0);
            self.camera.set_zoom(fit);
            self.camera.set_pan_target(Vec2::ZERO);
        }
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
        let text = fmt_ev(self.exposure);
        self.show_toast(text);
    }

    fn show_gamma_toast(&mut self) {
        let text = format!("Gamma {:.1}", self.gamma);
        self.show_toast(text);
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

    /// Target 2D zoom as a percentage where 100% == 1 image px : 1 monitor px.
    fn flat_zoom_percent(&self) -> Option<f32> {
        let img_h = self.file_info.height;
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
        let img_h = self.file_info.height.max(1) as f32;
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
        self.follow_zoom_with_window();
        if let Some(z0) = z0 {
            self.zoom_toward_cursor(z0);
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
                let image_aspect = self
                    .gfx
                    .as_ref()
                    .and_then(|g| g.renderer.image_aspect())
                    .unwrap_or(1.0);
                let sx = inv_zoom * (vw / vh) / image_aspect.max(1e-4);
                let sy = inv_zoom;
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
        if let Camera::Flat { pan, .. } = &mut self.camera.camera {
            pan.x = (pan.x + 0.5).rem_euclid(1.0) - 0.5;
            pan.y = (pan.y + 0.5).rem_euclid(1.0) - 0.5;
        }
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
        self.follow_zoom_with_window();
        if let Some(z0) = z0 {
            self.zoom_toward_cursor(z0);
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
        let aspect = self
            .gfx
            .as_ref()
            .and_then(|g| g.renderer.image_aspect())
            .unwrap_or(1.0)
            .max(1e-4);
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
                let image_aspect = self
                    .gfx
                    .as_ref()
                    .and_then(|g| g.renderer.image_aspect())
                    .unwrap_or(1.0);
                let sx = inv_zoom * (vw / vh) / image_aspect.max(1e-4);
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
                    self.gamma = (self.gamma - 0.1).max(0.1);
                    self.show_gamma_toast();
                } else {
                    self.exposure -= 0.5;
                    self.show_exposure_toast();
                }
            }
            (_, Some(".")) => {
                if ctrl {
                    self.gamma = (self.gamma + 0.1).min(4.0);
                    self.show_gamma_toast();
                } else {
                    self.exposure += 0.5;
                    self.show_exposure_toast();
                }
            }
            // Ctrl+R: reset exposure & gamma.
            (_, Some("r")) | (_, Some("R")) if ctrl => {
                self.exposure = 0.0;
                self.gamma = 1.0;
                self.show_toast(format!("{}   Gamma {:.1}", fmt_ev(0.0), 1.0));
            }
            (_, Some("p")) | (_, Some("P")) => {
                let want = !self.camera.is_panorama();
                self.camera.set_mode(want);
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
                self.nearest_filter = !self.nearest_filter;
                self.show_toast(
                    if self.nearest_filter {
                        "Nearest"
                    } else {
                        "Bilinear"
                    }
                    .to_string(),
                );
            }
            (_, Some("q")) | (_, Some("Q")) => self.escape_or_exit(event_loop),
            (Key::Named(NamedKey::F2), _) => {
                self.show_metadata = !self.show_metadata;
                log::info!("metadata overlay -> {}", self.show_metadata);
            }
            (Key::Named(NamedKey::Home), _) | (Key::Named(NamedKey::Backspace), _) => {
                self.reset_view_full()
            }
            (Key::Named(NamedKey::F11), _) => self.toggle_fullscreen(),
            (Key::Named(NamedKey::ArrowRight), _) => self.navigate(1),
            (Key::Named(NamedKey::ArrowLeft), _) => self.navigate(-1),
            (Key::Named(NamedKey::Escape), _) => {
                if self.ui_state.show_help {
                    self.ui_state.show_help = false;
                } else {
                    self.escape_or_exit(event_loop);
                }
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
        self.fullscreen = on;
        if let Some(gfx) = &self.gfx {
            gfx.window
                .set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
        }
    }

    /// True when the window is too small to comfortably overlay the metadata box
    /// (so its hover auto-reveal is suppressed).
    fn window_is_small(&self) -> bool {
        let (vw, vh) = self.viewport();
        vw < 480.0 || vh < 360.0
    }

    /// The titlebar reveals only while the cursor is within the window and near
    /// its top edge (so it doesn't cover the image while looking around lower
    /// down). Hidden in fullscreen.
    fn titlebar_should_show(&self) -> bool {
        if self.fullscreen || !self.cursor_in_window {
            return false;
        }
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor())
            .unwrap_or(1.0);
        self.cursor_pos.y <= 56.0 * scale
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
        let aspect = self
            .gfx
            .as_ref()
            .and_then(|g| g.renderer.image_aspect())
            .unwrap_or(1.0);
        let fit = (vw / vh / aspect.max(1e-4)).min(1.0);
        zoom <= fit * 1.001
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
        self.gfx
            .as_ref()
            .is_some_and(|g| g.window.drag_resize_window(dir).is_ok())
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

    /// Begin an Alt+right-drag resize from the cursor's third; returns whether
    /// one started.
    fn start_third_resize(&mut self) -> bool {
        let Some(dir) = self.resize_third_at_cursor() else {
            return false;
        };
        self.gfx
            .as_ref()
            .is_some_and(|g| g.window.drag_resize_window(dir).is_ok())
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        // Don't begin a pan/look gesture when pressing inside the metadata box,
        // so its text stays selectable.
        if state == ElementState::Pressed && self.ui_state.pointer_over_metadata {
            return;
        }
        match (state, button) {
            (ElementState::Pressed, MouseButton::Left) => {
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
            (ElementState::Pressed, MouseButton::Middle) => self.start_drag(),
            (ElementState::Released, MouseButton::Left) => {
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
            (ElementState::Released, MouseButton::Middle) => self.end_drag(),
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
        }
        if let Some(v) = f("IMGVWR_DEBUG_GAMMA") {
            self.gamma = v;
        }
        if let Ok(p) = std::env::var("IMGVWR_DEBUG_PROJECTION") {
            self.camera.set_mode(p.eq_ignore_ascii_case("pano"));
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
            Camera::Flat { zoom, .. } => {
                if let Some(v) = f("IMGVWR_DEBUG_ZOOM") {
                    *zoom = v;
                }
            }
        }
    }

    // ---- UI --------------------------------------------------------------

    fn update_window_title(&self) {
        if let Some(gfx) = &self.gfx {
            let title = if self.file_info.name.is_empty() {
                "imgvwr".to_string()
            } else {
                format!("{} · imgvwr", self.file_info.name)
            };
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
        let loading = self.is_busy() || forced == Some("loading");
        // Determinate during the GPU upload phase, indeterminate while decoding.
        let progress = self.pending.as_ref().map(|_| self.upload_progress);
        let error = match &self.load_state {
            LoadState::Failed(e) => Some(e.clone()),
            _ if forced == Some("error") => Some("Example decode error: unsupported format".into()),
            _ => None,
        };
        let show_hint = (!has_image && !loading && error.is_none()) || forced == Some("hint");

        UiInputs {
            toolbar_visible: self.toolbar_visible,
            has_image,
            display_views,
            active,
            ocio_available: !self.ocio.display_views().is_empty(),
            exposure: self.exposure,
            gamma: self.gamma,
            loading,
            progress,
            loading_name: self.pending_name.clone(),
            error,
            show_hint,
            // F2 always shows it; the top-right hover auto-reveal is suppressed
            // on a small window (it would cover too much of the image).
            show_metadata: self.show_metadata
                || (self.metadata_hover && !self.window_is_small())
                || self.ui_state.pointer_over_metadata
                || forced == Some("metadata"),
            metadata: self.metadata_lines(),
            show_help: self.ui_state.show_help || forced == Some("help"),
            toast: self.toast_render(),
            slot_labels: self.slot_labels(),
            active_slot: self.active_slot,
            titlebar_alpha: if self.fullscreen {
                0.0
            } else if forced == Some("titlebar") {
                1.0
            } else {
                self.titlebar_alpha
            },
            title: self.file_info.name.clone(),
            is_maximized: self.gfx.as_ref().is_some_and(|g| g.window.is_maximized()),
            resize_cursor: if self.dragging || self.window_drag_armed {
                None
            } else {
                self.resize_edge_at_cursor().map(|d| match d {
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
            },
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
        let view = self
            .ocio
            .active()
            .map(|dv| format!("{}/{}", dv.display, dv.view))
            .unwrap_or_else(|| "gamma 2.2".to_string());
        vec![
            ("File".into(), fi.name.clone()),
            ("Size".into(), format!("{}×{}", fi.width, fi.height)),
            ("Channels".into(), fi.channels.to_string()),
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
            ("View".into(), view),
        ]
    }

    /// Update toolbar show/hide from the cursor position and panel hover (§12.1).
    fn tick_toolbar(&mut self) {
        if self.force_toolbar {
            self.toolbar_visible = true;
            self.ui_state.show_view_submenu = true;
            self.ui_state.show_display_submenu = true;
            return;
        }
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor() as f32)
            .unwrap_or(1.0);
        let near_left = self.cursor_pos.x <= (28.0 * scale) as f64;
        if near_left || self.ui_state.pointer_over_panel {
            self.toolbar_visible = true;
            self.toolbar_hide_deadline = None;
        } else if self.toolbar_visible {
            match self.toolbar_hide_deadline {
                None => {
                    self.toolbar_hide_deadline = Some(Instant::now() + Duration::from_millis(100));
                }
                Some(t) if Instant::now() >= t => {
                    self.toolbar_visible = false;
                    self.toolbar_hide_deadline = None;
                }
                Some(_) => {}
            }
        }
    }

    /// Temporarily reveal the F2 metadata box when the cursor is near the
    /// top-right corner (or hovering the box itself).
    fn tick_metadata(&mut self) {
        let scale = self
            .gfx
            .as_ref()
            .map(|g| g.window.scale_factor() as f32)
            .unwrap_or(1.0);
        let (vw, _) = self.viewport();
        let near_corner = self.cursor_pos.x >= (vw - 240.0 * scale) as f64
            && self.cursor_pos.y <= (140.0 * scale) as f64;
        if near_corner || self.ui_state.pointer_over_metadata {
            self.metadata_hover = true;
            self.metadata_hide_deadline = None;
        } else if self.metadata_hover {
            match self.metadata_hide_deadline {
                None => {
                    self.metadata_hide_deadline = Some(Instant::now() + Duration::from_millis(120));
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
            UiAction::Reload => {
                if let Some(path) = self.loaded_path.clone() {
                    self.ocio.reload();
                    self.rebuild_ocio();
                    self.load_path(path);
                }
            }
            UiAction::SetView { display, view } => {
                if self.ocio.set_active(&display, &view) {
                    log::info!("view transform -> {display}/{view}");
                    // Update the T-toggle baseline so it returns here.
                    if !view.eq_ignore_ascii_case("standard") {
                        self.last_view = Some(view.clone());
                    }
                    self.rebuild_ocio();
                    self.persist_view_if_panorama();
                    self.show_toast(view.clone());
                }
            }
            UiAction::DismissError => {
                self.load_state = LoadState::Idle;
                self.request_redraw();
            }
            UiAction::CloseHelp => {
                self.ui_state.show_help = false;
                self.request_redraw();
            }
            UiAction::RecallSlot(i) => self.recall_slot(i + 1),
            UiAction::SetDefaultApp => match register_default_app() {
                Ok(n) => self.show_toast(format!("Default viewer for {n} file types")),
                Err(e) => {
                    log::error!("set-default failed: {e}");
                    self.show_toast("Could not set default".to_string());
                }
            },
            // Borderless titlebar controls.
            UiAction::DragWindow => {
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
        let file = rfd::FileDialog::new()
            .add_filter("Images", &supported_extensions())
            .pick_file();
        if let Some(path) = file {
            // A manual open ends any arrow-nav preload chain (saved comparator
            // slots persist; only the A/B scratch is dropped).
            self.preload_armed = false;
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

    fn render(&mut self) -> RenderOutcome {
        // Advance any in-progress incremental upload before drawing this frame.
        if self.pending.is_some() {
            self.pump_upload();
        }
        // Advance the zoom/pan easing toward the target (frame-rate independent;
        // dt is clamped so a long idle gap can't cause a jump). `animating`
        // drives the redraw scheduling in `about_to_wait`.
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|prev| now.saturating_duration_since(prev).as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1);
        let cam_moving = self.camera.animate(dt);
        // Ease the titlebar opacity toward shown only while the cursor is near
        // the top edge of the window.
        let tb_target = if self.titlebar_should_show() {
            1.0
        } else {
            0.0
        };
        let tb_k = 1.0 - (-dt / 0.10).exp();
        self.titlebar_alpha += (tb_target - self.titlebar_alpha) * tb_k;
        let tb_settled = (self.titlebar_alpha - tb_target).abs() <= 0.01;
        if tb_settled {
            self.titlebar_alpha = tb_target;
        }
        // Keep scheduling frames while the camera or the titlebar is still
        // moving; both settle, after which about_to_wait returns to Wait (idle).
        self.animating = cam_moving || !tb_settled;
        self.tick_toolbar();
        self.tick_metadata();

        // Gather everything the frame needs before the mutable gfx/ui borrows.
        let inputs = self.ui_inputs();
        let cam = self.camera.camera;
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
            nearest: self.nearest_filter,
        };
        let capture_ready = self.capture_ready();

        let mut actions: Vec<UiAction> = Vec::new();
        let mut grabbed: Option<(i32, i32, Vec<u8>)> = None;

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

            let params = RenderParams {
                viewport: (w, h),
                ..base
            };
            gfx.renderer.render(&params);

            // egui overlay on top of the scene.
            gfx.egui.run(&gfx.window, |ctx| {
                ui::build(ctx, &inputs, ui_state, &mut actions);
            });
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

            if let Err(e) = gfx.gl_surface.swap_buffers(&gfx.gl_context) {
                log::error!("swap_buffers failed: {e}");
            }
        }

        for action in actions {
            self.handle_ui_action(action);
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

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match self.create_gfx(event_loop) {
            Ok(gfx) => {
                gfx.window.request_redraw();
                self.gfx = Some(gfx);
                if let Some(c) = &mut self.capture {
                    c.start = Instant::now();
                }
                // Build the OCIO program (or gamma fallback) before first draw.
                self.rebuild_ocio();
                self.load_initial_image();
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
                if let Some(gfx) = &self.gfx {
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        gfx.gl_surface.resize(&gfx.gl_context, w, h);
                    }
                }
                // Redraw synchronously rather than via request_redraw so the new
                // surface size is presented this frame — otherwise the previous
                // frame is shown stretched to the new size until the next redraw.
                if matches!(self.render(), RenderOutcome::Captured) {
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
                self.tick_toolbar();
                self.tick_metadata();
                self.request_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                self.cursor_in_window = true;
                self.request_redraw();
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_in_window = false;
                self.request_redraw();
            }
            // A press that starts a borderless resize is handled before egui so
            // it wins over the toolbar/titlebar overlapping the window border:
            // left-press on an edge/corner hit-zone, or Alt+right-press anywhere
            // (direction from the cursor's third of the window). Otherwise
            // egui-consumed events fall through to `_ => {}`.
            WindowEvent::MouseInput { state, button, .. } => {
                let resized = state == ElementState::Pressed
                    && !self.fullscreen
                    && match button {
                        MouseButton::Left => self.start_edge_resize(),
                        MouseButton::Right if self.modifiers.alt_key() => self.start_third_resize(),
                        _ => false,
                    };
                if !resized && !egui_consumed {
                    self.on_mouse_button(state, button);
                }
            }
            // Block wheel input when the toolbar is hovered (§11.3).
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => self.on_wheel(delta),
            WindowEvent::DroppedFile(path) => {
                // A manually-dropped file ends any arrow-nav preload chain.
                self.preload_armed = false;
                self.image_cache.clear();
                self.compare_prev = None;
                self.load_path(path);
            }
            WindowEvent::KeyboardInput { event, .. }
                if !egui_consumed && event.state == ElementState::Pressed =>
            {
                // Numpad digits = exact zoom; top-row digits = comparator slots
                // (Ctrl+N saves, N recalls).
                if let Some(digit) = numpad_digit(&event.physical_key) {
                    self.set_exact_zoom(digit, self.ctrl());
                } else if let Some(slot) = toprow_digit(&event.physical_key) {
                    if self.ctrl() {
                        self.save_slot(slot as usize);
                    } else {
                        self.recall_slot(slot as usize);
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
            if self.dragging {
                self.on_drag_motion(delta.0 as f32, delta.1 as f32);
            } else if self.window_drag_armed {
                // Past the click threshold, hand off to the OS move loop (so
                // Aero Snap works); a smaller travel stays a click.
                self.window_drag_motion += (delta.0 * delta.0 + delta.1 * delta.1).sqrt() as f32;
                if self.window_drag_motion >= DBLCLICK_DRAG_TOL {
                    self.window_drag_armed = false;
                    self.pending_dblclick = false;
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
        } else if self.toast_active() || self.animating {
            // Drive ~60 fps while the toast fades or the zoom/pan eases. Both
            // settle quickly, after which we fall through to `Wait` (idle 0% CPU).
            let next = Instant::now() + Duration::from_millis(16);
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            self.request_redraw();
        } else if let Some(deadline) = [self.toolbar_hide_deadline, self.metadata_hide_deadline]
            .into_iter()
            .flatten()
            .min()
        {
            // Wake at the earliest hide deadline (toolbar / metadata box).
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            if Instant::now() >= deadline {
                self.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Persist the windowed (non-fullscreen) geometry for next launch.
        if !self.fullscreen {
            if let Some(gfx) = &self.gfx {
                if let Ok(pos) = gfx.window.outer_position() {
                    let size = gfx.window.inner_size();
                    self.prefs.window = Some(WindowGeometry {
                        x: pos.x,
                        y: pos.y,
                        width: size.width,
                        height: size.height,
                    });
                }
            }
        }
        self.prefs.save();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::LoadFinished(_gen) => self.poll_loads(),
            UserEvent::PreloadFinished(_gen) => self.poll_preloads(),
        }
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

/// Compute the window's inner size and outer position at creation, so it opens
/// already in the right place (no post-load jump):
///
/// * With a probed initial image → frame it (capped to the monitor), centred on
///   the saved window's centre if there is one, else on the screen (first launch).
/// * No image but saved geometry → restore the saved window exactly.
/// * Neither → a centred default.
fn startup_geometry(
    saved: Option<WindowGeometry>,
    probed: Option<(u32, u32)>,
    monitor: Option<&MonitorHandle>,
    scale: f64,
) -> (PhysicalSize<u32>, Option<PhysicalPosition<i32>>) {
    let size = if let Some((iw, ih)) = probed {
        let (w, h) = fit_to_monitor(iw as f32, ih as f32, monitor);
        PhysicalSize::new(w, h)
    } else if let Some(g) = saved {
        PhysicalSize::new(g.width.max(MIN_DIM), g.height.max(MIN_DIM))
    } else {
        PhysicalSize::new((1280.0 * scale) as u32, (720.0 * scale) as u32)
    };

    // No image to size to but a saved window → restore it verbatim.
    if probed.is_none() {
        if let Some(g) = saved {
            return (size, Some(PhysicalPosition::new(g.x, g.y)));
        }
    }

    // Otherwise centre: on the saved window's centre (returning user) or the
    // screen (first launch), clamped on-screen.
    let position = monitor.map(|m| {
        let (mp, ms) = (m.position(), m.size());
        let (cx, cy) = match saved {
            Some(g) => (g.x + g.width as i32 / 2, g.y + g.height as i32 / 2),
            None => (mp.x + ms.width as i32 / 2, mp.y + ms.height as i32 / 2),
        };
        let x = (cx - size.width as i32 / 2)
            .clamp(mp.x, mp.x + (ms.width as i32 - size.width as i32).max(0));
        let y = (cy - size.height as i32 / 2)
            .clamp(mp.y, mp.y + (ms.height as i32 - size.height as i32).max(0));
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
/// Register imgvwr (per-user, no admin) as the handler for every supported
/// extension: a ProgID with an open command + icon, each extension's
/// `OpenWithProgids`, and the classic default association. Returns the number of
/// extensions associated. Note: Windows protects an extension's *current*
/// default with a hashed UserChoice, so already-defaulted types (e.g. .jpg) may
/// still need confirmation in Settings → Default apps; unassociated types
/// (most HDR/EXR/RAW) take effect immediately.
#[cfg(windows)]
fn register_default_app() -> Result<usize, String> {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
    use winreg::RegKey;

    let exe = std::env::current_exe().map_err(|e| format!("exe path: {e}"))?;
    let exe = exe.to_string_lossy().into_owned();
    let progid = "imgvwr.Image";

    let classes = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags(r"Software\Classes", KEY_READ | KEY_WRITE)
        .map_err(|e| format!("open HKCU Classes: {e}"))?;

    // ProgID: friendly name, icon, and open command.
    let (prog, _) = classes.create_subkey(progid).map_err(|e| e.to_string())?;
    prog.set_value("", &"imgvwr Image")
        .map_err(|e| e.to_string())?;
    let (icon, _) = prog
        .create_subkey("DefaultIcon")
        .map_err(|e| e.to_string())?;
    icon.set_value("", &format!("{exe},0"))
        .map_err(|e| e.to_string())?;
    let (cmd, _) = prog
        .create_subkey(r"shell\open\command")
        .map_err(|e| e.to_string())?;
    cmd.set_value("", &format!("\"{exe}\" \"%1\""))
        .map_err(|e| e.to_string())?;

    // Associate each supported extension.
    let mut count = 0usize;
    for ext in crate::image_loader::supported_extensions() {
        let Ok((key, _)) = classes.create_subkey(format!(".{ext}")) else {
            continue;
        };
        if let Ok((owp, _)) = key.create_subkey("OpenWithProgids") {
            let _ = owp.set_value(progid, &"");
        }
        let _ = key.set_value("", &progid);
        count += 1;
    }

    // Refresh shell file-association state (icons / defaults).
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
fn register_default_app() -> Result<usize, String> {
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

/// Format an exposure value as e.g. "+3 EV" or "+0.05 EV".
fn fmt_ev(ev: f32) -> String {
    if ev.fract().abs() < 1e-3 {
        format!("{:+} EV", ev as i32)
    } else {
        format!("{:+.2} EV", ev)
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
