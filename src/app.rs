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
//!   * `IMGVWR_DEBUG_ZOOM` (2-D)
//!   * `IMGVWR_DEBUG_PROJECTION` = `pano` | `flat`
//!   * `IMGVWR_DEBUG_WRAP` = `1`

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
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::camera::{Camera, CameraController};
use crate::image_loader::{load_image, ImageData};
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
}

impl FileInfo {
    fn summary(&self) -> String {
        if self.width == 0 {
            return String::new();
        }
        format!(
            "{}×{} ch:{} {}",
            self.width, self.height, self.channels, self.dtype
        )
    }
}

/// Hard cap on how long a capture run waits for a load before grabbing anyway.
const CAPTURE_LOAD_CAP: Duration = Duration::from_secs(120);

/// Max interval between two left-clicks to count as a double-click.
const DOUBLE_CLICK: Duration = Duration::from_millis(350);

/// Result of a background decode, tagged with its generation id.
struct LoadResult {
    gen: u64,
    result: Result<ImageData, String>,
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
    last_left_press: Option<Instant>,
    fullscreen: bool,
    /// True once the user resizes the window (suppresses auto-resize).
    user_resized: bool,
    auto_resized_done: bool,
    /// Set when we request a programmatic resize, to ignore the resulting event.
    auto_resize_pending: bool,
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
            force_toolbar: std::env::var_os("IMGVWR_DEBUG_TOOLBAR").is_some(),
            force_overlay: std::env::var("IMGVWR_DEBUG_OVERLAY").ok(),
            modifiers: ModifiersState::empty(),
            dragging: false,
            // Start far from the left edge so the toolbar stays hidden until the
            // cursor actually moves there (and so headless captures are clean).
            cursor_pos: PhysicalPosition::new(1.0e6, 1.0e6),
            last_left_press: None,
            fullscreen: false,
            user_resized: false,
            auto_resized_done: false,
            auto_resize_pending: false,
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
            }
        }
    }

    /// Pick the OCIO view for a freshly-loaded image (§13). Panoramas restore the
    /// saved per-extension view; 2-D images always use Standard.
    fn select_view_for_load(&mut self, panorama: bool, path: &Path) {
        let applied = if panorama {
            path.extension()
                .and_then(|e| e.to_str())
                .and_then(|ext| self.prefs.preferred_view(ext).cloned())
                .map(|pv| self.ocio.set_active(&pv.display, &pv.view))
                .unwrap_or(false)
        } else {
            false
        };
        if !applied {
            self.select_standard_view();
        }
        self.rebuild_ocio();
    }

    /// Select a "Standard" view (preferring the current display), if available.
    fn select_standard_view(&mut self) {
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
                .find(|(d, v)| d == display && v.eq_ignore_ascii_case("standard"))
            {
                self.ocio.set_active(d, v);
                return;
            }
        }
        if let Some((d, v)) = pairs
            .iter()
            .find(|(_, v)| v.eq_ignore_ascii_case("standard"))
        {
            self.ocio.set_active(d, v);
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
        let window_attributes = Window::default_attributes()
            .with_title("imgvwr")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_min_inner_size(LogicalSize::new(170.0, 170.0));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_depth_size(0)
            .with_stencil_size(0);

        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));
        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|a, b| {
                        if b.num_samples() > a.num_samples() {
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

    /// On the first successful 2-D load, size the window to the image (clamped to
    /// the monitor) unless the user has already resized it (§16 Commit 10).
    fn maybe_autosize(&mut self, width: u32, height: u32) {
        if self.user_resized || self.auto_resized_done {
            return;
        }
        let Some(gfx) = &self.gfx else { return };
        let (max_w, max_h) = gfx
            .window
            .current_monitor()
            .map(|m| {
                let s = m.size();
                (
                    (s.width as f32 * 0.92) as u32,
                    (s.height as f32 * 0.92) as u32,
                )
            })
            .unwrap_or((width, height));
        let w = width.min(max_w).max(170);
        let h = height.min(max_h).max(170);
        self.auto_resized_done = true;
        self.auto_resize_pending = true;
        let _ = gfx.window.request_inner_size(PhysicalSize::new(w, h));
        log::debug!("auto-resized window to {w}x{h} for image {width}x{height}");
    }

    fn load_path(&mut self, path: PathBuf) {
        self.load_gen += 1;
        let gen = self.load_gen;
        self.load_state = LoadState::Loading;
        self.load_start = Instant::now();
        self.pending_name = path.file_name().map(|s| s.to_string_lossy().into_owned());
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
                    let equirect = data.is_equirectangular();
                    log::info!(
                        "loaded {}x{} ({} ch, {}) {} from {}",
                        data.width,
                        data.height,
                        data.channels,
                        data.dtype_name,
                        if equirect { "[panorama]" } else { "[2-D]" },
                        data.path.display()
                    );
                    if let Some(gfx) = &mut self.gfx {
                        gfx.renderer.set_image(&data);
                    }
                    log::info!(
                        "load-to-ready: {:.2}s for {}",
                        self.load_start.elapsed().as_secs_f32(),
                        data.path.display()
                    );
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
                    self.camera = CameraController::for_image(equirect);
                    self.exposure = 0.0;
                    self.gamma = 1.0;
                    self.wrap_2d = false;
                    self.load_state = LoadState::Loaded;
                    self.update_window_title();
                    // Choose the OCIO view: panoramas restore the saved view for
                    // their extension; 2-D images always default to Standard.
                    self.select_view_for_load(equirect, &data.path);
                    // Size the window to the first 2-D image (panoramas keep the
                    // current window).
                    if !equirect {
                        self.maybe_autosize(data.width, data.height);
                    }
                    self.apply_debug_overrides();
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
            gfx.window.set_cursor_visible(true);
        }
    }

    /// Apply a relative drag delta (raw device motion, pixels).
    fn on_drag_motion(&mut self, dx: f32, dy: f32) {
        let (vw, vh) = self.viewport();
        match self.camera.camera {
            Camera::Pano { pitch_rad, .. } => {
                // Pixels-per-radian from the current vertical FOV.
                let rad_per_px = (2.0 * self.camera.camera.half_fov_radians()) / vh;
                // Latitude-based horizontal multiplier (same formula as original).
                let h_mult = (1.0 / pitch_rad.abs().cos().max(0.25)).min(2.5);
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
                // Grab feel: content follows the cursor.
                let du = -(dx / vw) * sx;
                let dv = -(dy / vh) * sy;
                self.camera.pan(Vec2::new(du, dv));
                if !self.wrap_2d {
                    self.clamp_2d_pan();
                }
            }
        }
        self.request_redraw();
    }

    /// Keep the 2-D image within the viewport (no-wrap mode).
    fn clamp_2d_pan(&mut self) {
        let (vw, vh) = self.viewport();
        let image_aspect = self
            .gfx
            .as_ref()
            .and_then(|g| g.renderer.image_aspect())
            .unwrap_or(1.0);
        if let Camera::Flat { pan, zoom } = &mut self.camera.camera {
            let inv_zoom = 1.0 / zoom.max(1e-4);
            let sx = inv_zoom * (vw / vh) / image_aspect.max(1e-4);
            let sy = inv_zoom;
            // Visible half-span; if the image is smaller than the view, centre it.
            let lim_u = (0.5 - 0.5 * sx).max(0.0);
            let lim_v = (0.5 - 0.5 * sy).max(0.0);
            pan.x = pan.x.clamp(-lim_u, lim_u);
            pan.y = pan.y.clamp(-lim_v, lim_v);
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

        if self.ctrl() {
            // Ctrl + wheel: exposure.
            self.exposure += steps * 0.1;
            self.request_redraw();
            return;
        }

        match self.camera.camera {
            Camera::Pano { fov_deg, .. } => {
                // Progressive feel: step scaled by current FOV.
                let step = (fov_deg / 90.0) * 5.0;
                self.camera.adjust_fov(-steps * step);
            }
            Camera::Flat { .. } => {
                self.camera.adjust_zoom(1.1_f32.powf(steps));
                if !self.wrap_2d {
                    self.clamp_2d_pan();
                }
            }
        }
        self.request_redraw();
    }

    fn on_key(&mut self, event_loop: &ActiveEventLoop, key: &Key, is_char: Option<&str>) {
        let ctrl = self.ctrl();
        match (key, is_char) {
            (_, Some(",")) => {
                if ctrl {
                    self.gamma = (self.gamma - 0.1).max(0.1);
                } else {
                    self.exposure -= 1.0;
                }
            }
            (_, Some(".")) => {
                if ctrl {
                    self.gamma = (self.gamma + 0.1).min(4.0);
                } else {
                    self.exposure += 1.0;
                }
            }
            (_, Some("p")) | (_, Some("P")) => {
                let want = !self.camera.is_panorama();
                self.camera.set_mode(want);
                if !self.camera.is_panorama() && !self.wrap_2d {
                    self.clamp_2d_pan();
                }
                log::info!(
                    "projection -> {}",
                    if self.camera.is_panorama() {
                        "panorama"
                    } else {
                        "2-D"
                    }
                );
            }
            (_, Some("w")) | (_, Some("W")) => {
                self.wrap_2d = !self.wrap_2d;
                if !self.wrap_2d {
                    self.clamp_2d_pan();
                }
                log::info!("2-D wrap -> {}", self.wrap_2d);
            }
            (_, Some("t")) | (_, Some("T")) => self.toggle_view_transform(),
            (_, Some("q")) | (_, Some("Q")) => self.escape_or_exit(event_loop),
            (Key::Named(NamedKey::F2), _) => {
                self.show_metadata = !self.show_metadata;
                log::info!("metadata overlay -> {}", self.show_metadata);
            }
            (Key::Named(NamedKey::Home), _) => self.camera.reset_view(),
            (Key::Named(NamedKey::F11), _) => self.toggle_fullscreen(),
            (Key::Named(NamedKey::Escape), _) => self.escape_or_exit(event_loop),
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

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        match (state, button) {
            (ElementState::Pressed, MouseButton::Left) => {
                let now = Instant::now();
                let double = self
                    .last_left_press
                    .is_some_and(|t| now.duration_since(t) < DOUBLE_CLICK);
                self.last_left_press = Some(now);
                if double {
                    self.toggle_fullscreen();
                    self.request_redraw();
                } else {
                    self.start_drag();
                }
            }
            (ElementState::Pressed, MouseButton::Middle) => self.start_drag(),
            (ElementState::Released, MouseButton::Left | MouseButton::Middle) => self.end_drag(),
            _ => {}
        }
    }

    /// Apply `IMGVWR_DEBUG_*` overrides after load (headless verification only).
    fn apply_debug_overrides(&mut self) {
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
                format!("{} — imgvwr", self.file_info.name)
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
        let loading = self.is_loading() || forced == Some("loading");
        let error = match &self.load_state {
            LoadState::Failed(e) => Some(e.clone()),
            _ if forced == Some("error") => Some("Example decode error: unsupported format".into()),
            _ => None,
        };
        let show_hint = (!has_image && !loading && error.is_none()) || forced == Some("hint");

        UiInputs {
            toolbar_visible: self.toolbar_visible,
            has_image,
            file_info: self.file_info.summary(),
            display_views,
            active,
            ocio_available: !self.ocio.display_views().is_empty(),
            loading,
            loading_name: self.pending_name.clone(),
            error,
            show_hint,
            show_metadata: self.show_metadata || forced == Some("metadata"),
            metadata: self.metadata_lines(),
        }
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
                "Mode".into(),
                if fi.panorama {
                    "Panorama".into()
                } else {
                    "2-D".into()
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
                }
            }
            UiAction::DismissError => {
                self.load_state = LoadState::Idle;
                self.request_redraw();
            }
        }
    }

    fn open_file_dialog(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif", "ico", "tga", "pnm",
                    "hdr", "pic", "exr", "nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2",
                ],
            )
            .pick_file();
        if let Some(path) = file {
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

    fn capture_ready(&self) -> bool {
        self.capture.as_ref().is_some_and(|c| {
            let elapsed = c.start.elapsed();
            !c.done
                && elapsed >= c.delay
                && (!self.is_loading() || elapsed >= c.delay + CAPTURE_LOAD_CAP)
        })
    }

    fn render(&mut self) -> RenderOutcome {
        self.tick_toolbar();

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
                    gfx.window.request_redraw();
                }
                // Distinguish a user resize from our own auto-resize so the
                // latter doesn't suppress future auto-sizing semantics.
                if self.auto_resize_pending {
                    self.auto_resize_pending = false;
                } else if self.loaded_path.is_some() {
                    self.user_resized = true;
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
                self.tick_toolbar();
                self.request_redraw();
            }
            // egui-consumed pointer/wheel/key events fall through to `_ => {}`.
            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                self.on_mouse_button(state, button)
            }
            // Block wheel input when the toolbar is hovered (§11.3).
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => self.on_wheel(delta),
            WindowEvent::DroppedFile(path) => self.load_path(path),
            WindowEvent::KeyboardInput { event, .. }
                if !egui_consumed && event.state == ElementState::Pressed =>
            {
                let text = match &event.logical_key {
                    Key::Character(s) => Some(s.as_str().to_string()),
                    _ => None,
                };
                self.on_key(event_loop, &event.logical_key, text.as_deref());
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
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.capture_active() {
            // Drive continuous frames while a capture is pending.
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        } else if self.is_loading() {
            // Keep the loading spinner animating while a decode is in flight.
            event_loop.set_control_flow(ControlFlow::Poll);
            self.request_redraw();
        } else if let Some(deadline) = self.toolbar_hide_deadline {
            // Wake at the deadline to evaluate hiding the toolbar.
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            if Instant::now() >= deadline {
                self.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::LoadFinished(_gen) => self.poll_loads(),
        }
    }
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
