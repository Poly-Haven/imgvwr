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
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use glow::HasContext as _;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext,
    Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use glam::Vec2;
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::camera::{Camera, CameraController};
use crate::image_loader::{load_image, ImageData};
use crate::renderer::{RenderParams, Renderer};
use crate::UserEvent;

/// Hard-coded fixture used to verify the pipeline end-to-end before drag-drop /
/// CLI loading exists. Removed in Commit 10.
const TEST_IMAGE: &str = r"C:\tmp\imgvwr_test_files\american_walnut_veneer_rough_1k.png";

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

    // View state.
    camera: CameraController,
    exposure: f32,
    gamma: f32,
    wrap_2d: bool,
    show_metadata: bool,

    // Input state.
    modifiers: ModifiersState,
    dragging: bool,
    cursor_pos: PhysicalPosition<f64>,
    last_left_press: Option<Instant>,
    fullscreen: bool,
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

        Self {
            proxy,
            initial_path,
            gfx: None,
            capture,
            load_tx,
            load_rx,
            load_gen: 0,
            load_state: LoadState::Idle,
            camera: CameraController::for_image(false),
            exposure: 0.0,
            gamma: 1.0,
            wrap_2d: false,
            show_metadata: false,
            modifiers: ModifiersState::empty(),
            dragging: false,
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            last_left_press: None,
            fullscreen: false,
        }
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

        let display_builder =
            DisplayBuilder::new().with_window_attributes(Some(window_attributes));
        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|a, b| if b.num_samples() > a.num_samples() { b } else { a })
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
            if let Err(e) = gl_surface.set_swap_interval(
                &gl_context,
                SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
            ) {
                log::warn!("could not set vsync swap interval: {e}");
            }
        }

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

        Ok(Gfx {
            gl,
            gl_surface,
            gl_context,
            window,
            renderer,
        })
    }

    fn load_initial_image(&mut self) {
        let path = self.initial_path.clone().or_else(|| {
            let p = PathBuf::from(TEST_IMAGE);
            p.exists().then_some(p)
        });
        match path {
            Some(p) => self.load_path(p),
            None => log::info!("no initial image to load"),
        }
    }

    fn load_path(&mut self, path: PathBuf) {
        self.load_gen += 1;
        let gen = self.load_gen;
        self.load_state = LoadState::Loading;
        log::info!("loading (gen {gen}) {}", path.display());

        let tx = self.load_tx.clone();
        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name(format!("image-load-{gen}"))
            .spawn(move || {
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| load_image(&path))) {
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
                    self.camera = CameraController::for_image(equirect);
                    self.exposure = 0.0;
                    self.gamma = 1.0;
                    self.wrap_2d = false;
                    self.load_state = LoadState::Loaded;
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
                    if self.camera.is_panorama() { "panorama" } else { "2-D" }
                );
            }
            (_, Some("w")) | (_, Some("W")) => {
                self.wrap_2d = !self.wrap_2d;
                if !self.wrap_2d {
                    self.clamp_2d_pan();
                }
                log::info!("2-D wrap -> {}", self.wrap_2d);
            }
            (_, Some("t")) | (_, Some("T")) => {
                // View-transform toggle is wired to OCIO in Commit 7 / 12.
                log::info!("T (view-transform toggle) — pending OCIO");
            }
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
                .set_fullscreen(on.then(|| Fullscreen::Borderless(None)));
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
        let mut grabbed: Option<(i32, i32, Vec<u8>)> = None;
        {
            let Some(gfx) = &self.gfx else {
                return RenderOutcome::Idle;
            };
            let size = gfx.window.inner_size();
            if size.width == 0 || size.height == 0 {
                return RenderOutcome::Idle;
            }
            let (w, h) = (size.width as i32, size.height as i32);

            let cam = &self.camera.camera;
            let params = RenderParams {
                viewport: (w, h),
                exposure: self.exposure,
                gamma: self.gamma,
                projection_mode: cam.projection_mode(),
                yaw: cam.yaw(),
                pitch: cam.pitch(),
                half_fov_radians: cam.half_fov_radians(),
                tan_half_fov: cam.tan_half_fov(),
                wrap_2d: self.wrap_2d,
            };
            gfx.renderer.render(&params);

            if self.capture_ready() {
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
            }
            WindowEvent::RedrawRequested => {
                if matches!(self.render(), RenderOutcome::Captured) {
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(mods) => self.modifiers = mods.state(),
            WindowEvent::CursorMoved { position, .. } => self.cursor_pos = position,
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_mouse_button(state, button)
            }
            WindowEvent::MouseWheel { delta, .. } => self.on_wheel(delta),
            WindowEvent::DroppedFile(path) => self.load_path(path),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
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
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.capture_active() {
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::LoadFinished(_gen) => self.poll_loads(),
        }
    }
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
