//! Application struct and winit event loop.
//!
//! Brings up the window, the glutin OpenGL 4.3 core context, and a glow
//! context; loads images on a background thread; and renders each frame.
//!
//! ## Headless framebuffer capture
//!
//! Because `imgvwr.exe` is a dev binary (no Start-menu registration), the
//! computer-use screenshot tool cannot see its window contents. To verify
//! rendering headlessly we instead read the GL back buffer with `glReadPixels`
//! and write a PNG. Driven by env vars:
//!   * `IMGVWR_CAPTURE`           – output PNG path (enables capture)
//!   * `IMGVWR_CAPTURE_DELAY_MS`  – minimum wait after the window appears before
//!                                  capturing. Capture also waits for any
//!                                  in-flight load to finish (up to a cap), then
//!                                  captures one frame and exits.

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
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::image_loader::{load_image, ImageData};
use crate::renderer::{RenderParams, Renderer};
use crate::UserEvent;

/// Hard-coded fixture used to verify the pipeline end-to-end before drag-drop /
/// CLI loading exists. Removed in Commit 10.
const TEST_IMAGE: &str = r"C:\tmp\imgvwr_test_files\american_walnut_veneer_rough_1k.png";

/// Hard cap on how long a capture run waits for a load before grabbing anyway.
const CAPTURE_LOAD_CAP: Duration = Duration::from_secs(120);

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
        }
    }

    /// Build the window, GL context, surface, and glow context.
    fn create_gfx(&mut self, event_loop: &ActiveEventLoop) -> Result<Gfx> {
        let window_attributes = Window::default_attributes()
            .with_title("imgvwr")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_min_inner_size(LogicalSize::new(170.0, 170.0));

        // 8-bit colour, no depth/stencil (we only ever draw a full-screen quad).
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

    /// Load the CLI path (if any) or the hard-coded test fixture. Removed in
    /// Commit 10 once drag-drop / CLI loading are wired up.
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

    /// Begin a background decode of `path`. Bumps the generation id so stale
    /// results from earlier loads are discarded (§5.2).
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
                // A panicking decoder becomes a Failed state, not a crash.
                let result = match std::panic::catch_unwind(AssertUnwindSafe(|| load_image(&path))) {
                    Ok(Ok(data)) => Ok(data),
                    Ok(Err(e)) => Err(format!("{e:#}")),
                    Err(_) => Err("decoder panicked".to_string()),
                };
                let _ = tx.send(LoadResult { gen, result });
                // Wake the reactive event loop so the result is observed.
                let _ = proxy.send_event(UserEvent::LoadFinished(gen));
            })
            .expect("spawn image-load thread");
    }

    /// Drain finished loads, adopting only the one matching the current gen.
    fn poll_loads(&mut self) {
        let mut adopted = false;
        while let Ok(msg) = self.load_rx.try_recv() {
            if msg.gen != self.load_gen {
                log::debug!("discarding stale load result (gen {})", msg.gen);
                continue;
            }
            match msg.result {
                Ok(data) => {
                    log::info!(
                        "loaded {}x{} ({} ch, {}) from {}",
                        data.width,
                        data.height,
                        data.channels,
                        data.dtype_name,
                        data.path.display()
                    );
                    if let Some(gfx) = &mut self.gfx {
                        gfx.renderer.set_image(&data);
                    }
                    self.load_state = LoadState::Loaded;
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

    fn is_loading(&self) -> bool {
        matches!(self.load_state, LoadState::Loading)
    }

    fn capture_active(&self) -> bool {
        self.capture.as_ref().is_some_and(|c| !c.done)
    }

    /// Whether this frame should be grabbed: capture configured, delay elapsed,
    /// and (any in-flight load finished, or the hard cap reached).
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

            // Fixed 2-D view until the camera lands in Commit 5.
            let params = RenderParams {
                viewport: (w, h),
                ..RenderParams::default()
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
            WindowEvent::DroppedFile(path) => self.load_path(path),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape))
                {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Drive continuous frames only while a capture is pending.
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
