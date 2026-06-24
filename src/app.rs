//! Application struct and winit event loop.
//!
//! Commit 2 brings up the window, the glutin OpenGL 4.3 core context, and a
//! glow context, and clears the framebuffer each frame. Rendering, input, and
//! image loading are layered on in later commits.
//!
//! ## Headless framebuffer capture
//!
//! Because `imgvwr.exe` is a dev binary (no Start-menu registration), the
//! computer-use screenshot tool cannot see its window contents. To verify
//! rendering headlessly we instead read the GL back buffer with `glReadPixels`
//! and write a PNG. Driven by env vars:
//!   * `IMGVWR_CAPTURE`           – output PNG path (enables capture)
//!   * `IMGVWR_CAPTURE_DELAY_MS`  – wait this long after the window appears
//!                                  before capturing (lets a load finish), then
//!                                  capture one frame and exit.

use std::num::NonZeroU32;
use std::path::PathBuf;
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

use crate::image_loader::load_image;
use crate::renderer::{RenderParams, Renderer};
use crate::UserEvent;

/// Hard-coded fixture used to verify the pipeline end-to-end before drag-drop /
/// CLI loading exists. Removed in Commit 10.
const TEST_IMAGE: &str = r"C:\tmp\imgvwr_test_files\american_walnut_veneer_rough_1k.png";

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
    #[allow(dead_code)] // used to wake the loop from load threads, from Commit 4.
    proxy: EventLoopProxy<UserEvent>,
    #[allow(dead_code)] // CLI path is loaded once GL is ready, from Commit 10.
    initial_path: Option<PathBuf>,
    gfx: Option<Gfx>,
    capture: Option<Capture>,
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
        Self {
            proxy,
            initial_path,
            gfx: None,
            capture,
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
                // Prefer the config with the most samples; otherwise the first.
                configs
                    .reduce(|a, b| if b.num_samples() > a.num_samples() { b } else { a })
                    .expect("at least one GL config")
            })
            .map_err(|e| anyhow::anyhow!("failed to build GL display: {e}"))?;
        let window = window.context("winit did not return a window")?;

        let gl_display = gl_config.display();
        let raw_window_handle = window.window_handle()?.as_raw();

        // OpenGL 4.3 core: 4.1 is the floor for OCIO's GLSL 4.0; 4.3 gives
        // KHR_debug on Windows (Windows-only target — see §9.1).
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

        // Enable vsync (best-effort). Disabled while capturing so we are not
        // throttled to the refresh rate during the (short) capture run.
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

        let (version, renderer) = unsafe {
            (
                gl.get_parameter_string(glow::VERSION),
                gl.get_parameter_string(glow::RENDERER),
            )
        };
        log::info!("OpenGL {version} on {renderer}");

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

    /// Commit 3: synchronously load the CLI path (if any) or the hard-coded test
    /// fixture so the texture pipeline can be verified. Background threading and
    /// full-format support arrive in Commit 4; this is removed in Commit 10.
    fn load_initial_image(&mut self) {
        let path = self.initial_path.clone().or_else(|| {
            let p = PathBuf::from(TEST_IMAGE);
            p.exists().then_some(p)
        });
        let Some(path) = path else {
            log::info!("no initial image to load");
            return;
        };
        match load_image(&path) {
            Ok(data) => {
                if let Some(gfx) = &mut self.gfx {
                    gfx.renderer.set_image(&data);
                    gfx.window.request_redraw();
                }
                log::info!("loaded {}", path.display());
            }
            Err(e) => log::error!("failed to load {}: {e:#}", path.display()),
        }
    }

    /// Whether a capture is configured and not yet taken.
    fn capture_active(&self) -> bool {
        self.capture.as_ref().is_some_and(|c| !c.done)
    }

    /// Whether the capture delay has elapsed so this frame should be grabbed.
    fn capture_ready(&self) -> bool {
        self.capture
            .as_ref()
            .is_some_and(|c| !c.done && c.start.elapsed() >= c.delay)
    }

    fn render(&mut self) -> RenderOutcome {
        let mut grabbed: Option<(i32, i32, Vec<u8>)> = None;
        {
            let Some(gfx) = &self.gfx else {
                return RenderOutcome::Idle;
            };
            let size = gfx.window.inner_size();
            // Guard against a zero-size framebuffer (minimised window).
            if size.width == 0 || size.height == 0 {
                return RenderOutcome::Idle;
            }
            let (w, h) = (size.width as i32, size.height as i32);

            // Commit 3: fixed 2-D view (90° FOV, no pan). Camera control and
            // exposure/gamma state are wired in at Commits 5 and 6.
            let params = RenderParams {
                viewport: (w, h),
                ..RenderParams::default()
            };
            gfx.renderer.render(&params);

            // Read the back buffer *before* swapping.
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

    /// Save a captured RGBA framebuffer to a PNG (flipped to top-down).
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
                // Reset the capture clock to when the window actually appeared.
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
        // While a capture is pending, drive continuous frames so the delay can
        // elapse; otherwise stay reactive.
        if self.capture_active() {
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: UserEvent) {
        // Load results are handled from Commit 4 onward.
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
