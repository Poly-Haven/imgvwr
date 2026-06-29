//! imgvwr library crate. The binary (`src/main.rs`) is a thin wrapper around
//! [`run`]; exposing the modules here also lets integration tests in `tests/`
//! exercise the image loader and pure logic.

pub mod app;
pub mod camera;
#[cfg(feature = "ocio")]
pub mod exr_native;
pub mod image_loader;
pub mod logging;
pub mod ocio;
pub mod prefs;
#[cfg(feature = "ocio")]
pub mod raw_native;
pub mod renderer;
pub mod ui;
pub mod update;

use std::path::PathBuf;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

/// Custom event used to wake the reactive event loop from background threads
/// (see plans/rewrite.md §5.2).
#[derive(Debug, Clone)]
pub enum UserEvent {
    /// A background image load finished; carries its generation id.
    LoadFinished(u64),
    /// A background *preload* (arrow-key look-ahead) finished; its generation id.
    PreloadFinished(u64),
    /// A background "is there a newer release?" check finished (see `update`).
    UpdateChecked,
}

/// Initialise logging and run the winit event loop until the window closes.
pub fn run() {
    if let Err(e) = logging::init() {
        eprintln!("Warning: failed to initialise logger: {e}");
    }

    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("failed to build event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let proxy = event_loop.create_proxy();
    let mut app = App::new(initial_path, proxy);

    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("event loop terminated with error: {e}");
    }
}
