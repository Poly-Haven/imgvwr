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

    let first = std::env::args_os().nth(1);

    // Headless: register imgvwr as the default viewer and exit. Same code path as
    // the Settings "Set as default" button; handy for scripts and for testing the
    // registration without opening the GUI.
    if first.as_deref().and_then(|s| s.to_str()) == Some("--register-default-app") {
        match app::register_default_app() {
            Ok(n) => println!("imgvwr registered as the default viewer for {n} file types"),
            Err(e) => {
                eprintln!("register-default-app failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let initial_path = first.map(PathBuf::from);

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
