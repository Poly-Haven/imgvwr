// imgvwr – a minimal, GPU-accelerated viewer for HDR panoramas and standard
// images. See plans/rewrite.md for the full design.

// On Windows, suppress the console window for GUI (release) builds while keeping
// it for debug builds so logs are visible during development.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod camera;
#[cfg(feature = "ocio")]
mod exr_native;
mod image_loader;
mod logging;
mod ocio;
mod prefs;
mod renderer;
mod ui;

use std::path::PathBuf;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

/// Custom event used to wake the reactive event loop from background threads
/// (see plans/rewrite.md §5.2). Without this, a completed load would not be
/// observed until the next unrelated event, making the app appear to hang.
#[derive(Debug, Clone)]
pub enum UserEvent {
    /// A background image load finished; carries its generation id.
    LoadFinished(u64),
}

fn main() {
    if let Err(e) = logging::init() {
        eprintln!("Warning: failed to initialise logger: {e}");
    }

    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("failed to build event loop");
    // Reactive: only redraw in response to events / explicit redraw requests.
    event_loop.set_control_flow(ControlFlow::Wait);

    let proxy = event_loop.create_proxy();
    let mut app = App::new(initial_path, proxy);

    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("event loop terminated with error: {e}");
    }
}
