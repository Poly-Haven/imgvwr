// imgvwr – a minimal, GPU-accelerated viewer for HDR panoramas and standard
// images. See plans/rewrite.md for the full design. The implementation lives in
// the library crate; this binary is a thin entry point.

// On Windows, suppress the console window for GUI (release) builds while keeping
// it for debug builds so logs are visible during development.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    imgvwr::run();
}
