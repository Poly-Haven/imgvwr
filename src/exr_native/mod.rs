//! OpenEXR-backed fallback decoder, used when the pure-Rust `exr` crate cannot
//! read a file (e.g. DWAA/DWAB compression). Only compiled with the `ocio`
//! feature (which provides the vcpkg C++ toolchain build path).

mod ffi;

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::image_loader::{DecodeIntent, ImageData, PixelBuffer};

/// Decode an EXR via the OpenEXR C++ library (handles all compressions).
pub fn load_exr_native(path: &Path, intent: DecodeIntent) -> Result<ImageData> {
    use std::ffi::CString;

    let c_path =
        CString::new(path.to_string_lossy().as_ref()).map_err(|_| anyhow!("path contains NUL"))?;
    let mut width = 0i32;
    let mut height = 0i32;
    let mut channels = 0i32;

    // OpenEXR's worker pool is library-global and starts empty, so without this
    // every decode runs on the calling thread alone. Idempotent in the shim.
    unsafe { ffi::exr_native_init_thread_pool(pool_size()) };

    // Two-phase so OpenEXR decodes straight into a buffer we own, with no
    // malloc'd intermediate copied across the FFI boundary (see shim.h).
    let handle = unsafe {
        ffi::exr_native_begin(
            c_path.as_ptr(),
            native_thread_count(intent),
            &mut width,
            &mut height,
            &mut channels,
        )
    };
    if handle.is_null() || width <= 0 || height <= 0 {
        if !handle.is_null() {
            unsafe { ffi::exr_native_abort(handle) };
        }
        return Err(anyhow!(
            "OpenEXR fallback failed to decode {}",
            path.display()
        ));
    }

    let len = width as usize * height as usize * 4;
    let mut data = Vec::<f32>::with_capacity(len);
    // SAFETY: `data` has `len` floats of spare capacity; the shim fills exactly
    // that many, and we only publish them via set_len once it reports success.
    let rc = unsafe { ffi::exr_native_finish(handle, data.as_mut_ptr()) };
    if rc != 0 {
        return Err(anyhow!(
            "OpenEXR fallback failed to decode {}",
            path.display()
        ));
    }
    unsafe { data.set_len(len) };

    Ok(ImageData {
        path: path.to_path_buf(),
        width: width as u32,
        height: height as u32,
        channels: channels.clamp(1, 4) as u8,
        dtype_name: "openexr->float32".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::F32(data),
        is_encoded_srgb: false,
        animation: None,
        camera: None,
        // OpenEXR fallback decodes to f32 scene-linear → unbounded, never clips.
        clip_max: crate::image_loader::CLIP_MAX_NONE,
    })
}

/// Size of OpenEXR's shared worker pool: one per core. This is a hard ceiling on
/// the library's total worker threads however many files are open at once, so
/// concurrent throughput-mode decodes cannot oversubscribe through it.
fn pool_size() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(i32::MAX as usize) as i32)
        .unwrap_or(0)
}

/// How many of that shared pool this one file may occupy.
fn native_thread_count(intent: DecodeIntent) -> i32 {
    if intent.is_serial() {
        // 0 = decompress on the calling thread. The parallelism is already being
        // spent on other frames; taking pool threads here would only contend.
        0
    } else {
        pool_size()
    }
}
