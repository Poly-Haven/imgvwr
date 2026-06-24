//! OpenEXR-backed fallback decoder, used when the pure-Rust `exr` crate cannot
//! read a file (e.g. DWAA/DWAB compression). Only compiled with the `ocio`
//! feature (which provides the vcpkg C++ toolchain build path).

mod ffi;

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::image_loader::{ImageData, PixelBuffer};

/// Decode an EXR via the OpenEXR C++ library (handles all compressions).
pub fn load_exr_native(path: &Path) -> Result<ImageData> {
    use std::ffi::CString;

    let c_path = CString::new(path.to_string_lossy().as_ref())
        .map_err(|_| anyhow!("path contains NUL"))?;
    let mut width = 0i32;
    let mut height = 0i32;
    let mut channels = 0i32;

    let ptr = unsafe {
        ffi::exr_native_load_rgba(c_path.as_ptr(), &mut width, &mut height, &mut channels)
    };
    if ptr.is_null() || width <= 0 || height <= 0 {
        return Err(anyhow!(
            "OpenEXR fallback failed to decode {}",
            path.display()
        ));
    }

    let len = width as usize * height as usize * 4;
    let data = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    unsafe { ffi::exr_native_free(ptr) };

    Ok(ImageData {
        path: path.to_path_buf(),
        width: width as u32,
        height: height as u32,
        channels: channels.clamp(1, 4) as u8,
        dtype_name: "openexr->float32".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::F32(data),
        is_encoded_srgb: false,
    })
}
