//! LibRaw-backed camera RAW decoder. Produces scene-linear float RGBA with a
//! linear camera response (demosaic + camera white balance + camera->sRGB
//! matrix, no tone curve), so RAW photos view accurately and their highlights
//! stay recoverable via exposure / Filmic-ACES — exactly like the EXR/HDR path
//! (`is_encoded_srgb = false`). Only compiled with the `ocio` feature (which
//! provides the vcpkg C++ toolchain), alongside [`crate::exr_native`]. Without
//! it, RAW falls back to the pure-Rust best-effort `rawler` path.

mod ffi;

use std::ffi::{CStr, CString};
use std::path::Path;

use anyhow::{anyhow, Result};

use crate::image_loader::{CameraMeta, ImageData, PixelBuffer};

/// Decode a camera RAW file into scene-linear float RGBA via LibRaw.
pub fn load_raw_native(path: &Path) -> Result<ImageData> {
    let c_path =
        CString::new(path.to_string_lossy().as_ref()).map_err(|_| anyhow!("path contains NUL"))?;

    let mut info = ffi::RawNativeInfo {
        width: 0,
        height: 0,
        colors: 0,
        make: [0; 64],
        model: [0; 64],
        lens: [0; 128],
        iso: 0.0,
        shutter: 0.0,
        aperture: 0.0,
        focal_len: 0.0,
    };

    // Two-phase so the shim expands straight into a buffer we own: `begin`
    // develops the frame and reports its size, then `finish` writes the float
    // RGBA into our allocation. (Previously the shim malloc'd its own buffer and
    // we copied it out — an extra ~700 MB memcpy and double the peak memory on a
    // 45 Mpx RAW.)
    let handle = unsafe { ffi::raw_native_begin(c_path.as_ptr(), &mut info) };
    if handle.is_null() || info.width <= 0 || info.height <= 0 {
        if !handle.is_null() {
            unsafe { ffi::raw_native_abort(handle) };
        }
        return Err(anyhow!("LibRaw failed to decode {}", path.display()));
    }

    let len = info.width as usize * info.height as usize * 4;
    let mut data = Vec::<f32>::with_capacity(len);
    // SAFETY: `data` has `len` floats of spare capacity; the shim fills exactly
    // that many, and we only publish them via set_len once it reports success.
    let rc = unsafe { ffi::raw_native_finish(handle, data.as_mut_ptr()) };
    if rc != 0 {
        return Err(anyhow!("LibRaw failed to develop {}", path.display()));
    }
    unsafe { data.set_len(len) };

    Ok(ImageData {
        path: path.to_path_buf(),
        width: info.width as u32,
        height: info.height as u32,
        channels: info.colors.clamp(1, 4) as u8,
        dtype_name: "raw->float32".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::F32(data),
        // Scene-linear (linear camera response, no tone curve) — like EXR/HDR.
        is_encoded_srgb: false,
        animation: None,
        camera: Some(camera_meta(&info)),
        // Clip the overlay at 1.0 (neutral diffuse white): a RAW's values above 1.0
        // are recoverable highlight headroom, but relying on that headroom is
        // generally unwise, so flag anything at/above white as clipped.
        clip_max: crate::image_loader::CLIP_MAX_NORM,
    })
}

/// Build a [`CameraMeta`] from the shim's info struct, dropping empty/zero fields.
fn camera_meta(info: &ffi::RawNativeInfo) -> CameraMeta {
    fn cstr(buf: &[i8]) -> Option<String> {
        // SAFETY: the shim always NUL-terminates these fixed buffers.
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string();
        (!s.is_empty()).then_some(s)
    }
    fn pos(v: f32) -> Option<f32> {
        (v > 0.0).then_some(v)
    }

    CameraMeta {
        make: cstr(&info.make),
        model: cstr(&info.model),
        lens: cstr(&info.lens),
        iso: pos(info.iso),
        shutter: pos(info.shutter),
        aperture: pos(info.aperture),
        focal_len: pos(info.focal_len),
    }
}
