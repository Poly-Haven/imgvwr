//! Image loading and the in-memory `ImageData` representation.
//!
//! Commit 3 implements only the JPEG/PNG path (8-bit, sRGB-encoded). The full
//! dispatch table (HDR, EXR, RAW, channel normalisation, background threading)
//! lands in Commit 4.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// A decoded image, normalised to 4-channel RGBA interleaved.
///
/// RGBA (not RGB) because 3-channel float textures are slow / unsupported on
/// some drivers; RGBA is the safe, fast, universally-supported layout. Alpha is
/// set to opaque when the source has none.
pub struct ImageData {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Original channel count (before RGBA normalisation).
    pub channels: u8,
    pub dtype_name: String,
    pub compression: String,
    pub pixels: PixelBuffer,
    /// True → source pixels are sRGB-encoded and need the GPU sRGB decode.
    pub is_encoded_srgb: bool,
}

impl ImageData {
    pub fn aspect(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    pub fn is_equirectangular(&self) -> bool {
        is_equirectangular(self.width, self.height)
    }
}

/// Interleaved HxWx4 RGBA pixel data, either 8-bit or 32-bit float.
pub enum PixelBuffer {
    /// HxWx4 interleaved uint8 (JPEG / LDR PNG).
    U8(Vec<u8>),
    /// HxWx4 interleaved float32 (EXR / HDR / RAW).
    F32(Vec<f32>),
}

/// Panorama detection: an image is treated as equirectangular when its width is
/// exactly twice its height.
pub fn is_equirectangular(width: u32, height: u32) -> bool {
    height > 0 && width == height * 2
}

/// Decode an image file into RGBA `ImageData`.
///
/// Commit 3: JPEG and PNG only. Other extensions return a descriptive error
/// until the full dispatch table is implemented in Commit 4.
pub fn load_image(path: &Path) -> Result<ImageData> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "jpg" | "jpeg" | "png" => load_ldr(path),
        other => Err(anyhow!(
            "format '.{other}' not supported yet (full loader lands in Commit 4)"
        )),
    }
}

/// 8-bit, sRGB-encoded path via the `image` crate (JPEG / PNG).
fn load_ldr(path: &Path) -> Result<ImageData> {
    let decoded =
        image::open(path).with_context(|| format!("failed to decode {}", path.display()))?;
    let channels = decoded.color().channel_count();
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();

    Ok(ImageData {
        path: path.to_path_buf(),
        width,
        height,
        channels,
        dtype_name: "uint8".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::U8(rgba.into_raw()),
        is_encoded_srgb: true,
    })
}
