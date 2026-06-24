//! Image loading and the in-memory `ImageData` representation.

mod formats;

use std::path::{Path, PathBuf};

use anyhow::Result;

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

/// Camera RAW extensions handled (best-effort) by `rawler`.
const RAW_EXTS: &[&str] = &[
    "nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "nrw", "pef", "rwl", "sr2", "srf",
    "crw", "raw",
];

/// Decode an image file into RGBA `ImageData`, dispatching on the (lower-cased)
/// file extension. See plans/rewrite.md §8.2.
pub fn load_image(path: &Path) -> Result<ImageData> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "exr" {
        formats::load_exr(path)
    } else if RAW_EXTS.contains(&ext.as_str()) {
        formats::load_raw(path)
    } else {
        // PNG/JPEG/BMP/TIFF/WebP/GIF/ICO/TGA/PNM, Radiance HDR, and anything
        // else the `image` crate recognises.
        formats::load_via_image(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equirectangular_detection() {
        assert!(is_equirectangular(4096, 2048));
        assert!(is_equirectangular(24576, 12288));
        assert!(is_equirectangular(2, 1));
        assert!(!is_equirectangular(1024, 1024));
        assert!(!is_equirectangular(2048, 4096));
        assert!(!is_equirectangular(100, 0));
        assert!(!is_equirectangular(0, 0));
    }
}
