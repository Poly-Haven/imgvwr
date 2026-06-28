//! Image loading and the in-memory `ImageData` representation.

mod formats;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;

/// File-read progress shared with the loader thread so the loading bar can show
/// real progress while reading a (possibly slow, network-drive) file rather than
/// an indeterminate spinner. `total == 0` means "unknown" (indeterminate).
#[derive(Default)]
pub struct ReadProgress {
    pub read: AtomicU64,
    pub total: AtomicU64,
}

impl ReadProgress {
    /// Fraction read so far (0..1), or `None` while the total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total.load(Ordering::Relaxed);
        (total > 0)
            .then(|| (self.read.load(Ordering::Relaxed) as f32 / total as f32).clamp(0.0, 1.0))
    }
}

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
    /// Per-channel (RGBA) maximum representable value in the *texel* value space,
    /// for the clipping overlay: a channel clips at `clip_max[c] * (1 - margin)`.
    /// Integer formats normalise their max to 1.0; unbounded float formats use
    /// `f32::MAX` (never clips); a 16-bit-half EXR uses the half max (65504); RAW
    /// uses the per-channel sensor-saturation level (differs by channel after WB).
    pub clip_max: [f32; 4],
    /// Frames of an animated GIF (`None` for static images and single-frame
    /// GIFs). `pixels`/`width`/`height` mirror frame 0 so the static code paths
    /// (initial upload, diff, auto-exposure) work unchanged.
    pub animation: Option<Animation>,
    /// Camera EXIF metadata, populated for RAW files (and `None` otherwise).
    /// Surfaced in the F2 info box.
    pub camera: Option<CameraMeta>,
}

/// Camera EXIF metadata read from a RAW file. Every field is optional — cameras
/// and formats vary in what they record. Shown in the F2 metadata box.
#[derive(Clone, Debug, Default)]
pub struct CameraMeta {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<f32>,
    /// Exposure time in seconds (formatted as a shutter fraction for display).
    pub shutter: Option<f32>,
    /// Aperture f-number.
    pub aperture: Option<f32>,
    /// Focal length in millimetres.
    pub focal_len: Option<f32>,
}

impl CameraMeta {
    /// Human-readable "1/800 s", "0.5 s" etc. for an exposure time in seconds.
    pub fn shutter_display(secs: f32) -> String {
        if secs <= 0.0 {
            String::new()
        } else if secs >= 1.0 {
            format!("{secs:.0} s")
        } else {
            format!("1/{:.0} s", (1.0 / secs).round())
        }
    }
}

/// An animated image (currently only GIF): a list of pre-composited RGBA8 frames
/// with their on-screen durations. Held in [`ImageData`] so it travels with the
/// image through the load / cache / comparator pipeline.
pub struct Animation {
    pub frames: Vec<AnimFrame>,
}

/// One animation frame: the full canvas as interleaved RGBA8, plus how long it
/// stays on screen.
pub struct AnimFrame {
    pub pixels: Vec<u8>,
    pub delay: std::time::Duration,
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

    /// True for 8-bit-per-channel (LDR) sources, uploaded as a `U8` texture. Used
    /// to pick Lanczos downscaling (8-bit only) vs bilinear (higher bit depths).
    pub fn is_u8(&self) -> bool {
        matches!(self.pixels, PixelBuffer::U8(_))
    }

    /// Mean linear luminance (Rec.709) of the image, estimated from a subsample
    /// for speed (~1M pixels, so even a 24k panorama stays fast). `None` for
    /// 8-bit images, whose pixels are sRGB-encoded rather than scene-linear.
    /// Used to auto-expose HDR panoramas on load.
    pub fn average_linear_luminance(&self) -> Option<f32> {
        let PixelBuffer::F32(v) = &self.pixels else {
            return None;
        };
        let px = (self.width as usize) * (self.height as usize);
        if px == 0 || v.len() < px * 4 {
            return None;
        }
        let stride = (px / 1_000_000).max(1);
        let (mut sr, mut sg, mut sb) = (0.0f64, 0.0f64, 0.0f64);
        let mut n = 0u64;
        let mut i = 0;
        while i < px {
            let o = i * 4;
            sr += v[o] as f64;
            sg += v[o + 1] as f64;
            sb += v[o + 2] as f64;
            n += 1;
            i += stride;
        }
        if n == 0 {
            return None;
        }
        let (r, g, b) = (sr / n as f64, sg / n as f64, sb / n as f64);
        Some((0.2126 * r + 0.7152 * g + 0.0722 * b) as f32)
    }
}

/// Interleaved HxWx4 RGBA pixel data, either 8-bit or 32-bit float.
pub enum PixelBuffer {
    /// HxWx4 interleaved uint8 (JPEG / LDR PNG).
    U8(Vec<u8>),
    /// HxWx4 interleaved float32 (EXR / HDR / RAW).
    F32(Vec<f32>),
}

/// `clip_max` for a format whose maximum is normalised to 1.0 in the texel value
/// space (all 8-bit and 16-bit integer formats — the loader maps their max to 1.0).
pub const CLIP_MAX_NORM: [f32; 4] = [1.0; 4];

/// `clip_max` for an unbounded float format (32-bit EXR / Radiance HDR): values
/// far exceed 1.0 legitimately, so nothing should ever read as clipped.
pub const CLIP_MAX_NONE: [f32; 4] = [f32::MAX; 4];

/// The largest finite IEEE-754 half-float — the clip point for a 16-bit-half EXR.
pub const HALF_MAX: f32 = 65504.0;

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

/// Non-RAW formats handled by the `image` crate plus `exr` (the EXR special
/// case). Kept separate from [`RAW_EXTS`] so both can compose [`SUPPORTED_EXTS`].
const IMAGE_EXTS: &[&str] = &[
    "png", "apng", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif", "ico", "tga", "pnm", "hdr",
    "pic", "exr",
];

/// True if `path`'s (lower-cased) extension is one imgvwr can decode. The single
/// source of truth for the open-dialog filter and folder (arrow-key) navigation.
pub fn is_supported(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let ext = ext.to_ascii_lowercase();
            IMAGE_EXTS.contains(&ext.as_str()) || RAW_EXTS.contains(&ext.as_str())
        }
        None => false,
    }
}

/// Every extension imgvwr accepts (for the open-file dialog filter).
pub fn supported_extensions() -> Vec<&'static str> {
    IMAGE_EXTS.iter().chain(RAW_EXTS.iter()).copied().collect()
}

/// True if `path`'s (lower-cased) extension is a camera RAW format. Used to apply
/// the RAW-specific load policy (e.g. no auto-exposure by default).
pub fn is_raw(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => RAW_EXTS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Cheaply read an image's *displayed* pixel dimensions from its header, without
/// decoding the pixels — so the window can be sized to the image before the
/// (potentially multi-second) decode completes. EXIF orientation is applied, so
/// a portrait photo stored sideways reports its upright dimensions.
///
/// Returns `None` for camera RAW (no cheap header probe; develop also transposes
/// for orientation) and on any I/O or parse error — callers fall back to sizing
/// the window once the full decode finishes.
pub fn probe_dimensions(path: &Path) -> Option<(u32, u32)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    if ext == "exr" {
        formats::probe_exr_dimensions(path)
    } else if RAW_EXTS.contains(&ext.as_str()) {
        None
    } else {
        formats::probe_image_dimensions(path)
    }
}

/// Decode an image file into RGBA `ImageData`, dispatching on the (lower-cased)
/// file extension. See plans/rewrite.md §8.2.
pub fn load_image(path: &Path, progress: &std::sync::Arc<ReadProgress>) -> Result<ImageData> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "exr" {
        formats::load_exr(path, progress)
    } else if ext == "gif" {
        // Decode every frame so animated GIFs can play; a single-frame GIF comes
        // back as a plain static image.
        formats::load_gif(path, progress)
    } else if ext == "webp" && formats::webp_is_animated(path) {
        // Animated WebP plays its frames; a *still* WebP falls through to the
        // standard image path below (which keeps full ICC / orientation handling).
        formats::load_animated_webp(path, progress)
    } else if (ext == "png" || ext == "apng") && formats::png_is_apng(path) {
        // Animated PNG (APNG) plays its frames; a plain PNG falls through below.
        formats::load_apng(path, progress)
    } else if RAW_EXTS.contains(&ext.as_str()) {
        // LibRaw (the `ocio`/vcpkg path) develops RAW to scene-linear float with a
        // linear camera response — accurate viewing with recoverable highlights.
        // Without the feature, fall back to the pure-Rust best-effort `rawler`
        // path (sRGB-developed). Both read the file internally; read stays
        // indeterminate.
        #[cfg(feature = "ocio")]
        {
            crate::raw_native::load_raw_native(path)
        }
        #[cfg(not(feature = "ocio"))]
        {
            formats::load_raw(path)
        }
    } else {
        // PNG/JPEG/BMP/TIFF/WebP/GIF/ICO/TGA/PNM, Radiance HDR, and anything
        // else the `image` crate recognises.
        formats::load_via_image(path, progress)
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
