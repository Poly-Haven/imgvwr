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

    /// True when the image is a real equirectangular panorama: it has the 2:1
    /// aspect ratio **and** its pixel content shows the structural signatures of
    /// an equirect (see [`equirect_content_scores`]). The content test rejects
    /// ordinary 2:1 images (crops, renders, side-by-side pairs) that merely share
    /// the aspect ratio. Falls back to the aspect-only test if the buffer can't be
    /// sampled.
    pub fn is_equirectangular(&self) -> bool {
        is_equirectangular(self.width, self.height)
            && equirect_content_scores(self.width, self.height, &self.pixels)
                .as_ref()
                .is_none_or(|s| s.looks_like_pano())
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

/// Panorama detection: an image has the *aspect ratio* of an equirectangular
/// panorama when its width is exactly twice its height. This is necessary but not
/// sufficient — see [`ImageData::is_equirectangular`] / [`equirect_content_scores`]
/// for the content test that separates real panoramas from ordinary 2:1 images.
pub fn is_equirectangular(width: u32, height: u32) -> bool {
    height > 0 && width == height * 2
}

/// Structural signatures that distinguish a real equirectangular panorama from an
/// ordinary 2:1 image, measured straight from the pixels (no metadata):
///
/// * **Pole convergence** — every pixel in the top row maps to the zenith and
///   every pixel in the bottom row to the nadir (one 3-D point each), so both edge
///   rows are nearly a constant colour. `pole_top` / `pole_bottom` are the mean
///   per-channel standard deviation along each row; ~0 means flat/pole-like.
/// * **Horizontal wrap** — a 360° panorama is seamless, so its left and right edge
///   columns match. `wrap` is the mean absolute left-vs-right difference.
///
/// All three are measured in a tonemapped space so unbounded HDR highlights can't
/// dominate, and lower is more panorama-like for each.
pub struct EquirectScores {
    pub pole_top: f32,
    pub pole_bottom: f32,
    pub wrap: f32,
}

// Thresholds sit in the empty gap measured across real files: genuine equirect
// HDRIs score poles ~0.000–0.003 and wrap ~0.000–0.001, while ordinary 2:1
// renders (including a deliberately hard case with near-matching side colours)
// score poles ~0.033–0.074 and wrap ~0.031–0.078. 0.015 / 0.012 leave a ~5×
// margin below the panoramas and ~2× above the non-panoramas.
/// A row counts as flat (pole-like) below this mean per-channel stddev.
const POLE_FLAT_MAX: f32 = 0.015;
/// Left/right edges count as seamless below this mean absolute difference.
const WRAP_SEAM_MAX: f32 = 0.012;
/// Points sampled along each edge. Subsampled for large images, so the whole test
/// touches at most a few thousand texels regardless of resolution.
const EDGE_SAMPLES: usize = 512;

impl EquirectScores {
    /// Whether the measured content looks like a real equirectangular panorama.
    ///
    /// Pole convergence is the primary signal — an ordinary 2:1 photo or render
    /// has detail along its top/bottom rows, while a panorama's poles collapse to
    /// near-constant colour. Both poles must be flat. A seamless horizontal wrap
    /// then *confirms* a flat-poled image, but is deliberately **not** trusted on
    /// its own: some 2:1 renders happen to have matching side colours, and a real
    /// panorama cropped below 360° won't wrap — so wrap only rescues the borderline
    /// case where one pole is slightly over the flatness threshold.
    pub fn looks_like_pano(&self) -> bool {
        let both_poles_flat = self.pole_top < POLE_FLAT_MAX && self.pole_bottom < POLE_FLAT_MAX;
        let one_pole_flat = self.pole_top < POLE_FLAT_MAX || self.pole_bottom < POLE_FLAT_MAX;
        both_poles_flat || (one_pole_flat && self.wrap < WRAP_SEAM_MAX)
    }
}

/// Measure the panorama-content signatures of an image (see [`EquirectScores`]).
/// Returns `None` when the buffer is too small to sample — callers then fall back
/// to the aspect-ratio-only heuristic.
pub fn equirect_content_scores(
    width: u32,
    height: u32,
    pixels: &PixelBuffer,
) -> Option<EquirectScores> {
    let (w, h) = (width as usize, height as usize);
    let len = match pixels {
        PixelBuffer::U8(v) => v.len(),
        PixelBuffer::F32(v) => v.len(),
    };
    if w == 0 || h == 0 || len < w * h * 4 {
        return None;
    }
    Some(EquirectScores {
        pole_top: row_roughness(pixels, w, 0),
        pole_bottom: row_roughness(pixels, w, h - 1),
        wrap: wrap_diff(pixels, w, h),
    })
}

/// Tonemapped (Reinhard) RGB of one texel, keyed by pixel index. Reinhard maps
/// `[0, ∞) → [0, 1)` so an unbounded HDR highlight can't dominate the variance /
/// difference sums; it's monotonic, so 8-bit values (already `0..1`) are unharmed.
fn texel_tonemapped(pixels: &PixelBuffer, idx: usize) -> [f32; 3] {
    let reinhard = |x: f32| {
        let x = x.max(0.0);
        x / (1.0 + x)
    };
    let o = idx * 4;
    match pixels {
        PixelBuffer::U8(v) => [
            reinhard(v[o] as f32 / 255.0),
            reinhard(v[o + 1] as f32 / 255.0),
            reinhard(v[o + 2] as f32 / 255.0),
        ],
        PixelBuffer::F32(v) => [reinhard(v[o]), reinhard(v[o + 1]), reinhard(v[o + 2])],
    }
}

/// Mean per-channel standard deviation of tonemapped colour along row `y`, sampled
/// at up to [`EDGE_SAMPLES`] points spread across the full width. Near-zero for a
/// pole row (all one 3-D point); larger for a row of real image detail.
fn row_roughness(pixels: &PixelBuffer, width: usize, y: usize) -> f32 {
    let n = EDGE_SAMPLES.min(width).max(1);
    let mut sum = [0.0f64; 3];
    let mut sumsq = [0.0f64; 3];
    for i in 0..n {
        let x = if n > 1 { i * (width - 1) / (n - 1) } else { 0 };
        let t = texel_tonemapped(pixels, y * width + x);
        for c in 0..3 {
            sum[c] += t[c] as f64;
            sumsq[c] += (t[c] as f64) * (t[c] as f64);
        }
    }
    let nf = n as f64;
    let mut acc = 0.0f64;
    for c in 0..3 {
        let mean = sum[c] / nf;
        acc += (sumsq[c] / nf - mean * mean).max(0.0).sqrt();
    }
    (acc / 3.0) as f32
}

/// Mean absolute tonemapped difference between the left and right edge columns,
/// sampled at up to [`EDGE_SAMPLES`] rows. Near-zero for a seamless 360° wrap.
fn wrap_diff(pixels: &PixelBuffer, width: usize, height: usize) -> f32 {
    let n = EDGE_SAMPLES.min(height).max(1);
    let mut acc = 0.0f64;
    for i in 0..n {
        let y = if n > 1 { i * (height - 1) / (n - 1) } else { 0 };
        let l = texel_tonemapped(pixels, y * width);
        let r = texel_tonemapped(pixels, y * width + width - 1);
        for c in 0..3 {
            acc += (l[c] - r[c]).abs() as f64;
        }
    }
    (acc / (n as f64 * 3.0)) as f32
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

    // Reject clearly-unsupported files up front with a readable message (it shows
    // in the centre error panel) instead of a cryptic decoder error from below.
    if !is_supported(path) {
        let what = if ext.is_empty() {
            "files without an extension".to_string()
        } else {
            format!(".{ext} files")
        };
        anyhow::bail!(
            "Unsupported file type: imgvwr can't open {what}.\n\nIt supports JPEG, PNG, \
             GIF, BMP, TIFF, WebP, HDR, OpenEXR and camera RAW."
        );
    }

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

    /// Build a greyscale RGBA8 buffer from a per-pixel value closure.
    fn buf(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> PixelBuffer {
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let v = f(x, y);
                px[o] = v;
                px[o + 1] = v;
                px[o + 2] = v;
                px[o + 3] = 255;
            }
        }
        PixelBuffer::U8(px)
    }

    fn make_img(w: u32, h: u32, pixels: PixelBuffer) -> ImageData {
        ImageData {
            path: PathBuf::from("test"),
            width: w,
            height: h,
            channels: 4,
            dtype_name: "test".into(),
            compression: "test".into(),
            pixels,
            is_encoded_srgb: true,
            clip_max: CLIP_MAX_NORM,
            animation: None,
            camera: None,
        }
    }

    #[test]
    fn looks_like_pano_decision() {
        let s = |pole_top, pole_bottom, wrap| EquirectScores {
            pole_top,
            pole_bottom,
            wrap,
        };
        // Both poles flat -> panorama regardless of wrap.
        assert!(s(0.001, 0.002, 0.5).looks_like_pano());
        // Neither pole flat -> not a panorama (the hard 2:1 render case), even
        // with a smallish wrap.
        assert!(!s(0.033, 0.034, 0.031).looks_like_pano());
        // One pole flat + a seamless wrap -> rescued (e.g. a busy nadir).
        assert!(s(0.001, 0.05, 0.005).looks_like_pano());
        // One pole flat but the wrap isn't seamless -> not enough on its own.
        assert!(!s(0.001, 0.05, 0.05).looks_like_pano());
    }

    #[test]
    fn content_scores_flat_poles_vs_detail() {
        // Vertical gradient: every row is constant across x, so both pole rows are
        // flat and the left/right columns match -> reads as a panorama.
        let pano = buf(64, 32, |_, y| (y * 6) as u8);
        let s = equirect_content_scores(64, 32, &pano).unwrap();
        assert!(s.pole_top < 0.001, "pole_top={}", s.pole_top);
        assert!(s.pole_bottom < 0.001, "pole_bottom={}", s.pole_bottom);
        assert!(s.wrap < 0.001, "wrap={}", s.wrap);
        assert!(s.looks_like_pano());

        // Horizontal ramp: colour varies along x, so the pole rows carry detail
        // and the side columns differ -> not a panorama.
        let flat2d = buf(64, 32, |x, _| ((x * 255) / 63) as u8);
        let s = equirect_content_scores(64, 32, &flat2d).unwrap();
        assert!(s.pole_top > 0.05, "pole_top={}", s.pole_top);
        assert!(!s.looks_like_pano());
    }

    #[test]
    fn is_equirectangular_requires_aspect_and_content() {
        // 2:1 with pano-like content -> panorama.
        assert!(make_img(64, 32, buf(64, 32, |_, y| (y * 6) as u8)).is_equirectangular());
        // Same flat content but a square aspect ratio -> not a panorama.
        assert!(!make_img(32, 32, buf(32, 32, |_, y| (y * 6) as u8)).is_equirectangular());
        // 2:1 aspect but ordinary 2D content -> not a panorama.
        assert!(!make_img(64, 32, buf(64, 32, |x, _| ((x * 255) / 63) as u8)).is_equirectangular());
    }
}
