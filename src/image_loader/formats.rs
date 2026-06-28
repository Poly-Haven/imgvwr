//! Per-format decode helpers (see plans/rewrite.md §8.2).
//!
//! Every helper normalises to 4-channel RGBA. Integer formats are treated as
//! sRGB-encoded (`is_encoded_srgb = true`); float formats (EXR / HDR / float
//! TIFF / developed RAW float) are treated as scene-linear.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, ImageDecoder};

use super::{AnimFrame, Animation, ImageData, PixelBuffer, ReadProgress};

/// Animation frame delays are clamped to this minimum (≈ 50 fps). Many GIFs (and
/// some WebP/APNG) encode a 0 or 10 ms delay meaning "as fast as possible";
/// without a floor that would busy-spin the loop, and browsers cap likewise.
const MIN_FRAME_DELAY: Duration = Duration::from_millis(20);

/// Cap on the total decoded-frame bytes kept for one animated image. Frames are
/// decoded lazily and decoding stops once this is exceeded (the animation is
/// truncated — it still plays/loops), so a pathologically large or long GIF /
/// WebP / APNG can't allocate gigabytes. 1 GiB comfortably fits any realistic one.
const MAX_ANIM_FRAME_BYTES: usize = 1024 * 1024 * 1024;

/// A `Read`/`Seek` wrapper that tallies bytes read into a shared [`ReadProgress`]
/// so the loading bar can track a slow (e.g. network-drive) file read. Streams —
/// it never buffers the whole file, so there's no extra memory or I/O cost.
struct CountingReader<R> {
    inner: R,
    progress: Arc<ReadProgress>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.progress.read.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

impl<R: Seek> Seek for CountingReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

/// Open `path`, record its size as the progress total, and wrap it in a
/// byte-counting reader.
fn counting_open(
    path: &Path,
    progress: &Arc<ReadProgress>,
) -> Result<CountingReader<std::fs::File>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let total = file.metadata().map(|m| m.len()).unwrap_or(0);
    progress.total.store(total, Ordering::Relaxed);
    progress.read.store(0, Ordering::Relaxed);
    Ok(CountingReader {
        inner: file,
        progress: progress.clone(),
    })
}

/// Decode an 8/16/32-bit image via the `image` crate (PNG, JPEG, BMP, TIFF,
/// WebP, GIF, ICO, TGA, PNM, **and Radiance HDR**, plus the generic fallback).
pub fn load_via_image(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    // Disable the decoder's default allocation limits: the primary workload is
    // very large (24k+) images and we assume sufficient memory (§1, §8.1). Go
    // through the decoder (not image::open) so the embedded ICC profile can be
    // read before the pixels are decoded. The file is read through a counting
    // reader so the loading bar tracks the (network-)read.
    let counting = counting_open(path, progress)?;
    let mut reader = image::ImageReader::new(std::io::BufReader::new(counting))
        .with_guessed_format()
        .with_context(|| format!("failed to detect format of {}", path.display()))?;
    reader.no_limits();
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("failed to decode {}", path.display()))?;
    let icc = extract_icc(&mut decoder);
    // Read EXIF orientation before `from_decoder` consumes the decoder, then
    // bake it into the pixels so rotated/flipped photos display upright (§8.2).
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder)
        .with_context(|| format!("failed to decode {}", path.display()))?;
    img.apply_orientation(orientation);
    let channels = img.color().channel_count();
    let mut data = dynamic_to_imagedata(path, img, channels, None);
    apply_icc(icc, &mut data);
    Ok(data)
}

/// Decode every frame of an `AnimationDecoder` (GIF / animated WebP / APNG) into
/// full-canvas RGBA8 frames with their delays, applying `orientation` to each
/// (the `image` decoders composite disposal/transparency, but do not apply EXIF
/// orientation). Decoding stops once [`MAX_ANIM_FRAME_BYTES`] is reached so a
/// pathological file can't OOM — a lazy iterator, not `collect_frames()`, so the
/// cap actually bounds the allocation. Returns the (post-orientation) dimensions
/// and the frames.
fn collect_animation<'a, D: image::AnimationDecoder<'a>>(
    decoder: D,
    orientation: image::metadata::Orientation,
    path: &Path,
) -> Result<(u32, u32, Vec<AnimFrame>)> {
    let mut dims: Option<(u32, u32)> = None;
    let mut frames: Vec<AnimFrame> = Vec::new();
    let mut total = 0usize;
    let mut truncated = false;
    for frame in decoder.into_frames() {
        let frame = frame
            .with_context(|| format!("failed to decode animation frames of {}", path.display()))?;
        // numer_denom_ms() gives the delay as a fraction of one millisecond.
        let (num, den) = frame.delay().numer_denom_ms();
        let ms = num as f64 / (den.max(1)) as f64;
        let delay = Duration::from_secs_f64(ms / 1000.0).max(MIN_FRAME_DELAY);
        // Apply EXIF orientation uniformly (a no-op for GIF / most APNG).
        let mut img = DynamicImage::ImageRgba8(frame.into_buffer());
        img.apply_orientation(orientation);
        if dims.is_none() {
            dims = Some((img.width(), img.height()));
        }
        let pixels = img.into_rgba8().into_raw();
        total += pixels.len();
        frames.push(AnimFrame { pixels, delay });
        if total > MAX_ANIM_FRAME_BYTES {
            truncated = true;
            break;
        }
    }
    let (width, height) = dims.ok_or_else(|| anyhow!("animation has no frames"))?;
    if truncated {
        log::warn!(
            "{} exceeds {} MB of frames; playing the first {}",
            path.display(),
            MAX_ANIM_FRAME_BYTES / (1024 * 1024),
            frames.len()
        );
    }
    Ok((width, height, frames))
}

/// Build an [`ImageData`] from decoded animation frames. Frame 0 doubles as the
/// static representation (initial upload / diff / auto-exposure). A single frame
/// yields `animation = None` (a plain static image). Any embedded ICC profile is
/// converted to sRGB across the static buffer *and* every frame.
fn finish_animation(
    path: &Path,
    width: u32,
    height: u32,
    frames: Vec<AnimFrame>,
    icc: Option<Vec<u8>>,
    dtype: &str,
    compression: &str,
) -> Result<ImageData> {
    if frames.is_empty() {
        return Err(anyhow!("animation has no frames"));
    }
    let pixels = frames[0].pixels.clone();
    let animation = (frames.len() > 1).then_some(Animation { frames });
    let mut data = ImageData {
        path: path.to_path_buf(),
        width,
        height,
        channels: 4,
        dtype_name: dtype.to_string(),
        compression: compression.to_string(),
        pixels: PixelBuffer::U8(pixels),
        is_encoded_srgb: true,
        animation,
        camera: None,
        clip_max: super::CLIP_MAX_NORM,
    };
    apply_icc(icc, &mut data);
    Ok(data)
}

/// Decode a GIF, keeping every frame so animated GIFs can play. A single-frame
/// GIF returns `animation = None` (a plain static image).
pub fn load_gif(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    let counting = counting_open(path, progress)?;
    let decoder = image::codecs::gif::GifDecoder::new(std::io::BufReader::new(counting))
        .with_context(|| format!("failed to open GIF {}", path.display()))?;
    // GIF carries no EXIF orientation or ICC profile.
    let (width, height, frames) =
        collect_animation(decoder, image::metadata::Orientation::NoTransforms, path)?;
    finish_animation(path, width, height, frames, None, "gif->uint8", "LZW")
}

/// Whether `path` is an animated WebP (cheap header probe; `false` on any error).
pub fn webp_is_animated(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    image::codecs::webp::WebPDecoder::new(std::io::BufReader::new(file))
        .map(|d| d.has_animation())
        .unwrap_or(false)
}

/// Whether `path` is an animated PNG (APNG) (cheap header probe; `false` on any
/// error or a plain still PNG).
pub fn png_is_apng(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    image::codecs::png::PngDecoder::new(std::io::BufReader::new(file))
        .and_then(|d| d.is_apng())
        .unwrap_or(false)
}

/// Decode an animated WebP into playable frames (EXIF orientation + any embedded
/// ICC profile applied to every frame). Only called for files [`webp_is_animated`]
/// confirmed; a *still* WebP keeps going through [`load_via_image`].
pub fn load_animated_webp(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    let counting = counting_open(path, progress)?;
    let mut decoder = image::codecs::webp::WebPDecoder::new(std::io::BufReader::new(counting))
        .with_context(|| format!("failed to open WebP {}", path.display()))?;
    let icc = extract_icc(&mut decoder);
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let (width, height, frames) = collect_animation(decoder, orientation, path)?;
    finish_animation(path, width, height, frames, icc, "webp->uint8", "WebP")
}

/// Decode an animated PNG (APNG) into playable frames. ICC / orientation are read
/// from the base `PngDecoder` before `apng()` consumes it (the `ApngDecoder`
/// itself exposes neither). Only called for files [`png_is_apng`] confirmed; a
/// still PNG keeps going through [`load_via_image`].
pub fn load_apng(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    let counting = counting_open(path, progress)?;
    let mut decoder = image::codecs::png::PngDecoder::new(std::io::BufReader::new(counting))
        .with_context(|| format!("failed to open PNG {}", path.display()))?;
    let icc = extract_icc(&mut decoder);
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let apng = decoder
        .apng()
        .with_context(|| format!("failed to read APNG frames of {}", path.display()))?;
    let (width, height, frames) = collect_animation(apng, orientation, path)?;
    finish_animation(path, width, height, frames, icc, "apng->uint8", "PNG")
}

/// Read pixel dimensions from an `image`-crate format header (PNG/JPEG/TIFF/…)
/// without decoding the pixels, applying EXIF orientation so the reported size
/// matches what will be displayed. `None` on any error.
pub fn probe_image_dimensions(path: &Path) -> Option<(u32, u32)> {
    let mut reader = image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?;
    reader.no_limits();
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let (w, h) = decoder.dimensions();
    Some(orient_dims(w, h, orientation))
}

/// Read an OpenEXR image's dimensions from its metadata only (no pixel decode).
/// Works even for DWAA/DWAB EXRs the pure-Rust decoder can't fully decode, since
/// only the header is parsed. `None` on any error.
pub fn probe_exr_dimensions(path: &Path) -> Option<(u32, u32)> {
    let meta = exr::prelude::MetaData::read_from_file(path, false).ok()?;
    let size = meta.headers.first()?.layer_size;
    Some((size.0 as u32, size.1 as u32))
}

/// Swap width/height for the EXIF orientations that transpose the image.
fn orient_dims(w: u32, h: u32, orientation: image::metadata::Orientation) -> (u32, u32) {
    use image::metadata::Orientation as O;
    match orientation {
        O::Rotate90 | O::Rotate270 | O::Rotate90FlipH | O::Rotate270FlipH => (h, w),
        _ => (w, h),
    }
}

#[cfg(feature = "icc")]
fn extract_icc(decoder: &mut impl image::ImageDecoder) -> Option<Vec<u8>> {
    decoder.icc_profile().ok().flatten()
}

#[cfg(not(feature = "icc"))]
fn extract_icc(_decoder: &mut impl image::ImageDecoder) -> Option<Vec<u8>> {
    None
}

/// Convert pixels from a non-sRGB embedded ICC profile to sRGB in place (§8.2).
/// Applies only to the 8-bit (display-encoded) path — the static buffer and, for
/// animated WebP / APNG, every frame (each is a separate buffer). Profiles whose
/// description already names sRGB are skipped to avoid a no-op transform.
#[cfg(feature = "icc")]
fn apply_icc(icc: Option<Vec<u8>>, data: &mut ImageData) {
    use lcms2::{InfoType, Intent, Locale, PixelFormat, Profile, Transform};

    let Some(icc) = icc else { return };
    if !matches!(data.pixels, PixelBuffer::U8(_)) {
        return;
    }
    let src = match Profile::new_icc(&icc) {
        Ok(p) => p,
        Err(_) => {
            log::warn!(
                "ignoring unparseable ICC profile in {}",
                data.path.display()
            );
            return;
        }
    };
    if let Some(desc) = src.info(InfoType::Description, Locale::none()) {
        if desc.to_lowercase().contains("srgb") {
            return; // already sRGB
        }
        log::info!("converting ICC profile '{desc}' to sRGB");
    }
    let srgb = Profile::new_srgb();
    let transform = match Transform::new(
        &src,
        PixelFormat::RGBA_8,
        &srgb,
        PixelFormat::RGBA_8,
        Intent::Perceptual,
    ) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("ICC transform setup failed ({e}); leaving pixels unconverted");
            return;
        }
    };
    if let PixelBuffer::U8(rgba) = &mut data.pixels {
        let px: &mut [[u8; 4]] = bytemuck::cast_slice_mut(rgba.as_mut_slice());
        transform.transform_in_place(px);
    }
    if let Some(anim) = data.animation.as_mut() {
        for f in &mut anim.frames {
            let px: &mut [[u8; 4]] = bytemuck::cast_slice_mut(f.pixels.as_mut_slice());
            transform.transform_in_place(px);
        }
    }
}

#[cfg(not(feature = "icc"))]
fn apply_icc(_icc: Option<Vec<u8>>, _data: &mut ImageData) {}

/// Best-effort camera RAW via `rawler`: decode, run the default develop pipeline
/// (demosaic + white balance + sRGB), and read the developed f32 samples
/// directly. The goal is a *viewable* image, not colour-accurate science (§8.2).
///
/// We read rawler's `Intermediate` rather than its `to_dynamic_image()` because
/// rawler links `image` 0.24 while the rest of imgvwr uses 0.25 — the
/// `DynamicImage` types are not interchangeable across that boundary.
#[cfg(not(feature = "ocio"))]
pub fn load_raw(path: &Path) -> Result<ImageData> {
    use rawler::imgop::develop::{Intermediate, RawDevelop};

    let raw = rawler::decode_file(path).map_err(|e| anyhow!("RAW decode failed: {e:?}"))?;
    let developed = RawDevelop::default()
        .develop_intermediate(&raw)
        .map_err(|e| anyhow!("RAW develop failed: {e:?}"))?;

    let dim = developed.dim();
    let (mut width, mut height) = (dim.w, dim.h);
    let n = width * height;
    if n == 0 {
        return Err(anyhow!("RAW develop produced an empty image"));
    }

    // The develop pipeline's final step is sRGB encoding, so the samples are
    // display-referred sRGB in [0, 1] -> is_encoded_srgb = true.
    let mut rgba = vec![0f32; n * 4];
    let channels: u8 = match developed {
        Intermediate::Monochrome(p) => {
            for i in 0..n {
                let v = p.data[i];
                rgba[i * 4] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = 1.0;
            }
            1
        }
        Intermediate::ThreeColor(p) => {
            for i in 0..n {
                rgba[i * 4] = p.data[i][0];
                rgba[i * 4 + 1] = p.data[i][1];
                rgba[i * 4 + 2] = p.data[i][2];
                rgba[i * 4 + 3] = 1.0;
            }
            3
        }
        Intermediate::FourColor(p) => {
            for i in 0..n {
                rgba[i * 4] = p.data[i][0];
                rgba[i * 4 + 1] = p.data[i][1];
                rgba[i * 4 + 2] = p.data[i][2];
                rgba[i * 4 + 3] = 1.0;
            }
            4
        }
    };

    // rawler 0.6 hardcodes `RawImage.orientation` to Normal, so the developed
    // buffer is in sensor order. Read the real EXIF orientation from the file
    // metadata and bake it into the pixels (portrait photos otherwise display
    // sideways).
    if let Some(flips) = raw_orientation_flips(path) {
        apply_orientation_f32(&mut rgba, &mut width, &mut height, flips);
    }

    Ok(ImageData {
        path: path.to_path_buf(),
        width: width as u32,
        height: height as u32,
        channels,
        dtype_name: "raw->float32".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::F32(rgba),
        is_encoded_srgb: true,
        animation: None,
        camera: None,
        // rawler develops to display-referred sRGB in [0, 1] → max 1.0.
        clip_max: super::CLIP_MAX_NORM,
    })
}

/// Read a RAW file's EXIF orientation as `(transpose, flip_h, flip_v)`.
///
/// rawler develops pixels in sensor order and its higher-level metadata parse is
/// brittle (it rejects some otherwise-decodable files on unrelated tags), so we
/// read the TIFF Orientation tag (0x0112 in IFD0) directly. This covers the
/// TIFF-based raws (NEF/CR2/DNG/ARW/ORF/RW2/PEF/SR2/NRW/RWL); non-TIFF raws fall
/// through to `None` (no rotation), matching the previous behaviour. Returns
/// `None` for missing/identity orientation.
#[cfg(not(feature = "ocio"))]
fn raw_orientation_flips(path: &Path) -> Option<(bool, bool, bool)> {
    let orientation = read_tiff_orientation(path)?;
    let flips = rawler::Orientation::from_u16(orientation).to_flips();
    (flips != (false, false, false)).then_some(flips)
}

/// Read the IFD0 Orientation tag (0x0112) from a little/big-endian TIFF-based
/// file. Returns the raw EXIF orientation value (1..=8), or `None`.
#[cfg(not(feature = "ocio"))]
fn read_tiff_orientation(path: &Path) -> Option<u16> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path).ok()?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr).ok()?;
    let le = match &hdr[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16f = |b: &[u8]| {
        if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }
    };
    let u32f = |b: &[u8]| {
        if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    if u16f(&hdr[2..4]) != 42 {
        return None; // not a TIFF magic
    }
    f.seek(SeekFrom::Start(u32f(&hdr[4..8]) as u64)).ok()?;
    let mut cnt = [0u8; 2];
    f.read_exact(&mut cnt).ok()?;
    let mut entry = [0u8; 12];
    for _ in 0..u16f(&cnt) {
        f.read_exact(&mut entry).ok()?;
        if u16f(&entry[0..2]) == 0x0112 {
            // SHORT value lives in the first two bytes of the value field.
            return Some(u16f(&entry[8..10]));
        }
    }
    None
}

/// Apply EXIF flips/transpose to an interleaved RGBA-f32 buffer in place,
/// updating `width`/`height` when transposed. Flips are applied before the
/// transpose, matching rawler's `Orientation::to_flips` contract.
#[cfg(not(feature = "ocio"))]
fn apply_orientation_f32(
    rgba: &mut Vec<f32>,
    width: &mut usize,
    height: &mut usize,
    flips: (bool, bool, bool),
) {
    let (transpose, flip_h, flip_v) = flips;
    let (w, h) = (*width, *height);
    if flip_h {
        for y in 0..h {
            for x in 0..w / 2 {
                let a = (y * w + x) * 4;
                let b = (y * w + (w - 1 - x)) * 4;
                for k in 0..4 {
                    rgba.swap(a + k, b + k);
                }
            }
        }
    }
    if flip_v {
        for y in 0..h / 2 {
            for x in 0..w {
                let a = (y * w + x) * 4;
                let b = ((h - 1 - y) * w + x) * 4;
                for k in 0..4 {
                    rgba.swap(a + k, b + k);
                }
            }
        }
    }
    if transpose {
        let mut out = vec![0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let s = (y * w + x) * 4;
                let d = (x * h + y) * 4; // new row stride is the old height
                out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
            }
        }
        *rgba = out;
        *width = h;
        *height = w;
    }
}

/// OpenEXR via the `exr` crate. Reads the first valid layer's largest level and
/// maps channels by name (R/G/B/A), replicating single channels and taking the
/// first up-to-three channels for arbitrary AOV/data layers. Never fails solely
/// because the layer is not named RGBA.
// Without the `ocio` feature the Err arm is a plain `Err(e)`, which clippy
// flags as a needless match; the OpenEXR fallback (with the feature) is not.
#[cfg_attr(not(feature = "ocio"), allow(clippy::needless_match))]
pub fn load_exr(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    match load_exr_rust(path, progress) {
        Ok(data) => Ok(data),
        Err(e) => {
            // The pure-Rust exr crate cannot decode DWAA/DWAB (and a few other)
            // compressions; fall back to the OpenEXR C++ shim when available.
            #[cfg(feature = "ocio")]
            {
                log::warn!(
                    "exr crate could not decode {} ({e}); trying OpenEXR fallback",
                    path.display()
                );
                crate::exr_native::load_exr_native(path)
            }
            #[cfg(not(feature = "ocio"))]
            {
                Err(e)
            }
        }
    }
}

fn load_exr_rust(path: &Path, progress: &Arc<ReadProgress>) -> Result<ImageData> {
    use exr::prelude::*;

    // Read through a counting reader so the loading bar tracks the file read.
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_unbuffered(counting_open(path, progress)?)
        .map_err(|e| anyhow!("EXR decode failed: {e}"))?;

    let layer = &image.layer_data;
    let width = layer.size.0;
    let height = layer.size.1;
    let pixel_count = width * height;
    let list = &layer.channel_data.list;
    if list.is_empty() || pixel_count == 0 {
        return Err(anyhow!("EXR has no channels or zero size"));
    }

    let names: Vec<String> = list.iter().map(|c| c.name.to_string()).collect();
    log::debug!("EXR {} channels: {:?}", names.len(), names);

    let dtype = match &list[0].sample_data {
        FlatSamples::F16(_) => "half",
        FlatSamples::F32(_) => "float32",
        FlatSamples::U32(_) => "uint32",
    };
    // Clip point for the overlay: a 16-bit-half EXR genuinely clips at the half
    // max (65504); 32-bit float is unbounded (never clips); 32-bit uint clips at
    // 2^32-1. The samples are uploaded as f32 in their native value space.
    let clip_max = match &list[0].sample_data {
        FlatSamples::F16(_) => [super::HALF_MAX; 4],
        FlatSamples::F32(_) => super::CLIP_MAX_NONE,
        FlatSamples::U32(_) => [u32::MAX as f32; 4],
    };
    let compression = format!("{:?}", layer.encoding.compression);

    let to_f32 = |samples: &FlatSamples| -> Vec<f32> {
        match samples {
            FlatSamples::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            FlatSamples::F32(v) => v.clone(),
            FlatSamples::U32(v) => v.iter().map(|x| *x as f32).collect(),
        }
    };
    // Channel key = the part after the last '.', upper-cased ("diffuse.R" -> "R").
    let key =
        |name: &str| -> String { name.rsplit('.').next().unwrap_or(name).to_ascii_uppercase() };
    let find = |k: &str| names.iter().position(|n| key(n) == k);

    let mut rgba = vec![0f32; pixel_count * 4];
    let original_channels: u8;

    if let (Some(ri), Some(gi), Some(bi)) = (find("R"), find("G"), find("B")) {
        let r = to_f32(&list[ri].sample_data);
        let g = to_f32(&list[gi].sample_data);
        let b = to_f32(&list[bi].sample_data);
        let a = find("A").map(|ai| to_f32(&list[ai].sample_data));
        ensure_len(&[&r, &g, &b], pixel_count)?;
        for i in 0..pixel_count {
            rgba[i * 4] = r[i];
            rgba[i * 4 + 1] = g[i];
            rgba[i * 4 + 2] = b[i];
            rgba[i * 4 + 3] = a.as_ref().map(|a| a[i]).unwrap_or(1.0);
        }
        original_channels = 3 + a.is_some() as u8;
    } else {
        // Arbitrary / single / dual channels: take the first up to three in
        // file order; replicate a lone channel to RGB.
        let take = list.len().min(3);
        let cols: Vec<Vec<f32>> = (0..take).map(|i| to_f32(&list[i].sample_data)).collect();
        ensure_len(&cols.iter().collect::<Vec<_>>(), pixel_count)?;
        for i in 0..pixel_count {
            let r = cols[0][i];
            let g = if take > 1 { cols[1][i] } else { r };
            let b = if take > 2 {
                cols[2][i]
            } else if take == 1 {
                r
            } else {
                0.0
            };
            rgba[i * 4] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 1.0;
        }
        original_channels = list.len() as u8;
    }

    Ok(ImageData {
        path: path.to_path_buf(),
        width: width as u32,
        height: height as u32,
        channels: original_channels,
        dtype_name: dtype.to_string(),
        compression,
        pixels: PixelBuffer::F32(rgba),
        is_encoded_srgb: false,
        animation: None,
        camera: None,
        clip_max,
    })
}

fn ensure_len(channels: &[&Vec<f32>], expected: usize) -> Result<()> {
    for c in channels {
        if c.len() < expected {
            return Err(anyhow!(
                "EXR channel shorter ({}) than image ({expected})",
                c.len()
            ));
        }
    }
    Ok(())
}

/// Normalise any `image::DynamicImage` to RGBA `ImageData`.
///
/// * 8-bit variants  → `U8`, sRGB-encoded.
/// * 16-bit variants → `F32` (value / 65535), sRGB-encoded.
/// * 32-bit float    → `F32`, scene-linear (HDR / float TIFF).
pub fn dynamic_to_imagedata(
    path: &Path,
    img: DynamicImage,
    original_channels: u8,
    dtype_prefix: Option<String>,
) -> ImageData {
    use image::DynamicImage as D;

    let width = img.width();
    let height = img.height();

    let is_float = matches!(img, D::ImageRgb32F(_) | D::ImageRgba32F(_));
    let is_16 = matches!(
        img,
        D::ImageLuma16(_) | D::ImageLumaA16(_) | D::ImageRgb16(_) | D::ImageRgba16(_)
    );

    let (pixels, dtype, is_srgb) = if is_float {
        let buf = img.to_rgba32f();
        (PixelBuffer::F32(buf.into_raw()), "float32", false)
    } else if is_16 {
        let buf = img.to_rgba16();
        let floats = buf.into_raw().iter().map(|&v| v as f32 / 65535.0).collect();
        (PixelBuffer::F32(floats), "uint16", true)
    } else {
        let buf = img.to_rgba8();
        (PixelBuffer::U8(buf.into_raw()), "uint8", true)
    };

    let dtype_name = match dtype_prefix {
        Some(prefix) => format!("{prefix}->{dtype}"),
        None => dtype.to_string(),
    };

    // 32-bit float (HDR / float TIFF) is unbounded → never clips; 8-/16-bit
    // integer formats normalise their max to 1.0.
    let clip_max = if is_float {
        super::CLIP_MAX_NONE
    } else {
        super::CLIP_MAX_NORM
    };

    ImageData {
        path: path.to_path_buf(),
        width,
        height,
        channels: original_channels,
        dtype_name,
        compression: "-".to_string(),
        pixels,
        is_encoded_srgb: is_srgb,
        animation: None,
        camera: None,
        clip_max,
    }
}

#[cfg(all(test, feature = "icc"))]
mod icc_tests {
    use super::*;
    use lcms2::{CIExyY, CIExyYTRIPLE, Profile, ToneCurve};

    fn solid_gray(value: u8) -> ImageData {
        ImageData {
            path: std::path::PathBuf::from("test.png"),
            width: 1,
            height: 1,
            channels: 3,
            dtype_name: "uint8".to_string(),
            compression: "-".to_string(),
            pixels: PixelBuffer::U8(vec![value, value, value, 255]),
            is_encoded_srgb: true,
            animation: None,
            camera: None,
            clip_max: crate::image_loader::CLIP_MAX_NORM,
        }
    }

    /// A linear-gamma (sRGB-primaries) profile must be re-encoded to sRGB:
    /// linear mid-gray 128 brightens to ~188 after the sRGB transfer function.
    #[test]
    fn linear_profile_converts_to_srgb() {
        let d65 = CIExyY {
            x: 0.3127,
            y: 0.3290,
            Y: 1.0,
        };
        let primaries = CIExyYTRIPLE {
            Red: CIExyY {
                x: 0.64,
                y: 0.33,
                Y: 1.0,
            },
            Green: CIExyY {
                x: 0.30,
                y: 0.60,
                Y: 1.0,
            },
            Blue: CIExyY {
                x: 0.15,
                y: 0.06,
                Y: 1.0,
            },
        };
        let linear = ToneCurve::new(1.0);
        let profile = Profile::new_rgb(&d65, &primaries, &[&linear, &linear, &linear]).unwrap();
        let icc = profile.icc().unwrap();

        let mut data = solid_gray(128);
        apply_icc(Some(icc), &mut data);
        let PixelBuffer::U8(v) = &data.pixels else {
            panic!("expected U8")
        };
        assert!(
            v[0] > 170 && v[0] < 200,
            "linear 128 should encode to ~188, got {}",
            v[0]
        );
        assert_eq!(v[0], v[1]);
        assert_eq!(v[0], v[2]);
        assert_eq!(v[3], 255, "alpha preserved");
    }

    /// An sRGB-named profile is skipped (or is an identity transform): no change.
    #[test]
    fn srgb_profile_is_noop() {
        let icc = Profile::new_srgb().icc().unwrap();
        let mut data = solid_gray(100);
        apply_icc(Some(icc), &mut data);
        let PixelBuffer::U8(v) = &data.pixels else {
            panic!("expected U8")
        };
        assert!(
            (v[0] as i32 - 100).abs() <= 2,
            "sRGB profile should be ~no-op, got {}",
            v[0]
        );
    }

    /// No ICC profile present -> pixels untouched.
    #[test]
    fn no_profile_leaves_pixels_untouched() {
        let mut data = solid_gray(77);
        apply_icc(None, &mut data);
        let PixelBuffer::U8(v) = &data.pixels else {
            panic!("expected U8")
        };
        assert_eq!(v[0], 77);
    }
}
