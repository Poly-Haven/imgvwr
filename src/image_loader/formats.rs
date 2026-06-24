//! Per-format decode helpers (see plans/rewrite.md §8.2).
//!
//! Every helper normalises to 4-channel RGBA. Integer formats are treated as
//! sRGB-encoded (`is_encoded_srgb = true`); float formats (EXR / HDR / float
//! TIFF / developed RAW float) are treated as scene-linear.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use image::DynamicImage;

use super::{ImageData, PixelBuffer};

/// Decode an 8/16/32-bit image via the `image` crate (PNG, JPEG, BMP, TIFF,
/// WebP, GIF, ICO, TGA, PNM, **and Radiance HDR**, plus the generic fallback).
pub fn load_via_image(path: &Path) -> Result<ImageData> {
    let img = image::open(path).with_context(|| format!("failed to decode {}", path.display()))?;
    let channels = img.color().channel_count();
    Ok(dynamic_to_imagedata(path, img, channels, None))
}

/// Best-effort camera RAW via `rawler`: decode, run the default develop pipeline
/// (demosaic + white balance + sRGB), and read the developed f32 samples
/// directly. The goal is a *viewable* image, not colour-accurate science (§8.2).
///
/// We read rawler's `Intermediate` rather than its `to_dynamic_image()` because
/// rawler links `image` 0.24 while the rest of imgvwr uses 0.25 — the
/// `DynamicImage` types are not interchangeable across that boundary.
pub fn load_raw(path: &Path) -> Result<ImageData> {
    use rawler::imgop::develop::{Intermediate, RawDevelop};

    let raw = rawler::decode_file(path).map_err(|e| anyhow!("RAW decode failed: {e:?}"))?;
    let developed = RawDevelop::default()
        .develop_intermediate(&raw)
        .map_err(|e| anyhow!("RAW develop failed: {e:?}"))?;

    let dim = developed.dim();
    let (width, height) = (dim.w, dim.h);
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

    Ok(ImageData {
        path: path.to_path_buf(),
        width: width as u32,
        height: height as u32,
        channels,
        dtype_name: "raw->float32".to_string(),
        compression: "-".to_string(),
        pixels: PixelBuffer::F32(rgba),
        is_encoded_srgb: true,
    })
}

/// OpenEXR via the `exr` crate. Reads the first valid layer's largest level and
/// maps channels by name (R/G/B/A), replicating single channels and taking the
/// first up-to-three channels for arbitrary AOV/data layers. Never fails solely
/// because the layer is not named RGBA.
pub fn load_exr(path: &Path) -> Result<ImageData> {
    match load_exr_rust(path) {
        Ok(data) => Ok(data),
        Err(e) => {
            // The pure-Rust exr crate cannot decode DWAA/DWAB (and a few other)
            // compressions; fall back to the OpenEXR C++ shim when available.
            #[cfg(feature = "ocio")]
            {
                log::warn!("exr crate could not decode {} ({e}); trying OpenEXR fallback", path.display());
                crate::exr_native::load_exr_native(path)
            }
            #[cfg(not(feature = "ocio"))]
            {
                Err(e)
            }
        }
    }
}

fn load_exr_rust(path: &Path) -> Result<ImageData> {
    use exr::prelude::*;

    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_file(path)
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
    let compression = format!("{:?}", layer.encoding.compression);

    let to_f32 = |samples: &FlatSamples| -> Vec<f32> {
        match samples {
            FlatSamples::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            FlatSamples::F32(v) => v.clone(),
            FlatSamples::U32(v) => v.iter().map(|x| *x as f32).collect(),
        }
    };
    // Channel key = the part after the last '.', upper-cased ("diffuse.R" -> "R").
    let key = |name: &str| -> String {
        name.rsplit('.').next().unwrap_or(name).to_ascii_uppercase()
    };
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

    ImageData {
        path: path.to_path_buf(),
        width,
        height,
        channels: original_channels,
        dtype_name,
        compression: "-".to_string(),
        pixels,
        is_encoded_srgb: is_srgb,
    }
}
