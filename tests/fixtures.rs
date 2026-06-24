//! Integration tests: load each committed fixture and assert dimensions,
//! channel normalisation (-> RGBA), pixel-buffer kind, and is_encoded_srgb
//! per format (see plans/rewrite.md §17.2).

use std::path::{Path, PathBuf};

use imgvwr::image_loader::{load_image, ImageData, PixelBuffer};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn load(name: &str) -> ImageData {
    load_image(&fixture(name)).unwrap_or_else(|e| panic!("failed to load {name}: {e:#}"))
}

#[test]
fn png_8bit_is_u8_srgb_rgba() {
    let d = load("tiny_rgb.png");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(d.is_encoded_srgb, "8-bit PNG should be sRGB-encoded");
    match &d.pixels {
        PixelBuffer::U8(v) => assert_eq!(v.len(), 4 * 4 * 4, "must be normalised to RGBA"),
        _ => panic!("expected U8 buffer"),
    }
}

#[test]
fn jpeg_is_u8_srgb() {
    let d = load("tiny.jpg");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(d.is_encoded_srgb);
    assert!(matches!(d.pixels, PixelBuffer::U8(_)));
}

#[test]
fn png_16bit_is_f32_srgb() {
    // 16-bit integer formats decode to F32 (value/65535) and stay sRGB-encoded.
    let d = load("tiny_gray16.png");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(d.is_encoded_srgb);
    match &d.pixels {
        PixelBuffer::F32(v) => assert_eq!(v.len(), 4 * 4 * 4),
        _ => panic!("expected F32 buffer for 16-bit PNG"),
    }
}

#[test]
fn exr_rgba_is_f32_linear() {
    let d = load("tiny_rgba.exr");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(!d.is_encoded_srgb, "EXR is scene-linear");
    assert_eq!(d.channels, 4);
    match &d.pixels {
        PixelBuffer::F32(v) => assert_eq!(v.len(), 4 * 4 * 4),
        _ => panic!("expected F32 buffer"),
    }
}

#[test]
fn exr_single_channel_replicates_to_rgb() {
    let d = load("tiny_gray.exr");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(!d.is_encoded_srgb);
    assert_eq!(d.channels, 1, "original channel count is recorded");
    match &d.pixels {
        PixelBuffer::F32(v) => {
            assert_eq!(v.len(), 4 * 4 * 4);
            // Each pixel: R == G == B (replicated), A == 1.
            for px in v.chunks_exact(4) {
                assert!((px[0] - px[1]).abs() < 1e-5);
                assert!((px[0] - px[2]).abs() < 1e-5);
                assert!((px[3] - 1.0).abs() < 1e-5, "alpha defaults to 1");
            }
        }
        _ => panic!("expected F32 buffer"),
    }
}

#[test]
fn hdr_is_f32_linear() {
    let d = load("tiny.hdr");
    assert_eq!((d.width, d.height), (4, 4));
    assert!(!d.is_encoded_srgb, "Radiance HDR is scene-linear");
    match &d.pixels {
        PixelBuffer::F32(v) => assert_eq!(v.len(), 4 * 4 * 4),
        _ => panic!("expected F32 buffer"),
    }
}
