//! Integration tests: load each committed fixture and assert dimensions,
//! channel normalisation (-> RGBA), pixel-buffer kind, and is_encoded_srgb
//! per format (see plans/rewrite.md §17.2).

use std::path::{Path, PathBuf};

use std::sync::Arc;

use imgvwr::image_loader::{
    load_image, probe_dimensions, DecodeIntent, ImageData, PixelBuffer, ReadProgress,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn load(name: &str) -> ImageData {
    load_with(name, DecodeIntent::Latency)
}

fn load_with(name: &str, intent: DecodeIntent) -> ImageData {
    let progress = Arc::new(ReadProgress::default());
    load_image(&fixture(name), &progress, intent)
        .unwrap_or_else(|e| panic!("failed to load {name}: {e:#}"))
}

/// The OpenEXR C++ fallback is normally reached only for compressions the
/// pure-Rust decoder rejects (DWAA/DWAB), so it is easy for a change to it to go
/// unexercised. Call it directly on an ordinary fixture — OpenEXR reads every
/// compression — and require it to agree with the Rust decoder, at both thread
/// counts (the fallback now opens files with an explicit worker count).
#[cfg(feature = "ocio")]
#[test]
fn openexr_fallback_matches_the_rust_decoder() {
    for name in ["tiny_rgba.exr", "tiny_gray.exr"] {
        let reference = load(name);
        for intent in [DecodeIntent::Latency, DecodeIntent::Throughput] {
            let native = imgvwr::exr_native::load_exr_native(&fixture(name), intent)
                .unwrap_or_else(|e| panic!("OpenEXR fallback failed on {name}: {e:#}"));
            assert_eq!(
                (native.width, native.height),
                (reference.width, reference.height),
                "{name} dimensions ({intent:?})"
            );
            match (&reference.pixels, &native.pixels) {
                (PixelBuffer::F32(a), PixelBuffer::F32(b)) => {
                    assert_eq!(
                        a, b,
                        "{name} pixels differ from the Rust decoder ({intent:?})"
                    )
                }
                _ => panic!("expected F32 buffers for {name}"),
            }
        }
    }
}

/// The latency/throughput split is a scheduling choice, not a quality one: a
/// serial decode must produce byte-identical pixels to a parallel one. EXR is
/// the only format that actually branches, so it is the one worth pinning.
#[test]
fn decode_intent_does_not_change_pixels() {
    for name in ["tiny_rgba.exr", "tiny_gray.exr"] {
        let fast = load_with(name, DecodeIntent::Latency);
        let serial = load_with(name, DecodeIntent::Throughput);
        assert_eq!((fast.width, fast.height), (serial.width, serial.height));
        assert_eq!(fast.channels, serial.channels);
        match (&fast.pixels, &serial.pixels) {
            (PixelBuffer::F32(a), PixelBuffer::F32(b)) => {
                assert_eq!(a, b, "{name} differs between decode intents")
            }
            _ => panic!("expected F32 buffers for {name}"),
        }
    }
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
fn probe_matches_decoded_dimensions() {
    // The cheap header probe (used to pre-size the window before decoding) must
    // agree with the dimensions the full decode produces, across formats.
    for name in [
        "tiny_rgb.png",
        "tiny.jpg",
        "tiny_gray16.png",
        "tiny_rgba.exr",
        "tiny_gray.exr",
        "tiny.hdr",
    ] {
        let probed = probe_dimensions(&fixture(name))
            .unwrap_or_else(|| panic!("probe returned None for {name}"));
        let decoded = load(name);
        assert_eq!(
            probed,
            (decoded.width, decoded.height),
            "probe disagrees with decode for {name}"
        );
    }
}

#[test]
fn probe_is_none_for_unprobeable() {
    // Camera RAW has no cheap header probe; a missing file errors out. Both must
    // yield None so the caller falls back to post-decode sizing.
    assert_eq!(probe_dimensions(Path::new("nope.nef")), None);
    assert_eq!(probe_dimensions(Path::new("missing.png")), None);
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
