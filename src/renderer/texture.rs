//! Image texture storage: a single GL texture for images within
//! `GL_MAX_TEXTURE_SIZE`, or a tiled `GL_TEXTURE_2D_ARRAY` for very large images
//! (the 24k+ primary workload — see plans/rewrite.md §9.7).
//!
//! Tiles are padded to a uniform layer size and carry a multi-texel border
//! (`BORDER`) replicated from neighbours, so bilinear filtering and the capped
//! mip chain stay seamless across tile boundaries (the shader also derives the
//! LOD from the continuous pixel coordinate via `textureGrad`). A single
//! `sampler2DArray` (one texture unit) holds every tile, sidestepping the
//! bound-sampler-count limit regardless of tile count.

use glow::HasContext as _;

use crate::image_loader::{ImageData, PixelBuffer};

/// `__IMAGE_SAMPLER__` for the single-texture path.
///
/// `sample_image` uses implicit derivatives (correct for the continuous 2D
/// coordinate); `sample_image_grad` takes explicit, seam-corrected derivatives
/// for the equirectangular path (see the pano branch in the fragment template).
pub const SINGLE_TEXTURE_SAMPLER: &str = "\
uniform sampler2D u_image;
vec4 sample_image(vec2 uv) { return texture(u_image, uv); }
vec4 sample_image_grad(vec2 uv, vec2 ddx, vec2 ddy) {
    return textureGrad(u_image, uv, ddx, ddy);
}
// Original-resolution high-pass (LOD 0, 2px blur) for the sharpness checker.
vec3 sharp_highpass(vec2 uv) {
    vec2 ts = 2.0 / vec2(textureSize(u_image, 0));
    vec3 c = textureLod(u_image, uv, 0.0).rgb;
    vec3 b = (c
        + textureLod(u_image, uv + vec2(ts.x, 0.0), 0.0).rgb
        + textureLod(u_image, uv - vec2(ts.x, 0.0), 0.0).rgb
        + textureLod(u_image, uv + vec2(0.0, ts.y), 0.0).rgb
        + textureLod(u_image, uv - vec2(0.0, ts.y), 0.0).rgb) * 0.2;
    return vec3(0.5) + (c - b) * 4.0;
}";

/// `__IMAGE_SAMPLER__` for the tiled path.
///
/// The mip LOD is derived from the *continuous* pixel coordinate (`px`) via
/// `textureGrad`, not from the per-tile `tile_uv` which jumps at boundaries —
/// otherwise the hardware LOD spikes at every seam. Combined with the multi-texel
/// border and the capped mip chain (see `BORDER` / `MIP_LEVELS`) this keeps tile
/// boundaries seamless even when the whole image is minified into the viewport.
pub const TILED_SAMPLER: &str = "\
uniform sampler2DArray u_tiles;
uniform int   u_tile_cols;
uniform int   u_tile_rows;
uniform vec2  u_tiled_image_size;
uniform float u_tile_size;
uniform float u_layer_size;
uniform float u_tile_border;

vec4 sample_tiles(vec2 px, vec2 ddx, vec2 ddy) {
    vec2 cpx = clamp(px, vec2(0.0), u_tiled_image_size - vec2(0.5));
    float fcol = clamp(floor(cpx.x / u_tile_size), 0.0, float(u_tile_cols - 1));
    float frow = clamp(floor(cpx.y / u_tile_size), 0.0, float(u_tile_rows - 1));
    vec2 in_tile = cpx - vec2(fcol, frow) * u_tile_size;
    vec2 tile_uv = (in_tile + vec2(u_tile_border)) / u_layer_size;
    float layer = frow * float(u_tile_cols) + fcol;
    return textureGrad(u_tiles, vec3(tile_uv, layer), ddx, ddy);
}

vec4 sample_image(vec2 uv) {
    // Gradients from the continuous pixel coordinate (no jump across tile seams).
    vec2 px = uv * u_tiled_image_size;
    return sample_tiles(px, dFdx(px) / u_layer_size, dFdy(px) / u_layer_size);
}

// Equirect path: caller supplies seam-corrected UV-space derivatives.
vec4 sample_image_grad(vec2 uv, vec2 duvdx, vec2 duvdy) {
    vec2 px = uv * u_tiled_image_size;
    return sample_tiles(px, (duvdx * u_tiled_image_size) / u_layer_size,
                            (duvdy * u_tiled_image_size) / u_layer_size);
}

// LOD-0 point sample of the tiled image at UV.
vec4 sample_tile_lod0(vec2 uv) {
    vec2 cpx = clamp(uv * u_tiled_image_size, vec2(0.0), u_tiled_image_size - vec2(0.5));
    float fcol = clamp(floor(cpx.x / u_tile_size), 0.0, float(u_tile_cols - 1));
    float frow = clamp(floor(cpx.y / u_tile_size), 0.0, float(u_tile_rows - 1));
    vec2 in_tile = cpx - vec2(fcol, frow) * u_tile_size;
    vec2 tile_uv = (in_tile + vec2(u_tile_border)) / u_layer_size;
    float layer = frow * float(u_tile_cols) + fcol;
    return textureLod(u_tiles, vec3(tile_uv, layer), 0.0);
}

// Original-resolution high-pass (LOD 0, 2px blur) for the sharpness checker.
vec3 sharp_highpass(vec2 uv) {
    vec2 ts = 2.0 / u_tiled_image_size;
    vec3 c = sample_tile_lod0(uv).rgb;
    vec3 b = (c
        + sample_tile_lod0(uv + vec2(ts.x, 0.0)).rgb
        + sample_tile_lod0(uv - vec2(ts.x, 0.0)).rgb
        + sample_tile_lod0(uv + vec2(0.0, ts.y)).rgb
        + sample_tile_lod0(uv - vec2(0.0, ts.y)).rgb) * 0.2;
    return vec3(0.5) + (c - b) * 4.0;
}";

/// Border replicated around each tile. Mip level L stays seamless while
/// `BORDER >= 2^L`, so the mip chain is capped at `log2(BORDER)` levels.
const BORDER: i32 = 8;
const MIP_LEVELS: i32 = 4; // log2(BORDER) + 1
const TILE_CAP: i32 = 8192;

// Anisotropic filtering enums (core 4.6 / EXT_texture_filter_anisotropic).
const TEXTURE_MAX_ANISOTROPY: u32 = 0x84FE;
const MAX_TEXTURE_MAX_ANISOTROPY: u32 = 0x84FF;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SamplerKind {
    Single,
    Tiled,
}

pub struct TiledTexture {
    pub array: glow::Texture,
    pub cols: i32,
    pub rows: i32,
    pub tile_size: i32,
    pub layer_size: i32,
    pub border: i32,
    pub image_w: i32,
    pub image_h: i32,
}

pub enum ImageTextureKind {
    Single(glow::Texture),
    Tiled(TiledTexture),
}

pub struct ImageTexture {
    pub kind: ImageTextureKind,
    pub aspect: f32,
    pub is_encoded_srgb: bool,
}

impl ImageTexture {
    pub fn sampler_kind(&self) -> SamplerKind {
        match self.kind {
            ImageTextureKind::Single(_) => SamplerKind::Single,
            ImageTextureKind::Tiled(_) => SamplerKind::Tiled,
        }
    }

    /// # Safety: the GL context must be current.
    pub unsafe fn delete(self, gl: &glow::Context) {
        match self.kind {
            ImageTextureKind::Single(t) => gl.delete_texture(t),
            ImageTextureKind::Tiled(t) => gl.delete_texture(t.array),
        }
    }
}

/// The texture-size threshold above which an image must be tiled. Reads
/// `GL_MAX_TEXTURE_SIZE`, but can be lowered via `IMGVWR_MAX_TEXTURE_SIZE` to
/// exercise the tiled path on GPUs with a very large native limit.
pub fn effective_threshold(gl_max: i32) -> i32 {
    match std::env::var("IMGVWR_MAX_TEXTURE_SIZE")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
    {
        Some(n) if n >= 64 => n.min(gl_max),
        _ => gl_max,
    }
}

/// The tiling grid for an image, or `None` if it fits in a single texture.
/// Returns `(cols, rows, tile_size, layer_size)`.
pub fn tile_grid(width: i32, height: i32, threshold: i32) -> Option<(i32, i32, i32, i32)> {
    if width <= threshold && height <= threshold {
        return None;
    }
    let tile_size = threshold.min(TILE_CAP);
    let cols = (width + tile_size - 1) / tile_size;
    let rows = (height + tile_size - 1) / tile_size;
    Some((cols, rows, tile_size, tile_size + 2 * BORDER))
}

/// Bytes uploaded per [`pump_upload`] call. Large enough that total upload time
/// is close to a single blocking upload (performance is the priority), small
/// enough that each frame's hitch stays short so the UI keeps responding and
/// progress animates. ~256 MB ≈ a handful of frames even for a multi-GB texture.
const UPLOAD_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// An in-progress, incremental GPU upload. Built by [`begin_upload`] and advanced
/// a budget at a time by [`pump_upload`] across frames, so a multi-GB texture
/// uploads without freezing the UI and with real progress.
pub struct UploadJob {
    kind: JobKind,
    /// Next row (single) or next layer (tiled) to upload.
    cursor: i32,
    /// Total rows (single) or layers (tiled).
    total: i32,
    aspect: f32,
    is_encoded_srgb: bool,
}

enum JobKind {
    Single {
        texture: glow::Texture,
        width: i32,
        height: i32,
    },
    Tiled {
        tiled: TiledTexture,
        /// Reused per-tile staging buffer (bordered tile pixels).
        staging: Vec<u8>,
    },
}

impl UploadJob {
    /// Fraction uploaded so far, 0..=1.
    pub fn progress(&self) -> f32 {
        if self.total <= 0 {
            1.0
        } else {
            (self.cursor as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }

    /// Delete the partially-built texture (when an upload is abandoned).
    /// # Safety: the GL context must be current.
    pub unsafe fn discard(self, gl: &glow::Context) {
        match self.kind {
            JobKind::Single { texture, .. } => gl.delete_texture(texture),
            JobKind::Tiled { tiled, .. } => gl.delete_texture(tiled.array),
        }
    }
}

/// Allocate the texture(s) for `data` and return an [`UploadJob`] to pump. No
/// pixels are uploaded yet. # Safety: the GL context must be current.
pub unsafe fn begin_upload(
    gl: &glow::Context,
    data: &ImageData,
    threshold: i32,
) -> Option<UploadJob> {
    let w = data.width as i32;
    let h = data.height as i32;
    let (internal, _, _, bpp, _) = pixel_format(data);

    let kind = if let Some((cols, rows, tile_size, layer_size)) = tile_grid(w, h, threshold) {
        let layers = cols * rows;
        let levels = MIP_LEVELS.min((layer_size as f32).log2().floor() as i32 + 1);
        log::info!(
            "tiling {w}x{h} into {cols}x{rows} = {layers} tiles \
             (tile {tile_size}, layer {layer_size}, border {BORDER}, {levels} mips)"
        );
        let array = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(array));
        gl.tex_storage_3d(
            glow::TEXTURE_2D_ARRAY,
            levels,
            internal,
            layer_size,
            layer_size,
            layers,
        );
        set_filters(gl, glow::TEXTURE_2D_ARRAY);
        set_anisotropy(gl, glow::TEXTURE_2D_ARRAY);
        gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
        let layer_w = layer_size as usize;
        JobKind::Tiled {
            tiled: TiledTexture {
                array,
                cols,
                rows,
                tile_size,
                layer_size,
                border: BORDER,
                image_w: w,
                image_h: h,
            },
            staging: vec![0u8; layer_w * layer_w * bpp],
        }
    } else {
        let texture = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        // Immutable storage (all mip levels): the fast path for the per-band
        // tex_sub_image uploads that pump performs (a mutable tex_image_2d +
        // sub-image path is several times slower on NVIDIA for huge textures).
        let levels = (w.max(h) as f32).log2().floor() as i32 + 1;
        gl.tex_storage_2d(glow::TEXTURE_2D, levels, internal, w, h);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        set_filters(gl, glow::TEXTURE_2D);
        set_anisotropy(gl, glow::TEXTURE_2D);
        gl.bind_texture(glow::TEXTURE_2D, None);
        JobKind::Single {
            texture,
            width: w,
            height: h,
        }
    };

    let total = match &kind {
        JobKind::Single { height, .. } => *height,
        JobKind::Tiled { tiled, .. } => tiled.cols * tiled.rows,
    };
    Some(UploadJob {
        kind,
        cursor: 0,
        total,
        aspect: data.aspect(),
        is_encoded_srgb: data.is_encoded_srgb,
    })
}

/// Upload up to [`UPLOAD_BUDGET_BYTES`] of `job`. Returns `Some(ImageTexture)`
/// once complete (mipmaps generated), else `None` — call again next frame.
/// # Safety: GL context current; `data` must match what `begin_upload` received.
pub unsafe fn pump_upload(
    gl: &glow::Context,
    job: &mut UploadJob,
    data: &ImageData,
) -> Option<ImageTexture> {
    let (_, format, ty, bpp, src) = pixel_format(data);
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
    let (aspect, is_srgb) = (job.aspect, job.is_encoded_srgb);

    match &mut job.kind {
        JobKind::Single {
            texture,
            width,
            height,
        } => {
            let row_bytes = (*width as usize) * bpp;
            let band =
                ((UPLOAD_BUDGET_BYTES / row_bytes.max(1)).max(1) as i32).min(*height - job.cursor);
            let byte0 = job.cursor as usize * row_bytes;
            let byte1 = (job.cursor + band) as usize * row_bytes;
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                job.cursor,
                *width,
                band,
                format,
                ty,
                glow::PixelUnpackData::Slice(Some(&src[byte0..byte1])),
            );
            job.cursor += band;
            if job.cursor >= *height {
                gl.generate_mipmap(glow::TEXTURE_2D);
                gl.bind_texture(glow::TEXTURE_2D, None);
                return Some(ImageTexture {
                    kind: ImageTextureKind::Single(*texture),
                    aspect,
                    is_encoded_srgb: is_srgb,
                });
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
            None
        }
        JobKind::Tiled { tiled, staging } => {
            let (w, h) = (tiled.image_w, tiled.image_h);
            let (cols, tile_size, layer_size) = (tiled.cols, tiled.tile_size, tiled.layer_size);
            let row_stride = w as usize * bpp;
            let layer_w = layer_size as usize;
            let tile_bytes = layer_w * layer_w * bpp;
            let budget = (UPLOAD_BUDGET_BYTES / tile_bytes.max(1)).max(1) as i32;
            let end = (job.cursor + budget).min(job.total);
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tiled.array));
            for layer in job.cursor..end {
                let row = layer / cols;
                let col = layer % cols;
                for ly in 0..layer_size {
                    let sy = (row * tile_size + ly - BORDER).clamp(0, h - 1) as usize;
                    let src_row = sy * row_stride;
                    let dst_row = ly as usize * layer_w * bpp;
                    for lx in 0..layer_size {
                        let sx = (col * tile_size + lx - BORDER).clamp(0, w - 1) as usize;
                        let s = src_row + sx * bpp;
                        let d = dst_row + lx as usize * bpp;
                        staging[d..d + bpp].copy_from_slice(&src[s..s + bpp]);
                    }
                }
                gl.tex_sub_image_3d(
                    glow::TEXTURE_2D_ARRAY,
                    0,
                    0,
                    0,
                    layer,
                    layer_size,
                    layer_size,
                    1,
                    format,
                    ty,
                    glow::PixelUnpackData::Slice(Some(staging.as_slice())),
                );
            }
            job.cursor = end;
            if job.cursor >= job.total {
                gl.generate_mipmap(glow::TEXTURE_2D_ARRAY);
                gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
                let out = TiledTexture {
                    array: tiled.array,
                    cols: tiled.cols,
                    rows: tiled.rows,
                    tile_size: tiled.tile_size,
                    layer_size: tiled.layer_size,
                    border: tiled.border,
                    image_w: tiled.image_w,
                    image_h: tiled.image_h,
                };
                return Some(ImageTexture {
                    kind: ImageTextureKind::Tiled(out),
                    aspect,
                    is_encoded_srgb: is_srgb,
                });
            }
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
            None
        }
    }
}

/// Set min/mag filters for the default bilinear mode (the I-key toggle re-sets
/// them per frame in the renderer).
unsafe fn set_filters(gl: &glow::Context, target: u32) {
    gl.tex_parameter_i32(
        target,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    if target == glow::TEXTURE_2D_ARRAY {
        gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    }
}

/// Synchronously upload `data` as a single mip-mapped 2D texture for the slot
/// difference checker. Returns `None` if it exceeds `max_size` (would need
/// tiling, which the diff sampler doesn't support). # Safety: GL context current.
pub unsafe fn upload_diff_texture(
    gl: &glow::Context,
    data: &ImageData,
    max_size: i32,
) -> Option<glow::Texture> {
    let (w, h) = (data.width as i32, data.height as i32);
    if w <= 0 || h <= 0 || w.max(h) > max_size {
        return None;
    }
    let (internal, format, ty, _bpp, bytes) = pixel_format(data);
    let levels = (w.max(h) as f32).log2().floor() as i32 + 1;
    let tex = gl.create_texture().ok()?;
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    gl.tex_storage_2d(glow::TEXTURE_2D, levels.max(1), internal, w, h);
    gl.tex_sub_image_2d(
        glow::TEXTURE_2D,
        0,
        0,
        0,
        w,
        h,
        format,
        ty,
        glow::PixelUnpackData::Slice(Some(bytes)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    let lin = glow::LINEAR as i32;
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, lin);
    let clamp = glow::CLAMP_TO_EDGE as i32;
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
    gl.bind_texture(glow::TEXTURE_2D, None);
    Some(tex)
}

fn pixel_format(data: &ImageData) -> (u32, u32, u32, usize, &[u8]) {
    // (internal, format, type, bytes-per-pixel, raw bytes)
    match &data.pixels {
        PixelBuffer::U8(v) => (
            glow::RGBA8,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            4,
            v.as_slice(),
        ),
        PixelBuffer::F32(v) => (
            glow::RGBA32F,
            glow::RGBA,
            glow::FLOAT,
            16,
            bytemuck::cast_slice(v),
        ),
    }
}

unsafe fn set_anisotropy(gl: &glow::Context, target: u32) {
    let max_aniso = gl.get_parameter_f32(MAX_TEXTURE_MAX_ANISOTROPY);
    if max_aniso > 1.0 {
        gl.tex_parameter_f32(target, TEXTURE_MAX_ANISOTROPY, max_aniso);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn images_within_threshold_are_not_tiled() {
        assert_eq!(tile_grid(1024, 1024, 16384), None);
        assert_eq!(tile_grid(16384, 16384, 16384), None);
    }

    #[test]
    fn large_images_tile_into_the_expected_grid() {
        // 24576x12288 with a 16384 threshold -> tile_size capped at 8192.
        assert_eq!(
            tile_grid(24576, 12288, 16384),
            Some((3, 2, 8192, 8192 + 2 * BORDER))
        );
        // Exceeding in one dimension only still tiles.
        assert_eq!(
            tile_grid(40000, 1000, 16384),
            Some((5, 1, 8192, 8192 + 2 * BORDER))
        );
    }

    #[test]
    fn layer_size_includes_both_borders() {
        let (_, _, tile, layer) = tile_grid(20000, 20000, 8192).unwrap();
        assert_eq!(layer, tile + 2 * BORDER);
    }
}
