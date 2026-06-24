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
pub const SINGLE_TEXTURE_SAMPLER: &str = "\
uniform sampler2D u_image;
vec3 sample_image(vec2 uv) { return texture(u_image, uv).rgb; }";

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

vec3 sample_image(vec2 uv) {
    vec2 px = uv * u_tiled_image_size;
    // Gradients from the continuous coordinate (no jump across tile seams).
    vec2 ddx = dFdx(px) / u_layer_size;
    vec2 ddy = dFdy(px) / u_layer_size;
    vec2 cpx = clamp(px, vec2(0.0), u_tiled_image_size - vec2(0.5));
    float fcol = clamp(floor(cpx.x / u_tile_size), 0.0, float(u_tile_cols - 1));
    float frow = clamp(floor(cpx.y / u_tile_size), 0.0, float(u_tile_rows - 1));
    vec2 in_tile = cpx - vec2(fcol, frow) * u_tile_size;
    vec2 tile_uv = (in_tile + vec2(u_tile_border)) / u_layer_size;
    float layer = frow * float(u_tile_cols) + fcol;
    return textureGrad(u_tiles, vec3(tile_uv, layer), ddx, ddy).rgb;
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

/// Build a single or tiled texture for `data` based on `threshold`.
///
/// # Safety: the GL context must be current.
pub unsafe fn build_image_texture(
    gl: &glow::Context,
    data: &ImageData,
    threshold: i32,
) -> Option<ImageTexture> {
    let w = data.width as i32;
    let h = data.height as i32;
    let kind = if tile_grid(w, h, threshold).is_none() {
        ImageTextureKind::Single(upload_single(gl, data)?)
    } else {
        ImageTextureKind::Tiled(upload_tiled(gl, data, threshold)?)
    };
    Some(ImageTexture {
        kind,
        aspect: data.aspect(),
        is_encoded_srgb: data.is_encoded_srgb,
    })
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

unsafe fn upload_single(gl: &glow::Context, data: &ImageData) -> Option<glow::Texture> {
    let (internal, format, ty, _, bytes) = pixel_format(data);
    let texture = gl.create_texture().ok()?;
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        internal as i32,
        data.width as i32,
        data.height as i32,
        0,
        format,
        ty,
        glow::PixelUnpackData::Slice(Some(bytes)),
    );
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_T,
        glow::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MAG_FILTER,
        glow::LINEAR as i32,
    );
    set_anisotropy(gl, glow::TEXTURE_2D);
    gl.generate_mipmap(glow::TEXTURE_2D);
    gl.bind_texture(glow::TEXTURE_2D, None);
    Some(texture)
}

unsafe fn upload_tiled(
    gl: &glow::Context,
    data: &ImageData,
    threshold: i32,
) -> Option<TiledTexture> {
    let w = data.width as i32;
    let h = data.height as i32;
    let (internal, format, ty, bpp, src) = pixel_format(data);

    let (cols, rows, tile_size, layer_size) =
        tile_grid(w, h, threshold).expect("upload_tiled called for a non-tiled image");
    let layers = cols * rows;
    // Cap the mip chain at log2(BORDER) levels so every generated mip stays
    // within the border's seamless range.
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
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

    let row_stride = w as usize * bpp;
    let layer_w = layer_size as usize;
    let mut staging = vec![0u8; layer_w * layer_w * bpp];

    for row in 0..rows {
        for col in 0..cols {
            // Fill the staging buffer for this tile, replicating image-edge
            // pixels into the border.
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
            let layer = row * cols + col;
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
                glow::PixelUnpackData::Slice(Some(&staging)),
            );
        }
    }

    gl.tex_parameter_i32(
        glow::TEXTURE_2D_ARRAY,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D_ARRAY,
        glow::TEXTURE_MAG_FILTER,
        glow::LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D_ARRAY,
        glow::TEXTURE_WRAP_S,
        glow::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D_ARRAY,
        glow::TEXTURE_WRAP_T,
        glow::CLAMP_TO_EDGE as i32,
    );
    set_anisotropy(gl, glow::TEXTURE_2D_ARRAY);
    gl.generate_mipmap(glow::TEXTURE_2D_ARRAY);
    gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);

    Some(TiledTexture {
        array,
        cols,
        rows,
        tile_size,
        layer_size,
        border: BORDER,
        image_w: w,
        image_h: h,
    })
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
