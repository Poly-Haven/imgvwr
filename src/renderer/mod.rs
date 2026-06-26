//! OpenGL renderer: owns the shader program, the full-screen quad, the image
//! texture (single or tiled), the OCIO LUTs, and the per-frame uniforms. The
//! only module that calls `glow` directly (besides `texture`).

mod post;
mod texture;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use glow::HasContext as _;

use crate::image_loader::ImageData;
use crate::ocio::{OcioLut, OcioShader};
use texture::{ImageTexture, ImageTextureKind, SamplerKind, SINGLE_TEXTURE_SAMPLER, TILED_SAMPLER};

const VERTEX_SRC: &str = include_str!("../../resources/shaders/vertex.glsl");
const FRAGMENT_TEMPLATE: &str = include_str!("../../resources/shaders/fragment_template.glsl");

/// `__OCIO_APPLY__` when OCIO is absent (gamma-2.2 fallback).
const GAMMA_FALLBACK_APPLY: &str = "color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));";

/// `__OCIO_APPLY__` when an OCIO program is present.
const OCIO_APPLY: &str =
    "vec4 _ocio_out = ocio_display_transform(vec4(color, 1.0)); color = _ocio_out.rgb;";

/// Image texture occupies unit 0; OCIO LUTs start at unit 1.
const OCIO_FIRST_UNIT: u32 = 1;

/// Texture unit for the slot-difference image (above any plausible LUT count).
const DIFF_UNIT: u32 = 12;

/// Maximum simultaneous guide lines. Sized to hold a full 1/32 grid on both
/// axes (31 + 31 = 62 lines). Must match the `u_guides[..]` array in the
/// fragment shader.
pub const MAX_GUIDES: usize = 64;

/// Per-frame uniform values supplied by the application.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    pub viewport: (i32, i32),
    pub exposure: f32,
    pub gamma: f32,
    /// 0 = equirectangular panorama, 1 = 2D pan/zoom.
    pub projection_mode: i32,
    pub yaw: f32,
    pub pitch: f32,
    pub half_fov_radians: f32,
    pub tan_half_fov: f32,
    /// 2D mode: repeat the image (GL_REPEAT both axes) instead of clamping.
    pub wrap_2d: bool,
    /// Nearest-neighbour texture filtering instead of the default bilinear.
    pub nearest: bool,
    /// Background colour (sRGB 0–1) cleared behind the alpha-blended image.
    pub background: [f32; 3],
    /// Channel to isolate as greyscale: -1 = all, 0=R 1=G 2=B 3=A.
    pub isolate_channel: i32,
    /// Per-axis image squash/stretch (1,1 = none).
    pub stretch: [f32; 2],
    /// Show the original-resolution sharpness high-pass instead of the image.
    pub sharpness: bool,
    /// Show the absolute difference against the loaded comparator-slot image.
    pub diff: bool,
    /// Guide lines: `[image_coord (0..1), orientation (0=vertical/1=horizontal)]`.
    /// Capped at [`MAX_GUIDES`] (a full 1/32 grid on both axes = 62 lines).
    pub guides: [[f32; 2]; MAX_GUIDES],
    pub guide_count: i32,
    /// Guide-line colour (display-encoded sRGB 0–1), drawn after the view
    /// transform straight into the framebuffer.
    pub guide_color: [f32; 3],
    /// Index of the guide under the pointer (drawn in the hover colour), or -1.
    pub guide_hover: i32,
    /// Hover colour (inverse hue of `guide_color`), display-encoded sRGB 0–1.
    pub guide_hover_color: [f32; 3],
    /// Clarity (local-contrast) strength; 0 bypasses the whole post chain.
    pub clarity_amount: f32,
    /// Clarity unsharp-mask blur radius, in viewport pixels.
    pub clarity_radius: f32,
    /// Whole-pass opacity (1 = opaque). The minimap thumbnail pass uses this to
    /// fade itself over the scene; the main scene always renders at 1.0.
    pub global_alpha: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        let half_fov = std::f32::consts::FRAC_PI_4; // 45° => 90° FOV
        Self {
            viewport: (1, 1),
            exposure: 0.0,
            gamma: 1.0,
            projection_mode: 1,
            yaw: 0.0,
            pitch: 0.0,
            half_fov_radians: half_fov,
            tan_half_fov: half_fov.tan(),
            wrap_2d: false,
            nearest: false,
            background: [0.02, 0.02, 0.02],
            isolate_channel: -1,
            stretch: [1.0, 1.0],
            sharpness: false,
            diff: false,
            guides: [[0.0; 2]; MAX_GUIDES],
            guide_count: 0,
            guide_color: [1.0, 0.314, 0.314],
            guide_hover: -1,
            guide_hover_color: [0.314, 1.0, 1.0],
            clarity_amount: 0.0,
            clarity_radius: 64.0,
            global_alpha: 1.0,
        }
    }
}

struct Uniforms {
    yaw: Option<glow::UniformLocation>,
    pitch: Option<glow::UniformLocation>,
    half_fov_radians: Option<glow::UniformLocation>,
    tan_half_fov: Option<glow::UniformLocation>,
    aspect: Option<glow::UniformLocation>,
    exposure: Option<glow::UniformLocation>,
    gamma: Option<glow::UniformLocation>,
    projection_mode: Option<glow::UniformLocation>,
    image_aspect: Option<glow::UniformLocation>,
    input_is_encoded_srgb: Option<glow::UniformLocation>,
    wrap_2d: Option<glow::UniformLocation>,
    isolate_channel: Option<glow::UniformLocation>,
    stretch: Option<glow::UniformLocation>,
    sharpness: Option<glow::UniformLocation>,
    diff: Option<glow::UniformLocation>,
    diff_image: Option<glow::UniformLocation>,
    guide_count: Option<glow::UniformLocation>,
    guides: Option<glow::UniformLocation>,
    guide_color: Option<glow::UniformLocation>,
    guide_hover: Option<glow::UniformLocation>,
    guide_hover_color: Option<glow::UniformLocation>,
    global_alpha: Option<glow::UniformLocation>,
    // Single-texture sampler.
    image: Option<glow::UniformLocation>,
    // Tiled-texture sampler + grid.
    tiles: Option<glow::UniformLocation>,
    tile_cols: Option<glow::UniformLocation>,
    tile_rows: Option<glow::UniformLocation>,
    tiled_image_size: Option<glow::UniformLocation>,
    tile_size: Option<glow::UniformLocation>,
    layer_size: Option<glow::UniformLocation>,
    tile_border: Option<glow::UniformLocation>,
}

impl Uniforms {
    fn fetch(gl: &glow::Context, program: glow::Program) -> Self {
        let u = |name: &str| unsafe { gl.get_uniform_location(program, name) };
        Self {
            yaw: u("u_yaw"),
            pitch: u("u_pitch"),
            half_fov_radians: u("u_half_fov_radians"),
            tan_half_fov: u("u_tan_half_fov"),
            aspect: u("u_aspect"),
            exposure: u("u_exposure"),
            gamma: u("u_gamma"),
            projection_mode: u("u_projection_mode"),
            image_aspect: u("u_image_aspect"),
            input_is_encoded_srgb: u("u_input_is_encoded_srgb"),
            wrap_2d: u("u_wrap_2d"),
            isolate_channel: u("u_isolate_channel"),
            stretch: u("u_stretch"),
            sharpness: u("u_sharpness"),
            diff: u("u_diff"),
            diff_image: u("u_diff_image"),
            guide_count: u("u_guide_count"),
            guides: u("u_guides"),
            guide_color: u("u_guide_color"),
            guide_hover: u("u_guide_hover"),
            guide_hover_color: u("u_guide_hover_color"),
            global_alpha: u("u_global_alpha"),
            image: u("u_image"),
            tiles: u("u_tiles"),
            tile_cols: u("u_tile_cols"),
            tile_rows: u("u_tile_rows"),
            tiled_image_size: u("u_tiled_image_size"),
            tile_size: u("u_tile_size"),
            layer_size: u("u_layer_size"),
            tile_border: u("u_tile_border"),
        }
    }
}

/// An uploaded OCIO LUT texture bound to a fixed texture unit.
struct OcioGlTexture {
    texture: glow::Texture,
    target: u32,
    unit: u32,
    sampler_name: String,
}

struct OcioGlState {
    signature: String,
    textures: Vec<OcioGlTexture>,
}

impl OcioGlState {
    fn empty() -> Self {
        Self {
            signature: String::new(),
            textures: Vec::new(),
        }
    }
}

pub struct Renderer {
    gl: Arc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    uniforms: Uniforms,
    image: Option<ImageTexture>,
    /// An in-progress incremental upload (the previous image stays displayed
    /// until it completes).
    upload: Option<texture::UploadJob>,
    ocio: OcioGlState,
    /// Current `__IMAGE_SAMPLER__` kind (drives shader rebuilds, §9.4).
    sampler_kind: SamplerKind,
    /// Stored OCIO injection so the program can be rebuilt on a sampler change.
    ocio_decls: String,
    ocio_apply: String,
    threshold: i32,
    max_texture_size: i32,
    /// Uploaded comparator-slot image for the difference checker (unit DIFF_UNIT).
    diff_texture: Option<glow::Texture>,
    /// Reusable offscreen post-process chain (Clarity, and future review tools).
    post: post::PostChain,
}

impl Renderer {
    pub fn new(gl: Arc<glow::Context>) -> Result<Self> {
        unsafe {
            let program = build_program(&gl, SINGLE_TEXTURE_SAMPLER, "", GAMMA_FALLBACK_APPLY)?;
            let uniforms = Uniforms::fetch(&gl, program);
            let (vao, vbo) = build_quad(&gl, program)?;

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            let max_texture_size = gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE);
            let threshold = texture::effective_threshold(max_texture_size);
            log::debug!("GL_MAX_TEXTURE_SIZE = {max_texture_size}, tiling threshold = {threshold}");

            let post = post::PostChain::new(&gl)
                .map_err(|e| anyhow!("failed to build post-process chain: {e}"))?;

            Ok(Self {
                gl,
                program,
                vao,
                vbo,
                uniforms,
                image: None,
                upload: None,
                ocio: OcioGlState::empty(),
                sampler_kind: SamplerKind::Single,
                ocio_decls: String::new(),
                ocio_apply: GAMMA_FALLBACK_APPLY.to_string(),
                threshold,
                max_texture_size,
                diff_texture: None,
                post,
            })
        }
    }

    pub fn max_texture_size(&self) -> i32 {
        self.max_texture_size
    }

    /// Upload (or clear with `None`) the comparator-slot image used by the slot
    /// difference checker. Returns false if the image is too large to upload as a
    /// single texture (the diff sampler isn't tiled).
    pub fn set_diff_image(&mut self, data: Option<&ImageData>) -> bool {
        unsafe {
            if let Some(t) = self.diff_texture.take() {
                self.gl.delete_texture(t);
            }
            match data {
                Some(d) => {
                    self.diff_texture =
                        texture::upload_diff_texture(&self.gl, d, self.max_texture_size);
                    self.diff_texture.is_some()
                }
                None => true,
            }
        }
    }

    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    pub fn image_aspect(&self) -> Option<f32> {
        self.image.as_ref().map(|i| i.aspect)
    }

    pub fn ocio_signature(&self) -> &str {
        &self.ocio.signature
    }

    /// Begin an incremental upload of `data`. The previously-displayed image
    /// stays until [`pump_upload`](Self::pump_upload) reports completion. Any
    /// in-progress upload is abandoned.
    pub fn start_upload(&mut self, data: &ImageData) {
        let gl = self.gl.clone();
        if let Some(job) = self.upload.take() {
            unsafe { job.discard(&gl) };
        }
        self.upload = unsafe { texture::begin_upload(&gl, data, self.threshold) };
        if self.upload.is_none() {
            log::error!("failed to begin image upload for {}", data.path.display());
        }
    }

    /// True while an incremental upload is in progress.
    pub fn is_uploading(&self) -> bool {
        self.upload.is_some()
    }

    /// Advance the in-progress upload by one budget; returns the fraction done
    /// (1.0 when complete or idle). On completion the new image is installed and
    /// the program rebuilt if the sampler kind changed.
    pub fn pump_upload(&mut self, data: &ImageData) -> f32 {
        let gl = self.gl.clone();
        let finished = match &mut self.upload {
            Some(job) => unsafe { texture::pump_upload(&gl, job, data) },
            None => return 1.0,
        };
        match finished {
            Some(tex) => {
                if let Some(prev) = self.image.take() {
                    unsafe { prev.delete(&gl) };
                }
                let kind = tex.sampler_kind();
                if kind != self.sampler_kind {
                    self.sampler_kind = kind;
                    unsafe { self.rebuild_program() };
                    log::info!("image sampler kind -> {kind:?}");
                }
                self.image = Some(tex);
                self.upload = None;
                log::info!(
                    "uploaded image {}x{} ({} ch, {})",
                    data.width,
                    data.height,
                    data.channels,
                    data.dtype_name
                );
                1.0
            }
            None => self.upload.as_ref().map(|j| j.progress()).unwrap_or(1.0),
        }
    }

    /// Rebuild the fragment program for `shader` and upload its LUTs (§9.3/§9.4).
    pub fn set_ocio_shader(&mut self, shader: &OcioShader) {
        let gl = self.gl.clone();
        for t in self.ocio.textures.drain(..) {
            unsafe { gl.delete_texture(t.texture) };
        }

        if shader.is_fallback() {
            self.ocio_decls = String::new();
            self.ocio_apply = GAMMA_FALLBACK_APPLY.to_string();
        } else {
            self.ocio_decls = shader.glsl_text.clone();
            self.ocio_apply = OCIO_APPLY.to_string();
        }
        unsafe { self.rebuild_program() };

        let mut textures = Vec::new();
        unsafe {
            gl.use_program(Some(self.program));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            let mut unit = OCIO_FIRST_UNIT;
            for lut in &shader.luts {
                if let Some((texture, target)) = upload_lut(&gl, lut) {
                    set_sampler_uniform(&gl, self.program, &lut.sampler, unit);
                    textures.push(OcioGlTexture {
                        texture,
                        target,
                        unit,
                        sampler_name: lut.sampler.clone(),
                    });
                    unit += 1;
                }
            }
        }

        self.ocio = OcioGlState {
            signature: shader.signature.clone(),
            textures,
        };
        log::info!(
            "OCIO program active: {} ({} LUTs)",
            self.ocio.signature,
            shader.luts.len()
        );
    }

    /// Rebuild the program from the current sampler kind + OCIO injection, and
    /// re-point existing LUT sampler uniforms at their units.
    unsafe fn rebuild_program(&mut self) {
        let image_sampler = match self.sampler_kind {
            SamplerKind::Single => SINGLE_TEXTURE_SAMPLER,
            SamplerKind::Tiled => TILED_SAMPLER,
        };
        let program =
            match build_program(&self.gl, image_sampler, &self.ocio_decls, &self.ocio_apply) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("shader rebuild failed: {e}\n-- keeping previous program --");
                    return;
                }
            };
        self.gl.delete_program(self.program);
        self.program = program;
        self.uniforms = Uniforms::fetch(&self.gl, program);
        self.gl.use_program(Some(program));
        for t in &self.ocio.textures {
            set_sampler_uniform(&self.gl, program, &t.sampler_name, t.unit);
        }
    }

    /// Render a frame. With clarity off (strength 0) this draws the scene
    /// straight to the default framebuffer exactly as before — zero overhead.
    /// Otherwise the scene is rendered into an offscreen target and the post
    /// chain composites a Clarity (local-contrast) pass to the default fb.
    pub fn render(&mut self, params: &RenderParams) {
        let gl = self.gl.clone();
        let (w, h) = params.viewport;
        if params.clarity_amount <= 0.0 || w <= 0 || h <= 0 {
            unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
            self.draw_scene(params);
            return;
        }
        // Scene → offscreen target, then post-process → default framebuffer.
        let scene_fbo = unsafe { self.post.scene_framebuffer(&gl, w, h) };
        unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scene_fbo)) };
        self.draw_scene(params);
        unsafe {
            self.post
                .apply_clarity(&gl, params.clarity_radius, params.clarity_amount, w, h);
        }
    }

    /// Draw the scene (textured quad through the OCIO/exposure pipeline) into the
    /// currently-bound framebuffer, clearing the whole viewport first.
    fn draw_scene(&self, params: &RenderParams) {
        let gl = &self.gl;
        let (w, h) = params.viewport;
        let bg = params.background;
        unsafe {
            gl.enable(glow::BLEND);
            gl.viewport(0, 0, w.max(1), h.max(1));
            gl.clear_color(bg[0], bg[1], bg[2], 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.draw_quad(params, 0, 0, w.max(1), h.max(1));
    }

    /// Draw the textured quad into an explicit sub-viewport `(x, y, w, h)` of the
    /// bound framebuffer (GL origin bottom-left). The scene aspect is taken from
    /// `(w, h)`, so a sub-viewport matching the image aspect fits the whole image
    /// exactly. Does NOT clear — the caller owns clearing / scissoring. Shared by
    /// the full-screen scene draw and the minimap thumbnail pass.
    fn draw_quad(&self, params: &RenderParams, vx: i32, vy: i32, vw: i32, vh: i32) {
        let gl = &self.gl;
        let Some(image) = &self.image else {
            return;
        };
        let aspect = if vh > 0 { vw as f32 / vh as f32 } else { 1.0 };

        unsafe {
            gl.use_program(Some(self.program));
            gl.viewport(vx, vy, vw.max(1), vh.max(1));

            let u = &self.uniforms;
            gl.uniform_1_f32(u.yaw.as_ref(), params.yaw);
            gl.uniform_1_f32(u.pitch.as_ref(), params.pitch);
            gl.uniform_1_f32(u.half_fov_radians.as_ref(), params.half_fov_radians);
            gl.uniform_1_f32(u.tan_half_fov.as_ref(), params.tan_half_fov);
            gl.uniform_1_f32(u.aspect.as_ref(), aspect);
            gl.uniform_1_f32(u.exposure.as_ref(), params.exposure);
            gl.uniform_1_f32(u.gamma.as_ref(), params.gamma);
            gl.uniform_1_f32(u.global_alpha.as_ref(), params.global_alpha);
            gl.uniform_1_i32(u.projection_mode.as_ref(), params.projection_mode);
            gl.uniform_1_f32(u.image_aspect.as_ref(), image.aspect);
            gl.uniform_1_i32(
                u.input_is_encoded_srgb.as_ref(),
                image.is_encoded_srgb as i32,
            );
            gl.uniform_1_i32(u.wrap_2d.as_ref(), params.wrap_2d as i32);
            gl.uniform_1_i32(u.isolate_channel.as_ref(), params.isolate_channel);
            gl.uniform_2_f32(u.stretch.as_ref(), params.stretch[0], params.stretch[1]);
            gl.uniform_1_i32(u.sharpness.as_ref(), params.sharpness as i32);
            // Slot-difference image on its own unit (only meaningful when u_diff).
            let diffing = params.diff && self.diff_texture.is_some();
            gl.uniform_1_i32(u.diff.as_ref(), diffing as i32);
            gl.uniform_1_i32(u.diff_image.as_ref(), DIFF_UNIT as i32);
            if diffing {
                gl.active_texture(glow::TEXTURE0 + DIFF_UNIT);
                gl.bind_texture(glow::TEXTURE_2D, self.diff_texture);
            }
            let n = params.guide_count.clamp(0, MAX_GUIDES as i32) as usize;
            gl.uniform_1_i32(u.guide_count.as_ref(), n as i32);
            let gc = params.guide_color;
            gl.uniform_3_f32(u.guide_color.as_ref(), gc[0], gc[1], gc[2]);
            let gh = params.guide_hover_color;
            gl.uniform_3_f32(u.guide_hover_color.as_ref(), gh[0], gh[1], gh[2]);
            gl.uniform_1_i32(u.guide_hover.as_ref(), params.guide_hover);
            if n > 0 {
                gl.uniform_2_f32_slice(
                    u.guides.as_ref(),
                    bytemuck::cast_slice(&params.guides[..n]),
                );
            }

            // Min/mag filters for the active interpolation mode (I key).
            let (min_f, mag_f) = if params.nearest {
                (glow::NEAREST_MIPMAP_NEAREST, glow::NEAREST)
            } else {
                (glow::LINEAR_MIPMAP_LINEAR, glow::LINEAR)
            };

            gl.active_texture(glow::TEXTURE0);
            match &image.kind {
                ImageTextureKind::Single(tex) => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(*tex));
                    let wrap_t = if params.wrap_2d {
                        glow::REPEAT
                    } else {
                        glow::CLAMP_TO_EDGE
                    };
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap_t as i32);
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, min_f as i32);
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, mag_f as i32);
                    gl.uniform_1_i32(u.image.as_ref(), 0);
                }
                ImageTextureKind::Tiled(t) => {
                    gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(t.array));
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D_ARRAY,
                        glow::TEXTURE_MIN_FILTER,
                        min_f as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D_ARRAY,
                        glow::TEXTURE_MAG_FILTER,
                        mag_f as i32,
                    );
                    gl.uniform_1_i32(u.tiles.as_ref(), 0);
                    gl.uniform_1_i32(u.tile_cols.as_ref(), t.cols);
                    gl.uniform_1_i32(u.tile_rows.as_ref(), t.rows);
                    gl.uniform_2_f32(
                        u.tiled_image_size.as_ref(),
                        t.image_w as f32,
                        t.image_h as f32,
                    );
                    gl.uniform_1_f32(u.tile_size.as_ref(), t.tile_size as f32);
                    gl.uniform_1_f32(u.layer_size.as_ref(), t.layer_size as f32);
                    gl.uniform_1_f32(u.tile_border.as_ref(), t.border as f32);
                }
            }

            for lut in &self.ocio.textures {
                gl.active_texture(glow::TEXTURE0 + lut.unit);
                gl.bind_texture(lut.target, Some(lut.texture));
            }

            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Draw the minimap thumbnail into a sub-rectangle of the default framebuffer
    /// (GL origin bottom-left), scissored so it touches only that corner. Composes
    /// over the already-drawn scene with `params.global_alpha` (the fade), so it is
    /// NOT cleared first — the thumbnail's own opaque pixels cover the rect, and a
    /// translucent pass reveals the scene behind as it fades out.
    ///
    /// `params` is expected to be a fit-the-whole-image 2D view (projection_mode 1,
    /// pan 0, `tan_half_fov` 1) whose `(w, h)` aspect matches the image; the caller
    /// builds it. Clarity is intentionally skipped (this draws the quad directly).
    pub fn render_minimap(&self, params: &RenderParams, x: i32, y: i32, w: i32, h: i32) {
        if w <= 0 || h <= 0 || self.image.is_none() {
            return;
        }
        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
            // The shader outputs straight (non-premultiplied) alpha, so the fade
            // needs SRC_ALPHA / ONE_MINUS_SRC_ALPHA. egui's painter leaves a
            // *premultiplied* blend func set from the previous frame's paint and
            // never restores it; without resetting here the fade would composite
            // the thumbnail additively (blown out) instead of cross-fading.
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(x, y, w, h);
        }
        self.draw_quad(params, x, y, w, h);
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let gl = &self.gl;
        unsafe {
            if let Some(image) = self.image.take() {
                image.delete(gl);
            }
            if let Some(job) = self.upload.take() {
                job.discard(gl);
            }
            for t in self.ocio.textures.drain(..) {
                gl.delete_texture(t.texture);
            }
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
            gl.delete_program(self.program);
        }
    }
}

unsafe fn build_quad(
    gl: &glow::Context,
    program: glow::Program,
) -> Result<(glow::VertexArray, glow::Buffer)> {
    #[rustfmt::skip]
    let verts: [f32; 8] = [
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ];

    let vao = gl
        .create_vertex_array()
        .map_err(|e| anyhow!("create VAO: {e}"))?;
    let vbo = gl.create_buffer().map_err(|e| anyhow!("create VBO: {e}"))?;
    gl.bind_vertex_array(Some(vao));
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(
        glow::ARRAY_BUFFER,
        bytemuck::cast_slice(&verts),
        glow::STATIC_DRAW,
    );

    let loc = gl.get_attrib_location(program, "in_position").unwrap_or(0);
    gl.enable_vertex_attrib_array(loc);
    gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 0, 0);

    gl.bind_vertex_array(None);
    gl.bind_buffer(glow::ARRAY_BUFFER, None);
    Ok((vao, vbo))
}

/// Assemble the fragment template, compile, and link the program.
unsafe fn build_program(
    gl: &glow::Context,
    image_sampler: &str,
    ocio_declarations: &str,
    ocio_apply: &str,
) -> Result<glow::Program> {
    let fragment_src = FRAGMENT_TEMPLATE
        .replace("__IMAGE_SAMPLER__", image_sampler)
        .replace("__OCIO_DECLARATIONS__", ocio_declarations)
        .replace("__OCIO_APPLY__", ocio_apply);

    let program = gl
        .create_program()
        .map_err(|e| anyhow!("create_program: {e}"))?;
    let vs = compile_shader(gl, glow::VERTEX_SHADER, VERTEX_SRC)?;
    let fs = compile_shader(gl, glow::FRAGMENT_SHADER, &fragment_src)?;
    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);
    let linked = gl.get_program_link_status(program);
    gl.detach_shader(program, vs);
    gl.detach_shader(program, fs);
    gl.delete_shader(vs);
    gl.delete_shader(fs);
    if !linked {
        let log = gl.get_program_info_log(program);
        gl.delete_program(program);
        return Err(anyhow!("shader program link failed:\n{log}"));
    }
    Ok(program)
}

/// Link [`VERTEX_SRC`] with a post-process fragment shader. Used by the
/// reusable post-process chain ([`post::PostChain`]).
unsafe fn build_post_program(gl: &glow::Context, frag: &str) -> Result<glow::Program> {
    let program = gl
        .create_program()
        .map_err(|e| anyhow!("create_program: {e}"))?;
    let vs = compile_shader(gl, glow::VERTEX_SHADER, VERTEX_SRC)?;
    let fs = compile_shader(gl, glow::FRAGMENT_SHADER, frag)?;
    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);
    let ok = gl.get_program_link_status(program);
    gl.detach_shader(program, vs);
    gl.detach_shader(program, fs);
    gl.delete_shader(vs);
    gl.delete_shader(fs);
    if !ok {
        let log = gl.get_program_info_log(program);
        gl.delete_program(program);
        return Err(anyhow!("post program link failed:\n{log}"));
    }
    Ok(program)
}

unsafe fn compile_shader(gl: &glow::Context, kind: u32, src: &str) -> Result<glow::Shader> {
    let shader = gl
        .create_shader(kind)
        .map_err(|e| anyhow!("create_shader: {e}"))?;
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        let log = gl.get_shader_info_log(shader);
        gl.delete_shader(shader);
        let stage = if kind == glow::VERTEX_SHADER {
            "vertex"
        } else {
            "fragment"
        };
        return Err(anyhow!("{stage} shader compile failed:\n{log}"));
    }
    Ok(shader)
}

/// Upload an OCIO LUT as a 1D/2D/3D float texture.
unsafe fn upload_lut(gl: &glow::Context, lut: &OcioLut) -> Option<(glow::Texture, u32)> {
    let texture = gl.create_texture().ok()?;
    let (internal, format) = if lut.channels == 1 {
        (glow::R32F as i32, glow::RED)
    } else {
        (glow::RGB32F as i32, glow::RGB)
    };
    let filter = if lut.linear {
        glow::LINEAR
    } else {
        glow::NEAREST
    } as i32;
    let bytes: &[u8] = bytemuck::cast_slice(&lut.data);

    let target = match lut.dim {
        1 => glow::TEXTURE_1D,
        3 => glow::TEXTURE_3D,
        _ => glow::TEXTURE_2D,
    };
    gl.bind_texture(target, Some(texture));

    match lut.dim {
        1 => gl.tex_image_1d(
            target,
            0,
            internal,
            lut.width,
            0,
            format,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(Some(bytes)),
        ),
        3 => gl.tex_image_3d(
            target,
            0,
            internal,
            lut.width,
            lut.height,
            lut.depth,
            0,
            format,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(Some(bytes)),
        ),
        _ => gl.tex_image_2d(
            target,
            0,
            internal,
            lut.width,
            lut.height,
            0,
            format,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(Some(bytes)),
        ),
    }

    gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, filter);
    gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, filter);
    gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    if lut.dim >= 2 {
        gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    }
    if lut.dim == 3 {
        gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_R, glow::CLAMP_TO_EDGE as i32);
    }
    gl.bind_texture(target, None);
    Some((texture, target))
}

unsafe fn set_sampler_uniform(gl: &glow::Context, program: glow::Program, name: &str, unit: u32) {
    if let Some(loc) = gl.get_uniform_location(program, name) {
        gl.uniform_1_i32(Some(&loc), unit as i32);
    } else {
        log::debug!("OCIO sampler uniform '{name}' not found in program");
    }
}
