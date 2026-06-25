//! OpenGL renderer: owns the shader program, the full-screen quad, the image
//! texture (single or tiled), the OCIO LUTs, and the per-frame uniforms. The
//! only module that calls `glow` directly (besides `texture`).

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
            })
        }
    }

    pub fn max_texture_size(&self) -> i32 {
        self.max_texture_size
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

    pub fn render(&self, params: &RenderParams) {
        let gl = &self.gl;
        let (w, h) = params.viewport;
        unsafe {
            gl.viewport(0, 0, w.max(1), h.max(1));
            gl.clear_color(0.02, 0.02, 0.02, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }

        let Some(image) = &self.image else {
            return;
        };
        let aspect = if h > 0 { w as f32 / h as f32 } else { 1.0 };

        unsafe {
            gl.use_program(Some(self.program));

            let u = &self.uniforms;
            gl.uniform_1_f32(u.yaw.as_ref(), params.yaw);
            gl.uniform_1_f32(u.pitch.as_ref(), params.pitch);
            gl.uniform_1_f32(u.half_fov_radians.as_ref(), params.half_fov_radians);
            gl.uniform_1_f32(u.tan_half_fov.as_ref(), params.tan_half_fov);
            gl.uniform_1_f32(u.aspect.as_ref(), aspect);
            gl.uniform_1_f32(u.exposure.as_ref(), params.exposure);
            gl.uniform_1_f32(u.gamma.as_ref(), params.gamma);
            gl.uniform_1_i32(u.projection_mode.as_ref(), params.projection_mode);
            gl.uniform_1_f32(u.image_aspect.as_ref(), image.aspect);
            gl.uniform_1_i32(
                u.input_is_encoded_srgb.as_ref(),
                image.is_encoded_srgb as i32,
            );
            gl.uniform_1_i32(u.wrap_2d.as_ref(), params.wrap_2d as i32);

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
