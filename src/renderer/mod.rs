//! OpenGL renderer: owns the shader program, the full-screen quad, the image
//! texture, and the per-frame uniforms. This is the only module that calls
//! `glow` directly.
//!
//! Commit 3 implements the single-texture path with the gamma-2.2 fallback view
//! transform. OCIO LUTs (Commit 7) and large-image tiling (Commit 11) extend it.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use glow::HasContext as _;

use crate::image_loader::{ImageData, PixelBuffer};

const VERTEX_SRC: &str = include_str!("../../resources/shaders/vertex.glsl");
const FRAGMENT_TEMPLATE: &str = include_str!("../../resources/shaders/fragment_template.glsl");

/// `__IMAGE_SAMPLER__` substitution for the single-texture path.
const SINGLE_TEXTURE_SAMPLER: &str = "\
uniform sampler2D u_image;
vec3 sample_image(vec2 uv) { return texture(u_image, uv).rgb; }";

/// `__OCIO_APPLY__` substitution when OCIO is absent (gamma-2.2 fallback).
const GAMMA_FALLBACK_APPLY: &str = "color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));";

// Anisotropic filtering enums (core in 4.6 / EXT_texture_filter_anisotropic).
const TEXTURE_MAX_ANISOTROPY: u32 = 0x84FE;
const MAX_TEXTURE_MAX_ANISOTROPY: u32 = 0x84FF;

/// Per-frame uniform values supplied by the application.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    pub viewport: (i32, i32),
    pub exposure: f32,
    pub gamma: f32,
    /// 0 = equirectangular panorama, 1 = 2-D pan/zoom.
    pub projection_mode: i32,
    pub yaw: f32,
    pub pitch: f32,
    pub half_fov_radians: f32,
    pub tan_half_fov: f32,
    /// 2-D mode: repeat the image (GL_REPEAT both axes) instead of clamping.
    pub wrap_2d: bool,
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
    image: Option<glow::UniformLocation>,
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
        }
    }
}

/// GPU state for the currently-loaded image.
struct ImageTexture {
    texture: glow::Texture,
    aspect: f32,
    is_encoded_srgb: bool,
}

pub struct Renderer {
    gl: Arc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    uniforms: Uniforms,
    image: Option<ImageTexture>,
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
            log::debug!("GL_MAX_TEXTURE_SIZE = {max_texture_size}");

            Ok(Self {
                gl,
                program,
                vao,
                vbo,
                uniforms,
                image: None,
                max_texture_size,
            })
        }
    }

    pub fn max_texture_size(&self) -> i32 {
        self.max_texture_size
    }

    /// Upload `data` as the current image (single-texture path).
    pub fn set_image(&mut self, data: &ImageData) {
        let gl = &self.gl;
        // Release any previous image texture.
        if let Some(prev) = self.image.take() {
            unsafe { gl.delete_texture(prev.texture) };
        }

        unsafe {
            let texture = match gl.create_texture() {
                Ok(t) => t,
                Err(e) => {
                    log::error!("create_texture failed: {e}");
                    return;
                }
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            // Rows are not guaranteed 4-byte aligned; default of 4 corrupts
            // odd-width RGBA8 / non-aligned data.
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            let (internal, format, ty, bytes): (i32, u32, u32, &[u8]) = match &data.pixels {
                PixelBuffer::U8(v) => {
                    (glow::RGBA8 as i32, glow::RGBA, glow::UNSIGNED_BYTE, v.as_slice())
                }
                PixelBuffer::F32(v) => {
                    (glow::RGBA32F as i32, glow::RGBA, glow::FLOAT, bytemuck::cast_slice(v))
                }
            };

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal,
                data.width as i32,
                data.height as i32,
                0,
                format,
                ty,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );

            // Equirectangular needs seamless horizontal wrap; vertical clamps.
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

            let max_aniso = gl.get_parameter_f32(MAX_TEXTURE_MAX_ANISOTROPY);
            if max_aniso > 1.0 {
                gl.tex_parameter_f32(glow::TEXTURE_2D, TEXTURE_MAX_ANISOTROPY, max_aniso);
            }

            gl.generate_mipmap(glow::TEXTURE_2D);
            gl.bind_texture(glow::TEXTURE_2D, None);

            self.image = Some(ImageTexture {
                texture,
                aspect: data.aspect(),
                is_encoded_srgb: data.is_encoded_srgb,
            });
        }

        log::info!(
            "uploaded image texture {}x{} ({} ch, {})",
            data.width,
            data.height,
            data.channels,
            data.dtype_name
        );
    }

    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Aspect ratio (w/h) of the current image, if any.
    pub fn image_aspect(&self) -> Option<f32> {
        self.image.as_ref().map(|i| i.aspect)
    }

    /// Clear and draw the current frame to the (already-current) framebuffer.
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

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(image.texture));
            // Vertical wrap follows the 2-D wrap toggle; horizontal stays REPEAT
            // (panorama needs seamless horizontal wrap regardless).
            let wrap_t = if params.wrap_2d {
                glow::REPEAT
            } else {
                glow::CLAMP_TO_EDGE
            };
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap_t as i32);
            gl.uniform_1_i32(u.image.as_ref(), 0);

            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let gl = &self.gl;
        unsafe {
            if let Some(image) = self.image.take() {
                gl.delete_texture(image.texture);
            }
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
            gl.delete_program(self.program);
        }
    }
}

/// Build the full-screen quad VAO/VBO. Triangle strip of clip-space corners.
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

    let vao = gl.create_vertex_array().map_err(|e| anyhow!("create VAO: {e}"))?;
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
pub(crate) unsafe fn build_program(
    gl: &glow::Context,
    image_sampler: &str,
    ocio_declarations: &str,
    ocio_apply: &str,
) -> Result<glow::Program> {
    let fragment_src = FRAGMENT_TEMPLATE
        .replace("__IMAGE_SAMPLER__", image_sampler)
        .replace("__OCIO_DECLARATIONS__", ocio_declarations)
        .replace("__OCIO_APPLY__", ocio_apply);

    let program = gl.create_program().map_err(|e| anyhow!("create_program: {e}"))?;
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
    let shader = gl.create_shader(kind).map_err(|e| anyhow!("create_shader: {e}"))?;
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
