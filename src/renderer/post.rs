//! A small, reusable multi-pass screen-space post-process chain.
//!
//! The scene is first rendered into an offscreen RGBA16F texture at viewport
//! resolution (`scene`), then post effects run as fullscreen-quad passes and the
//! final result is composited to the default framebuffer. Today it implements
//! **Clarity** (a large-radius unsharp mask: `base + amount*(base - blur)` with a
//! midtone luminance mask), but the FBO scaffold is deliberately generic so other
//! review tools (focus peaking, false-colour, slot diff) can reuse it.
//!
//! Cost scales with the viewport (a few megapixels), not the source image, so it
//! stays real-time at 4K even on a 24k EXR: the blur runs on quarter-resolution
//! targets. When clarity is off the whole chain is bypassed (see `Renderer`).

use anyhow::Result;
use glow::HasContext as _;

const BLIT_SRC: &str = include_str!("../../resources/shaders/post_blit.glsl");
const BLUR_SRC: &str = include_str!("../../resources/shaders/post_blur.glsl");
const CLARITY_SRC: &str = include_str!("../../resources/shaders/post_clarity.glsl");

/// A colour texture and the framebuffer that renders into it.
struct Target {
    tex: glow::Texture,
    fbo: glow::Framebuffer,
}

impl Target {
    unsafe fn new(gl: &glow::Context, w: i32, h: i32) -> Self {
        let tex = gl.create_texture().expect("post target texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA16F, w.max(1), h.max(1));
        let lin = glow::LINEAR as i32;
        let clamp = glow::CLAMP_TO_EDGE as i32;
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, lin);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, lin);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
        let fbo = gl.create_framebuffer().expect("post target fbo");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(tex),
            0,
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        Target { tex, fbo }
    }

    unsafe fn delete(&self, gl: &glow::Context) {
        gl.delete_framebuffer(self.fbo);
        gl.delete_texture(self.tex);
    }
}

/// The viewport-sized scene target plus the half/quarter-res blur targets.
struct Targets {
    size: (i32, i32),
    scene: Target, // full-res scene (display-encoded)
    half: Target,  // w/2 × h/2 downsample
    q_a: Target,   // w/4 × h/4 (downsample, then H blur)
    q_b: Target,   // w/4 × h/4 (V blur)
}

impl Targets {
    unsafe fn new(gl: &glow::Context, w: i32, h: i32) -> Self {
        let (hw, hh) = ((w / 2).max(1), (h / 2).max(1));
        let (qw, qh) = ((w / 4).max(1), (h / 4).max(1));
        Targets {
            size: (w, h),
            scene: Target::new(gl, w, h),
            half: Target::new(gl, hw, hh),
            q_a: Target::new(gl, qw, qh),
            q_b: Target::new(gl, qw, qh),
        }
    }

    unsafe fn delete(&self, gl: &glow::Context) {
        for t in [&self.scene, &self.half, &self.q_a, &self.q_b] {
            t.delete(gl);
        }
    }
}

pub struct PostChain {
    blit: glow::Program,
    blur: glow::Program,
    clarity: glow::Program,
    blur_step: Option<glow::UniformLocation>,
    clarity_amount: Option<glow::UniformLocation>,
    vao: glow::VertexArray,
    targets: Option<Targets>,
}

impl PostChain {
    /// # Safety: the GL context must be current.
    pub unsafe fn new(gl: &glow::Context) -> Result<Self> {
        let blit = super::build_post_program(gl, BLIT_SRC)?;
        let blur = super::build_post_program(gl, BLUR_SRC)?;
        let clarity = super::build_post_program(gl, CLARITY_SRC)?;
        let (vao, _vbo) = super::build_quad(gl, blit)?;
        // Bind sampler units once: blit/blur read u_src on unit 0; clarity reads
        // u_base on unit 0 and u_blur on unit 1.
        let blur_step = gl.get_uniform_location(blur, "u_step");
        let clarity_amount = gl.get_uniform_location(clarity, "u_amount");
        for (p, name, unit) in [
            (blit, "u_src", 0),
            (blur, "u_src", 0),
            (clarity, "u_base", 0),
            (clarity, "u_blur", 1),
        ] {
            gl.use_program(Some(p));
            if let Some(loc) = gl.get_uniform_location(p, name) {
                gl.uniform_1_i32(Some(&loc), unit);
            }
        }
        gl.use_program(None);
        Ok(PostChain {
            blit,
            blur,
            clarity,
            blur_step,
            clarity_amount,
            vao,
            targets: None,
        })
    }

    /// (Re)create the offscreen targets if the viewport size changed. Returns the
    /// scene framebuffer the caller should render the scene into.
    ///
    /// # Safety: the GL context must be current; `w`/`h` must be > 0.
    pub unsafe fn scene_framebuffer(
        &mut self,
        gl: &glow::Context,
        w: i32,
        h: i32,
    ) -> glow::Framebuffer {
        let stale = self.targets.as_ref().map(|t| t.size) != Some((w, h));
        if stale {
            if let Some(old) = self.targets.take() {
                old.delete(gl);
            }
            self.targets = Some(Targets::new(gl, w, h));
        }
        self.targets.as_ref().unwrap().scene.fbo
    }

    /// Run the clarity passes on the already-rendered scene target and composite
    /// the result to the default framebuffer (left bound on return).
    ///
    /// `radius` is the unsharp blur radius in viewport pixels; `amount` the
    /// strength. # Safety: the GL context must be current.
    pub unsafe fn apply_clarity(
        &self,
        gl: &glow::Context,
        radius: f32,
        amount: f32,
        w: i32,
        h: i32,
    ) {
        let Some(t) = &self.targets else {
            return;
        };
        // Post passes overwrite every texel — no blending, no depth.
        gl.disable(glow::BLEND);
        gl.bind_vertex_array(Some(self.vao));
        gl.active_texture(glow::TEXTURE0);

        let (hw, hh) = ((w / 2).max(1), (h / 2).max(1));
        let (qw, qh) = ((w / 4).max(1), (h / 4).max(1));

        // Downsample scene → half → quarter (bilinear box-ish).
        self.blit_pass(gl, &t.half, t.scene.tex, hw, hh);
        self.blit_pass(gl, &t.q_a, t.half.tex, qw, qh);

        // Separable Gaussian at quarter-res. The per-tap UV step maps `radius`
        // viewport pixels into UV space (resolution-independent), so the same
        // formula blurs by the same radius regardless of the quarter-res grid.
        gl.use_program(Some(self.blur));
        // Horizontal: q_a → q_b.
        gl.uniform_2_f32(self.blur_step.as_ref(), radius / (4.0 * w as f32), 0.0);
        self.draw_into(gl, &t.q_b, t.q_a.tex, qw, qh);
        // Vertical: q_b → q_a.
        gl.uniform_2_f32(self.blur_step.as_ref(), 0.0, radius / (4.0 * h as f32));
        self.draw_into(gl, &t.q_a, t.q_b.tex, qw, qh);

        // Composite base (full-res scene) + amount*(base-blur) → default fb.
        gl.use_program(Some(self.clarity));
        gl.uniform_1_f32(self.clarity_amount.as_ref(), amount);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(0, 0, w.max(1), h.max(1));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(t.scene.tex));
        gl.active_texture(glow::TEXTURE1);
        gl.bind_texture(glow::TEXTURE_2D, Some(t.q_a.tex));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

        // Restore state for the next frame's scene draw / egui.
        gl.active_texture(glow::TEXTURE1);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.bind_vertex_array(None);
        gl.enable(glow::BLEND);
    }

    /// One bilinear blit of `src` into `dst` at `(w, h)`.
    unsafe fn blit_pass(
        &self,
        gl: &glow::Context,
        dst: &Target,
        src: glow::Texture,
        w: i32,
        h: i32,
    ) {
        gl.use_program(Some(self.blit));
        self.draw_into(gl, dst, src, w, h);
    }

    /// Bind `dst`, set the viewport, bind `src` on unit 0, and draw the quad
    /// (the program is assumed already bound).
    unsafe fn draw_into(
        &self,
        gl: &glow::Context,
        dst: &Target,
        src: glow::Texture,
        w: i32,
        h: i32,
    ) {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst.fbo));
        gl.viewport(0, 0, w, h);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(src));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
    }
}
