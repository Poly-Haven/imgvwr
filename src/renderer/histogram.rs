//! GPU histogram of the image *as it is actually displayed*.
//!
//! There is no CPU-side display transform anywhere in imgvwr — OCIO is only ever
//! built as a GPU program ([`crate::ocio::OcioManager::build_gpu_shader`]), which
//! is why even the colour-pick tooltip reads its "Display" value back out of the
//! framebuffer. So the histogram is measured where the transform lives: the whole
//! image is drawn flat into a small offscreen target through the *same* fragment
//! program the screen uses, and a compute shader bins the result.
//!
//! Two properties are load-bearing:
//!
//! * **Point sampling, not minification.** The offscreen target is far smaller
//!   than the image, so the draw asks for [`RenderParams::point_sample`] — a
//!   plain `GL_NEAREST` min filter with no mip chain. Mip-averaged texels blend
//!   neighbours together and quietly erase the extremes, which are precisely the
//!   clipped highlights and crushed blacks the graph exists to show.
//! * **Nothing blocks the UI thread.** The dispatch is fenced; each frame polls
//!   the fence with a zero timeout and only copies the 3 KB result back once the
//!   GPU says it has finished. At most one dispatch is ever in flight.
//!
//! The cost of subsampling is that a *handful* of clipped pixels in a 24k
//! panorama can fall between samples. Finding those is what the clipping overlay
//! (`C`, backed by a max-mipped mask) is for; this is a tonal overview.
//!
//! Compute shaders are core in GL 4.3, which is the context imgvwr asks for, but
//! a driver that hands back something older simply leaves [`HistogramPass::new`]
//! returning `None` and the F2 box omits the graph.

use glow::HasContext as _;

const BIN_SRC: &str = include_str!("../../resources/shaders/histogram.glsl");

/// Bins per channel. 256 matches 8-bit display quantisation, so one bin is one
/// output code — finer would only resolve the display's own rounding.
pub const BINS: usize = 256;

/// Pixels measured per pass. Progressive refinement means this no longer sets
/// *accuracy* — every setting converges on having read the whole image — only
/// how the work is sliced: bigger passes finish sooner but cost more per frame.
///
/// Benchmarked on a 24576×12288 EXR (302 Mpx), where this stride of 6 gives 36
/// passes: 0.93 ms each and 33 ms of GPU in total. Both neighbours are worse on
/// one axis or the other — a stride of 4 costs 2.16 ms per frame for the same
/// total, and a stride of 8 needs 64 passes and 47 ms in total because the
/// fixed per-pass overhead gets paid more often. It also lands under the pass
/// cap for any plausible image, so the refinement always completes.
const SAMPLES_PER_PASS: u32 = 8_000_000;

/// `over[4]` in the shader — three channels plus one slot of padding.
const OVER_SLOTS: usize = 4;
const BUF_U32: usize = BINS * 3 + OVER_SLOTS;

/// Must match `layout(local_size_x/y)` in `histogram.glsl`.
const GROUP: i32 = 16;

/// A completed histogram of the displayed image.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Per-channel sample counts (R, G, B), indexed by bin. Bin `i` covers
    /// display values `[i / BINS, (i + 1) / BINS)`.
    pub bins: [[u32; BINS]; 3],
    /// Per-channel samples at or above the display maximum — everything the
    /// screen can only render as full white. Drawn as the 1px right-edge spike.
    pub over: [u32; 3],
}

impl Histogram {
    /// The tallest ordinary bin across all three channels, used to normalise the
    /// graph. The over-range spike is deliberately excluded: it is a single bin
    /// wide and can dwarf the distribution it sits beside, flattening the very
    /// shape the graph is for. The spike simply clips at full height instead.
    pub fn peak(&self) -> u32 {
        self.bins
            .iter()
            .flat_map(|c| c.iter().copied())
            .max()
            .unwrap_or(0)
    }
}

/// A colour texture and the framebuffer that renders into it.
struct Target {
    tex: glow::Texture,
    fbo: glow::Framebuffer,
    size: (i32, i32),
}

impl Target {
    unsafe fn new(gl: &glow::Context, w: i32, h: i32) -> Option<Self> {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        // RGBA16F keeps values above 1.0 measurable (that is the whole point of
        // the over-range spike) while staying far finer than 1/256 below it.
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA16F, w.max(1), h.max(1));
        let nearest = glow::NEAREST as i32;
        let clamp = glow::CLAMP_TO_EDGE as i32;
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, nearest);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, nearest);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
        let fbo = match gl.create_framebuffer() {
            Ok(f) => f,
            Err(e) => {
                log::warn!("histogram: create framebuffer failed: {e}");
                gl.bind_texture(glow::TEXTURE_2D, None);
                gl.delete_texture(tex);
                return None;
            }
        };
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(tex),
            0,
        );
        // An allocation the driver refused leaves an incomplete framebuffer that
        // silently discards every draw, which would show up as a permanently
        // empty graph rather than as an error. Check, and tear down cleanly.
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        if status != glow::FRAMEBUFFER_COMPLETE {
            log::warn!("histogram: {w}x{h} target incomplete ({status:#x}); skipping");
            gl.delete_framebuffer(fbo);
            gl.delete_texture(tex);
            return None;
        }
        Some(Target {
            tex,
            fbo,
            size: (w, h),
        })
    }

    unsafe fn delete(&self, gl: &glow::Context) {
        gl.delete_framebuffer(self.fbo);
        gl.delete_texture(self.tex);
    }
}

/// The offscreen target, the binning program and the fenced result buffer.
pub struct HistogramPass {
    program: glow::Program,
    src_loc: Option<glow::UniformLocation>,
    ssbo: glow::Buffer,
    target: Option<Target>,
    /// A target size the driver refused. Remembered so the failure isn't retried
    /// — and re-logged — on every single frame; a different size tries again.
    failed_size: Option<(i32, i32)>,
    /// Set between [`dispatch`](Self::dispatch) and the fence being signalled.
    fence: Option<glow::Fence>,
    /// Staging for both the pre-dispatch zeroing and the readback, so neither
    /// allocates. Held as `u32` because casting *down* to a byte view is always
    /// correctly aligned, whereas a `Vec<u8>` read back as `u32` need not be.
    scratch: Vec<u32>,
}

impl HistogramPass {
    /// Build the pass, or `None` when the driver has no compute support (the
    /// caller then just never shows a histogram). # GL context must be current.
    pub unsafe fn new(gl: &glow::Context) -> Option<Self> {
        let v = gl.version();
        if v.is_embedded || (v.major, v.minor) < (4, 3) {
            log::info!(
                "histogram: GL {}.{} has no compute shaders; graph disabled",
                v.major,
                v.minor
            );
            return None;
        }

        let program = build_compute_program(gl, BIN_SRC)?;
        let src_loc = gl.get_uniform_location(program, "u_src");

        let ssbo = match gl.create_buffer() {
            Ok(b) => b,
            Err(e) => {
                log::warn!("histogram: create SSBO failed: {e}");
                gl.delete_program(program);
                return None;
            }
        };
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(ssbo));
        gl.buffer_data_size(
            glow::SHADER_STORAGE_BUFFER,
            (BUF_U32 * 4) as i32,
            glow::DYNAMIC_READ,
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        Some(Self {
            program,
            src_loc,
            ssbo,
            target: None,
            failed_size: None,
            fence: None,
            scratch: vec![0u32; BUF_U32],
        })
    }

    /// True while a dispatch is awaiting its fence — the caller should keep
    /// requesting frames so [`poll`](Self::poll) gets a chance to land it, and
    /// must not start another dispatch.
    pub fn is_pending(&self) -> bool {
        self.fence.is_some()
    }

    /// The offscreen grid to render `image_w × image_h` into: the image's own
    /// size, scaled down by one uniform factor until it fits `samples` pixels,
    /// preserving the aspect ratio and never upscaling.
    ///
    /// Because the draw then reads exactly one source texel per target texel
    /// (`point_sample`), that factor *is* the sampling stride — a 4096×2048
    /// image at a 1M budget becomes a 1414×707 grid, every ~3rd pixel each way,
    /// spread evenly over the whole image. Anything already at or under the
    /// budget is measured in full.
    ///
    /// A total-pixel budget alone doesn't bound either side — a 1×200000 strip is
    /// well under the budget yet far past `GL_MAX_TEXTURE_SIZE`, and the
    /// allocation would simply fail. Each axis is clamped as well; the resulting
    /// aspect skew only re-weights which parts of such a pathological image get
    /// sampled, which beats having no histogram at all.
    pub fn grid_size(image_w: i32, image_h: i32, max_texture: i32) -> (i32, i32) {
        let (w, h) = (image_w.max(1), image_h.max(1));
        let s = Self::stride(w, h, max_texture);
        // Round *up*, so the grid covers the full extent even when the stride
        // doesn't divide it. The overhang wraps (`wrap_2d`) onto columns already
        // measured, which double-counts a sliver — far better than leaving an
        // unmeasured strip down one edge.
        // (`i32::div_ceil` is still unstable; both operands are positive here.)
        ((w + s - 1) / s, (h + s - 1) / s)
    }

    /// Sampling stride: take every `stride`-th pixel on each axis.
    ///
    /// An integer stride is what makes progressive refinement exact — the
    /// `stride²` phase offsets then partition the image into residue classes
    /// that tile it perfectly, so each pass measures a disjoint set of pixels
    /// and the accumulated result is identical to reading every pixel.
    pub fn stride(image_w: i32, image_h: i32, max_texture: i32) -> i32 {
        let (w, h) = (image_w.max(1), image_h.max(1));
        let px = w as f64 * h as f64;
        let ideal = (px / SAMPLES_PER_PASS as f64).sqrt().round() as i32;
        // Also enough stride to keep the grid inside the driver's texture limit.
        let by_limit = (w.max(h) as f64 / max_texture.max(1) as f64).ceil() as i32;
        ideal.max(by_limit).max(1)
    }

    /// Phase-offset passes needed to visit every source pixel: `stride²`, capped
    /// so a very small budget on a very large image can't schedule a refinement
    /// that runs for many seconds. Past the cap the graph simply stops improving,
    /// still having measured far more than a single pass.
    pub fn passes_for(stride: i32) -> u32 {
        const MAX_PASSES: u32 = 128; // ~2s at 60fps
        ((stride as u32).saturating_mul(stride as u32)).min(MAX_PASSES)
    }

    /// The grid for measuring the viewport: one sample per screen pixel, no
    /// budget applied.
    ///
    /// The budget exists because an image can be 300 Mpx; a viewport cannot —
    /// it is bounded by the display. Measured on a 24k HDRI, a full-resolution
    /// 4K viewport costs 0.76 ms and a 1080p one 0.34 ms, so subsampling it
    /// would save nothing worth having and would cost the property that makes
    /// the reading easy to trust: that it is exactly the pixels on screen.
    pub fn viewport_grid(w: i32, h: i32, max_texture: i32) -> (i32, i32) {
        let cap = max_texture.max(1);
        (w.clamp(1, cap), h.clamp(1, cap))
    }

    /// Bind (creating or resizing as needed) the offscreen framebuffer the caller
    /// should draw the flat image into. `None` if it could not be allocated.
    ///
    /// # Safety: the GL context must be current.
    pub unsafe fn begin(
        &mut self,
        gl: &glow::Context,
        w: i32,
        h: i32,
    ) -> Option<glow::Framebuffer> {
        if self.failed_size == Some((w, h)) {
            return None;
        }
        if self.target.as_ref().map(|t| t.size) != Some((w, h)) {
            if let Some(old) = self.target.take() {
                old.delete(gl);
            }
            self.target = Target::new(gl, w, h);
            self.failed_size = self.target.is_none().then_some((w, h));
        }
        let fbo = self.target.as_ref()?.fbo;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        // The image draw must land in the target verbatim; blending it against
        // whatever the previous dispatch left there would corrupt every bin.
        gl.disable(glow::BLEND);
        gl.viewport(0, 0, w.max(1), h.max(1));
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        Some(fbo)
    }

    /// Bin whatever [`begin`](Self::begin) left in the target and fence the
    /// result. Restores the default framebuffer and the scene's blend state.
    ///
    /// # Safety: the GL context must be current, and `begin` must have succeeded.
    /// With `accumulate`, the counters are left alone so this pass adds to the
    /// previous one — the shader only ever `atomicAdd`s, so summing several
    /// offset passes is free. That is the whole mechanism behind progressive
    /// refinement: same cost per frame, more of the image measured each time.
    pub unsafe fn dispatch(&mut self, gl: &glow::Context, accumulate: bool) {
        let Some(target) = &self.target else {
            return;
        };
        let (w, h) = target.size;

        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.ssbo));
        // Zero the counters: the shader only ever adds. Safe to write directly
        // because a dispatch is never started while another is in flight.
        if !accumulate {
            self.scratch.fill(0);
            gl.buffer_sub_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                0,
                bytemuck::cast_slice(&self.scratch),
            );
        }
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.ssbo));

        gl.use_program(Some(self.program));
        gl.uniform_1_i32(self.src_loc.as_ref(), 0);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(target.tex));
        let group = GROUP as u32;
        gl.dispatch_compute(
            (w.max(1) as u32).div_ceil(group),
            (h.max(1) as u32).div_ceil(group),
            1,
        );
        gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::BUFFER_UPDATE_BARRIER_BIT);

        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.use_program(None);

        match gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) {
            Ok(f) => self.fence = Some(f),
            // Without a fence there is no way to read back without stalling, so
            // drop this round rather than freeze the UI; the next change retries.
            Err(e) => log::warn!("histogram: fence_sync failed: {e}"),
        }

        // Leave the GL state as the scene draw expects to find it. egui and the
        // minimap both rely on the alpha-blend func being live (see CLAUDE.md).
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.enable(glow::BLEND);
        gl.blend_equation(glow::FUNC_ADD);
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
    }

    /// Collect a finished dispatch, or `None` if the GPU is still working (this
    /// never waits). # Safety: the GL context must be current.
    pub unsafe fn poll(&mut self, gl: &glow::Context) -> Option<Histogram> {
        let fence = self.fence?;
        // Zero timeout: ask, never block. The flush bit guarantees the dispatch
        // has actually been submitted, so this can't wait on a command that is
        // still sitting unflushed in the driver's queue.
        let status = gl.client_wait_sync(fence, glow::SYNC_FLUSH_COMMANDS_BIT, 0);
        match status {
            glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => {}
            glow::TIMEOUT_EXPIRED => return None,
            _ => {
                // WAIT_FAILED — drop the fence so a later change can retry.
                log::warn!("histogram: client_wait_sync failed ({status:#x})");
                gl.delete_sync(fence);
                self.fence = None;
                return None;
            }
        }
        gl.delete_sync(fence);
        self.fence = None;

        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.ssbo));
        gl.get_buffer_sub_data(
            glow::SHADER_STORAGE_BUFFER,
            0,
            bytemuck::cast_slice_mut(&mut self.scratch),
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        let words = &self.scratch;
        let mut hist = Histogram {
            bins: [[0u32; BINS]; 3],
            over: [0u32; 3],
        };
        for (ch, dst) in hist.bins.iter_mut().enumerate() {
            dst.copy_from_slice(&words[ch * BINS..(ch + 1) * BINS]);
        }
        let over = hist.over.len();
        hist.over.copy_from_slice(&words[BINS * 3..BINS * 3 + over]);
        Some(hist)
    }

    /// # Safety: the GL context must be current.
    pub unsafe fn delete(&mut self, gl: &glow::Context) {
        if let Some(f) = self.fence.take() {
            gl.delete_sync(f);
        }
        if let Some(t) = self.target.take() {
            t.delete(gl);
        }
        gl.delete_buffer(self.ssbo);
        gl.delete_program(self.program);
    }
}

/// Compile and link a standalone compute program. Returns `None` (having logged)
/// rather than erroring, since every failure here just means no histogram.
unsafe fn build_compute_program(gl: &glow::Context, src: &str) -> Option<glow::Program> {
    let shader = match gl.create_shader(glow::COMPUTE_SHADER) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("histogram: create compute shader failed: {e}");
            return None;
        }
    };
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        log::warn!(
            "histogram: compute shader compile failed:\n{}",
            gl.get_shader_info_log(shader)
        );
        gl.delete_shader(shader);
        return None;
    }
    let program = match gl.create_program() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("histogram: create_program failed: {e}");
            gl.delete_shader(shader);
            return None;
        }
    };
    gl.attach_shader(program, shader);
    gl.link_program(program);
    let ok = gl.get_program_link_status(program);
    gl.detach_shader(program, shader);
    gl.delete_shader(shader);
    if !ok {
        log::warn!(
            "histogram: compute program link failed:\n{}",
            gl.get_program_info_log(program)
        );
        gl.delete_program(program);
        return None;
    }
    Some(program)
}
