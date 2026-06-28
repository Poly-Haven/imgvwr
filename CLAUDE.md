# imgvwr — notes for Claude

Minimal Windows-only GPU image viewer for HDR panoramas and standard images
(Rust + OpenGL 4.3 + egui). The full design lives in `plans/rewrite.md` (gitignored).

## Building

Native deps (OpenColorIO / OpenEXR / lcms) come from vcpkg; `bindgen` needs libclang.
Set `VCPKG_ROOT` to your vcpkg checkout and have LLVM installed (or `LIBCLANG_PATH`
pointing at a `libclang.dll`); see the README "Building from source" section for the
full prerequisites, then:

```powershell
cargo build --release
```

`cargo build --no-default-features` builds the gamma-only fallback and needs neither vcpkg nor libclang.

## Conventions & lessons

- **Adding a user-facing feature?** Always update the help popup (`KEYS` in
  [`src/ui/overlay.rs`](src/ui/overlay.rs) `help_dialog`) **and** the README controls table.
- Run clippy on **both** feature sets before declaring done:
  `cargo clippy --all-targets -- -D warnings` and the same with `--no-default-features`.
- **UI icons: prefer the *filled* Bootstrap variant** (`*-fill.svg`) over the thin outline
  one. At the small titlebar/HUD sizes the hairline strokes of the outline icons render muddy;
  the filled glyphs stay bold and crisp. Done so far for Open (`folder-fill`) and Settings
  (`gear-fill`); use the `-fill` version for any new icon when one exists.
- The fragment shader is templated — `__IMAGE_SAMPLER__` / `__OCIO_*__` are substituted in
  `renderer/mod.rs`. The panorama path must sample via `sample_image_grad` (seam-corrected
  derivatives), never plain `sample_image`, or the longitude wrap shows a mip-LOD seam.
- Headless verification (no visible window needed): `IMGVWR_CAPTURE=out.png` renders one frame
  to a PNG; `IMGVWR_DEBUG_*` env vars (debug builds only) force camera/exposure/projection state
  (e.g. `IMGVWR_DEBUG_CLARITY`, `IMGVWR_DEBUG_ISOLATE`, `IMGVWR_DEBUG_BOTTOM`).
- Screen-space review effects belong in the reusable post chain (`renderer/post.rs`): the scene
  renders into an offscreen RGBA16F target and effects run as fullscreen passes composited to the
  default framebuffer. Clarity lives there; focus peaking / false-colour / slot-diff should reuse
  it. When the effect is off, `Renderer::render` bypasses the whole chain (zero overhead).
- **Any GL pass drawn after the scene but before `egui.paint` must set its own `blend_func`.**
  egui_glow sets a *premultiplied* func (`ONE, ONE_MINUS_SRC_ALPHA`) every paint and never restores
  it, so the next frame the default-framebuffer blend func is egui's, not the renderer's init-time
  `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`. `draw_scene` now sets `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` itself
  every frame (it used to rely on luck — opaque src over an opaque cleared bg is identical under
  either func — which silently mis-composited *transparent* images and the checkerboard showing
  through them). `render_minimap` likewise sets its own func for the fade. Keep this rule for any new
  pass.
- **Background backdrop is a `BgPreset` cycle (B key): configured colour / black / grey checkerboard
  / white** (`App::bg_preset`, session-only — not persisted, not in `UndoState`). Solid presets feed
  `RenderParams::background` (the `glClear`); the checkerboard is a dedicated fullscreen pass
  (`bg_program`, `resources/shaders/background.glsl`, keyed off `gl_FragCoord`) drawn in `draw_scene`
  *before* the image when `params.bg_checker`, so transparency / letterboxing reveals it. The default
  `background_color` is 25% grey (`[64,64,64]`); existing users keep their saved value. Force a preset
  headlessly with `IMGVWR_DEBUG_BG=black|checker|white|user`.
- **The colour-management UI is two single-level dropdowns (Display + View), not a nested submenu.**
  egui 0.31 has no API to open a submenu leftward (`SubMenu` always positions right, then clamps
  against the screen edge — cramped, since the metadata box is top-right). So `view_dropdown` +
  `display_dropdown` are separate `menu_custom_button`s; the View list follows the active display, and
  switching display keeps the current view if valid else reverts to Standard→Raw→first
  (`eq_ignore_ascii_case`, matching the rest of the OCIO view selection).
- The navigation minimap (`M`) reuses the scene shader via a scissored second `draw_quad` into the
  bottom-right corner (`Renderer::render_minimap`) — so the thumbnail is tone-mapped and tiled like
  the main view; egui only strokes the border + view box on top. Its fade rides `u_global_alpha`.
- **8-bit images downscale with Lanczos-3, not bilinear** (`sample_image_lanczos` in the
  `SINGLE_TEXTURE_SAMPLER` injection; gated by `u_lanczos`, set in `draw_quad` from
  `image.is_u8 && !nearest`). It mip-prefilters to the nearest level (`round(lod)`) so the GPU mip
  chain anti-aliases the bulk minification, then reconstructs with a separable Lanczos-3 kernel over
  that level's texels via `texelFetch`. Upscaling (`lod <= 0`) and the nearest toggle fall back to
  `texture()`/bilinear, so only minification pays the cost. Bit depth ≠ sRGB: a 16-bit PNG is
  `F32`+sRGB, so the gate is `is_u8` (`ImageData::is_u8`, threaded onto `ImageTexture`), *not*
  `is_encoded_srgb`. Tiled (huge) images keep bilinear. Its `texelFetch` indices must mirror the
  *live* wrap modes `draw_quad` sets for the bilinear path — S always `REPEAT`, T `REPEAT` only when
  `u_wrap_2d` (else `CLAMP_TO_EDGE`); a static clamp on T stretched the top edge instead of tiling
  vertically when wrap (W) was on. `IMGVWR_DEBUG_NO_LANCZOS` forces it off for A/B capture; verify
  via `IMGVWR_DEBUG_ZOOM` < 1 (downscale, differs) vs > 1 (upscale, identical), and with
  `IMGVWR_DEBUG_WRAP` for the tiling path.
- **The `nearest` sampling flag is computed, not just the I-key toggle** (`App::effective_nearest`
  → `pick_nearest`). By default (`nearest_auto`) a 2D image magnified past 200% (on-screen scale
  `zoom * vh / img_h > 2.0`, via `flat_scale_now`) samples nearest for crisp pixels; panoramas and
  everything ≤ 200% stay bilinear. The I key pins the *current* effective mode and clears
  `nearest_auto`, so the manual choice persists. Both `nearest_auto`/`nearest_filter` are in
  `UndoState`. This effective value feeds `RenderParams::nearest`, so the Lanczos gate above
  (`is_u8 && !nearest`) tracks it automatically. Verify the auto switch with a checkerboard at
  `IMGVWR_DEBUG_ZOOM=1.9` (bilinear, soft) vs `2.1` (nearest, crisp).
- **2D display rotation (Up/Down) is a per-image session property** (`App::rotation` 0–3 CW
  quarter-turns; remembered in `image_rotations` by path; not reset by R/Ctrl+R). It's applied
  purely in the shader: `u_image_aspect` is fed the *rotated* aspect and the sampler permutes the
  uv (`rotate_uv`); the texture is never re-uploaded. Any 2D code that needs the image's aspect or
  pixel dimensions must use `display_dims()` / `display_aspect()` / `frame_dims()` (rotation-aware),
  NOT the raw `file_info` dims or `renderer.image_aspect()` — rulers, fit, the minimap, guides and
  window-framing all do. Guides stay in *displayed* uv (the shader compares them against the
  pre-permute coordinate).
- The camera is an enum (`Pano` | `Flat`) — keep the two states distinct; convert through
  `center_uv` on the P-toggle rather than reusing fields.
- **Window-follow zoom ease (`ease_window`) must be advanced from `about_to_wait`, never from
  the `Resized` handler.** Two failure modes this balances, both shipped as bugs before: (1) if
  the timed render loop *and* a synchronous `Resized` render both present, you get a double
  present → half-rate "slow-mo" zoom; (2) if the ease self-perpetuates from `Resized` (each step
  posts the next `SetWindowPos` before the loop yields), the whole chain drains before
  `about_to_wait` runs → OS scroll input is starved → fast-scroll "shudder". The working shape:
  advance one step per loop iteration in `about_to_wait` (input already processed), `Poll` while
  easing, and let the single synchronous `Resized` render present each size once (vsync-paced).
  `follow_zoom_with_window` frames against the in-flight `window_anim_target` height (not the
  lagging current size) so zoom gain per notch is scroll-speed-independent.
- **The borderless window keeps `WS_CAPTION | WS_SIZEBOX`.** winit's `with_decorations(false)`
  does NOT strip the caption/sizing-border styles (it needs them for Aero-snap) — it hides them
  via `WM_NCCALCSIZE` only. So `DefWindowProc` still paints the classic GDI frame on
  `WM_NCACTIVATE` / `WM_NCPAINT` (focus change, restore-from-minimize), flashing an old-style
  titlebar. `suppress_nonclient_frame` subclasses the proc to swallow `WM_NCPAINT` and forward
  `WM_NCACTIVATE` with `lParam = -1`; don't "fix" this by stripping styles (breaks snap/resize)
  or by leaning on `DWMWA_NCRENDERING_POLICY` (that's the DWM frame, not the GDI one).
- rawler's `raw_metadata` is brittle (rejects some decodable files); raw EXIF orientation is read
  directly from the TIFF tag instead (`read_tiff_orientation` in `image_loader/formats.rs`).
- **Animated GIFs play by re-uploading the current frame to the *same* single texture** each tick
  (`Renderer::update_animation_frame` → `tex_sub_image_2d` + `generate_mipmap`), never re-running the
  whole upload pipeline. `load_gif` decodes every frame up front (the `image` crate composites disposal
  / transparency, so each `AnimFrame` is a full-canvas RGBA8) into `ImageData::animation`; frame 0 also
  fills `pixels` so the static paths (initial upload, diff, auto-exposure) are unchanged. `App::anim`
  (`AnimState`) tracks the current frame + `next_at` + `paused`; `advance_animation` (called from
  `render`) flips frames when their delay elapses, and `about_to_wait` schedules a `WaitUntil(next_at)`
  so the loop sleeps between frames (no 60 fps spin). Space toggles `paused`. Per-frame delays are
  floored at 20 ms (`MIN_GIF_DELAY`) so a 0-delay GIF can't busy-spin. Verify headlessly with
  `IMGVWR_DEBUG_GIF_FRAME=k` (debug builds: pin + pause frame k) for exact frame pixels, or time-spaced
  `IMGVWR_CAPTURE_DELAY_MS` captures of an N-colour-cycle GIF to confirm advancement + looping.
