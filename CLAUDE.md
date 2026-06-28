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
- **The T key toggles Standard ↔ `prefs.default_view_transform`** (a Settings dropdown, default
  "Filmic"), NOT a session "last view" — the old `App::last_view` was removed because it could drift
  to whatever was last picked (e.g. ACES). The same pref is the on-load default for HDRIs
  (`select_view_for_load`). View names match case-insensitively against the active display's views.
- **Continuous adjustment keys auto-repeat (hold to ramp); toggles/actions don't.** The
  `window_event` repeat guard allows OS auto-repeat for `← →` (folder nav) and the `Key::Character`
  adjustment keys `, . [ ] ; '` (exposure / gamma / clarity); everything else returns on
  `event.repeat`. A held ramp coalesces into ONE undo entry: each press/repeat pushes
  `App::adjust_repeat_until` forward (`ADJUST_COALESCE` 350 ms), `undo_gesture_active` treats that
  window as a gesture, and `commit_undo_if_changed` clears it + commits once the window closes
  (`about_to_wait` lists it as a wake deadline so the deferred commit fires after the tone ease).
- **Opening Settings grows the window to fit the dialog** (`App::sync_settings_window`, called after
  the per-frame UI actions): a too-small image window is eased up to `SETTINGS_MIN_LOGICAL`
  (540×720 pt) via `resize_window_centered` and restored on close (`settings_restore_size`); no-op in
  fullscreen / maximized / when already large enough. The dialog itself (`settings_dialog` in
  `ui/overlay.rs`) is a fixed-`SETTINGS_WIDTH` (440 pt) scrollable panel grouped into labelled
  sections (Display & view / Appearance / On open / Review tools / System), each a 2-column
  label+control `egui::Grid`. (A true second OS window was scoped out: it needs multi-window event
  routing + dual GL contexts + Win32-subclass changes and can't be validated headlessly.) Verify the
  dialog with `IMGVWR_DEBUG_OVERLAY=settings` + `IMGVWR_CAPTURE`; the grow itself only logs/eases
  interactively (headless capture grabs before the ease and `outer_position()` is unavailable).
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
  → `pick_nearest`). By default (`nearest_auto`) an image magnified past 200% at the view centre
  samples nearest for crisp pixels; less-magnified views stay bilinear. 2D uses on-screen scale
  `zoom * vh / img_h` (`flat_scale_now`); panoramas use a FOV/dimension-aware centre scale
  `π * vh / (2·tan(½fov)·img_h)` (`pano_scale_now`) — the equirect is `H/π` texels per radian and the
  rectilinear projection is `vh/(2·tan½fov)` screen px per radian at centre, so > 2.0 means one texel
  spans two screen px, same rule as 2D. The I key pins the *current* effective mode and clears
  `nearest_auto`, so the manual choice persists (works in both modes). Both `nearest_auto`/
  `nearest_filter` are in `UndoState`. This effective value feeds `RenderParams::nearest`, so the
  Lanczos gate above (`is_u8 && !nearest`) tracks it automatically. Verify the 2D switch with a
  checkerboard at `IMGVWR_DEBUG_ZOOM=1.9` (soft) vs `2.1` (crisp); the pano switch with an equirect
  checkerboard at `IMGVWR_DEBUG_PROJECTION=pano` + a wide `IMGVWR_DEBUG_FOV` (soft) vs narrow (crisp).
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
- **Camera RAW is developed by LibRaw to scene-linear float** (`src/raw_native/`, a C++ shim +
  bindgen like `exr_native`, gated on the `ocio` feature; vcpkg port `libraw`, links `raw_r`,
  `raw_r.dll` auto-staged by build.rs). `load_raw_native` sets a *view-faithful linear* develop:
  `use_camera_wb=1`, `output_color=1` (sRGB primaries), `gamm={1,1}` (no tone curve),
  `no_auto_bright=1` (respect the actual exposure), `output_bps=16`, `highlight=1` (unclip — keep
  recoverable highlights), `user_qual=3` (AHD). `is_encoded_srgb=false` so it rides the EXR/HDR
  linear path (exposure → OCIO view transform). **Highlight headroom:** with `highlight=1` LibRaw
  normalises its 16-bit output by the *max* WB multiplier (nothing overflows the container); the
  shim then rescales by `1/min(nonzero pre_mul)` (read *after* `dcraw_process`, which normalises
  `pre_mul` in place) so the neutral (green) reference channel saturates at 1.0 while the other
  channels keep their headroom *above* 1.0 — so lowering exposure / Filmic-ACES recovers blown
  regions (verified: a clipped sky lands at R≈2.24 in the float). Without `highlight=1` (i.e.
  clip) blown areas flatten to a neutral 1.0 and there is nothing to recover. Orientation is
  applied natively by LibRaw (`flip` in `dcraw_process`); no manual TIFF-tag read needed. Camera
  EXIF (make/model/lens/ISO/shutter/aperture/focal) flows via `ImageData::camera` (`CameraMeta`)
  into the F2 box. RAW does **not** auto-expose on load by default (`prefs.raw_auto_exposure`,
  off; `image_loader::is_raw` gates it in `finalize_adopt`) — distinct from the equirect-HDR
  `auto_exposure` path. No new keybinding (the toggle is a Settings checkbox), so the help KEYS
  popup is unchanged. Verify headlessly: `IMGVWR_CAPTURE` + `IMGVWR_DEBUG_EXPOSURE=-2` recovers a
  blown highlight that is white at EV 0. A standalone `LibRaw` probe (see git history /
  scratchpad) measuring output percentiles is the way to re-calibrate the headroom if needed.
- **Without the `ocio` feature** RAW falls back to the pure-Rust best-effort `rawler` path
  (`formats::load_raw`, sRGB-developed, `is_encoded_srgb=true`) — kept `#[cfg(not(feature =
  "ocio"))]` along with its helpers. rawler's `raw_metadata` is brittle (rejects some decodable
  files); that path reads EXIF orientation directly from the TIFF tag (`read_tiff_orientation`).
- **Clipping overlay (C key) is drawn in the main fragment shader, judged on the *original* sampled
  texel** (`clip_src` = the full RGBA `texel`, captured right after sampling — *before* sRGB-decode /
  exposure / view transform and before the diff/sharpness overrides), NOT a post pass (the post chain
  only has the display-encoded scene). A channel clips when `clip_src[c] >= u_clip_max[c] * (1 -
  margin)`. **`u_clip_max` is per-channel and per-format**, in *texel value space* (`ImageData::clip_max`,
  threaded onto `ImageTexture`, set in `draw_quad`): integer formats normalise their max to 1.0
  (`CLIP_MAX_NORM`); unbounded 32-bit float / Radiance HDR use `f32::MAX` (`CLIP_MAX_NONE`, never
  clips — this is the fix for "HDR values >1.0 falsely flagged"); a **16-bit-half EXR** clips at the
  half max `65504` (`HALF_MAX`) — note a 16-bit *integer* PNG/TIFF is `CLIP_MAX_NORM` (1.0), a
  16-bit *half* EXR is 65504, they are different. RAW uses `CLIP_MAX_NORM` (1.0): its >1.0 values are
  recoverable headroom, but relying on that is generally unwise, so anything at/above neutral white
  is flagged as clipped (a deliberate choice — not the per-channel sensor-saturation level). The mask is
  sampled `LINEAR_MIPMAP_LINEAR` + `step(0.15, …)` (not NEAREST): the max already can't be diluted
  by minification, and LINEAR spreads a clipped texel over ~1 texel so it isn't missed *between*
  screen-pixel sample points at high zoom-out. The lit stripe colour is `vec3(clip)` (R→red,
  R+G→yellow, all→white) alternating with black, diagonal in `gl_FragCoord` space, scrolled by
  `u_time`, composited at 0.75 only where a channel clips. **The composite mixes over
  `clamp(color,0,1)`, NOT the raw display colour** — clipped regions are exactly where the display
  blows past 1.0, and `0.25*huge` would re-clip to white, making the stripes invisible on HDR/EXR
  (8-bit looked fine only because its clipped display value is already 1.0). This was the bug that
  made the overlay silently no-op on float formats. **When a
  channel is isolated (F2 boxes, `u_isolate_channel>=0`) only that channel is evaluated** and the
  stripe takes its colour (R/G/B, or white for alpha). **Detection is max-not-average mipped:** a
  per-channel clip MASK (`texture::build_clip_mask`, RGBA8 255-where-clipped, MAX-reduced 2×2 mips,
  NEAREST sampling, unit `CLIP_MASK_UNIT`=13) is built from the *original* pixels so even a few-pixel
  blown region (e.g. a sun disc) survives minification — averaged image mips would dilute it away when
  zoomed out. The shader samples it `textureGrad(u_clip_mask, src_uv, …)` (seam-unwrapped grad in
  pano) and thresholds `>0.5`; `clip_src`/`u_clip_max` are only the *fallback* for **tiled** images
  (no single-texture mask). The mask is rebuilt lazily: `App::clip_mask_dirty` set on image load
  (`finalize_adopt`) and margin change (`SetClipMargin`/debug), consumed in `render` only while the
  overlay is on (`Renderer::set_clip_mask`, mirrors `set_diff_image`). The margin is *baked* into the
  mask, hence the rebuild-on-margin. State: `App::clip_overlay` (in `UndoState` like `sharpness`),
  margin `prefs.clip_margin` (Settings slider), `u_time` from `App::app_epoch`. `about_to_wait` keeps
  ~60 fps while it's on; `mm_params` sets `clip_overlay:false` so the minimap has no stripes. Verify
  with `IMGVWR_CAPTURE` + `IMGVWR_DEBUG_CLIP=1` (debug-only; `IMGVWR_DEBUG_CLIP_MARGIN` overrides the
  margin, clamped 0..1 in debug) on a synthetic 255-patch image, and on the 16-bit-half vs 32-bit EXR
  pair (`suburban_garden_4k_16b.exr` clips at the sun; the 32-bit `.exr`/`.hdr` never clip).
- **Animated images (GIF, animated WebP, APNG) play by re-uploading the current frame to the *same*
  single texture** each tick (`Renderer::update_animation_frame` → `tex_sub_image_2d` +
  `generate_mipmap`), never re-running the whole upload pipeline. All three share `collect_animation`
  (in `image_loader/formats.rs`) — a lazy `AnimationDecoder::into_frames` loop that applies EXIF
  orientation per frame and caps total bytes (`MAX_ANIM_FRAME_BYTES`, 1 GiB) — and `finish_animation`,
  which fills `ImageData::animation` (each `AnimFrame` a full-canvas RGBA8) plus frame 0 into `pixels`
  so the static paths (initial upload, diff, auto-exposure) are unchanged, and runs `apply_icc` over
  the static buffer **and every frame**. Dispatch: `gif` → `load_gif`; `webp` → `load_animated_webp`
  only when `webp_is_animated` (a still WebP keeps going through `load_via_image`); `png`/`apng` →
  `load_apng` only when `png_is_apng` (ICC/orientation read from `PngDecoder` *before* `.apng()`
  consumes it; the `ApngDecoder` exposes neither). `App::anim` (`AnimState`) tracks the current frame
  + `next_at` + `paused`; `advance_animation` (called from `render`) flips frames when their delay
  elapses, and `about_to_wait` schedules a `WaitUntil(next_at)` so the loop sleeps between frames (no
  60 fps spin). Space toggles `paused`. Per-frame delays are floored at 20 ms (`MIN_FRAME_DELAY`).
  Verify headlessly with `IMGVWR_DEBUG_GIF_FRAME=k` (debug builds: pin + pause frame k of *any*
  animation) for exact frame pixels, or time-spaced `IMGVWR_CAPTURE_DELAY_MS` captures of an
  N-colour-cycle file to confirm advancement + looping.
