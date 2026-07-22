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

## Batch changes

When asked to develop multiple changes or features in a batch, always commit them separately.

Never push without explicit permission.

## Adding to CLAUDE.md

Do not bloat the file. Keep it limited to information that's hard to discover in the code, lessons we're likely to need to know again, and info that's important for every session.

## Testing

The user has given you a folder full of various test images to use. If not, ask them for one.

If no relevant test images are found for the current task, generate one.

If during testing or validation you are struggling to screenshot the correct region of interest over and over again, instead crop an existing image carefully and reuse it.

## Release

Built **locally** (vcpkg deps are too slow for CI). With `VCPKG_ROOT` set: bump `version` in `Cargo.toml` (the tag derives from it), commit as `Release vX.Y.Z`, push `master`, then `pwsh scripts/release.ps1 -Publish` (builds, packages the self-contained zip, tags, pushes, creates the release).

- It writes only a placeholder note — replace with detailed notes covering everything since the previous tag: `gh release edit vX.Y.Z --notes-file <file>` (`## What's new`, then `### Features` / `### Fixes` bullets, self-contained/SmartScreen footer).
- Smoke-test the zip is self-contained: extract it and run with the PATH stripped to System32 plus `IMGVWR_CAPTURE` — it must render with no missing-DLL error.

## Conventions & lessons

- **Adding a user-facing feature?** Always update the help popup (`KEYS` in
  [`src/ui/overlay.rs`](src/ui/overlay.rs) `help_dialog`) **and** the README controls table.
- Run clippy on **both** feature sets before declaring done:
  `cargo clippy --all-targets -- -D warnings` and the same with `--no-default-features`.
- **UI icons come from `resources/icons`, never from a font character.** The UI font (Inter)
  doesn't cover most symbols — `+`, `✕`, `✓`, arrows, etc. render as a tofu box (a missing-glyph
  square). Any icon in a widget must be an SVG rendered through `ui::icon_button` (or
  `egui::Image`/`Button::image` with `egui::include_image!("../../resources/icons/…")`), not a
  `Button`/`label` with a glyph string. If the icon you need isn't in `resources/icons/ui`, add the
  Bootstrap SVG for it. Prefer the *filled* Bootstrap variant (`*-fill.svg`): at titlebar/HUD sizes
  the hairline outline strokes render muddy; filled glyphs stay crisp.
- **Headless verification:** `IMGVWR_CAPTURE=out.png` renders one frame to a PNG; `IMGVWR_DEBUG_*`
  env vars (debug builds only) force state — e.g. `IMGVWR_DEBUG_ZOOM`, `IMGVWR_DEBUG_EXPOSURE`,
  `IMGVWR_DEBUG_PROJECTION=pano`, `IMGVWR_DEBUG_OVERLAY=settings|metadata|error|loading|hint|help`,
  `IMGVWR_DEBUG_BG=black|checker|white|user`,
  `IMGVWR_DEBUG_LEVELS=black,white`, `IMGVWR_DEBUG_HIST_VIEWPORT=1`. Playback:
  `IMGVWR_DEBUG_PLAY=1` (play), `IMGVWR_DEBUG_PLAY_FRAME=<n>` (park *paused* on a frame — racing
  the clock is not reproducible), `IMGVWR_DEBUG_PLAY_STOP=1` (then leave playback, to check the
  ring teardown), `IMGVWR_DEBUG_CACHE_MB=<n>`; plus `IMGVWR_PLAYBACK_WORKERS` /
  `IMGVWR_PLAYBACK_RING_BYTES`, which work in release too. `RUST_LOG=debug` also dumps
  each landed histogram as a 16-bucket digest, which is the only way to check the *numbers* rather
  than the picture — and the digest is what proves e.g. that viewport-mode sampling really tracks
  the zoom (`_ZOOM=4` on a full-range ramp collapses it to the middle quarter).
- **Panorama sampling must use `sample_image_grad`, never plain `sample_image`.** The grad variant
  applies seam-corrected derivatives; plain sampling produces a mip-LOD seam at the longitude wrap.
- **Every GL pass drawn after the scene but before `egui.paint` must set its own `blend_func`.**
  egui_glow sets a premultiplied func (`ONE, ONE_MINUS_SRC_ALPHA`) every paint and never restores it,
  so the next frame's default-framebuffer blend is egui's. This silently mis-composited transparent
  images — the bug only appeared with transparency because opaque src is identical under either func.
- **Any code that needs the image's dimensions must use `display_dims()` / `display_aspect()` /
  `frame_dims()`, NOT `file_info` dims or `renderer.image_aspect()`.** Display rotation (Up/Down,
  `App::rotation`) is applied in the shader only; the texture is never re-uploaded. Raw dims are
  wrong whenever the image is rotated.
- **Window-follow zoom ease must advance from `about_to_wait`, never from the `Resized` handler.**
  Advancing from `Resized` causes the chain to drain before `about_to_wait` runs, starving OS scroll
  input → fast-scroll "shudder". Advancing from the timed loop *and* `Resized` double-presents →
  half-rate "slow-mo" zoom. The working shape: one step per `about_to_wait`, `Poll` while easing,
  single synchronous `Resized` render per size.
- **Don't strip `WS_CAPTION | WS_SIZEBOX` from the borderless window.** winit's
  `with_decorations(false)` keeps these styles for Aero-snap and hides them via `WM_NCCALCSIZE`.
  Stripping them breaks snap/resize. The GDI frame flash on focus-change is suppressed by
  `suppress_nonclient_frame` (swallows `WM_NCPAINT`, forwards `WM_NCACTIVATE` with `lParam = -1`);
  `DWMWA_NCRENDERING_POLICY` is the DWM frame, not the GDI one — wrong lever.
  **Corollary: don't call `window.set_title` on a hot path.** `SetWindowText`
  forces a native non-client caption repaint that flashes the GDI frame through,
  even with `suppress_nonclient_frame` active — sequence playback hit this by
  updating the OS title per frame. The egui custom titlebar tracks state via
  `file_info.name` (a plain egui redraw, no flash); the OS title only feeds the
  taskbar/alt-tab, so set it rarely (playback sets it once on entry/exit).
- **LibRaw must be built with OpenMP, or RAW loads are ~2.6× slower.** vcpkg's `libraw` port
  defaults to *no* OpenMP, and nothing warns you — the demosaic just runs single-threaded.
  `vcpkg.json` pins `libraw[openmp]`; classic-mode installs need `libraw[openmp]` explicitly.
  Verify with `strings raw_r.dll | grep -i vcomp` (case-**insensitive** — MSVC writes
  `VCOMP140.DLL`; a case-sensitive grep gives a false negative). Measured on a 45 Mpx NEF,
  32 cores: decode 4.24 s → 1.83 s from OpenMP alone, then → 1.62 s after the shim stopped
  handing a malloc'd buffer across the FFI for Rust to copy (`begin`/`finish` now expand
  straight into a Rust-owned `Vec`, which also halves peak memory: no 727 MB duplicate).
  Stage breakdown at that point: unpack 0.57 s (LibRaw, serial — not parallelised upstream),
  dcraw_process 0.74 s, make_mem_image 0.26 s. Don't "fix" the remaining time by changing
  `user_qual` (AHD) — that changes the pixels. OpenMP does not: serial vs 32 threads renders
  bit-identical. The packaged zip must bundle **vcomp140.dll** (`package.ps1`) or a machine
  without the VC++ redist can't start it.
- **RAW develops to scene-linear float with highlight headroom above 1.0.** The shim rescales by
  `1/min(nonzero pre_mul)` so the green channel saturates at 1.0 and other channels can exceed 1.0 —
  lowering exposure / applying Filmic recovers blown regions. Don't change `highlight=1` (unclip)
  or the rescale: removing either makes blown highlights flatten to 1.0 with nothing to recover.
- **There is no CPU-side display transform.** OCIO is only ever built as a *GPU* program
  (`build_gpu_shader`; the shim never creates a CPU processor), so anything needing on-screen
  values must read them back from the GPU. That's why the colour-pick tooltip's "Display" row is a
  1×1 `glReadPixels`, and why the F2 histogram renders the image flat into an offscreen target
  through the same fragment program and bins it with a GL 4.3 compute shader. Don't reach for a
  `cpu_processor` — there isn't one.
- **Anything collected inside the gfx borrow is one frame late, and must request its own redraw.**
  `ui_inputs()` snapshots App state *before* that borrow, so a value produced inside it (histogram
  readback, colour-pick `glReadPixels`) can't reach the paint that just happened. The colour-pick
  gets away with it because a drag emits continuous mouse events; the histogram doesn't — once the
  tone ease settles, `about_to_wait` parks on `Wait` and the box would keep showing the previous
  measurement until an unrelated event. **`IMGVWR_CAPTURE` cannot catch this class of bug**: capture
  mode forces `ControlFlow::Poll` and a redraw every iteration, so the missing request is invisible.
- **Bar charts narrower than a pixel drop bars.** The F2 histogram has 256 bins in ~205 points, so
  a per-bin quad is ~0.8px wide; `Shape::mesh` goes straight to the GPU with no anti-aliasing, and
  a quad containing no pixel *centre* rasterises to nothing. Contiguous columns still leave no
  blank pixels, which is why a smooth distribution looks fine — but an isolated spike silently
  vanishes (a flat-colour test image lost exactly one of its three channel spikes). Draw one column
  per device pixel taking the **max** of the bins that map to it, and snap any lone 1px bar (the
  over-range spike) to a whole pixel.
- **egui widgets CAN be tested — drive synthetic input through a headless `Context`.** The
  borderless window is invisible to computer-use, so pointer gestures used to be untestable and got
  "fixed" twice by reasoning alone. `ui::overlay::tests::drag_levels` is the pattern: build an
  `egui::Context::default()`, call `ctx.run` once per frame with `RawInput { screen_rect, time,
  events }`, and assert on the `UiAction`s that come out. Two gotchas the harness encodes: a press
  is hit-tested against the **previous** frame's widget rects, so the widget needs ~2 hover frames
  before a press is credited to it; and a drag only starts on the frame the pointer passes egui's
  click threshold, which is exactly the lag that broke the levels handles.
- **`Response::interact_pointer_pos()` is the pointer's position NOW, not where it was pressed.**
  Despite the name and doc wording it's `pointer.interact_pos()`. Because `drag_started()` fires
  several points into the movement, using it to decide *what* a drag grabbed picks whatever the
  pointer already moved onto. Use `ctx.input(|i| i.pointer.press_origin())` for that.
- **Don't use `ui.available_width()` inside the metadata box.** It's an auto-sizing `egui::Area`,
  so available width reflects how wide the box already is; padding it out feeds straight back into
  the box's width and balloons it every frame (the histogram header did this — a 208pt graph in a
  600pt box). Lay out against a width you computed yourself, e.g.
  `allocate_ui_with_layout(vec2(known_w, h), Layout::right_to_left(..))`.
- **The histogram's sample budget is benchmarked, not guessed** — `IMGVWR_DEBUG_HIST_BENCH=1`
  sweeps budgets × {point, mip} and prints GPU ms, VRAM-shaped grid dims, and how many of the 256
  bins each one resolves. On a 24k EXR (302 Mpx): cost is *flat* below ~8M (0.69 ms at 1M, 0.82 ms
  at 8M — the pass is overhead-bound down there), then 2.3 ms at 32M, 15.8 ms to read every pixel
  (plus 2.4 GB for the target). Hence `SAMPLES_PER_PASS = 8M`. There is deliberately **no user
  setting** for it: with progressive refinement every value converges on the same exact answer, so
  it only picks how the work is sliced, and 8M measured best on both axes (lowest total work *and*
  sub-millisecond frames) — a smaller value is strictly worse, needing more passes, more total
  work, and risking the pass cap.
- **The histogram refines progressively, and the half-pixel phase offset is load-bearing.** A big
  image is measured a strided slice per frame; `dispatch(accumulate)` just skips zeroing the SSBO
  (the shader only `atomicAdd`s), and the offset rides the existing 2D pan — `pan_u = yaw/2π`, so
  `yaw = 2π·k/width` shifts every sample by *k* source pixels, wrapping via `wrap_2d`. With an
  integer stride S the S² phases tile the image and the accumulation is *bit-identical* to reading
  every pixel (verified: 36 × 4096×2048 = 301,989,888 samples, zero delta vs a full-res pass).
  **But the grid lands at `S·i + S/2`, an exact texel boundary for even S**, where `GL_NEAREST`'s
  floor tips either way on float error — visiting some pixels twice and others never. The offset is
  `(k + 0.5) − S/2` (`phase_px`) to land on texel *centres*; without the half pixel, totals look
  perfect while per-bin counts are off by ~0.16%.
- **"Every texel of mip N" was measured against the grid and rejected — but for the right reason.**
  The bench compares them at *matched sample counts* (render at exactly `dims >> N` and the implicit
  LOD lands on mip N with no blend, so one output texel = one mip texel). Mip has 100% spatial
  coverage and the grid does not, so mip genuinely never misses a feature — but each sample is a box
  average, and that *fabricates values the image doesn't contain* and inflates sparse ones. On a
  synthetic 4096×2048 with 682 known blown pixels (0.00813%): the grid reports 0.0084% at both
  6.25% and 1.56% coverage (unbiased), while mip reports 0.0153% and 0.0244% — 2×–3× too much,
  because a single blown pixel becomes one whole mip texel. On the real 24k it resolves fewer
  distinct bins at *every* matched level (247 vs 254 at 4.7M). Mip *is* 2–5× faster at low sample
  counts (cache locality — the grid strides badly at LOD 0), which is the one thing worth
  remembering if a cheap continuous readout is ever needed. **Detection of tiny clipped regions is
  not the histogram's job**: the `C` overlay already does it exactly, with a *max*-mipped mask
  (`build_clip_mask`), which never dilutes and never fabricates.
- **The histogram pass must point-sample, and must see identity levels.** It sets
  `RenderParams::point_sample` (plain `GL_NEAREST`, *not* the I key's `NEAREST_MIPMAP_NEAREST`,
  which still picks a pre-averaged mip and would smooth away the clipped highlights the graph
  exists to report), and `levels: [0.0, 1.0]` so the graph describes the values *entering* the
  levels adjustment — otherwise it chases its own handles. It also forces `wrap_2d`: the integer
  sample grid can only approximate the image aspect, and without it the shader's out-of-bounds
  branch returns transparent black and fabricates a spike in bin 0.
- **Clip overlay composites over `clamp(color, 0, 1)`, NOT the raw display colour.** Clipped regions
  blow past 1.0, so `0.25 * huge` re-clips to white and makes the stripes invisible on float formats.
  This was a silent no-op on EXR/HDR (8-bit looked fine only because its clipped display value is
  already 1.0). Also: 16-bit-half EXR clips at `65504` (`HALF_MAX`), not 1.0 — integer 16-bit
  PNG/TIFF uses 1.0 (`CLIP_MAX_NORM`); they are different clip thresholds.
- **The live two-sequences diff must sample A and B with the *same* operation, or identical
  frames don't cancel.** Alt+N between two playing sequences diffs on the GPU: sequence A is the
  ring texture, B rides in the diff texture (`upload_diff_frame`, reused frame-to-frame), and the
  shader does `abs(A − B)` when `u_diff_live`. The trap: the display path samples A through the
  *Lanczos* minifier and the ring's sampler carries REPEAT-wrap + anisotropy, while the diff texture
  was plain trilinear/CLAMP/no-aniso — so a self-diff traced faint edges instead of pure black. Fix
  is twofold: the shader re-samples A via `sample_image_grad` (plain `textureGrad`, matching B) in
  the live branch instead of reusing the Lanczos `texel`; and `refresh_diff_texture` re-matches the
  ring's wrap+anisotropy on B. Verified: a fit-view self-diff is exactly 0 (`IMGVWR_DEBUG_SEQ_DIFF=1`).
  Only the *live* path re-matches — the still precompute keeps `upload_diff_texture`'s defaults. The
  live diff is deliberately not suspended while playing (unlike the CPU precompute); `sync_diff_with_playback`
  early-returns when `diff_playback` is set.
- **Out-of-band adopts (slot recall, drop, open dialog) must bump `load_gen` and clear
  `nav_pending` before `begin_adopt`.** Otherwise a still-running nav decode's result passes
  `poll_loads`'s gen check and clobbers the recall.
- **`DragWindow` and `ToggleMaximize` must call `set_fullscreen(false)` first when fullscreen.**
  That's the only call that clears both winit's `Fullscreen(Borderless)` and `self.fullscreen`
  together. Skipping it leaves OS window state and app state disagreeing.
- **The window is created hidden (`with_visible(false)`) to kill the startup flash.** The sequence
  in `resumed` is: create gfx → `rebuild_ocio` → `load_initial_image` → render one frame →
  `set_visible(true)`. Don't move `set_visible` before the first `render()` and don't drop
  `with_visible(false)` — either reintroduces the flash.
- **Default-app registration uses one ProgID *per extension*, never a shared one.** Explorer's Type
  column reads the (default) value of the extension's ProgID, so a single shared `imgvwr.Image` made
  every type show "imgvwr Image" and broke sort-by-type. `register_default_app` writes
  `imgvwr.<ext>` each with its own `friendly_type_name(ext)` (RAW share a "Camera RAW Image (…)"
  prefix so they cluster yet stay per-brand). It also deletes the legacy `imgvwr.Image`. The button
  can only set the *classic* default (`.ext\(default)` + `OpenWithProgids`); a Windows-hashed
  `UserChoice` overrides it, so types the user already defaulted elsewhere keep their old type until
  re-picked in Settings (that's why `.cr2`/`.dng` may still show "CR2 File"). Run the exact code
  path headlessly with `imgvwr.exe --register-default-app`; verify the resulting Type strings with
  `SHGetFileInfo(..., SHGFI_TYPENAME | SHGFI_USEFILEATTRIBUTES)` from a fresh process (the shell
  caches associations per-process).
- **Sequence playback: the frame advance must never touch the adopt pipeline.** `load_path` →
  `begin_adopt` → `finalize_adopt` resets the camera, re-runs panorama detection, re-frames the
  window, auto-exposes, clears undo and re-runs OCIO view selection — all wrong (and slow) 24 times
  a second. `show_playback_frame` swaps the texture and does *only* the bookkeeping that depends on
  which pixels are showing. Consequences that fall out of that and are easy to break:
  - The ring holds `ImageTextureKind::RingFrame`, a **borrowed** texture name. `ImageTexture::delete`
    must ignore it, and `end_sequence` must re-upload the kept frame *before* freeing the ring.
  - The frame to keep on Stop is `pb.shown`, not the playhead — a seek can leave the playhead on a
    frame that has not decoded, and keeping that one deletes the texture still being drawn from.
  - `invalidate_histogram()` per frame changes the epoch every frame, so the landing check that
    normally discards a stale measurement would discard *every* one. While playing, a one-frame-old
    graph is the point; the check relaxes (see `render`).
  - Rotation is remembered against `seq.identity()`, not the frame path — the frame path changes
    under it and `↑`/`↓` would appear to do nothing.
  - An animation's frames all share one `ImageData` whose own buffer mirrors frame 0, so stopping on
    frame *n* has to lift frame *n*'s pixels into a still (`FrameSource::still_of`).
- **Panorama detection is format- + content-gated, not just 2:1 aspect.** `ImageData::is_equirectangular()`
  gates in order: `can_be_panorama(path)` (float HDR only — `PANO_EXTS` = exr/hdr/pic; 8-bit/LDR is
  *always* 2D by deliberate choice, since 360 JPEGs exist but the user wants them flat by default,
  `P` to override) → 2:1 aspect → content. The free `is_equirectangular(w,h)` is the aspect gate
  *only* — it's all that's available pre-decode (window sizing at `resumed`), and it stays aspect-only
  on purpose. The content stage runs `equirect_content_scores`: a true equirect collapses its top/bottom rows to single points
  (near-constant `pole_top`/`pole_bottom`) and wraps seamlessly L↔R (`wrap`), all measured on
  Reinhard-tonemapped edge samples so HDR highlights can't dominate. The scan is ~15µs and
  resolution-independent (≤512 samples/edge). Real HDRIs score poles ~0.000–0.003; ordinary 2:1
  renders ~0.03–0.07 — thresholds sit in the gap (`POLE_FLAT_MAX`/`WRAP_SEAM_MAX` in `mod.rs`). A
  2:1 image now classed 2D still fits its window (post-decode `resize_window_to_image` at the
  `!want_pano` branch of `finalize_adopt`).
