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
- **UI icons: prefer the *filled* Bootstrap variant** (`*-fill.svg`). At titlebar/HUD sizes the
  hairline outline strokes render muddy; filled glyphs stay crisp.
- **Headless verification:** `IMGVWR_CAPTURE=out.png` renders one frame to a PNG; `IMGVWR_DEBUG_*`
  env vars (debug builds only) force state — e.g. `IMGVWR_DEBUG_ZOOM`, `IMGVWR_DEBUG_EXPOSURE`,
  `IMGVWR_DEBUG_PROJECTION=pano`, `IMGVWR_DEBUG_OVERLAY=settings|metadata|error|loading|hint|help`,
  `IMGVWR_DEBUG_BG=black|checker|white|user`,
  `IMGVWR_DEBUG_LEVELS=black,white`. `RUST_LOG=debug` also dumps each landed histogram as a
  16-bucket digest, which is the only way to check the *numbers* rather than the picture.
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
- **Bar charts narrower than a pixel drop bars.** The F2 histogram has 256 bins in ~205 points, so
  a per-bin quad is ~0.8px wide; `Shape::mesh` goes straight to the GPU with no anti-aliasing, and
  a quad containing no pixel *centre* rasterises to nothing. Contiguous columns still leave no
  blank pixels, which is why a smooth distribution looks fine — but an isolated spike silently
  vanishes (a flat-colour test image lost exactly one of its three channel spikes). Draw one column
  per device pixel taking the **max** of the bins that map to it, and snap any lone 1px bar (the
  over-range spike) to a whole pixel.
- **Don't use `ui.available_width()` inside the metadata box.** It's an auto-sizing `egui::Area`,
  so available width reflects how wide the box already is; padding it out feeds straight back into
  the box's width and balloons it every frame (the histogram header did this — a 208pt graph in a
  600pt box). Lay out against a width you computed yourself, e.g.
  `allocate_ui_with_layout(vec2(known_w, h), Layout::right_to_left(..))`.
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
