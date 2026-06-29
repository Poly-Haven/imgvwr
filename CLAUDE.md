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

## Conventions & lessons

- **Adding a user-facing feature?** Always update the help popup (`KEYS` in
  [`src/ui/overlay.rs`](src/ui/overlay.rs) `help_dialog`) **and** the README controls table.
- Run clippy on **both** feature sets before declaring done:
  `cargo clippy --all-targets -- -D warnings` and the same with `--no-default-features`.
- **UI icons: prefer the *filled* Bootstrap variant** (`*-fill.svg`). At titlebar/HUD sizes the
  hairline outline strokes render muddy; filled glyphs stay crisp.
- **Headless verification:** `IMGVWR_CAPTURE=out.png` renders one frame to a PNG; `IMGVWR_DEBUG_*`
  env vars (debug builds only) force state — e.g. `IMGVWR_DEBUG_ZOOM`, `IMGVWR_DEBUG_EXPOSURE`,
  `IMGVWR_DEBUG_PROJECTION=pano`, `IMGVWR_DEBUG_OVERLAY=settings|error`, `IMGVWR_DEBUG_BG=black|checker|white|user`.
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
