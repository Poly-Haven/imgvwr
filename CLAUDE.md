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
