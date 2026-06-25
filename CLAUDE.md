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
- The fragment shader is templated — `__IMAGE_SAMPLER__` / `__OCIO_*__` are substituted in
  `renderer/mod.rs`. The panorama path must sample via `sample_image_grad` (seam-corrected
  derivatives), never plain `sample_image`, or the longitude wrap shows a mip-LOD seam.
- Headless verification (no visible window needed): `IMGVWR_CAPTURE=out.png` renders one frame
  to a PNG; `IMGVWR_DEBUG_*` env vars (debug builds only) force camera/exposure/projection state.
- The camera is an enum (`Pano` | `Flat`) — keep the two states distinct; convert through
  `center_uv` on the P-toggle rather than reusing fields.
- rawler's `raw_metadata` is brittle (rejects some decodable files); raw EXIF orientation is read
  directly from the TIFF tag instead (`read_tiff_orientation` in `image_loader/formats.rs`).
