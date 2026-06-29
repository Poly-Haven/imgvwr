# imgvwr

A minimal, GPU-accelerated image viewer for HDR panoramas and standard images,
for Windows.

- **Two display modes:** 2D pan/zoom for ordinary images; rectilinear
  equirectangular projection for panoramas. Auto-detects panoramas
  (`width == height * 2`) and opens them in panorama mode. Press **P** to toggle.
- **Full-quality HDR:** very high resolution (24k+) 32-bit float images are
  uploaded at full bit depth, tiled across multiple GPU textures when they exceed
  `GL_MAX_TEXTURE_SIZE`. If the GPU runs out of memory the viewer frees what it
  can and retries, and reports a clear message rather than showing a blank frame;
  an optional *Store 32-bit float as 16-bit* setting halves the VRAM such images
  use.
- **High-quality downscaling:** 8-bit images are minified with a separable
  Lanczos-3 filter (sharper than bilinear, with mip pre-filtering so it stays
  alias-free at any zoom-out); higher bit depths and all upscaling use bilinear.
- **OCIO v2 colour management:** bundled Blender/AgX config, with AgX/Filmic and
  other view transforms selectable from the metadata box's View dropdown.
- **Accurate camera RAW:** RAW photos are developed (demosaiced) to scene-linear
  with a **linear camera response** — camera white balance and the camera→sRGB
  matrix, but *no* tone curve — so you see the sensor data as it is, not a
  "make it pretty" rendering. White maps to 1.0 and highlights keep their
  headroom, so lowering the exposure (or a Filmic/ACES view transform) recovers
  detail in regions that first looked clipped.
- **Review tools:** per-channel isolation, a **clipping overlay** (`C`) that marks
  at-/near-max pixels with animated per-channel stripes (judged on the original
  data, not the adjusted view), an auto-exposure pick for HDR panoramas, and a GPU
  **Clarity** (local-contrast) filter that can be cranked well past photographic
  levels to make issues pop.
- **Minimal UI:** the image *is* the window; a borderless titlebar (Open,
  Settings, window controls) reveals at the top edge, an adjustment-slider panel
  at the bottom edge, and the metadata box at the top-right — all auto-hiding.

## Controls

### View & zoom

| Input | Action |
|---|---|
| Left / middle mouse drag | Pan (2D) or look around (panorama) |
| Mouse wheel | Zoom (2D, toward the cursor) or FOV (panorama) |
| `Shift` / `Ctrl` + wheel | Pan horizontally / vertically |
| Shift / Ctrl + drag | Lock pan/look to the horizontal / vertical axis |
| Numpad `1`–`9` | Zoom in `2^(N-1)×` (1=100%, 2=200%, 3=400%…); with `Ctrl`, zoom out |
| `Home` / `R` | Reset the view and window to the default |
| `Backspace` | Centre the image and fit the window to it at the current zoom |
| `↑` / `↓` | Rotate the image 90° (counter-clockwise / clockwise); remembered per image for the session |
| `P` | Toggle 2D ↔ panorama mode |
| `W` | Toggle tiled wrap (2D) |
| `I` | Pin nearest ↔ bilinear filtering (by default it's automatic: a 2D image magnified past 200% samples nearest for crisp pixels, everything else bilinear; `I` overrides that and persists) |
| `M` | Toggle the navigation minimap (auto-shows on 2D pan/zoom; click/drag it to jump) |
| `B` | Cycle the background backdrop (your setting → black → grey checkerboard → white) |

### Tone & files

| Input | Action |
|---|---|
| `,` / `.` | Exposure −/+ 0.5 stops (hold to ramp) |
| `Ctrl + ,` / `Ctrl + .` | Gamma −/+ 0.1 (hold to ramp) |
| `[` / `]` | Clarity (local contrast) radius −/+ (hold to ramp) |
| `;` / `'` | Clarity strength −/+ (0 = off; crank high to make issues pop; hold to ramp) |
| `Ctrl + R` | Reset all adjustments (exposure, gamma, clarity, channel, sharpness, diff) |
| `Ctrl + Z` / `Ctrl + Shift + Z` (or `Ctrl + Y`) | Undo / redo edits — guides, adjustments and toggle modes (not navigation); up to 256 steps, per image |
| `T` | Toggle Standard ↔ the Default View Transform (set in Settings; defaults to Filmic) |
| `O` | Open file… |
| `←` / `→` | Previous / next image in the folder (alphabetical) |
| `Space` | Pause / play an animation (GIF / animated WebP / APNG) |
| `F2` | Toggle metadata overlay (also appears on top-right hover) |

### Inspect / review

| Input | Action |
|---|---|
| `S` | Sharpness checker — `\|original − blurred\|`, amplifiable by exposure |
| `C` | Clipping overlay — animated diagonal stripes over regions at/near the format max, judged on the **original** data (pre-adjustment); per channel (red stripes = red clipped, white = all three), margin configurable in Settings |
| `G` | Add the next guide *level* — density doubles each press (½, then ¼, ⅛ … down to 1/32 on each axis) |
| Pull from a ruler | Drag a guide out of a ruler — left ruler → vertical, bottom ruler → horizontal (2D pixels or panorama degrees) |
| Drag / right-click a guide | Grab a guide to move it, or delete it (drag off the image, or right-click) — 2D and panorama |
| `Alt` + middle-drag | Squash / stretch the image non-uniformly, unbounded (line straightness) |
| Channel boxes (`F2`) | Click to isolate R / G / B / A as greyscale |
| Guides list (`F2`) | Shows each guide (`V 425px` / `H 312px`); × removes; `Ctrl+R` resets all |

### Window (borderless)

| Input | Action |
|---|---|
| `Alt` + drag · titlebar drag · 2D-fit body drag | Move the window |
| Drag a window edge / corner | Resize |
| `Alt` + right-drag | Resize; edge(s) chosen by the cursor's third of the window |
| `Alt` + scroll | Grow / shrink the window (both modes; zooms the whole view) |
| `A` | Toggle always-on-top |
| `F` / `F11` / double-click | Toggle fullscreen (2D images fit the screen but smaller-than-screen ones show at 1:1; the cursor auto-hides when idle) |
| `Escape` / `Q` | Exit fullscreen or close |
| Move cursor to top edge | Show the titlebar (Open, Settings, window controls) — also in fullscreen; dragging it there exits fullscreen and moves the window |
| Move cursor to bottom edge | Show the adjustment sliders panel (Exposure, Gamma) |

### Comparator & help

| Input | Action |
|---|---|
| `Ctrl` + `1`–`9` | Save the current image to a comparator slot |
| `1`–`9` (top row) | Recall a saved slot; press again to toggle back |
| `Alt` + `1`–`9` | Show the difference vs that slot's image (processing applies to the diff) |
| `L` | Lock zoom/pan across images |
| `H` | Show keyboard / mouse help (with version + GitHub link) |

Arrow-key navigation pre-decodes the next image in the background so stepping
through a folder is instant. **L** locks the current zoom/pan (and exposure) so
they carry across images of the same kind for side-by-side comparison.

**Comparator:** `Ctrl`+`1`…`9` pins the current image in a numbered slot (kept
in memory across navigation). Pressing the bare number recalls it for an instant
A/B comparison — preserving the view and showing each image at its native
resolution (different-sized images are not scaled to match). Pressing the same
number again flips back to what you were viewing. Saved slots appear as small
numbered flags at the top-right; the active one is highlighted.

## Supported formats

JPEG, PNG / APNG, BMP, TIFF, WebP, GIF, ICO, TGA, PNM, Radiance HDR, OpenEXR, and
camera RAW: NEF, CR2, CR3, ARW, DNG, RAF, ORF, RW2, PEF, and similar.

Camera RAW is developed with **LibRaw** to scene-linear float (demosaic + camera
white balance + camera→sRGB matrix, linear response — no tone curve), matching
the HDR/EXR pipeline so exposure and view transforms behave the same. RAW photos
do **not** auto-expose by default (the develop already respects the real
exposure); enable *Auto-expose RAW photos on open* in Settings to brighten dark
shots automatically. The F2 box shows camera, lens, ISO, shutter, aperture and
focal length when the file records them. (Without the `ocio` feature, RAW falls
back to a best-effort pure-Rust decode.)

**Animation:** GIF, animated WebP, and APNG (animated PNG) play automatically and
loop; press `Space` to pause/resume.

## Usage

```
imgvwr [path-to-image]
```

Open an image from the command line, by dragging it onto the window, the `O`
key, or the **📂** button in the titlebar (revealed at the top edge).

## Building from source

Target platform is **Windows only** (x64).

### Host prerequisites

- **Rust** (stable, MSVC toolchain).
- **MSVC C++ build tools** (Visual Studio 2019/2022 or the standalone Build
  Tools) — used by the `cc` crate to compile the OCIO / OpenEXR / LibRaw shims.
- **LLVM/Clang on `PATH`** (provides `libclang`) — required by `bindgen`.
- **vcpkg** providing OpenColorIO, lcms, OpenEXR, and LibRaw. Either workflow
  works:

  *Classic mode* (set `VCPKG_ROOT`):
  ```bat
  git clone https://github.com/microsoft/vcpkg %USERPROFILE%\vcpkg
  %USERPROFILE%\vcpkg\bootstrap-vcpkg.bat
  setx VCPKG_ROOT %USERPROFILE%\vcpkg
  %VCPKG_ROOT%\vcpkg install opencolorio lcms openexr libraw --triplet x64-windows
  ```

  *Manifest mode* (uses `vcpkg.json`; no `VCPKG_ROOT` needed):
  ```bat
  %VCPKG_ROOT%\vcpkg install --triplet x64-windows   :: run from the project root
  ```

  `build.rs` looks for the libraries in `./vcpkg_installed` (manifest) first, then
  `%VCPKG_ROOT%\installed` (classic). `openexr` provides the fallback decoder for
  DWAA/DWAB-compressed EXRs; `lcms` powers ICC-profile conversion (JPEGs/PNGs with
  a non-sRGB embedded profile are converted to sRGB on load); `libraw` develops
  camera RAW files to scene-linear float.

### Packaging & releasing

`scripts\package.ps1` bundles `imgvwr.exe`, the full runtime DLL closure, the
`resources/` directory, and the licenses into a self-contained
`dist\imgvwr-<version>-windows-x64.zip`.

Releases are built **locally** rather than in CI — the native deps
(OpenColorIO / OpenEXR) compile from source via vcpkg, which takes 20+ minutes
on a clean GitHub runner. From a shell with `VCPKG_ROOT` and `LIBCLANG_PATH`
set (see the prerequisites above):

```powershell
# Bump the version in Cargo.toml first, then:
pwsh scripts\release.ps1            # build + package only -> dist\
pwsh scripts\release.ps1 -Publish   # also tag vX.Y.Z and create the GitHub release (needs gh)
```

`-Publish` derives the tag from `Cargo.toml`, pushes it, and uploads the zip to
a new GitHub release via the `gh` CLI.

### Build

```bat
cargo build --release
```

For a quick development build **without** OCIO (gamma-2.2 fallback, no vcpkg /
LLVM required):

```bat
cargo build --no-default-features
```

> A gamma-only build is a development convenience only. Shipped release builds
> must include OCIO (this is enforced by `build.rs` when the `ocio` feature is
> enabled).

### Running the distributed build

The release zip is self-contained: extract it and run `imgvwr.exe`. No
prerequisites, vcpkg, or environment setup are required. Because the build is
unsigned, Windows SmartScreen may prompt on first launch — choose *More info →
Run anyway*.

## License

`imgvwr` is MIT licensed (see `LICENSE`). Bundled third-party libraries and OCIO
assets carry their own terms; see `THIRD_PARTY_LICENSES.md`.
