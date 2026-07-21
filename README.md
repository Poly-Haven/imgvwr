# imgvwr

A minimal, GPU-accelerated image viewer for HDR panoramas and standard images.

![screenshot](https://u.polyhaven.org/PAb/2026-06-29_13-45-18.jpg)

Oh, were you looking for a screenshot of imgvwr? Sure, here's one:

![screenshot2](https://u.polyhaven.org/MmH/2026-06-29_16-10-46.jpg)

That's it. That's the whole thing.

I told you it was minimal, there isn't even a titlebar - until you need it.

imgvwr was built around three paradigms:

1. The image is the window.
2. Nothing gets in your way.
3. But it can do everything you need an image viewer to do.

At Poly Haven we look at images all day, it's kind of our whole job. I was sick of juggling different programs for different purposes and built one that fits all my needs:

> ### ⬇️ [Download now](https://github.com/Poly-Haven/imgvwr/releases)

## Viewing HDRIs

Equirectangular HDR panoramas (`.exr` / `.hdr`) are automatically detected and displayed in an interactive 3D view. Detection looks past the 2:1 aspect ratio at the actual pixels — an equirectangular projection collapses its top and bottom rows to single points and wraps seamlessly left-to-right — so ordinary 2:1 HDRs stay in 2D. 8-bit images (JPEG, PNG, …) always open flat; press `P` to view any image as a panorama.

Left/middle click to pan around, scroll to zoom.

You can adjust the exposure (`,`&`.`) to see the full dynamic range of HDRIs.

It doesn't try to make *your images* look pretty, it just shows them to you as truthfully as possible. If you want to, you can enable OCIO view transforms (Filmic, ACES, AGX, etc) to make high-dynamic-range content easier on the eyes.

Press `P` to toggle between 3D mode and 2D mode and view the panorama as a plain image.

## Viewing Textures

The color space and bit-depth of your images are respected.

Toggle wrapping/tiling preview with `W`, pan infinitely far away.

If you want to inspect the same tiny region in multiple PBR maps, press `L` to lock the view and navigate with `←` and `→`, or open another image entirely.

Want to compare two images? Save them in a slot with `Ctrl`-`1-9`, and recall with `1-9` or click on the flags.

Want to see the mathematical difference between two images? save it in a slot and press `Alt`-`1-9` (same as *Difference* blending mode). Exposure controls still work if you want to amplify the difference.

## Viewing RAW Photos

Most image viewers will simply show you the embedded JPG/TIFF inside the RAW file for a fast preview. That's nice and fast, but I want to see the real data, the *actual* linear RAW image with only demosaicing and basic black/white point handling.

## Reviewing Images

One of my roles at Poly Haven is QC, I spend a lot of time inspecting pixels and looking for flaws. So in addition to the comparison tools above, I added a few features to help me find issues:

- **Clarity filter:** A large scale local contrast filter that can be cranked well past photographic
  levels to make issues pop.
- **Sharpness filter:** Not for *sharpening*, but for seeing *how* sharp an image is. `S` to toggle, darker regions are less sharp/detailed.
- **Clipping overlay:** Press `C` to highlight pixels that are near to the limit of what the image format can store (RGB~=256 for 8-bit images and RAW photos, 65536 for 16-bit float, etc). Note that this can't tell you if an HDRI has clipped lighting, but it can tell you if the source images you start with are clipped.
- **Guides:** Vertical and horizontal guides can be added for straight edge references. Helps you check your HDRIs are level. Press `G` to add them in automatic subdivided increments.
- **Metadata:** Hover near the top right or press `F2` to show the file metadata (bit depth, resolution, etc). Inspect channels individually.
- **Levels:** Two handles under the histogram set the display black and white points, stretching that part of the range back out to full. The graph is the control, not just a picture of it — grab either of its vertical edges to move that point, or anywhere else to slide the whole range. And because the graph is measured *before* the adjustment, it stays put while you drag: you're always reading the data you're cutting against, not a picture chasing your own handle.
- **Histogram:** The `F2` box also plots the tonal distribution of what's actually on screen — measured *after* exposure and the view transform, so it follows every adjustment you make. R/G/B are drawn as additive translucent areas (red + green reads yellow, all three read white), and a 1px spike on the right counts everything past the top of the displayable range. Lower the exposure and you'll watch that spike drain back into the graph. Isolate a channel and the graph follows it as a single white area, alpha included. Pick a Linear, square-root or logarithmic vertical scale with the `L` / `Sq` / `Log` buttons, and use the eye button to measure just the visible region instead of the whole image — zoom in for a reading of one area. A large image is measured a slice at a time over the following moment, so within about half a second the graph has counted *every* pixel — without ever costing a dropped frame.

## OCIO Color Management

This is the standard in VFX and digital content creation. Using the same system ensures that what we see is what we get further down the pipeline.

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
| `P` | Toggle 2D ↔ panorama mode |
| `W` | Toggle tiled wrap (2D) |
| `I` | Pin nearest ↔ bilinear filtering (by default it's automatic: a 2D image magnified past 200% samples nearest for crisp pixels, everything else bilinear; `I` overrides that and persists) |
| `M` | Toggle the navigation minimap (auto-shows on 2D pan/zoom; click/drag it to jump) |
| `B` | Cycle the background color (your setting → black → grey checkerboard → white) |
| `↑` / `↓` | Rotate the image 90° (counter-clockwise / clockwise); remembered per image for the session |

### Tone & files

| Input | Action |
|---|---|
| `,` / `.` | Exposure −/+ 0.5 stops (hold to ramp) |
| `Ctrl + ,` / `Ctrl + .` | Gamma −/+ 0.1 (hold to ramp) |
| `;` / `'` | Clarity strength −/+ (0 = off; crank high to make issues pop; hold to ramp) |
| `[` / `]` | Clarity (local contrast) radius −/+ (hold to ramp) |
| `Ctrl + R` | Reset all adjustments (exposure, gamma, levels, clarity, channel, sharpness, diff) |
| `Ctrl + Z` / `Ctrl + Shift + Z` (or `Ctrl + Y`) | Undo / redo edits — guides, adjustments and toggle modes (not navigation); up to 256 steps, per image |
| `T` | Toggle Standard ↔ the Default View Transform (set in Settings; defaults to Filmic) |
| `O` | Open file… |
| `Ctrl + C` | Copy the current window render to the clipboard — the displayed region at the current window size, adjustments and view transform baked in, guides/minimap/other UI included |
| `Delete` | Prompts, then permanently deletes the current file from disk and steps to the next image in the folder |
| `←` / `→` | Previous / next image in the folder (alphabetical) |
| `Space` | Pause / play an animation (GIF / animated WebP / APNG) |
| `F2` | Toggle metadata overlay (also appears on top-right hover) |

### Inspect / review

| Input | Action |
|---|---|
| `S` | Sharpness checker — `\|original − blurred\|`, amplifiable by exposure |
| `C` | Clipping overlay — animated diagonal stripes over regions at/near the format max, judged on the **original** data (pre-adjustment); per channel (red stripes = red clipped, white = all three), margin configurable in Settings |
| `G` | Show / hide the existing guides without touching them; adds the first one if there are none yet |
| `Shift+G` | Add the next guide *level* — density doubles each press (½, then ¼, ⅛ … down to 1/32 on each axis). In panorama, completing the first vertical level (180°) also drops its 0° partner so the pair bisects the sphere |
| `Ctrl+G` | Remove one guide level (undoes a `Shift+G` step) — aware of guides added/moved/removed by hand; falls back to clearing everything once no clean level remains |
| Pull from a ruler | Drag a guide out of a ruler — left ruler → vertical, bottom ruler → horizontal (2D pixels or panorama degrees) |
| Drag / right-click a guide | Grab a guide to move it, or delete it (drag off the image, or right-click) — 2D and panorama |
| `Ctrl` + drag a guide | Snap it to 10px increments (2D) or whole degrees (panorama), whether moving an existing guide or pulling a new one out of a ruler |
| `Alt` + middle-drag | Squash / stretch the image non-uniformly, unbounded (line straightness) |
| Channel boxes (`F2`) | Click to isolate R / G / B / A as greyscale |
| Histogram (`F2`) | Distribution of the *displayed* values (after exposure / view transform), as additive R/G/B areas — or one white area when a channel is isolated (alpha included). One bin per pixel column, 256 of them. The 1px spike at the right edge counts samples past the displayable maximum — drop the exposure to bring them back into range. `L` / `Sq` / `Log` pick the vertical scale; the eye button measures only the visible region, so you can zoom in for a focused reading |
| Levels (`F2`) | The two triangles under the histogram set the display black and white points, stretching that slice of the range back out to full. The graph itself is the control: drag either of its vertical edges to move that point, or drag anywhere else on the graph to slide both at once, keeping the range's width. Double-click a handle to reset it, the rest of the graph to reset both; `Ctrl+R` resets them along with the other adjustments |
| Guides list (`F2`) | Shows each guide (`V 425px` / `H 312px`, plus degrees in panorama); × removes; `Ctrl+R` resets all |
| Hover / drag a guide | Shows a colour-coded tooltip near the cursor with its coordinate — blue while hovering or grabbing, green while dragging a new one out of a ruler |
| Right-click + hold-drag | Colour-pick tooltip: swatch, pixel (and panorama degree) coordinates, and Linear / Display values under the cursor — R/G/B(/A) or L(/A) depending on the image's channels, coloured like the F2 box's channel swatches. Suppresses the other auto-hiding toolbars while active; a plain right-click still deletes a guide |

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

*Settings → Appearance → Titlebar → "Always show"* keeps the titlebar permanently
revealed instead of auto-hiding.

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

Camera RAW is slow since we decode and demosaic the image ourselves, skipping the usual embedded JPG/TIFF that other image viewers settle for.

RAW files are developed with **LibRaw** to scene-linear float (demosaic + camera
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

## License

`imgvwr` is MIT licensed (see `LICENSE`). Bundled third-party libraries and OCIO
assets carry their own terms; see `THIRD_PARTY_LICENSES.md`.
