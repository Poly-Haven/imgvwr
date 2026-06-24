# imgvwr

A minimal, GPU-accelerated image viewer for HDR panoramas and standard images,
for Windows.

- **Two display modes:** 2D pan/zoom for ordinary images; rectilinear
  equirectangular projection for panoramas. Auto-detects panoramas
  (`width == height * 2`) and opens them in panorama mode. Press **P** to toggle.
- **Full-quality HDR:** very high resolution (24k+) 32-bit float images are
  uploaded at full bit depth, tiled across multiple GPU textures when they exceed
  `GL_MAX_TEXTURE_SIZE`.
- **OCIO v2 colour management:** bundled Blender/AgX config, with AgX/Filmic and
  other view transforms selectable from the toolbar.
- **Minimal UI:** the viewer fills the window; a semi-transparent toolbar hovers
  at the left edge.

## Controls

| Input | Action |
|---|---|
| Left / middle mouse drag | Pan (2D) or look around (panorama) |
| Mouse wheel | Zoom (2D) or FOV (panorama) |
| `Ctrl` + wheel | Exposure ±0.1 stops per notch |
| `,` | Exposure −1 stop |
| `.` | Exposure +1 stop |
| `Ctrl + ,` | Gamma −0.1 |
| `Ctrl + .` | Gamma +0.1 |
| `P` | Toggle 2D ↔ panorama mode |
| `W` | Toggle tiled wrap (2D) |
| `T` | Toggle Standard ↔ last-used view transform |
| `F2` | Toggle metadata overlay |
| `Home` | Reset view to state at load time |
| Move cursor to left edge | Show toolbar overlay |
| `F11` / double-click | Toggle fullscreen |
| `Escape` / `Q` | Exit fullscreen or close |

## Supported formats

JPEG, PNG, BMP, TIFF, WebP, GIF, ICO, TGA, PNM, Radiance HDR, OpenEXR, and
(best-effort) camera RAW: NEF, CR2, CR3, ARW, DNG, RAF, ORF, RW2, and similar.

## Usage

```
imgvwr [path-to-image]
```

Open an image from the command line, by dragging it onto the window, or via the
toolbar's **Open file…** button.

## Building from source

Target platform is **Windows only** (x64).

### Host prerequisites

- **Rust** (stable, MSVC toolchain).
- **MSVC C++ build tools** (Visual Studio 2019/2022 or the standalone Build
  Tools) — used by the `cc` crate to compile the OCIO shim.
- **LLVM/Clang on `PATH`** (provides `libclang`) — required by `bindgen`.
- **vcpkg** with `VCPKG_ROOT` set, with OpenColorIO and lcms installed:

  ```bat
  git clone https://github.com/microsoft/vcpkg %USERPROFILE%\vcpkg
  %USERPROFILE%\vcpkg\bootstrap-vcpkg.bat
  setx VCPKG_ROOT %USERPROFILE%\vcpkg
  %VCPKG_ROOT%\vcpkg install opencolorio lcms openexr --triplet x64-windows
  ```

  (`openexr` provides the fallback decoder for DWAA/DWAB-compressed EXRs.)

### Packaging

`scripts\package.ps1` bundles `imgvwr.exe`, the full runtime DLL closure, the
`resources/` directory, and the licenses into a self-contained
`dist\imgvwr-<version>-windows-x64.zip`. CI does this automatically on `v*` tags
(`.github/workflows/release.yml`).

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
