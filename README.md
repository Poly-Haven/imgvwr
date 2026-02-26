# imgvwr

`imgvwr` is a modern, minimal image viewer with support for not just normal image formats, but HDRI panoramas and camera raw files too.

## Controls

- Left or middle mouse drag: pan/rotate view
- Mouse wheel: zoom
- Ctrl + mouse wheel: smooth exposure change
- `,`: exposure -1 stop
- `.`: exposure +1 stop
- `Ctrl + ,`: gamma -0.1
- `Ctrl + .`: gamma +0.1
- Right-click: context menu (open/reload/view transform/file info)
- `P`: Toggle 2D projection and equirectangular
- `Home`: Reset zoom and pan to the image's original opened view
- `W`: In 2D mode, toggle tiled wrapping on/off (both axes, default: off)
- `F`: Toggle fisheye and rectilinear lens (rectilinear is default; disabled in 2D mode)
- `T`: Toggle view transform
- `F2`: Toggle image metadata overlay (top-left)
- Double-click: Full screen
- `Esc`: Exit

For panorama images, (2:1 aspect, hdr or exr) the last view transform used persists. For all others, Standard is always loaded by default.

---

# Development

- Python 3.11+
- PyQt6 windowing
- moderngl renderer
- OpenImageIO loading
- rawpy camera RAW decode
- OpenColorIO v2 display transforms

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

## Run

```bash
imgvwr
# or open directly
imgvwr path/to/file.exr
```

## Tests

```bash
pytest
```

## Type checking

```bash
mypy hdri_viewer tests
```

## Formatting

```bash
black .
```

## OCIO configs

- Default config: `hdri_viewer/resources/config.ocio`
- Custom config folder: `hdri_viewer/resources/ocio_configs`

Drop a `.ocio` file in the custom folder and choose **Reload** from the context menu.
