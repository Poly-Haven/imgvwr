from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys


def _load_config(ocio: object, config_path: str | None):
    if config_path:
        path = Path(config_path)
        if path.exists():
            return ocio.Config.CreateFromFile(str(path))
    return ocio.Config.CreateRaw()


def _enumerate(config_path: str | None) -> dict[str, object]:
    import PyOpenColorIO as ocio

    config = _load_config(ocio, config_path)
    display_views: list[list[str]] = []
    for display in config.getDisplays():
        for view in config.getViews(display):
            display_views.append([str(display), str(view)])
    return {"display_views": display_views}


def _build_shader(config_path: str | None, display: str, view: str) -> dict[str, object]:
    import PyOpenColorIO as ocio

    config = _load_config(ocio, config_path)
    gpu_desc = ocio.GpuShaderDesc.CreateShaderDesc()
    gpu_desc.setLanguage(ocio.GPU_LANGUAGE_GLSL_4_0)
    gpu_desc.setFunctionName("ocio_display_transform")
    gpu_desc.setAllowTexture1D(False)

    processor = config.getProcessor(
        ocio.ROLE_SCENE_LINEAR,
        display,
        view,
        ocio.TRANSFORM_DIR_FORWARD,
    )
    processor.getDefaultGPUProcessor().extractGpuShaderInfo(gpu_desc)

    textures_2d: list[dict[str, object]] = []
    for texture in gpu_desc.getTextures():
        values = texture.getValues()
        values_b64 = base64.b64encode(values.astype("float32", copy=False).tobytes()).decode("ascii")
        textures_2d.append(
            {
                "sampler_name": str(texture.samplerName),
                "binding_index": int(texture.textureShaderBindingIndex),
                "width": int(texture.width),
                "height": int(texture.height),
                "channel": str(texture.channel),
                "interpolation": str(texture.interpolation),
                "values_b64": values_b64,
            }
        )

    textures_3d: list[dict[str, object]] = []
    for texture in gpu_desc.get3DTextures():
        values = texture.getValues()
        values_b64 = base64.b64encode(values.astype("float32", copy=False).tobytes()).decode("ascii")
        textures_3d.append(
            {
                "sampler_name": str(texture.samplerName),
                "binding_index": int(texture.textureShaderBindingIndex),
                "edge_len": int(texture.edgeLen),
                "interpolation": str(texture.interpolation),
                "values_b64": values_b64,
            }
        )

    return {
        "shader_text": str(gpu_desc.getShaderText()),
        "function_name": str(gpu_desc.getFunctionName()),
        "signature": f"{display}/{view}",
        "textures_2d": textures_2d,
        "textures_3d": textures_3d,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OCIO subprocess helper for imgvwr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enum_parser = subparsers.add_parser("enumerate")
    enum_parser.add_argument("--config", default="")

    shader_parser = subparsers.add_parser("shader")
    shader_parser.add_argument("--config", default="")
    shader_parser.add_argument("--display", required=True)
    shader_parser.add_argument("--view", required=True)

    args = parser.parse_args()
    try:
        if args.command == "enumerate":
            payload = _enumerate(args.config or None)
        else:
            payload = _build_shader(args.config or None, args.display, args.view)
        sys.stdout.write(json.dumps(payload))
        return 0
    except Exception as error:
        sys.stderr.write(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
