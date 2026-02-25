from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np

from hdri_viewer.color.ocio_manager import OcioShader, OcioTexture2D, OcioTexture3D
from hdri_viewer.io.image_loader import ImageData
from hdri_viewer.viewer.camera import CameraState


@dataclass(slots=True)
class RenderState:
    """Current renderer state for uniform and resource tracking."""

    exposure_stops: float = 0.0
    viewport_width: int = 1
    viewport_height: int = 1


class PanoramaRenderer:
    """OpenGL renderer responsible for GPU sampling and OCIO display conversion."""

    def __init__(self, shaders_dir: Path) -> None:
        self._shaders_dir = shaders_dir
        self._ctx: Any | None = None
        self._program: Any | None = None
        self._vao: Any | None = None
        self._vbo: Any | None = None
        self._texture: Any | None = None
        self._state = RenderState()
        self._uniform_cache: dict[str, float] = {}
        self._fragment_template: str = ""
        self._vertex_source: str = ""
        self._ocio_shader = OcioShader(shader_text="", function_name="")
        self._shader_cache_key: tuple[str, str, str] | None = None
        self._projection_2d_enabled = False
        self._image_aspect = 1.0
        self._ocio_lut_textures: list[tuple[Any, int]] = []

    @property
    def has_texture(self) -> bool:
        """Returns whether an image texture is currently available."""

        return self._texture is not None

    @property
    def image_aspect(self) -> float:
        """Returns currently loaded image aspect ratio (width / height)."""

        return self._image_aspect

    def initialize(self) -> None:
        """Initializes ModernGL resources from the current OpenGL context."""

        import moderngl

        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.BLEND)

        self._fragment_template = (self._shaders_dir / "fragment_template.glsl").read_text(encoding="utf-8")
        self._vertex_source = (self._shaders_dir / "vertex.glsl").read_text(encoding="utf-8")

        vertices = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(vertices.tobytes())
        self._rebuild_program_if_needed(force=True)

    def update_ocio_shader(self, ocio_shader: OcioShader) -> None:
        """Updates OCIO shader snippet and rebuilds GLSL program when required."""

        self._ocio_shader = ocio_shader
        self._rebuild_program_if_needed(force=False)

    def set_exposure(self, exposure_stops: float) -> None:
        """Updates linear exposure in stops."""

        self._state.exposure_stops = exposure_stops

    def set_projection_2d_enabled(self, enabled: bool) -> None:
        """Sets whether rendering uses 2D UV pan/zoom instead of equirectangular projection."""

        self._projection_2d_enabled = enabled

    def set_viewport(self, width: int, height: int) -> None:
        """Updates viewport dimensions for rendering and aspect ratio calculations."""

        self._state.viewport_width = max(width, 1)
        self._state.viewport_height = max(height, 1)
        if self._ctx is not None:
            self._ctx.viewport = (0, 0, self._state.viewport_width, self._state.viewport_height)

    def set_image(self, image_data: ImageData) -> None:
        """Uploads a float32 RGB texture without mipmaps for HDR display."""

        if self._ctx is None:
            raise RuntimeError("Renderer not initialized.")

        if self._texture is not None:
            self._texture.release()

        texture = self._ctx.texture(
            size=(image_data.width, image_data.height),
            components=3,
            data=image_data.pixels.tobytes(),
            dtype="f4",
        )

        texture.repeat_x = True
        texture.repeat_y = False
        texture.filter = (self._ctx.LINEAR, self._ctx.LINEAR)
        texture.use(location=0)
        self._texture = texture
        self._image_aspect = float(image_data.width) / max(float(image_data.height), 1.0)

    def render(self, camera: CameraState) -> None:
        """Renders the panorama using current camera, exposure, and OCIO transform."""

        if self._ctx is None or self._program is None or self._vao is None:
            return

        framebuffer = self._ctx.detect_framebuffer()
        framebuffer.use()
        self._ctx.viewport = (0, 0, self._state.viewport_width, self._state.viewport_height)
        self._ctx.clear(0.02, 0.02, 0.02, 1.0)
        if self._texture is None:
            return

        try:
            self._program["u_image"].value = 0
        except KeyError:
            pass

        aspect = self._state.viewport_width / self._state.viewport_height
        tan_half_fov = math.tan(math.radians(camera.fov_degrees) * 0.5)

        self._set_uniform_if_changed("u_aspect", float(aspect))
        self._set_uniform_if_changed("u_tan_half_fov", float(tan_half_fov))
        self._set_uniform_if_changed("u_yaw", float(camera.yaw_radians))
        self._set_uniform_if_changed("u_pitch", float(camera.pitch_radians))
        self._set_uniform_if_changed("u_exposure", float(self._state.exposure_stops))
        self._set_uniform_if_changed("u_projection_mode", float(1.0 if self._projection_2d_enabled else 0.0))
        self._set_uniform_if_changed("u_image_aspect", float(self._image_aspect))

        for texture, binding_index in self._ocio_lut_textures:
            texture.use(location=binding_index)

        self._texture.use(location=0)
        self._vao.render(mode=self._ctx.TRIANGLE_STRIP)

    def _set_uniform_if_changed(self, name: str, value: float) -> None:
        """Writes a program uniform only when its value changed."""

        if self._program is None:
            return

        try:
            uniform = self._program[name]
        except KeyError:
            return

        cached_value = self._uniform_cache.get(name)
        if cached_value is not None and abs(cached_value - value) < 1e-10:
            return

        uniform.value = value
        self._uniform_cache[name] = value

    def _rebuild_program_if_needed(self, force: bool) -> None:
        """Recompiles shader only when injected OCIO content changes."""

        if self._ctx is None:
            return

        key = (self._ocio_shader.shader_text, self._ocio_shader.function_name)
        key = (self._ocio_shader.shader_text, self._ocio_shader.function_name, self._ocio_shader.signature)
        if not force and self._shader_cache_key == key:
            return

        declarations = self._ocio_shader.shader_text
        if self._ocio_shader.function_name:
            ocio_apply = (
                f"vec4 ocio_rgba = {self._ocio_shader.function_name}(vec4(color, 1.0));\n" "    color = ocio_rgba.rgb;"
            )
        else:
            ocio_apply = "color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));"

        fragment_source = self._fragment_template.replace("__OCIO_DECLARATIONS__", declarations)
        fragment_source = fragment_source.replace("__OCIO_APPLY__", ocio_apply)

        if self._program is not None:
            self._program.release()

        self._release_ocio_lut_textures()

        self._program = self._ctx.program(
            vertex_shader=self._vertex_source,
            fragment_shader=fragment_source,
        )

        self._bind_ocio_lut_textures()

        self._vao = self._ctx.vertex_array(self._program, [(self._vbo, "2f", "in_position")])
        self._uniform_cache.clear()
        self._shader_cache_key = key

    def _release_ocio_lut_textures(self) -> None:
        """Releases OCIO-generated LUT textures bound to the current program."""

        for texture, _ in self._ocio_lut_textures:
            try:
                texture.release()
            except Exception:
                pass
        self._ocio_lut_textures.clear()

    def _bind_ocio_lut_textures(self) -> None:
        """Uploads OCIO LUT textures and binds corresponding sampler uniforms."""

        if self._ctx is None or self._program is None:
            return

        for descriptor in self._ocio_shader.textures_2d:
            self._bind_ocio_2d_texture(descriptor)

        for descriptor in self._ocio_shader.textures_3d:
            self._bind_ocio_3d_texture(descriptor)

    def _bind_ocio_2d_texture(self, descriptor: OcioTexture2D) -> None:
        """Uploads one OCIO 2D LUT texture and links it to a sampler uniform."""

        if self._ctx is None or self._program is None:
            return

        texture = self._ctx.texture(
            size=(descriptor.width, descriptor.height),
            components=descriptor.components,
            data=descriptor.values,
            dtype="f4",
        )

        interpolation = descriptor.interpolation.upper()
        if "NEAREST" in interpolation:
            texture.filter = (self._ctx.NEAREST, self._ctx.NEAREST)
        else:
            texture.filter = (self._ctx.LINEAR, self._ctx.LINEAR)
        texture.repeat_x = False
        if hasattr(texture, "repeat_y"):
            texture.repeat_y = False
        texture.use(location=descriptor.binding_index)
        self._ocio_lut_textures.append((texture, descriptor.binding_index))

        try:
            self._program[descriptor.sampler_name].value = descriptor.binding_index
        except KeyError:
            pass

    def _bind_ocio_3d_texture(self, descriptor: OcioTexture3D) -> None:
        """Uploads one OCIO 3D LUT texture and links it to a sampler uniform."""

        if self._ctx is None or self._program is None:
            return

        texture = self._ctx.texture3d(
            size=(descriptor.edge_len, descriptor.edge_len, descriptor.edge_len),
            components=descriptor.components,
            data=descriptor.values,
            dtype="f4",
        )

        interpolation = descriptor.interpolation.upper()
        if "NEAREST" in interpolation:
            texture.filter = (self._ctx.NEAREST, self._ctx.NEAREST)
        else:
            texture.filter = (self._ctx.LINEAR, self._ctx.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        texture.repeat_z = False
        texture.use(location=descriptor.binding_index)
        self._ocio_lut_textures.append((texture, descriptor.binding_index))

        try:
            self._program[descriptor.sampler_name].value = descriptor.binding_index
        except KeyError:
            pass
