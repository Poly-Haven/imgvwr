#version 400 core

in vec2 v_uv;
out vec4 frag_color;

uniform float u_yaw;
uniform float u_pitch;
uniform float u_half_fov_radians;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_exposure;            // stops; applied in scene-linear BEFORE the view transform
uniform float u_gamma;               // output tweak; applied ONCE after the display transform (default 1.0)
uniform int   u_projection_mode;     // 0 = equirectangular panorama, 1 = 2-D pan/zoom
uniform float u_image_aspect;
uniform bool  u_input_is_encoded_srgb;  // true when source pixels are sRGB-encoded (JPEG / LDR PNG)
uniform bool  u_wrap_2d;             // 2-D mode: repeat the image instead of clamping

// Declares the image sampler(s) and `vec3 sample_image(vec2 uv)`.
// Single texture  -> returns texture(u_image, uv).rgb
// Tiled image     -> selects and samples the correct tile (§9.7)
__IMAGE_SAMPLER__

__OCIO_DECLARATIONS__

const float PI = 3.14159265358979323846;

mat3 rotation_yaw_pitch(float yaw, float pitch) {
    float cy = cos(yaw),  sy = sin(yaw);
    float cp = cos(pitch), sp = sin(pitch);
    mat3 my = mat3(cy, 0.0, sy,  0.0, 1.0, 0.0,  -sy, 0.0, cy);
    mat3 mp = mat3(1.0, 0.0, 0.0,  0.0, cp, -sp,  0.0, sp, cp);
    return my * mp;
}

vec2 direction_to_equirect_uv(vec3 dir) {
    float lon = atan(dir.z, dir.x);
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    float u = 1.0 - (lon / (2.0 * PI) + 0.5);
    float v = 0.5 - lat / PI;
    return vec2(u, v);
}

vec3 srgb_to_linear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92,
               pow((c + 0.055) / 1.055, vec3(2.4)),
               step(0.04045, c));
}

void main() {
    vec2 uv;

    if (u_projection_mode == 1) {
        // -- 2-D pan / zoom ----------------------------------------------
        float inv_zoom = max(u_tan_half_fov, 0.02);
        vec2 centered = v_uv - vec2(0.5);
        float pan_u = u_yaw  / (2.0 * PI);
        float pan_v = -u_pitch / PI;
        float sx = inv_zoom * (u_aspect / max(u_image_aspect, 0.0001));
        float sy = inv_zoom;
        vec2 raw_uv = vec2(0.5 + pan_u + centered.x * sx,
                           0.5 + pan_v - centered.y * sy);
        if (!u_wrap_2d &&
            (raw_uv.x < 0.0 || raw_uv.x > 1.0 ||
             raw_uv.y < 0.0 || raw_uv.y > 1.0)) {
            frag_color = vec4(0.02, 0.02, 0.02, 1.0);
            return;
        }
        // When wrapping, GL_REPEAT on both axes tiles the image seamlessly.
        uv = raw_uv;
    } else {
        // -- Rectilinear equirectangular projection ----------------------
        vec2 ndc = v_uv * 2.0 - 1.0;
        vec3 ray = normalize(vec3(ndc.x * u_aspect * u_tan_half_fov,
                                  ndc.y * u_tan_half_fov,
                                  1.0));
        vec3 world_dir = normalize(rotation_yaw_pitch(u_yaw, u_pitch) * ray);
        uv = direction_to_equirect_uv(world_dir);
    }

    // 1. Fetch source pixel and bring it to scene-linear.
    vec3 color = sample_image(uv);
    if (u_input_is_encoded_srgb) {
        color = srgb_to_linear(color);
    }

    // 2. Scene-linear exposure (BEFORE the view transform).
    color *= pow(2.0, u_exposure);

    // 3. View transform -> display-encoded output.
    //    OCIO path: the display/view already encodes for the display.
    //    Fallback : approximate sRGB display encoding.
    __OCIO_APPLY__

    // 4. Optional user output gamma. Default 1.0 = no-op. This is a deliberate
    //    post-display tweak applied exactly ONCE in both the OCIO and fallback
    //    paths (do not also fold gamma into __OCIO_APPLY__).
    color = pow(max(color, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)));
    frag_color = vec4(color, 1.0);
}
