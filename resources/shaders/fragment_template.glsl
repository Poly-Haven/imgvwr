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
uniform int   u_projection_mode;     // 0 = equirectangular panorama, 1 = 2D pan/zoom
uniform float u_image_aspect;
uniform bool  u_input_is_encoded_srgb;  // true when source pixels are sRGB-encoded (JPEG / LDR PNG)
uniform bool  u_wrap_2d;             // 2D mode: repeat the image instead of clamping
uniform int   u_isolate_channel;     // -1 = all channels; 0=R 1=G 2=B 3=A shown as greyscale

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
    vec3 color;
    vec4 texel;

    if (u_projection_mode == 1) {
        // -- 2D pan / zoom ----------------------------------------------
        // Floor only guards against division by zero; keep it tiny so deep
        // zoom-in (well past 100%) is not silently capped.
        float inv_zoom = max(u_tan_half_fov, 1e-6);
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
            // Outside the image: stay transparent so the cleared background
            // colour shows through (the letterbox uses the chosen background).
            frag_color = vec4(0.0);
            return;
        }
        // When wrapping, GL_REPEAT on both axes tiles the image seamlessly.
        // The 2D coordinate is screen-space-continuous, so implicit-derivative
        // sampling (and its mip LOD) is correct here.
        uv = raw_uv;
        texel = sample_image(uv);
    } else {
        // -- Rectilinear equirectangular projection ----------------------
        vec2 ndc = v_uv * 2.0 - 1.0;
        vec3 ray = normalize(vec3(ndc.x * u_aspect * u_tan_half_fov,
                                  ndc.y * u_tan_half_fov,
                                  1.0));
        vec3 world_dir = normalize(rotation_yaw_pitch(u_yaw, u_pitch) * ray);
        uv = direction_to_equirect_uv(world_dir);
        // The equirect U coordinate is discontinuous at the longitude wrap
        // (atan2 jumps ~1→0 across one pixel column). Left to implicit
        // derivatives, that one column's huge dFdx(u) forces the coarsest mip,
        // producing a flickering/dashed seam. Unwrap the derivative there and
        // sample with an explicit, seam-continuous gradient.
        vec2 ddx = dFdx(uv);
        vec2 ddy = dFdy(uv);
        if (abs(ddx.x) > 0.5) ddx.x -= sign(ddx.x);
        if (abs(ddy.x) > 0.5) ddy.x -= sign(ddy.x);
        texel = sample_image_grad(uv, ddx, ddy);
    }

    color = texel.rgb;
    float out_alpha = texel.a;
    // Channel isolation (F2 metadata box): show one channel as greyscale, fully
    // opaque, processed through the normal exposure/view pipeline below.
    if (u_isolate_channel >= 0) {
        color = vec3(texel[u_isolate_channel]);
        out_alpha = 1.0;
    }

    // 1. Source pixel is now in `color`; bring it to scene-linear.
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
    frag_color = vec4(color, out_alpha);
}
