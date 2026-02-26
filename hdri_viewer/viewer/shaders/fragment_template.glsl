#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_half_fov_radians;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_exposure;
uniform float u_projection_mode;
uniform float u_projection_2d_wrap_enabled;
uniform float u_fisheye_enabled;
uniform float u_image_aspect;
uniform float u_input_is_encoded_srgb;

__OCIO_DECLARATIONS__

vec3 rotate_yaw_pitch(vec3 dir, float yaw, float pitch) {
    float cy = cos(yaw);
    float sy = sin(yaw);
    float cp = cos(pitch);
    float sp = sin(pitch);

    mat3 yaw_mat = mat3(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    );

    mat3 pitch_mat = mat3(
        1.0, 0.0, 0.0,
        0.0, cp, -sp,
        0.0, sp, cp
    );

    return yaw_mat * pitch_mat * dir;
}

vec2 direction_to_equirect_uv(vec3 dir) {
    float longitude = atan(dir.z, dir.x);
    float latitude = asin(clamp(dir.y, -1.0, 1.0));

    float u = 1.0 - ((longitude / (2.0 * 3.14159265358979323846)) + 0.5);
    float v = 0.5 - (latitude / 3.14159265358979323846);
    return vec2(u, v);
}

vec3 srgb_to_linear(vec3 encoded) {
    vec3 clipped = clamp(encoded, vec3(0.0), vec3(1.0));
    bvec3 lower = lessThanEqual(clipped, vec3(0.04045));
    vec3 low = clipped / 12.92;
    vec3 high = pow((clipped + 0.055) / 1.055, vec3(2.4));
    return vec3(
        lower.x ? low.x : high.x,
        lower.y ? low.y : high.y,
        lower.z ? low.z : high.z
    );
}

void main() {
    vec2 uv;
    if (u_projection_mode >= 0.5) {
        float inv_zoom = max(u_tan_half_fov, 0.02);
        vec2 centered = v_uv - vec2(0.5, 0.5);
        float pan_u = u_yaw / (2.0 * 3.14159265358979323846);
        float pan_v = -u_pitch / 3.14159265358979323846;
        float scale_x = inv_zoom * (u_aspect / max(u_image_aspect, 0.0001));
        float scale_y = inv_zoom;

        vec2 raw_uv = vec2(
            0.5 + pan_u + centered.x * scale_x,
            0.5 + pan_v - centered.y * scale_y
        );

        if (u_projection_2d_wrap_enabled >= 0.5) {
            uv = fract(raw_uv);
        } else {
            if (raw_uv.x < 0.0 || raw_uv.x > 1.0 || raw_uv.y < 0.0 || raw_uv.y > 1.0) {
                frag_color = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }
            uv = raw_uv;
        }
    } else {
        vec2 ndc = v_uv * 2.0 - 1.0;
        vec3 ray;
        if (u_fisheye_enabled >= 0.5) {
            vec2 lens_xy = vec2(ndc.x * u_aspect, ndc.y);
            float lens_radius = length(lens_xy);
            float max_lens_radius = max(length(vec2(u_aspect, 1.0)), 0.0001);
            float half_fov = clamp(u_half_fov_radians, 0.0, 3.12413936107);
            float theta = (lens_radius / max_lens_radius) * half_fov;

            if (lens_radius > 0.000001) {
                vec2 lens_dir = lens_xy / lens_radius;
                float sin_theta = sin(theta);
                ray = normalize(vec3(lens_dir.x * sin_theta, lens_dir.y * sin_theta, cos(theta)));
            } else {
                ray = vec3(0.0, 0.0, 1.0);
            }
        } else {
            ray = normalize(vec3(ndc.x * u_aspect * u_tan_half_fov, ndc.y * u_tan_half_fov, 1.0));
        }

        vec3 world_dir = normalize(rotate_yaw_pitch(ray, u_yaw, u_pitch));
        uv = direction_to_equirect_uv(world_dir);
    }

    vec3 color = texture(u_image, uv).rgb;
    if (u_input_is_encoded_srgb >= 0.5) {
        color = srgb_to_linear(color);
    }
    color *= pow(2.0, u_exposure);
    __OCIO_APPLY__

    frag_color = vec4(color, 1.0);
}
