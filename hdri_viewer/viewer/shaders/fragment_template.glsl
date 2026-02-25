#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_exposure;
uniform float u_projection_mode;
uniform float u_image_aspect;

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

void main() {
    vec2 uv;
    if (u_projection_mode >= 0.5) {
        float inv_zoom = max(u_tan_half_fov, 0.02);
        vec2 centered = v_uv - vec2(0.5, 0.5);
        float pan_u = u_yaw / (2.0 * 3.14159265358979323846);
        float pan_v = -u_pitch / 3.14159265358979323846;
        float scale_x = inv_zoom * (u_aspect / max(u_image_aspect, 0.0001));
        float scale_y = inv_zoom;

        uv.x = fract(0.5 + pan_u + centered.x * scale_x);
        uv.y = clamp(0.5 + pan_v - centered.y * scale_y, 0.0, 1.0);
    } else {
        vec2 ndc = v_uv * 2.0 - 1.0;
        vec3 ray = normalize(vec3(ndc.x * u_aspect * u_tan_half_fov, ndc.y * u_tan_half_fov, 1.0));

        vec3 world_dir = normalize(rotate_yaw_pitch(ray, u_yaw, u_pitch));
        uv = direction_to_equirect_uv(world_dir);
    }

    vec3 color = texture(u_image, uv).rgb;
    color *= pow(2.0, u_exposure);
    __OCIO_APPLY__

    frag_color = vec4(color, 1.0);
}
