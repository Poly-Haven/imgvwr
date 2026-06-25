#version 400 core
// Clarity composite: a large-radius unsharp mask in display space.
//   out = base + amount * (base - blur) * midtone_mask
// The midtone luminance mask attenuates the effect in deep shadows and bright
// highlights, which is what keeps halos from forming around a bright sky/sun on
// an HDRI (the blur is of the already display-encoded scene, not raw HDR).
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_base;   // full-res scene (display-encoded)
uniform sampler2D u_blur;   // downsampled, blurred copy
uniform float u_amount;
void main() {
    vec3 base = texture(u_base, v_uv).rgb;
    vec3 blur = texture(u_blur, v_uv).rgb;
    vec3 detail = base - blur;
    float luma = dot(base, vec3(0.2126, 0.7152, 0.0722));
    // Full effect across the midtones, ramping off at the extremes.
    float mask = smoothstep(0.05, 0.25, luma) * (1.0 - smoothstep(0.80, 0.98, luma));
    vec3 outc = base + u_amount * detail * mask;
    frag_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
