#version 400 core
// One direction of a separable 9-tap Gaussian (sigma ~2). `u_step` is the
// per-tap UV offset (direction * spacing), so a horizontal then vertical pass
// gives a 2D blur. Run on a downsampled target so a large radius stays cheap.
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_src;
uniform vec2 u_step;
void main() {
    float w[5] = float[](0.20416, 0.18017, 0.12383, 0.06637, 0.02763);
    vec4 c = texture(u_src, v_uv) * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_src, v_uv + u_step * float(i)) * w[i];
        c += texture(u_src, v_uv - u_step * float(i)) * w[i];
    }
    frag_color = c;
}
