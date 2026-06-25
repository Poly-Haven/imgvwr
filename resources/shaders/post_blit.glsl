#version 400 core
// Passthrough / bilinear downsample for the post-process chain.
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_src;
void main() {
    frag_color = texture(u_src, v_uv);
}
