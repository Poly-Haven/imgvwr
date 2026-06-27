#version 400 core

// Fullscreen checkerboard backdrop, drawn behind the image when the background
// preset is "checkerboard" (the B key). Pairs with the shared fullscreen-quad
// vertex shader; v_uv is ignored — the pattern is keyed off screen pixels
// (gl_FragCoord) so it stays fixed while the image pans/zooms over it.
//
// Colours are written as display-encoded sRGB 0–1 directly (the default
// framebuffer is not sRGB; the scene shader and the clear colour do the same),
// so no decode is needed.

out vec4 frag_color;

const float SQUARE = 12.0;          // checker square size, screen pixels
const vec3  LIGHT  = vec3(0.50);    // #808080
const vec3  DARK   = vec3(0.38);    // #616161

void main() {
    vec2 cell = floor(gl_FragCoord.xy / SQUARE);
    float parity = mod(cell.x + cell.y, 2.0);
    frag_color = vec4(mix(LIGHT, DARK, parity), 1.0);
}
