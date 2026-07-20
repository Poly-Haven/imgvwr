#version 430 core
// Histogram binning pass (compute). Driven by src/renderer/histogram.rs.
//
// `u_src` is the whole image already drawn flat through the *display* pipeline —
// exposure, the OCIO view transform and output gamma — into a small offscreen
// RGBA16F target. So these are the values that actually reach the screen, which
// is what the graph is supposed to describe.
//
// Each workgroup accumulates into shared memory and flushes once at the end. A
// flat sky or a letterboxed black frame drops every one of its samples into a
// single bin, and without the shared-memory stage that would serialise the
// whole dispatch onto one global address.

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_src;

const uint BINS = 256u;
const uint SLOTS = 3u * BINS; // channel-major: R bins, then G, then B

layout(std430, binding = 0) buffer HistBuf {
    uint bins[SLOTS];
    // Samples at or above the display maximum — "outside the displayable range",
    // drawn as the 1px spike at the right of the graph. Only [0..2] are used;
    // the fourth keeps the block a tidy 16-byte multiple.
    uint over[4];
};

shared uint s_bins[SLOTS];
shared uint s_over[4];

void main() {
    uint lidx = gl_LocalInvocationIndex;
    uint threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    for (uint i = lidx; i < SLOTS; i += threads) s_bins[i] = 0u;
    if (lidx < 4u) s_over[lidx] = 0u;
    barrier();

    // The dispatch is rounded up to whole workgroups, so the last row/column of
    // groups runs partly outside the target.
    ivec2 size = textureSize(u_src, 0);
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x < size.x && p.y < size.y) {
        vec3 c = texelFetch(u_src, p, 0).rgb;
        for (uint ch = 0u; ch < 3u; ch++) {
            float v = c[ch];
            // Spelled `!(v < 1.0)` rather than `v >= 1.0` so that NaN — which
            // compares false against everything, and does turn up in damaged
            // EXRs — lands in the over-range spike instead of an arbitrary bin.
            if (!(v < 1.0)) {
                atomicAdd(s_over[ch], 1u);
            } else {
                // Negative display values (some view transforms undershoot)
                // read as black on screen, so they belong in bin 0.
                uint b = min(uint(clamp(v, 0.0, 1.0) * float(BINS)), BINS - 1u);
                atomicAdd(s_bins[ch * BINS + b], 1u);
            }
        }
    }
    barrier();

    for (uint i = lidx; i < SLOTS; i += threads) {
        uint v = s_bins[i];
        if (v != 0u) atomicAdd(bins[i], v);
    }
    if (lidx < 3u) {
        uint v = s_over[lidx];
        if (v != 0u) atomicAdd(over[lidx], v);
    }
}
