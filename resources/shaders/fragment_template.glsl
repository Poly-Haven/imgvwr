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
uniform vec2  u_levels;              // display black/white points (0,1 = no-op); see step 4a
uniform int   u_projection_mode;     // 0 = equirectangular panorama, 1 = 2D pan/zoom
uniform float u_image_aspect;
uniform bool  u_input_is_encoded_srgb;  // true when source pixels are sRGB-encoded (JPEG / LDR PNG)
uniform bool  u_wrap_2d;             // 2D mode: repeat the image instead of clamping
uniform int   u_isolate_channel;     // -1 = all channels; 0=R 1=G 2=B 3=A shown as greyscale
uniform vec2  u_stretch;             // per-axis image squash/stretch (1,1 = none)
uniform int   u_sharpness;           // 1 = show the original-resolution high-pass
uniform int   u_diff;                // 1 = show the slot difference (the slot diff checker)
uniform int   u_diff_live;           // 1 = u_diff_image is sequence B's raw frame; subtract it live
uniform sampler2D u_diff_image;      // the comparator slot's image (for u_diff)
// Guide lines, each .x = image coordinate (0..1), .y = 0 vertical / 1 horizontal.
uniform int   u_guide_count;
uniform vec2  u_guides[64];
uniform vec3  u_guide_color;         // display-encoded sRGB 0–1
uniform int   u_guide_hover;         // index of the hovered guide, or -1
uniform vec3  u_guide_hover_color;   // inverse-hue colour for the hovered guide
uniform float u_global_alpha;        // whole-frame opacity (1 = opaque); the
                                     // minimap pass fades its thumbnail with this
uniform int   u_rotation;            // 2D display rotation, 90° CW quarter-turns 0–3
                                     // (u_image_aspect is already the rotated aspect)
uniform int   u_lanczos;             // 1 = Lanczos-3 minification (8-bit images);
                                     // 0 = bilinear/trilinear. Upscaling is always bilinear.
uniform int   u_clip_overlay;        // 1 = draw the clipping overlay (C key)
uniform float u_clip_margin;         // clip when a channel >= clip_max*(1 - margin)
uniform vec4  u_clip_max;            // per-channel format max in texel value space
uniform sampler2D u_clip_mask;       // max-mipped per-channel clip mask (1=clipped)
uniform int   u_clip_mask_valid;     // 1 = use the mask; 0 = per-texel fallback
uniform float u_time;                // wall-clock seconds, animates the clip stripes

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

// Map a displayed image-uv to the source texture uv for a 90°-CW quarter-turn
// display rotation `rot` (0–3). The displayed aspect is handled separately (the
// app feeds the rotated aspect as u_image_aspect), so this is a pure unit-square
// coordinate swap — undistorted for any image.
vec2 rotate_uv(vec2 d, int rot) {
    if (rot == 1) return vec2(d.y, 1.0 - d.x);
    if (rot == 2) return vec2(1.0 - d.x, 1.0 - d.y);
    if (rot == 3) return vec2(1.0 - d.y, d.x);
    return d;
}

vec3 srgb_to_linear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92,
               pow((c + 0.055) / 1.055, vec3(2.4)),
               step(0.04045, c));
}

void main() {
    vec2 uv;       // displayed image uv (guides / bounds compare against this)
    vec2 src_uv;   // source-texture uv to sample (uv permuted by 2D rotation)
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
        // Dividing the screen->image scale by u_stretch makes the image appear
        // wider/taller (squash/stretch for inspecting line straightness).
        float sx = inv_zoom * (u_aspect / max(u_image_aspect, 0.0001)) / u_stretch.x;
        float sy = inv_zoom / u_stretch.y;
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
        // sampling (and its mip LOD) is correct here. `uv` stays in DISPLAYED
        // space (guides / bounds compare against it); the image is sampled at the
        // rotation-permuted source coordinate.
        uv = raw_uv;
        src_uv = rotate_uv(uv, u_rotation);
        // Lanczos-3 minification for 8-bit images (sharper downscaling); bilinear
        // otherwise and for any upscaling (handled inside sample_image_lanczos).
        texel = (u_lanczos != 0) ? sample_image_lanczos(src_uv) : sample_image(src_uv);
    } else {
        // -- Rectilinear equirectangular projection ----------------------
        vec2 ndc = (v_uv * 2.0 - 1.0) / u_stretch;
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
        src_uv = uv; // no rotation in panorama
    }

    // Original per-channel sample (before exposure / view transform and before the
    // diff/sharpness overrides below) — the clipping overlay must judge the source
    // data, not the displayed result. Compared against the format's per-channel max
    // (u_clip_max): 1.0 for integer formats, the half max for a 16-bit-half EXR,
    // f32::MAX (never) for unbounded 32-bit float / HDR.
    vec4 clip_src = texel;

    // Slot difference. Two ways in:
    //   * Still diff (u_diff_live == 0): u_diff_image holds the absolute per-pixel
    //     difference, PRECOMPUTED at base resolution on the CPU with its own mip
    //     chain. Sampling it normally shows the *average of the differences* when
    //     minified, so identical regions stay exactly 0 at every zoom. (Diffing
    //     two separately mip-averaged images instead bled nearby differences into
    //     identical regions when zoomed out, vanishing only at LOD 0.)
    //   * Live sequence diff (u_diff_live != 0): u_diff_image holds sequence B's
    //     raw frame (the primary sequence A is the image sampled above), so we
    //     subtract them here per frame — no CPU precompute can keep up at 24 fps.
    //     Correct at LOD 0; when minified each side is box-averaged before the
    //     subtract, which is the best-effort the playing comparison accepts.
    // Either way it is kept in source space so exposure / view / clarity amplify it.
    if (u_diff != 0) {
        // Sample with a seam-corrected gradient (the same longitude unwrap the
        // image sampler uses), so a minified panorama diff doesn't flash the
        // coarsest-mip column at the wrap where dFdx(uv.x) ≈ 1. In 2D the
        // gradient is unchanged.
        vec2 ddxd = dFdx(uv);
        vec2 ddyd = dFdy(uv);
        if (u_projection_mode != 1) {
            if (abs(ddxd.x) > 0.5) ddxd.x -= sign(ddxd.x);
            if (abs(ddyd.x) > 0.5) ddyd.x -= sign(ddyd.x);
        }
        // Sample in SOURCE space (src_uv = uv rotated) so the diff lines up with
        // the displayed, rotated image. The 2D gradient is a pure axis swap under a
        // 90° turn, so the displayed-space derivatives select the same mip.
        vec3 other = textureGrad(u_diff_image, src_uv, ddxd, ddyd).rgb;
        if (u_diff_live != 0) {
            // Re-sample A (the image) with the SAME operation as B — a plain
            // trilinear textureGrad at src_uv, NOT the Lanczos/bilinear display
            // path `texel` came from — so two identical frames cancel to exactly
            // black at every zoom instead of tracing faint edges where the two
            // samplers (Lanczos vs trilinear) disagreed. `sample_image_grad` is
            // that textureGrad, and the diff texture is configured to match the
            // image sampler (wrap + anisotropy), so the two sides are identical.
            texel = vec4(abs(sample_image_grad(src_uv, ddxd, ddyd).rgb - other), 1.0);
        } else {
            texel = vec4(other, 1.0);
        }
    }
    // Sharpness checker: |original - 2px-blurred original|, from the ORIGINAL
    // full-resolution pixels (LOD 0), not the displayed mip. Done here in source
    // space — like the slot diff — so exposure/view below can amplify it (sampled
    // at src_uv so it matches the rotated image).
    if (u_sharpness != 0) {
        texel = vec4(sharp_diff(src_uv), 1.0);
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

    // 4a. Display levels — the two handles under the F2 histogram. Stretches the
    //     [black, white] slice of the display range back out to 0..1. Applied
    //     HERE, after the view transform and gamma, because that is exactly where
    //     the histogram is measured: the handles then line up 1:1 with the graph's
    //     x axis, which is the whole point of putting them under it.
    //
    //     Deliberately left unclamped. Pushing black up drives dark pixels
    //     negative and pulling white down drives bright ones past 1.0, and the
    //     clip overlay below still needs to see how far out of range a value went
    //     (it re-clamps for its own compositing). The histogram pass feeds an
    //     identity (0, 1) here so the graph never moves under its own handles.
    color = (color - u_levels.x) / max(u_levels.y - u_levels.x, 1e-4);

    // 4b. Clipping overlay (C): animated diagonal stripes over regions whose
    //     ORIGINAL per-channel value (clip_src, captured before any adjustment) is
    //     within u_clip_margin of that channel's format max (u_clip_max). Each
    //     clipped channel contributes its colour to the "lit" stripe band
    //     (R→red, R+G→yellow, all→white); the alternating band is black.
    //     Composited at 75% opacity only where a channel clips. When a single
    //     channel is isolated (F2 boxes), only that channel is evaluated and the
    //     stripe takes its colour. Drawn before the guides so they stay readable.
    if (u_clip_overlay != 0) {
        // Per-channel clip flags (0/1). Prefer the max-mipped mask so even a
        // few-pixel blown region survives minification (averaged image mips would
        // dilute it away when zoomed out); fall back to a per-texel test on the
        // displayed sample for tiled images that have no mask.
        vec4 clip4;
        if (u_clip_mask_valid != 0) {
            vec2 mdx = dFdx(src_uv);
            vec2 mdy = dFdy(src_uv);
            if (u_projection_mode != 1) {          // unwrap longitude seam (pano)
                if (abs(mdx.x) > 0.5) mdx.x -= sign(mdx.x);
                if (abs(mdy.x) > 0.5) mdy.x -= sign(mdy.x);
            }
            // Low threshold: the LINEAR-filtered max-mip spreads a clipped texel
            // over ~1 texel, so any partial coverage (a few clipped source pixels
            // anywhere in the footprint) lights the channel even when minified.
            clip4 = step(0.15, textureGrad(u_clip_mask, src_uv, mdx, mdy));
        } else {
            clip4 = step(u_clip_max * (1.0 - u_clip_margin), clip_src);
        }
        vec3 stripe_color;
        float clipped;
        if (u_isolate_channel >= 0) {
            clipped = clip4[u_isolate_channel];
            stripe_color = (u_isolate_channel == 0) ? vec3(1.0, 0.0, 0.0)
                         : (u_isolate_channel == 1) ? vec3(0.0, 1.0, 0.0)
                         : (u_isolate_channel == 2) ? vec3(0.0, 0.0, 1.0)
                         : vec3(1.0);              // alpha → white
        } else {
            clipped = max(max(clip4.r, clip4.g), clip4.b);
            stripe_color = clip4.rgb;              // additive per-channel colour
        }
        if (clipped > 0.5) {
            // Diagonal screen-space bands, scrolling with time. PERIOD = one
            // colour+black pair in px; half lit, half black.
            const float PERIOD = 14.0;
            const float SPEED = 36.0; // px/sec
            float diag = gl_FragCoord.x + gl_FragCoord.y - u_time * SPEED;
            float lit = step(0.5, fract(diag / PERIOD));
            vec3 stripe = stripe_color * lit; // channel colour on lit bands, black between
            // Composite over the DISPLAY-CLAMPED colour: clipped regions are exactly
            // where the display value blows past 1.0, and 0.25*huge would still clip
            // to white — so the stripes would be invisible without the clamp.
            color = mix(clamp(color, 0.0, 1.0), stripe, 0.75);
            out_alpha = 1.0; // make the warning visible even over transparency
        }
    }

    // 5. Guide lines, drawn last so they sit on top of every mode. The line
    //    sticks to the image (compared in image-uv space) but is a constant ~1px
    //    in screen space (distance normalised by the per-pixel uv derivative),
    //    with a soft dark halo on either side. Works for 2D and panorama alike.
    //    Anti-aliased via smoothstep on the pixel distance: the nearest pixel to
    //    the line (always within 0.5px) is fully coloured, so the line can never
    //    drop out at an unlucky sub-pixel alignment (the old hard `d < 0.5` test
    //    sometimes left only the halo, making the line look faint/invisible).
    for (int i = 0; i < u_guide_count; i++) {
        bool vertical = u_guides[i].y < 0.5; // a constant-uv.x (longitude) line
        float coord = vertical ? uv.x : uv.y;
        float dcdx = dFdx(coord);
        float dcdy = dFdy(coord);
        float dist = abs(coord - u_guides[i].x);
        // Longitude (uv.x) wraps in the panorama: unwrap the derivative across the
        // seam (as the image sampler does) and measure the circular distance, so
        // the wrap-around vertical guide isn't smeared into a fat aliased band on
        // the image-edge side. 2D / latitude lines don't wrap, so stay plain.
        if (u_projection_mode != 1 && vertical) {
            if (abs(dcdx) > 0.5) dcdx -= sign(dcdx);
            if (abs(dcdy) > 0.5) dcdy -= sign(dcdy);
            dist = min(dist, 1.0 - dist);
        }
        float d = dist / max(abs(dcdx) + abs(dcdy), 1e-9);
        float halo = (1.0 - smoothstep(1.0, 2.2, d)) * 0.5; // dark, just outside
        float line = 1.0 - smoothstep(0.5, 1.2, d);         // full within 0.5px
        if (halo > 0.0 || line > 0.0) out_alpha = 1.0;
        vec3 gcol = (i == u_guide_hover) ? u_guide_hover_color : u_guide_color;
        color = mix(color, vec3(0.0), halo);
        color = mix(color, gcol, line);
    }
    frag_color = vec4(color, out_alpha * u_global_alpha);
}
