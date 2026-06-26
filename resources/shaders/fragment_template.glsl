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
uniform vec2  u_stretch;             // per-axis image squash/stretch (1,1 = none)
uniform int   u_sharpness;           // 1 = show the original-resolution high-pass
uniform int   u_diff;                // 1 = show |image - slot| (the slot diff checker)
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

    // Slot difference: the absolute per-pixel difference vs the comparator slot,
    // PRECOMPUTED at base resolution on the CPU and uploaded as u_diff_image with
    // its own mip chain. Sampling it normally (implicit-derivative mip selection)
    // therefore shows the *average of the differences* when minified — identical
    // regions stay exactly 0 at every zoom. (Differencing two separately
    // mip-averaged images instead bled nearby differences into identical regions
    // when zoomed out, vanishing only at LOD 0.) Kept in source space so the
    // exposure / view / clarity below still amplify it.
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
        texel = vec4(textureGrad(u_diff_image, src_uv, ddxd, ddyd).rgb, 1.0);
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
