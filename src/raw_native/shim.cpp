// shim.cpp - LibRaw-backed RAW decoder producing scene-linear float RGBA.
//
// Pipeline (LibRaw): black-level subtraction, camera white balance, demosaic
// (AHD), and the camera->sRGB-primaries colour matrix, with a LINEAR transfer
// (gamm = {1,1}) and auto-bright disabled so the actual photo exposure is
// preserved. highlight = 1 (unclip) keeps each channel's data up to its own clip
// point instead of flattening blown areas to white, so highlights stay
// recoverable. LibRaw normalises its 16-bit output by the largest WB multiplier
// (so nothing overflows the container); we then rescale by 1/min(pre_mul) so the
// neutral (smallest-multiplier, i.e. green) reference channel saturates at 1.0
// and the other channels keep headroom above 1.0. The result is scene-linear,
// matching the EXR/HDR float path (is_encoded_srgb = false on the Rust side).
//
// All exceptions / LibRaw error codes are handled at the C boundary (NULL).

#include "shim.h"

#include <libraw/libraw.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {

// Copy a fixed C string field into an info buffer, always NUL-terminated.
template <size_t N>
void copy_field(char (&dst)[N], const char *src) {
    if (!src) {
        dst[0] = '\0';
        return;
    }
    std::strncpy(dst, src, N - 1);
    dst[N - 1] = '\0';
}

// Live state between begin() and finish(): the LibRaw instance owning the
// developed image, LibRaw's 16-bit output bitmap, and the scale that maps it to
// scene-linear float.
struct RawHandle {
    LibRaw *rp = nullptr;
    libraw_processed_image_t *img = nullptr;
    float scale = 1.0f;
};

void destroy_handle(RawHandle *hn) {
    if (!hn) {
        return;
    }
    if (hn->img) {
        LibRaw::dcraw_clear_mem(hn->img);
    }
    if (hn->rp) {
        hn->rp->recycle();
        delete hn->rp;
    }
    delete hn;
}

} // namespace

extern "C" void *raw_native_begin(const char *path, RawNativeInfo *info) {
    if (!path || !info) {
        return nullptr;
    }
    RawHandle *hn = new RawHandle();
    hn->rp = new LibRaw();
    LibRaw &rp = *hn->rp;

    // --- View-faithful linear development parameters -----------------------
    rp.imgdata.params.use_camera_wb = 1;   // factual, camera-recorded WB
    rp.imgdata.params.use_auto_wb = 0;     // never histogram-derived WB
    rp.imgdata.params.output_color = 1;    // sRGB primaries (matrix only)
    rp.imgdata.params.gamm[0] = 1.0;       // linear transfer (no tone curve)
    rp.imgdata.params.gamm[1] = 1.0;
    rp.imgdata.params.no_auto_bright = 1;  // respect the actual exposure
    rp.imgdata.params.output_bps = 16;     // 16-bit develop, converted to float
    rp.imgdata.params.highlight = 1;       // unclip: keep recoverable highlights
    rp.imgdata.params.user_qual = 3;       // AHD demosaic (quality/speed balance)
    // user_flip left at LibRaw's default (-1): apply the camera's EXIF
    // orientation so portrait shots come out upright (output dims reflect it).

    if (rp.open_file(path) != LIBRAW_SUCCESS) {
        destroy_handle(hn);
        return nullptr;
    }
    if (rp.unpack() != LIBRAW_SUCCESS) {
        destroy_handle(hn);
        return nullptr;
    }

    // Metadata is available after open_file/unpack (before the heavy develop).
    copy_field(info->make, rp.imgdata.idata.make);
    copy_field(info->model, rp.imgdata.idata.model);
    // Prefer the maker-note lens name; fall back to the EXIF lens string.
    const char *lens = rp.imgdata.lens.Lens;
    if (!lens || lens[0] == '\0') {
        lens = rp.imgdata.lens.makernotes.Lens;
    }
    copy_field(info->lens, lens);
    info->iso = rp.imgdata.other.iso_speed;
    info->shutter = rp.imgdata.other.shutter;
    info->aperture = rp.imgdata.other.aperture;
    info->focal_len = rp.imgdata.other.focal_len;

    if (rp.dcraw_process() != LIBRAW_SUCCESS) {
        destroy_handle(hn);
        return nullptr;
    }

    // LibRaw normalised pre_mul in place during scale_colors (divided by the max
    // multiplier). The min non-zero entry is the neutral reference channel's
    // saturation fraction; 1/min maps neutral white to 1.0 and lifts the other
    // channels above 1.0 (their recoverable headroom). Clamp to a sane range.
    double mn = 1e30;
    for (int c = 0; c < 4; ++c) {
        double v = rp.imgdata.color.pre_mul[c];
        if (v > 1e-6 && v < mn) {
            mn = v;
        }
    }
    double headroom = (mn < 1e29 && mn > 1e-6) ? 1.0 / mn : 1.0;
    headroom = std::min(std::max(headroom, 1.0), 16.0);
    hn->scale = static_cast<float>(headroom / 65535.0);

    int errc = 0;
    hn->img = rp.dcraw_make_mem_image(&errc);
    if (!hn->img) {
        destroy_handle(hn);
        return nullptr;
    }
    // Expect a 16-bit bitmap; bail on anything unexpected.
    const libraw_processed_image_t *img = hn->img;
    if (img->type != LIBRAW_IMAGE_BITMAP || img->bits != 16 || img->colors < 1 ||
        img->colors > 4 || img->width == 0 || img->height == 0) {
        destroy_handle(hn);
        return nullptr;
    }

    info->width = static_cast<int>(img->width);
    info->height = static_cast<int>(img->height);
    info->colors = img->colors;
    return hn;
}

extern "C" int raw_native_finish(void *handle, float *dst) {
    RawHandle *hn = static_cast<RawHandle *>(handle);
    if (!hn || !hn->img || !dst) {
        destroy_handle(hn);
        return -1;
    }
    const libraw_processed_image_t *img = hn->img;
    const ptrdiff_t n =
        static_cast<ptrdiff_t>(img->width) * static_cast<ptrdiff_t>(img->height);
    const int colors = img->colors;
    const float scale = hn->scale;
    const unsigned short *src = reinterpret_cast<const unsigned short *>(img->data);

    // Expand LibRaw's tightly-packed 16-bit output to interleaved float RGBA,
    // writing straight into the caller's buffer. Parallel because this touches
    // ~1 GB for a 45 Mpx frame and is purely element-wise (OpenMP 2.0 via MSVC
    // needs a signed loop counter).
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        const unsigned short *p = src + i * colors;
        float r, g, b;
        if (colors == 1) {
            r = g = b = p[0] * scale; // monochrome -> grey
        } else {
            r = p[0] * scale;
            g = p[1] * scale;
            b = p[2] * scale; // colors==4 (rare): ignore the 4th component
        }
        dst[i * 4 + 0] = r;
        dst[i * 4 + 1] = g;
        dst[i * 4 + 2] = b;
        dst[i * 4 + 3] = 1.0f;
    }

    destroy_handle(hn);
    return 0;
}

extern "C" void raw_native_abort(void *handle) {
    destroy_handle(static_cast<RawHandle *>(handle));
}
