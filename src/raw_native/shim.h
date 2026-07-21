/* shim.h - plain C surface over LibRaw, used to decode camera RAW files into
 * scene-linear float RGBA for accurate viewing (demosaic + camera white balance
 * + camera->sRGB-primaries matrix, NO tone curve). See CLAUDE.md "RAW" notes.
 *
 * The output is normalised so a neutral diffuse white sits at ~1.0 while
 * per-channel highlights keep their headroom above 1.0 (recoverable by lowering
 * exposure or via a Filmic/ACES view transform). Only built with the `ocio`
 * feature (which provides the vcpkg C++ toolchain path), alongside exr_native. */
#ifndef IMGVWR_RAW_SHIM_H
#define IMGVWR_RAW_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Camera metadata read from the RAW file. Strings are NUL-terminated; empty
 * when the camera/format does not provide them. Numeric fields are 0 when
 * absent. Filled by raw_native_load on success. */
typedef struct {
    int width;
    int height;
    int colors; /* original developed channel count (1, 3 or 4) */
    char make[64];
    char model[64];
    char lens[128];
    float iso;        /* ISO speed */
    float shutter;    /* exposure time, seconds */
    float aperture;   /* f-number */
    float focal_len;  /* focal length, mm */
} RawNativeInfo;

/* Two-phase decode, so the float conversion can write straight into a buffer the
 * caller owns: no intermediate allocation is handed across the FFI boundary and
 * copied (which for a 45 Mpx RAW meant an extra ~700 MB memcpy and double peak
 * memory).
 *
 * `raw_native_begin` opens, unpacks and develops `path`, filling `*info` (dims +
 * camera metadata). Returns an opaque handle, or NULL on failure.
 *
 * `raw_native_finish` converts the developed image into `dst`, which the caller
 * must have sized `info.width * info.height * 4` floats: interleaved RGBA,
 * scene-linear with a linear camera response (no tone curve), demosaiced,
 * camera-white-balanced and converted to sRGB primaries. It consumes the handle
 * (valid or not afterwards) and returns 0 on success.
 *
 * `raw_native_abort` releases a handle without converting. Every successful
 * `begin` must be paired with exactly one `finish` or `abort`. */
void *raw_native_begin(const char *path, RawNativeInfo *info);
int raw_native_finish(void *handle, float *dst);
void raw_native_abort(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* IMGVWR_RAW_SHIM_H */
