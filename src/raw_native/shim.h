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

/* Decode `path` to interleaved RGBA float (width*height*4), scene-linear with a
 * linear camera response (no tone curve), demosaiced, camera-white-balanced and
 * converted to sRGB primaries. Returns NULL on failure. On success fills `*info`
 * and returns a buffer the caller must release with raw_native_free. */
float *raw_native_load(const char *path, RawNativeInfo *info);
void raw_native_free(float *data);

#ifdef __cplusplus
}
#endif

#endif /* IMGVWR_RAW_SHIM_H */
