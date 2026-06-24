/* shim.h - plain C surface over OpenEXR, used as a fallback decoder for EXR
 * files the pure-Rust `exr` crate cannot read (notably DWAA/DWAB compression).
 * See plans/rewrite.md §8.2 and the DWAA fallback task. */
#ifndef IMGVWR_EXR_SHIM_H
#define IMGVWR_EXR_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Decode `path` to interleaved RGBA float. On success returns a buffer of
 * width*height*4 floats and writes the dimensions / original channel count;
 * returns NULL on failure. Free the buffer with exr_native_free. */
float *exr_native_load_rgba(const char *path, int *out_width, int *out_height,
                            int *out_channels);
void exr_native_free(float *data);

#ifdef __cplusplus
}
#endif

#endif /* IMGVWR_EXR_SHIM_H */
