/* shim.h - plain C surface over OpenEXR, used as a fallback decoder for EXR
 * files the pure-Rust `exr` crate cannot read (notably DWAA/DWAB compression).
 * See plans/rewrite.md §8.2 and the DWAA fallback task. */
#ifndef IMGVWR_EXR_SHIM_H
#define IMGVWR_EXR_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Two-phase decode, so OpenEXR reads straight into a buffer the caller owns: no
 * intermediate allocation is handed across the FFI boundary and copied. (For a
 * 4K RGBA float frame that copy was 134 MB and doubled peak memory; at sequence
 * rates it would be per frame.)
 *
 * `exr_native_begin` opens `path` and reads its header, writing the data-window
 * dimensions and the original channel count. Returns an opaque handle, or NULL
 * on failure.
 *
 * `exr_native_init_thread_pool` sizes OpenEXR's *library-global* worker pool.
 * It must be called at least once before any decode: the pool defaults to ZERO
 * workers, meaning every file decompresses entirely on the calling thread, so
 * this fallback ran single-threaded for every file — including exactly the
 * DWAA/DWAB ones it exists to handle, which are among the more expensive to
 * decompress. Only the first call takes effect (it is `call_once`-guarded, so
 * concurrent decode threads cannot race over a global).
 *
 * `threads` on `begin` is then how many of that shared pool THIS file may
 * occupy, fixed when it is opened: a positive value for a single interactive
 * open (take the lot), and 0 when many frames are decoding at once — each of
 * those wants its own frame decompressed on its own thread, and handing them a
 * shared pool as well would only oversubscribe the machine.
 *
 * `exr_native_finish` decodes into `dst`, which the caller must have sized
 * `width * height * 4` floats (interleaved RGBA, scene-linear; missing colour
 * channels read 0 and missing alpha reads 1). It consumes the handle (valid or
 * not afterwards) and returns 0 on success.
 *
 * `exr_native_abort` releases a handle without decoding. Every successful
 * `begin` must be paired with exactly one `finish` or `abort`. */
void exr_native_init_thread_pool(int count);
void *exr_native_begin(const char *path, int threads, int *out_width, int *out_height,
                       int *out_channels);
int exr_native_finish(void *handle, float *dst);
void exr_native_abort(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* IMGVWR_EXR_SHIM_H */
