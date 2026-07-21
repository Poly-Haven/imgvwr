// shim.cpp - OpenEXR-backed EXR decoder fallback (handles all compressions,
// including DWAA/DWAB which the pure-Rust `exr` crate cannot decode).
//
// Reads the first part's R/G/B/A channels as FLOAT (defaulting missing colour
// channels to 0 and alpha to 1). If no R/G/B are present, the first channel is
// replicated to RGB. All exceptions are caught at the C boundary.

#include "shim.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfThreading.h>
#include <Imath/ImathBox.h>

#include <memory>
#include <mutex>
#include <string>

namespace IMF = OPENEXR_IMF_NAMESPACE;

namespace {

// Everything `begin` learned, carried to `finish` so the pixels can be decoded
// straight into the caller's buffer.
struct ExrJob {
    IMF::InputFile file;
    Imath::Box2i dw;
    int width;
    int height;
    bool has_rgb;
    bool has_alpha;
    std::string lone_channel; // only set when has_rgb is false

    ExrJob(const char *path, int threads) : file(path, threads) {}
};

std::once_flag g_pool_once;

} // namespace

extern "C" void exr_native_init_thread_pool(int count) {
    std::call_once(g_pool_once, [count] {
        try {
            IMF::setGlobalThreadCount(count < 0 ? 0 : count);
        } catch (...) {
            // A pool that cannot be created just means single-threaded decodes.
        }
    });
}

extern "C" void *exr_native_begin(const char *path, int threads, int *out_width, int *out_height,
                                  int *out_channels) {
    try {
        // unique_ptr so any throw after construction (e.g. bad_alloc filling
        // lone_channel) frees the job rather than leaking it.
        auto job = std::make_unique<ExrJob>(path, threads < 0 ? 0 : threads);
        job->dw = job->file.header().dataWindow();
        job->width = job->dw.max.x - job->dw.min.x + 1;
        job->height = job->dw.max.y - job->dw.min.y + 1;
        if (job->width <= 0 || job->height <= 0) {
            return nullptr;
        }

        const IMF::ChannelList &channels = job->file.header().channels();
        const bool hasR = channels.findChannel("R") != nullptr;
        const bool hasG = channels.findChannel("G") != nullptr;
        const bool hasB = channels.findChannel("B") != nullptr;
        job->has_alpha = channels.findChannel("A") != nullptr;
        job->has_rgb = hasR || hasG || hasB;

        int orig_channels = (hasR ? 1 : 0) + (hasG ? 1 : 0) + (hasB ? 1 : 0) +
                            (job->has_alpha ? 1 : 0);
        if (!job->has_rgb) {
            IMF::ChannelList::ConstIterator it = channels.begin();
            if (it == channels.end()) {
                return nullptr;
            }
            job->lone_channel = it.name();
            orig_channels = 1;
        }

        *out_width = job->width;
        *out_height = job->height;
        *out_channels = orig_channels > 0 ? orig_channels : 3;
        return job.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" int exr_native_finish(void *handle, float *dst) {
    ExrJob *job = static_cast<ExrJob *>(handle);
    if (!job || !dst) {
        delete job;
        return 1;
    }
    int rc = 1;
    try {
        const size_t n = static_cast<size_t>(job->width) * job->height;
        const size_t xs = 4 * sizeof(float);
        const size_t ys = static_cast<size_t>(job->width) * 4 * sizeof(float);
        // Base accounts for the data-window origin (OpenEXR slice convention).
        char *base = reinterpret_cast<char *>(dst) -
                     (static_cast<size_t>(job->dw.min.x) * xs +
                      static_cast<size_t>(job->dw.min.y) * ys);

        IMF::FrameBuffer fb;
        auto add = [&](const char *name, int offset, double dflt) {
            fb.insert(name, IMF::Slice(IMF::FLOAT,
                                       base + static_cast<size_t>(offset) * sizeof(float), xs, ys,
                                       1, 1, dflt));
        };

        if (job->has_rgb) {
            // Slices are inserted for all four channels unconditionally: OpenEXR
            // fills a slice whose channel is absent from the file with its
            // `fillValue`, which is exactly the 0/0/0/1 default we want. That is
            // what lets us skip pre-filling the destination — a full-buffer write
            // that was redundant whenever RGBA were all present, i.e. always for
            // the sequence frames this path now decodes.
            add("R", 0, 0.0);
            add("G", 1, 0.0);
            add("B", 2, 0.0);
            add("A", 3, 1.0);
            job->file.setFrameBuffer(fb);
            job->file.readPixels(job->dw.min.y, job->dw.max.y);
        } else {
            add(job->lone_channel.c_str(), 0, 0.0);
            add("A", 3, 1.0); // absent -> filled with 1.0
            job->file.setFrameBuffer(fb);
            job->file.readPixels(job->dw.min.y, job->dw.max.y);
            for (size_t i = 0; i < n; ++i) {
                dst[i * 4 + 1] = dst[i * 4 + 0];
                dst[i * 4 + 2] = dst[i * 4 + 0];
            }
        }
        rc = 0;
    } catch (...) {
        rc = 1;
    }
    delete job;
    return rc;
}

extern "C" void exr_native_abort(void *handle) { delete static_cast<ExrJob *>(handle); }
