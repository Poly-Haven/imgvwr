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
#include <Imath/ImathBox.h>

#include <cstdlib>
#include <string>

namespace IMF = OPENEXR_IMF_NAMESPACE;

extern "C" float *exr_native_load_rgba(const char *path, int *out_width, int *out_height,
                                       int *out_channels) {
    try {
        IMF::InputFile file(path);
        Imath::Box2i dw = file.header().dataWindow();
        const int width = dw.max.x - dw.min.x + 1;
        const int height = dw.max.y - dw.min.y + 1;
        if (width <= 0 || height <= 0) {
            return nullptr;
        }
        const size_t n = static_cast<size_t>(width) * height;
        float *rgba = static_cast<float *>(std::malloc(n * 4 * sizeof(float)));
        if (!rgba) {
            return nullptr;
        }
        for (size_t i = 0; i < n; ++i) {
            rgba[i * 4 + 0] = 0.0f;
            rgba[i * 4 + 1] = 0.0f;
            rgba[i * 4 + 2] = 0.0f;
            rgba[i * 4 + 3] = 1.0f;
        }

        const IMF::ChannelList &channels = file.header().channels();
        const bool hasR = channels.findChannel("R") != nullptr;
        const bool hasG = channels.findChannel("G") != nullptr;
        const bool hasB = channels.findChannel("B") != nullptr;
        const bool hasA = channels.findChannel("A") != nullptr;

        const size_t xs = 4 * sizeof(float);
        const size_t ys = static_cast<size_t>(width) * 4 * sizeof(float);
        // Base accounts for the data-window origin (OpenEXR slice convention).
        char *base = reinterpret_cast<char *>(rgba) -
                     (static_cast<size_t>(dw.min.x) * xs + static_cast<size_t>(dw.min.y) * ys);

        IMF::FrameBuffer fb;
        auto add = [&](const char *name, int offset, double dflt) {
            fb.insert(name, IMF::Slice(IMF::FLOAT,
                                       base + static_cast<size_t>(offset) * sizeof(float), xs, ys,
                                       1, 1, dflt));
        };

        int orig_channels = (hasR ? 1 : 0) + (hasG ? 1 : 0) + (hasB ? 1 : 0) + (hasA ? 1 : 0);

        if (hasR || hasG || hasB) {
            if (hasR) add("R", 0, 0.0);
            if (hasG) add("G", 1, 0.0);
            if (hasB) add("B", 2, 0.0);
            if (hasA) add("A", 3, 1.0);
            file.setFrameBuffer(fb);
            file.readPixels(dw.min.y, dw.max.y);
        } else {
            std::string first;
            for (IMF::ChannelList::ConstIterator it = channels.begin(); it != channels.end();
                 ++it) {
                first = it.name();
                break;
            }
            if (first.empty()) {
                std::free(rgba);
                return nullptr;
            }
            add(first.c_str(), 0, 0.0);
            file.setFrameBuffer(fb);
            file.readPixels(dw.min.y, dw.max.y);
            for (size_t i = 0; i < n; ++i) {
                rgba[i * 4 + 1] = rgba[i * 4 + 0];
                rgba[i * 4 + 2] = rgba[i * 4 + 0];
            }
            orig_channels = 1;
        }

        *out_width = width;
        *out_height = height;
        *out_channels = orig_channels > 0 ? orig_channels : 3;
        return rgba;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void exr_native_free(float *data) { std::free(data); }
