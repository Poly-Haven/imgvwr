// shim.cpp - C++ implementation of the plain-C OCIO surface declared in shim.h.
//
// All OCIO exceptions are caught at the boundary; nothing unwinds across the
// extern "C" interface. Everything returned to Rust is owned POD (flat arrays
// and C strings) that lives until the matching *_free call.

#include "shim.h"

#include <OpenColorIO/OpenColorIO.h>

#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace OCIO = OCIO_NAMESPACE;

namespace {

// Candidate source colour spaces, in priority order. We always feed the shader
// scene-linear data, so the OCIO transform must start from a scene-linear space.
const char *const kSourceCandidates[] = {
    nullptr, // placeholder for ROLE_SCENE_LINEAR, filled in at runtime
    "Linear Rec.709 (sRGB)",
    "Linear Rec.709",
    "lin_rec709",
    "Linear",
    "scene-linear Rec.709-sRGB",
};

std::string resolve_source_colorspace(const OCIO::ConstConfigRcPtr &config) {
    // Prefer the scene_linear role.
    try {
        auto cs = config->getColorSpace(OCIO::ROLE_SCENE_LINEAR);
        if (cs) {
            return std::string(cs->getName());
        }
    } catch (...) {
    }
    for (const char *name : kSourceCandidates) {
        if (!name) {
            continue;
        }
        try {
            auto cs = config->getColorSpace(name);
            if (cs) {
                return std::string(cs->getName());
            }
        } catch (...) {
        }
    }
    return std::string();
}

} // namespace

struct OcioConfig {
    OCIO::ConstConfigRcPtr config;
    std::string source_cs;
    // Backing storage for display/view name strings, kept alive for borrowers.
    std::vector<std::string> names;
    std::vector<OcioDisplayView> dvs;
    size_t default_index = 0;
};

struct OcioShaderResult {
    std::string glsl;
    std::string function;
    std::vector<std::string> sampler_names; // stable backing storage
    std::vector<std::vector<float>> buffers; // stable backing storage
    std::vector<OcioLutTexture> textures;
};

static OcioConfig *finalize_config(OCIO::ConstConfigRcPtr cfg) {
    if (!cfg) {
        return nullptr;
    }
    auto *out = new (std::nothrow) OcioConfig();
    if (!out) {
        return nullptr;
    }
    out->config = cfg;
    out->source_cs = resolve_source_colorspace(cfg);

    try {
        const int num_displays = cfg->getNumDisplays();
        // Reserve so the c_str() pointers we hand out stay stable.
        out->names.reserve(static_cast<size_t>(num_displays) * 8);
        for (int d = 0; d < num_displays; ++d) {
            const char *display = cfg->getDisplay(d);
            if (!display) {
                continue;
            }
            const int num_views = cfg->getNumViews(display);
            for (int v = 0; v < num_views; ++v) {
                const char *view = cfg->getView(display, v);
                if (!view) {
                    continue;
                }
                out->names.emplace_back(display);
                out->names.emplace_back(view);
            }
        }
        out->dvs.reserve(out->names.size() / 2);
        for (size_t i = 0; i + 1 < out->names.size(); i += 2) {
            OcioDisplayView dv;
            dv.display = out->names[i].c_str();
            dv.view = out->names[i + 1].c_str();
            out->dvs.push_back(dv);
        }

        // Resolve the config's default display + view to an index.
        const char *def_display = cfg->getDefaultDisplay();
        const char *def_view = def_display ? cfg->getDefaultView(def_display) : nullptr;
        if (def_display && def_view) {
            for (size_t i = 0; i < out->dvs.size(); ++i) {
                if (std::strcmp(out->dvs[i].display, def_display) == 0 &&
                    std::strcmp(out->dvs[i].view, def_view) == 0) {
                    out->default_index = i;
                    break;
                }
            }
        }
    } catch (...) {
        // Enumeration failure is non-fatal; the config still exists.
        out->dvs.clear();
    }
    return out;
}

extern "C" {

OcioConfig *ocio_config_load(const char *path) {
    if (!path) {
        return nullptr;
    }
    try {
        return finalize_config(OCIO::Config::CreateFromFile(path));
    } catch (...) {
        return nullptr;
    }
}

OcioConfig *ocio_config_builtin(const char *name_substr) {
    try {
        const auto &registry = OCIO::BuiltinConfigRegistry::Get();
        const size_t count = registry.getNumBuiltinConfigs();
        std::string want = name_substr ? name_substr : "";
        for (auto &c : want) {
            c = static_cast<char>(::tolower(c));
        }
        for (size_t i = 0; i < count; ++i) {
            std::string name = registry.getBuiltinConfigName(i);
            std::string lower = name;
            for (auto &c : lower) {
                c = static_cast<char>(::tolower(c));
            }
            if (want.empty() || lower.find(want) != std::string::npos) {
                return finalize_config(OCIO::Config::CreateFromBuiltinConfig(name.c_str()));
            }
        }
    } catch (...) {
    }
    return nullptr;
}

OcioConfig *ocio_config_raw(void) {
    try {
        return finalize_config(OCIO::Config::CreateRaw());
    } catch (...) {
        return nullptr;
    }
}

void ocio_config_free(OcioConfig *cfg) { delete cfg; }

const char *ocio_config_source_colorspace(const OcioConfig *cfg) {
    return cfg ? cfg->source_cs.c_str() : "";
}

size_t ocio_num_display_views(const OcioConfig *cfg) {
    return cfg ? cfg->dvs.size() : 0;
}

const OcioDisplayView *ocio_display_views(const OcioConfig *cfg) {
    return (cfg && !cfg->dvs.empty()) ? cfg->dvs.data() : nullptr;
}

size_t ocio_config_default_index(const OcioConfig *cfg) {
    return cfg ? cfg->default_index : 0;
}

OcioShaderResult *ocio_build_gpu_shader(const OcioConfig *cfg, const char *display,
                                        const char *view) {
    if (!cfg || !display || !view || cfg->source_cs.empty()) {
        return nullptr;
    }
    try {
        auto transform = OCIO::DisplayViewTransform::Create();
        transform->setSrc(cfg->source_cs.c_str());
        transform->setDisplay(display);
        transform->setView(view);

        auto processor = cfg->config->getProcessor(transform);
        auto gpu = processor->getDefaultGPUProcessor();

        auto desc = OCIO::GpuShaderDesc::CreateShaderDesc();
        desc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_4_0);
        desc->setFunctionName("ocio_display_transform");
        desc->setResourcePrefix("ocio_");
        gpu->extractGpuShaderInfo(desc);

        auto *out = new (std::nothrow) OcioShaderResult();
        if (!out) {
            return nullptr;
        }
        out->glsl = desc->getShaderText() ? desc->getShaderText() : "";
        out->function = "ocio_display_transform";

        const unsigned n2d = desc->getNumTextures();
        const unsigned n3d = desc->getNum3DTextures();
        const size_t total = static_cast<size_t>(n2d) + n3d;
        out->sampler_names.reserve(total);
        out->buffers.reserve(total);

        struct Meta {
            int dim, width, height, depth, channels, interp;
        };
        std::vector<Meta> metas;
        metas.reserve(total);

        // 2D / 1D-as-2D textures.
        for (unsigned i = 0; i < n2d; ++i) {
            const char *texName = nullptr;
            const char *sampName = nullptr;
            unsigned width = 0, height = 0;
            OCIO::GpuShaderDesc::TextureType channel = OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL;
            OCIO::GpuShaderDesc::TextureDimensions dims = OCIO::GpuShaderDesc::TEXTURE_2D;
            OCIO::Interpolation interp = OCIO::INTERP_LINEAR;
            desc->getTexture(i, texName, sampName, width, height, channel, dims, interp);

            const float *values = nullptr;
            desc->getTextureValues(i, values);

            const int channels = (channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL) ? 1 : 3;
            const size_t count = static_cast<size_t>(width) * height * channels;
            // OCIO emits 1D LUTs as sampler1D (small) or sampler2D (large); honour it.
            const int dim = (dims == OCIO::GpuShaderDesc::TEXTURE_1D) ? 1 : 2;
            out->sampler_names.emplace_back(sampName ? sampName : "");
            out->buffers.emplace_back(values, values + count);
            metas.push_back(Meta{dim, static_cast<int>(width), static_cast<int>(height), 1,
                                 channels, interp == OCIO::INTERP_NEAREST ? 0 : 1});
        }

        // 3D textures (always RGB).
        for (unsigned i = 0; i < n3d; ++i) {
            const char *texName = nullptr;
            const char *sampName = nullptr;
            unsigned edge = 0;
            OCIO::Interpolation interp = OCIO::INTERP_LINEAR;
            desc->get3DTexture(i, texName, sampName, edge, interp);

            const float *values = nullptr;
            desc->get3DTextureValues(i, values);

            const size_t count = static_cast<size_t>(edge) * edge * edge * 3;
            out->sampler_names.emplace_back(sampName ? sampName : "");
            out->buffers.emplace_back(values, values + count);
            metas.push_back(Meta{3, static_cast<int>(edge), static_cast<int>(edge),
                                 static_cast<int>(edge), 3,
                                 interp == OCIO::INTERP_NEAREST ? 0 : 1});
        }

        // Build the flat texture array now that backing storage is stable.
        out->textures.reserve(total);
        for (size_t i = 0; i < total; ++i) {
            OcioLutTexture t;
            t.sampler_name = out->sampler_names[i].c_str();
            t.dim = metas[i].dim;
            t.width = metas[i].width;
            t.height = metas[i].height;
            t.depth = metas[i].depth;
            t.channels = metas[i].channels;
            t.interpolation = metas[i].interp;
            t.data = out->buffers[i].data();
            t.data_len = out->buffers[i].size();
            out->textures.push_back(t);
        }
        return out;
    } catch (...) {
        return nullptr;
    }
}

const char *ocio_shader_glsl(const OcioShaderResult *r) {
    return r ? r->glsl.c_str() : "";
}

const char *ocio_shader_function(const OcioShaderResult *r) {
    return r ? r->function.c_str() : "";
}

size_t ocio_shader_num_textures(const OcioShaderResult *r) {
    return r ? r->textures.size() : 0;
}

const OcioLutTexture *ocio_shader_textures(const OcioShaderResult *r) {
    return (r && !r->textures.empty()) ? r->textures.data() : nullptr;
}

void ocio_shader_free(OcioShaderResult *r) { delete r; }

} // extern "C"
