/* shim.h - plain C surface over OpenColorIO. No C++ types cross this boundary.
 *
 * This is the only header bindgen ever sees (see plans/rewrite.md §7.3). The
 * implementation in shim.cpp does all the C++ work (exceptions, GpuShaderDesc
 * walking, std::string -> owned char*) and returns flat, owned POD structs. */
#ifndef IMGVWR_OCIO_SHIM_H
#define IMGVWR_OCIO_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcioConfig OcioConfig;             /* opaque */
typedef struct OcioShaderResult OcioShaderResult; /* opaque */

typedef struct {
    const char *display;
    const char *view;
} OcioDisplayView;

/* A single LUT texture extracted from the GPU shader description. */
typedef struct {
    const char *sampler_name; /* GLSL sampler uniform name to bind */
    int dim;                  /* 2 = sampler2D, 3 = sampler3D */
    int width;                /* texels in X */
    int height;               /* texels in Y (1 for a 1D-as-2D lut) */
    int depth;                /* edge length for 3D, else 1 */
    int channels;             /* 1 (R) or 3 (RGB) */
    int interpolation;        /* 0 = nearest, 1 = linear */
    const float *data;        /* width*height*depth*channels floats, owned */
    size_t data_len;          /* element count of `data` */
} OcioLutTexture;

/* --- config lifecycle --------------------------------------------------- */
OcioConfig *ocio_config_load(const char *path);           /* NULL on failure */
OcioConfig *ocio_config_builtin(const char *name_substr); /* "agx"/"aces"/... */
OcioConfig *ocio_config_raw(void);
void ocio_config_free(OcioConfig *);

/* The resolved scene-linear source colour space name, or "" if unresolved. */
const char *ocio_config_source_colorspace(const OcioConfig *);

/* --- display / view enumeration ----------------------------------------- */
size_t ocio_num_display_views(const OcioConfig *);
const OcioDisplayView *ocio_display_views(const OcioConfig *); /* borrowed */
/* Index into the display-views array of the config's default display+view. */
size_t ocio_config_default_index(const OcioConfig *);

/* --- GPU shader build --------------------------------------------------- */
/* Builds the GLSL_4_0 shader for source -> (display, view). NULL on failure. */
OcioShaderResult *ocio_build_gpu_shader(const OcioConfig *, const char *display,
                                        const char *view);
const char *ocio_shader_glsl(const OcioShaderResult *);
const char *ocio_shader_function(const OcioShaderResult *);
size_t ocio_shader_num_textures(const OcioShaderResult *);
const OcioLutTexture *ocio_shader_textures(const OcioShaderResult *); /* array */
void ocio_shader_free(OcioShaderResult *);

#ifdef __cplusplus
}
#endif

#endif /* IMGVWR_OCIO_SHIM_H */
