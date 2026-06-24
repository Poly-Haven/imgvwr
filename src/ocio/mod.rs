//! Safe RAII wrapper over the OCIO C++ shim (see plans/rewrite.md §7.4).
//!
//! When the `ocio` feature is disabled, `OcioManager` is a stub that always
//! yields an empty `OcioShader`, which drives the renderer's gamma-2.2 fallback.

mod ffi;

use std::path::{Path, PathBuf};

/// A `(display, view)` pair from the active config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayView {
    pub display: String,
    pub view: String,
}

/// A LUT texture to upload: `dim` is 1 (sampler1D), 2 (sampler2D), or 3
/// (sampler3D). `channels` is 1 (R32F) or 3 (RGB32F).
pub struct OcioLut {
    pub sampler: String,
    pub dim: i32,
    pub width: i32,
    pub height: i32,
    pub depth: i32,
    pub channels: i32,
    pub linear: bool,
    pub data: Vec<f32>,
}

/// The result of building a GPU shader for the active display/view. An empty
/// `glsl_text` signals the renderer to use the gamma-2.2 fallback.
pub struct OcioShader {
    pub glsl_text: String,
    pub function_name: String,
    pub luts: Vec<OcioLut>,
    pub signature: String,
}

impl OcioShader {
    fn gamma_fallback() -> Self {
        Self {
            glsl_text: String::new(),
            function_name: String::new(),
            luts: Vec::new(),
            signature: "gamma2.2".to_string(),
        }
    }

    /// True when there is no OCIO program (renderer uses the gamma fallback).
    pub fn is_fallback(&self) -> bool {
        self.glsl_text.is_empty()
    }
}

#[cfg(feature = "ocio")]
struct ConfigHandle(*mut ffi::OcioConfig);

#[cfg(feature = "ocio")]
impl Drop for ConfigHandle {
    fn drop(&mut self) {
        unsafe { ffi::ocio_config_free(self.0) }
    }
}

pub struct OcioManager {
    #[cfg(feature = "ocio")]
    config: Option<ConfigHandle>,
    display_views: Vec<DisplayView>,
    active: Option<DisplayView>,
    source_colorspace: String,
    resources_dir: PathBuf,
}

impl OcioManager {
    /// Resolve and load a config from `resources_dir` (see §7.5 ordering).
    pub fn new(resources_dir: PathBuf) -> Self {
        let mut manager = Self {
            #[cfg(feature = "ocio")]
            config: None,
            display_views: Vec::new(),
            active: None,
            source_colorspace: String::new(),
            resources_dir,
        };
        manager.reload();
        manager
    }

    pub fn display_views(&self) -> &[DisplayView] {
        &self.display_views
    }

    pub fn active(&self) -> Option<&DisplayView> {
        self.active.as_ref()
    }

    pub fn source_colorspace(&self) -> &str {
        &self.source_colorspace
    }

    /// Distinct display names, in first-seen order.
    pub fn displays(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for dv in &self.display_views {
            if !out.contains(&dv.display) {
                out.push(dv.display.clone());
            }
        }
        out
    }

    /// Views available for `display`.
    pub fn views_for(&self, display: &str) -> Vec<String> {
        self.display_views
            .iter()
            .filter(|dv| dv.display == display)
            .map(|dv| dv.view.clone())
            .collect()
    }

    /// Select an active `(display, view)` if it exists in the config.
    pub fn set_active(&mut self, display: &str, view: &str) -> bool {
        let wanted = DisplayView {
            display: display.to_string(),
            view: view.to_string(),
        };
        if self.display_views.contains(&wanted) {
            self.active = Some(wanted);
            true
        } else {
            false
        }
    }

    #[cfg(feature = "ocio")]
    pub fn reload(&mut self) {
        self.config = None;
        self.display_views.clear();
        self.active = None;
        self.source_colorspace.clear();

        let raw = unsafe { self.discover_config() };
        if raw.is_null() {
            log::warn!("OCIO: no config could be loaded; using gamma-2.2 fallback");
            return;
        }
        let handle = ConfigHandle(raw);

        unsafe {
            let src = ffi::ocio_config_source_colorspace(raw);
            self.source_colorspace = cstr_to_string(src);

            let count = ffi::ocio_num_display_views(raw);
            let arr = ffi::ocio_display_views(raw);
            if !arr.is_null() {
                for i in 0..count {
                    let dv = &*arr.add(i);
                    self.display_views.push(DisplayView {
                        display: cstr_to_string(dv.display),
                        view: cstr_to_string(dv.view),
                    });
                }
            }
            let default_idx = ffi::ocio_config_default_index(raw);
            self.active = self.display_views.get(default_idx).cloned();
        }

        self.config = Some(handle);
        log::info!(
            "OCIO: loaded config, source colour space '{}', {} display/view pairs, default {:?}",
            self.source_colorspace,
            self.display_views.len(),
            self.active,
        );
        if self.source_colorspace.is_empty() {
            log::warn!("OCIO: scene-linear source colour space unresolved; shader build will fall back to gamma");
        }
    }

    /// Try each config source in priority order, returning the first that loads.
    ///
    /// # Safety
    /// Returns a raw pointer the caller must wrap in a `ConfigHandle`.
    #[cfg(feature = "ocio")]
    unsafe fn discover_config(&self) -> *mut ffi::OcioConfig {
        use std::ffi::CString;

        // 1. resources/ocio_configs/*.ocio (first alphabetically).
        let custom_dir = self.resources_dir.join("ocio_configs");
        if let Ok(entries) = std::fs::read_dir(&custom_dir) {
            let mut configs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("ocio"))
                .collect();
            configs.sort();
            if let Some(first) = configs.first() {
                if let Some(ptr) = try_load_file(first) {
                    log::info!("OCIO: using custom config {}", first.display());
                    return ptr;
                }
            }
        }

        // 2. resources/config.ocio (bundled Blender/AgX config).
        let bundled = self.resources_dir.join("config.ocio");
        if bundled.exists() {
            if let Some(ptr) = try_load_file(&bundled) {
                log::info!("OCIO: using bundled config {}", bundled.display());
                return ptr;
            }
        }

        // 3. OCIO built-in config containing "agx" or "aces".
        for substr in ["agx", "aces"] {
            if let Ok(c) = CString::new(substr) {
                let ptr = ffi::ocio_config_builtin(c.as_ptr());
                if !ptr.is_null() {
                    log::info!("OCIO: using built-in config matching '{substr}'");
                    return ptr;
                }
            }
        }

        // 4. Raw pass-through config.
        log::warn!("OCIO: falling back to raw pass-through config");
        ffi::ocio_config_raw()
    }

    #[cfg(feature = "ocio")]
    pub fn build_gpu_shader(&self) -> OcioShader {
        use std::ffi::CString;

        let (Some(config), Some(active)) = (&self.config, &self.active) else {
            return OcioShader::gamma_fallback();
        };
        if self.source_colorspace.is_empty() {
            return OcioShader::gamma_fallback();
        }

        let (Ok(display), Ok(view)) = (
            CString::new(active.display.as_str()),
            CString::new(active.view.as_str()),
        ) else {
            return OcioShader::gamma_fallback();
        };

        unsafe {
            let res = ffi::ocio_build_gpu_shader(config.0, display.as_ptr(), view.as_ptr());
            if res.is_null() {
                log::warn!(
                    "OCIO: shader build failed for {}/{}; using gamma fallback",
                    active.display,
                    active.view
                );
                return OcioShader::gamma_fallback();
            }
            let shader = extract_shader(res, active);
            ffi::ocio_shader_free(res);
            shader
        }
    }

    // ---- gamma-only build (feature disabled) -----------------------------

    #[cfg(not(feature = "ocio"))]
    pub fn reload(&mut self) {
        log::info!("OCIO feature disabled; using gamma-2.2 fallback");
    }

    #[cfg(not(feature = "ocio"))]
    pub fn build_gpu_shader(&self) -> OcioShader {
        OcioShader::gamma_fallback()
    }
}

#[cfg(feature = "ocio")]
fn try_load_file(path: &Path) -> Option<*mut ffi::OcioConfig> {
    use std::ffi::CString;
    let c = CString::new(path.to_string_lossy().as_ref()).ok()?;
    let ptr = unsafe { ffi::ocio_config_load(c.as_ptr()) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

#[cfg(feature = "ocio")]
unsafe fn extract_shader(res: *mut ffi::OcioShaderResult, active: &DisplayView) -> OcioShader {
    let glsl = cstr_to_string(ffi::ocio_shader_glsl(res));
    let function_name = cstr_to_string(ffi::ocio_shader_function(res));

    let mut luts = Vec::new();
    let count = ffi::ocio_shader_num_textures(res);
    let arr = ffi::ocio_shader_textures(res);
    if !arr.is_null() {
        for i in 0..count {
            let t = &*arr.add(i);
            let data = std::slice::from_raw_parts(t.data, t.data_len).to_vec();
            luts.push(OcioLut {
                sampler: cstr_to_string(t.sampler_name),
                dim: t.dim,
                width: t.width,
                height: t.height,
                depth: t.depth,
                channels: t.channels,
                linear: t.interpolation != 0,
                data,
            });
        }
    }

    OcioShader {
        glsl_text: glsl,
        function_name,
        luts,
        signature: format!("{}/{}", active.display, active.view),
    }
}

#[cfg(feature = "ocio")]
unsafe fn cstr_to_string(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
