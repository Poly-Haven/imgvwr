//! Build script: compile the OCIO C++ shim, generate Rust bindings over its
//! plain-C header, link OpenColorIO, and stage the runtime DLL closure next to
//! the executable. See plans/rewrite.md §7.2.
//!
//! Gated on the `ocio` Cargo feature. With the feature off (an explicit dev
//! opt-out) this is a no-op and the crate builds the gamma-2.2 fallback.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/ocio/shim.cpp");
    println!("cargo:rerun-if-changed=src/ocio/shim.h");
    println!("cargo:rerun-if-changed=src/exr_native/shim.cpp");
    println!("cargo:rerun-if-changed=src/exr_native/shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");

    // Gamma-only fallback is an explicit opt-out, never a silent degrade.
    if std::env::var_os("CARGO_FEATURE_OCIO").is_none() {
        println!("cargo:warning=`ocio` feature disabled - building gamma-only fallback");
        return;
    }

    let vcpkg_root = std::env::var("VCPKG_ROOT").expect(
        "VCPKG_ROOT must be set to build with OCIO \
         (disable the `ocio` feature for a gamma-only dev build)",
    );
    let installed = PathBuf::from(&vcpkg_root).join("installed").join("x64-windows");
    let include_dir = installed.join("include");
    let lib_dir = installed.join("lib");
    let bin_dir = installed.join("bin");

    assert!(
        include_dir.join("OpenColorIO").exists(),
        "OpenColorIO headers not found under {} - run `vcpkg install opencolorio` (see README)",
        include_dir.display()
    );

    // 1. Compile the C++ shim with the same MSVC toolchain vcpkg used for OCIO.
    cc::Build::new()
        .cpp(true)
        .file("src/ocio/shim.cpp")
        .include(&include_dir)
        .std("c++17")
        .flag_if_supported("/EHsc")
        .compile("ocio_shim");

    // 2. Generate bindings over the shim's *C* header (not OCIO's C++ headers).
    ensure_libclang();
    let bindings = bindgen::Builder::default()
        .header("src/ocio/shim.h")
        .allowlist_function("ocio_.*")
        .allowlist_type("Ocio.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate OCIO shim bindings");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("ocio_shim_bindings.rs"))
        .expect("failed to write OCIO shim bindings");

    // 3. Compile the OpenEXR fallback shim (decodes DWAA/DWAB EXRs the pure-Rust
    //    `exr` crate cannot read) and bind it.
    cc::Build::new()
        .cpp(true)
        .file("src/exr_native/shim.cpp")
        .include(&include_dir)
        // OpenEXR's headers include Imath headers flat (e.g. "ImathBox.h"), so
        // the Imath (and OpenEXR) subdirs must be on the include path.
        .include(include_dir.join("Imath"))
        .include(include_dir.join("OpenEXR"))
        .std("c++17")
        .flag_if_supported("/EHsc")
        .compile("exr_shim");
    let exr_bindings = bindgen::Builder::default()
        .header("src/exr_native/shim.h")
        .allowlist_function("exr_native_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate OpenEXR shim bindings");
    exr_bindings
        .write_to_file(out_dir.join("exr_shim_bindings.rs"))
        .expect("failed to write OpenEXR shim bindings");

    // 4. Link OpenColorIO and OpenEXR (versioned import lib names are discovered
    //    from the vcpkg lib dir, e.g. OpenEXR-3_4.lib).
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=OpenColorIO");
    for stem in ["OpenEXR", "OpenEXRCore", "Imath", "Iex", "IlmThread"] {
        match find_versioned_lib(&lib_dir, stem) {
            Some(name) => println!("cargo:rustc-link-lib={name}"),
            None => println!("cargo:warning=could not find {stem}-*.lib under {}", lib_dir.display()),
        }
    }

    // 5. Stage the runtime DLL closure next to the executable so dev runs work
    //    without VCPKG bin on PATH. (CI packaging does its own bundling, §20.)
    stage_runtime_dlls(&bin_dir, &out_dir);
}

/// Find a versioned import lib like `OpenEXR-3_4.lib` and return its link name
/// (`OpenEXR-3_4`). Matches `{stem}-...` so "OpenEXR" doesn't match "OpenEXRCore".
fn find_versioned_lib(lib_dir: &Path, stem: &str) -> Option<String> {
    let prefix = format!("{stem}-");
    let entries = std::fs::read_dir(lib_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) && name.ends_with(".lib") {
            return Some(name.trim_end_matches(".lib").to_string());
        }
    }
    None
}

/// bindgen needs libclang. If `LIBCLANG_PATH` is not set, fall back to a known
/// local copy so a plain `cargo build` works on this machine. CI installs LLVM.
fn ensure_libclang() {
    if std::env::var_os("LIBCLANG_PATH").is_some() {
        return;
    }
    const FALLBACKS: &[&str] = &[
        r"C:\Program Files\LLVM\bin",
        r"C:\Program Files\Side Effects Software\Houdini 20.5.584\python311\lib\site-packages-forced\shiboken2_generator",
    ];
    for candidate in FALLBACKS {
        let dir = Path::new(candidate);
        if dir.join("libclang.dll").exists() {
            println!("cargo:warning=using libclang from {candidate}");
            std::env::set_var("LIBCLANG_PATH", candidate);
            return;
        }
    }
    println!(
        "cargo:warning=LIBCLANG_PATH not set and no fallback libclang found; \
         bindgen may fail. Install LLVM and set LIBCLANG_PATH."
    );
}

/// Copy every DLL from the vcpkg bin dir into the cargo target profile dir
/// (the parent of the build-script OUT_DIR chain), beside `imgvwr.exe`.
fn stage_runtime_dlls(bin_dir: &Path, out_dir: &Path) {
    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out  ->  profile dir is 3 up.
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.to_path_buf());

    let entries = match std::fs::read_dir(bin_dir) {
        Ok(e) => e,
        Err(e) => {
            println!("cargo:warning=cannot read vcpkg bin {}: {e}", bin_dir.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("dll") {
            continue;
        }
        if let Some(name) = path.file_name() {
            let dest = profile_dir.join(name);
            // Copy if missing or out of date (cheap mtime check).
            let needs_copy = match (std::fs::metadata(&dest), std::fs::metadata(&path)) {
                (Ok(d), Ok(s)) => match (d.modified(), s.modified()) {
                    (Ok(dm), Ok(sm)) => sm > dm,
                    _ => true,
                },
                _ => true,
            };
            if needs_copy {
                if let Err(e) = std::fs::copy(&path, &dest) {
                    println!("cargo:warning=failed to stage {}: {e}", name.to_string_lossy());
                }
            }
        }
    }
}
