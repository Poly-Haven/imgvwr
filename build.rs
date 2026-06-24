// Commit 1 placeholder build script.
//
// The real OCIO build logic (compile the C++ shim with `cc`, run `bindgen` over
// `src/ocio/shim.h`, and emit link directives for the vcpkg-installed
// OpenColorIO) is wired in at Commit 7 — see plans/rewrite.md §7.2.
//
// Until then this is intentionally a no-op so that `cargo build`
// (with or without the `ocio` feature) succeeds without vcpkg present.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:warning=imgvwr: build.rs placeholder (OCIO is wired in at Commit 7)");
}
