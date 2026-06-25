# Third-Party Licenses

`imgvwr` itself is distributed under the MIT License (see `LICENSE`). The
distributed application and its bundled `resources/` directory additionally
include third-party software and assets that carry their own terms. Each is
attributed below. Before any public release, re-audit every bundled LUT/config
for its actual license.

---

## Bundled C/C++ libraries (shipped as DLLs beside `imgvwr.exe`)

### OpenColorIO — BSD 3-Clause License
Copyright Contributors to the OpenColorIO Project.
<https://github.com/AcademySoftwareFoundation/OpenColorIO>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the BSD-3-Clause conditions are met.

OpenColorIO pulls in a transitive dependency closure that is also redistributed
as DLLs. Each carries its own permissive license:

- **Imath** — BSD 3-Clause (Contributors to the OpenEXR Project)
- **yaml-cpp** — MIT License (Jesse Beder)
- **expat** — MIT License (Thai Open Source Software Center Ltd, Clark Cooper)
- **pystring** — BSD 3-Clause (Contributors to the OpenColorIO Project)
- **minizip / zlib (`z`)** — zlib License (Jean-loup Gailly, Mark Adler)
- **zstd** — BSD 3-Clause / GPLv2 dual (Meta Platforms, Inc.) — BSD terms used
- **bzip2 (`bz2`)** — BSD-style License (Julian R. Seward)
- **xz / liblzma** — Public Domain / 0-BSD (Lasse Collin and contributors)

### OpenEXR — BSD 3-Clause License
Copyright Contributors to the OpenEXR Project.
<https://github.com/AcademySoftwareFoundation/OpenEXR>

Used as a fallback EXR decoder for compressions (DWAA/DWAB) the pure-Rust `exr`
crate cannot read. Ships `OpenEXR-*.dll`, `OpenEXRCore-*.dll`, `Iex-*.dll`, and
`IlmThread-*.dll` (plus Imath, above).

### Little CMS 2 (lcms2) — MIT License
Copyright (c) 1998-2024 Marti Maria Saguer.
<https://github.com/mm2/Little-CMS>

---

## Bundled OCIO configuration and LUTs (`resources/`)

### Blender / AgX OCIO configuration (`resources/config.ocio`)
The bundled config is the Blender colour-management configuration based on AgX
by Troy James Sobotka, further developed by Zijun Eary Zhou, Mark Faderbauer,
and Sakari Kapanen. See the header of `config.ocio` for the full attribution.

- AgX: <https://github.com/sobotka/AgX>
- AgX (this version): <https://github.com/EaryChow/AgX>

The Filmic Dynamic Range LUT configuration was crafted by Troy James Sobotka
with feedback from the Blender community (see the `config.ocio` header).

These configurations and their associated LUTs (`resources/luts/*`,
`resources/filmic/*`) originate from the Blender project and are distributed
under their respective open-source terms. Consult the upstream repositories for
the authoritative license text.

---

## Bundled font (`resources/fonts/`)

### Inter — SIL Open Font License 1.1
Copyright (c) 2016 The Inter Project Authors
(<https://github.com/rsms/inter>).

`Inter-Regular.otf` is bundled and used as the UI font so the interface renders
identically (and crisply) on every machine. The full license text ships
alongside it at `resources/fonts/OFL.txt`.

---

## Bundled icons (`resources/icons/ui/`)

### Bootstrap Icons — MIT License
Copyright (c) 2019–2024 The Bootstrap Authors
(<https://github.com/twbs/icons>).

A handful of Bootstrap Icons SVGs (gear, folder, window controls, chevron) are
bundled and rasterised at runtime for the titlebar / menu controls. Bootstrap
Icons is released under the MIT License.

---

## Rust crates

All Rust crate dependencies are licensed under MIT and/or Apache-2.0. Run
`cargo about` or `cargo license` to regenerate the full per-crate manifest.
