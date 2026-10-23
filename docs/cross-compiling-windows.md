# Cross-compiling for Windows (Linux → Windows x86_64)

This project can be cross-compiled from a Linux x86_64 host to Windows x86_64
using [llvm-mingw](https://github.com/mstorsjo/llvm-mingw): a self-contained
Clang/LLD toolchain that ships the mingw-w64 sysroot, compiler-rt, libunwind and
libc++ — **including the libc++ standard-library module metadata**, which is what
makes `import std;` work on the Windows target.

## 1. Prerequisites

- Linux x86_64 build host
- CMake 3.30–3.31, 4.3 or 4.4 (those are the releases whose `import std;`
  activation value `CMakeLists.txt` has verified). 4.2 introduced
  `CMAKE_CXX_STDLIB_MODULES_JSON`, which the cross toolchain uses to point CMake
  at the *target* libc++ module metadata. Any other release stops at configure
  with a message naming the variable to set, rather than failing later; if you
  use one, add its band to the `CMAKE_EXPERIMENTAL_CXX_IMPORT_STD` block in
  `CMakeLists.txt` using the value from that CMake's
  `Help/dev/experimental.rst`, or pass
  `-DCMAKE_EXPERIMENTAL_CXX_IMPORT_STD=<value>` yourself.
- Ninja
- vcpkg, with `VCPKG_ROOT` set
- *optional*: `wine`, to run the resulting executables

## 2. Download llvm-mingw into the project

The toolchain is expected **inside this project** at `toolchains/`. That
directory is listed in `.gitignore`, so the toolchain is never committed and
nothing outside the checkout is required.

Download the newest **ucrt** release for your Linux host from
<https://github.com/mstorsjo/llvm-mingw/releases> into `toolchains/`, then extract
it in place:

```bash
cd /path/to/MazeGenerator

mkdir -p toolchains
# put the newest llvm-mingw-*-ucrt-ubuntu-*-x86_64.tar.xz here
sha256sum toolchains/llvm-mingw-*.tar.xz   # compare against the release page

tar -xf toolchains/llvm-mingw-*.tar.xz -C toolchains
rm toolchains/llvm-mingw-*.tar.xz
```

After extracting you have a `toolchains/llvm-mingw-.../` directory.

Use the **ucrt** variant (the msvcrt variant is only for very old Windows).
`cmake/toolchains/llvm-mingw.cmake` auto-detects `toolchains/llvm-mingw*`, so no
environment variables are needed. To keep the toolchain somewhere else, pass
`-DVKENGINE_LLVM_MINGW_ROOT=/path/to/llvm-mingw` (or set the environment
variable of the same name).

Sanity check — the second file is the one that enables `import std;`:

```bash
toolchains/llvm-mingw-*/bin/x86_64-w64-mingw32-clang++ --version
ls toolchains/llvm-mingw-*/x86_64-w64-mingw32/lib/libc++.modules.json
```

## 3. Configure

```bash
export VCPKG_ROOT=/path/to/vcpkg   # only needed if not already set
cmake --preset windows
```

The `windows` preset selects the `x64-mingw-libcxx` vcpkg triplet, enables
libc++, and turns off the options that are not available for a MinGW target.

## 4. Build

```bash
cmake --build --preset windows-release        # or windows-debug / windows-relwithdebinfo
```

Resulting executables land in `build-windows/bin/apps/<Config>/<name>/`, e.g.
`build-windows/bin/apps/Release/MazeGeneratorApp/MazeGeneratorApp.exe`, together
with the runtime DLLs that `add_app` copies next to them.

## 5. Run (optional, via wine)

```bash
wine build-windows/bin/apps/Release/MazeGeneratorApp/MazeGeneratorApp.exe
```

The `vulkan-1.dll` next to the executable is deployed by the build, so wine does
not need a Windows driver installation. The `fixme:` lines wine prints are
normal noise.

## How this differs from the native Linux build

- **Shader hot reload is off** (`VKENGINE_HOT_RELOAD=OFF`). The only Windows
  Slang build available is MSVC-produced and cannot be linked from a MinGW
  target. Shaders are still compiled ahead of time, so the engine behaves
  normally; only runtime recompilation is unavailable.
- **`slang-spirv-compiler` is a build-time host tool.** It runs on the build
  machine, so for a cross build it is built for Linux in
  `build-windows/host-tools/` and executed during the build to emit the
  `.spv`/`.cppm` files that are then compiled for the Windows target. Its
  Shader-Slang dependency comes from this project's overlay port, so the host
  tool and the target always use the same Slang release.
- **Sanitizers are unavailable** on MinGW and are forced off.
- **Windows-only dependencies**: `vulkan-loader` is installed because imgui's
  CMake config hard-links `Vulkan::Vulkan`; `shader-slang` is *not* installed for
  the Windows triplet (see above).

## Troubleshooting

- **"Stale CMake build tree"** — the toolchain refuses to configure a build
  directory that was created before the cross toolchain existed (CMake keeps a
  cached `CMakeSystem.cmake` reporting a native system). Delete the directory
  and configure again:
  ```bash
  rm -rf build-windows && cmake --preset windows
  ```
- **Slang errors about missing built-ins (e.g. `CDataLayout`)** — the host
  shader compiler was built against a different Shader-Slang release than the
  target. Rebuild the host tool:
  ```bash
  rm -rf build-windows/host-tools build-windows/slang_spirv_compiler_host-prefix
  cmake --build --preset windows-release
  ```
- **`wine: Library libc++.dll (or libunwind.dll) not found`** — run the
  executable from its output directory (the build copies the llvm-mingw
  runtimes there), e.g.
  `wine build-windows/bin/apps/Release/MazeGeneratorApp/MazeGeneratorApp.exe`.
- **Updating llvm-mingw** — download a newer release into `toolchains/` (the
  newest `toolchains/llvm-mingw*` directory is picked up automatically) and
  delete `build-windows`.

## Related files

- `cmake/toolchains/llvm-mingw.cmake` — the cross toolchain file
- `cmake/triplets/x64-mingw-libcxx.cmake` — the vcpkg triplet
- `cmake/CopyRuntimeDlls.cmake` — runtime DLL deployment
- `CMakePresets.json` — the `windows` configure preset and `windows-*` build presets
