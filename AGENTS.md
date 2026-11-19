# AGENTS.md

## Comments

the style defined in here are non-negotiable. whenever you make a change it should always follow this style. do not touch existing comments that don't adhere to this style indiscriminately, but if you change the comment, also rewrite it to follow this style.

One thought per line. A thought is usually a sentence; keep closely related clauses of the same sentence together, and do not split at every comma. Two short sentences may share a line when they make one point. Never hard-wrap at 80 columns: break at a sentence boundary, even if the line runs long.

A colon that introduces a list normally ends the line. Each item gets its own line and its own period, and the joining "or"/"and" is dropped. A short list that fits on one continuation line may stay there. A colon that introduces an explanation may continue on the next line.

```cpp
// The solver picks a strategy:
// scan the wordlist for a single word.
// compose a phrase, when the clue contains spaces.
```

Start a new paragraph (blank comment line) when the topic changes. Copy-edit the seams ("plus" -> "and", comma before a trailing "when ..."), but keep the content: no shortening, no dropped opinions, no dashed bullets.

The same applies to `///` doc comments and `#` CMake comments.

## Commits

The comment rules apply to commit messages: one thought per line, break at a sentence boundary instead of at 80 columns, and a blank line between topics.

The subject is one imperative line with no prefix and no trailing period, and it names the change as a whole. "Let the scene tables own the storage and the writes" is a subject; "Update SceneRenderer" is a file list.

A trivial edit can stand on its subject alone. Anything a reader would have to reconstruct from the diff gets a body that says what changed and why, and the body covers the whole change: a message that describes the rename but not the comment rewrite, or that stops at the first file, misrepresents its own commit.

State what the code does now and the invariant it keeps, and give the measured effect where there is one, such as barriers per frame or a test count.

Optionally end with a `Verified:` paragraph when you built or ran something, naming the presets or commands and their result, for example `Verified: cmake --build --preset debug; ctest --preset debug (1060 tests, all pass)`. If nothing was built or run, leave it out rather than claim a check that did not happen.

## Build & test preflight

The `default` configure preset reads `VCPKG_ROOT`; CMake silently falls back to a
different preset if it is unset, first check if it exists, if not export it first:

```bash
export VCPKG_ROOT="$HOME/vcpkg"      # or wherever vcpkg lives
cmake --preset default -S . -B build
cmake --build --preset debug
ctest --preset debug
```

- Build presets: `debug`, `release`, `relwithdebinfo` (Clang + libc++),
  `msvc-debug`, `msvc-release`, `msvc-relwithdebinfo` (MSVC + Microsoft STL),
  and the `windows-*` cross presets.
- Test presets mirror them (device-free; they exclude the `gpu` label), plus
  `debug-gpu`, `release-gpu`, `relwithdebinfo-gpu` and `msvc-debug-gpu` for the
  opt-in GPU tests.
- The strict component-access gate is a separate configure tree, because it
  differs by an option value: `check` (inherits `default`, `build-check/`,
  `VKENGINE_COMPONENT_CHECKS=ON`) with `check-debug` / `check-release` build and
  test presets. `check-release` runs the component access validator under
  `NDEBUG`; the plain `release` preset is the shipping configuration where it is
  compiled out. Both must pass. See `docs/component-access-and-scheduling.md`.
- The `windows` preset sets `BUILD_TESTING=OFF` and `VKENGINE_HOT_RELOAD=OFF`.
- C++23 modules (`import std;`) are used throughout. Supported compilers:
  Clang with libc++ (`USE_LIBCPP=ON`, the `default` preset; static libc++ in
  Release) and MSVC 19.40+ with the Microsoft STL. The `msvc` preset needs
  Ninja Multi-Config (the Visual Studio generators cannot build the imported
  `std` BMI) and the *C++ Modules for the Standard Library* VS component.
  Clang against libstdc++ (`-DUSE_LIBCPP=OFF`) does not build: Clang 21 rejects
  libstdc++ 15's `<unordered_map>` ("cannot befriend target of using
  declaration").
- GCC is not supported: 15.3 ICEs in its module implementation on this
  project's module interfaces, and 16.2.1 produces corrupt BMIs (`bad file
  data` while reading a `.gcm`) and rejects the standard-header mixing in the
  glTF module. Do not add a `gcc` preset back until that is fixed; the
  Steam-runtime ABI (`GLIBCXX_3.4.33`) also rules out GCC 15/16 binaries.
- `VKENGINE_STEAM_RUNTIME` (`off|scout|sniper|steamrt4`) turns on a post-link
  check that fails shipped executables requiring a newer `GLIBCXX_*`,
  `CXXABI_*` or `GLIBC_*` symbol version than the target runtime, or linking
  libstdc++ statically. libc++ builds are skipped (they do not load the
  runtime's libstdc++), so today only the `GLIBC_*` ceiling is exercised.
- Slang/hot-reload is gated by two knobs: `VKENGINE_HOT_RELOAD` (Debug always,
  when ON) and `VKENGINE_HOT_RELOAD_RELWITHDEBINFO` (default ON; set OFF to
  ship a slang-free RelWithDebInfo). Slang-free configurations
  (`VKENGINE_STATIC_LIBCPP`, default ON) statically embed libc++ + libc++abi
  via the `project_libcpp_static` interface target, which every C++ executable
  must link last; the static link requires the `libcxx-static` archive
  (`libc++.a`), else configure warns and falls back to the dynamic link.
  Build-time host tools (slang-spirv-compiler) stay fully dynamic.
- `libcpp-sanitize` (top-level CMakeLists, after the lld probe): distro
  static libc++ archives ship with broken addrsig tables (RPM packaging
  zeroes sh_link), which makes lld warn once per member under `--icf=safe`.
  The build auto-traces which archives the linker resolves, generates
  sanitized copies into `<build>/libcpp-sanitized/` with a Ninja custom
  command (auto-regenerates when the source archive changes), and remaps
  the link to them via `-Wl,--remap-inputs`. Folding results are identical;
  the binaries just link quietly. Binaries stay byte-identical in size.
- Never add per-command `cd`; pass the working directory to the tool instead.

## Test conventions

- **Default tests must be device-free.** Test bodies must not create a Vulkan
  device. Device-bound tests live behind the `gpu` ctest label and are excluded
  by the default test presets.
- Register a GPU test target with `setup_test_target(<target> LABELS gpu)`.
- Guard every GPU test body with the availability probe:

  ```cpp
  if (!TestSupport::IsGpuDeviceAvailable()) {
      GTEST_SKIP() << "no Vulkan device available";
  }
  ```

  `test_gpu.cppm` creates only an instance and enumerates physical devices, so it
  is safe to call and costs nothing when no device exists.
- Shared fakes live in `tests/support/`. `TestSupport::FakeVulkanBootstrapBackend`
  (`test_vulkan_fakes.cppm`) implements the lifecycle surface; every accessor
  that would return a live Vulkan handle throws on purpose. Add the modules to a
  target with `enable_test_support_modules(<target>)`.
- `gtest_discover_tests` runs each test binary at build time to enumerate cases
  (`--gtest_list_tests`). Globals constructed before `main` must not open a
  device, or the build itself fails.
- Golden GPU hashes live in the test source; if a render is intentionally
  changed, update the constant in the same commit and say so in the message.
