# AGENTS.md

## Build & test preflight

The `default` configure preset reads `VCPKG_ROOT`; CMake silently falls back to a
different preset if it is unset, first check if it exists, if not export it first:

```bash
export VCPKG_ROOT="$HOME/vcpkg"      # or wherever vcpkg lives
cmake --preset default -S . -B build
cmake --build --preset debug
ctest --preset debug
```

- Build presets: `debug`, `release`, `relwithdebinfo`.
- Test presets: `debug`, `release`, `relwithdebinfo` (device-free; they exclude
  the `gpu` label), plus `debug-gpu`, `release-gpu`, `relwithdebinfo-gpu` for the
  opt-in GPU tests.
- The `windows` preset sets `BUILD_TESTING=OFF` and `VKENGINE_HOT_RELOAD=OFF`.
- Compiler is Clang only; C++23 modules (`import std;`) are used throughout, with
  libc++ (`USE_LIBCPP=ON` in the default preset).
- Slang/hot-reload is gated by two knobs: `VKENGINE_HOT_RELOAD` (Debug always,
  when ON) and `VKENGINE_HOT_RELOAD_RELWITHDEBINFO` (default ON; set OFF to
  ship a slang-free RelWithDebInfo). Slang-free configurations
  (`VKENGINE_STATIC_LIBCPP`, default ON) statically embed libc++ + libc++abi
  via the `project_libcpp_static` interface target, which every C++ executable
  must link last; the static link requires the `libcxx-static` archive
  (`libc++.a`), else configure warns and falls back to the dynamic link.
  Build-time host tools (slang-spirv-compiler) stay fully dynamic.
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