module;

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_hpp_macros.hpp>

export module test_gpu;

import std;

import vulkan;

export namespace TestSupport {

namespace detail {

// Creates a throwaway instance and enumerates physical devices. No logical
// device, surface, or swapchain is created, so this is cheap and cannot fail
// the build when `gtest_discover_tests` probes the binary.
[[nodiscard]] inline bool ProbeGpuDevice() {
    // Kept alive for the process lifetime: the dispatcher stores function
    // pointers resolved through it, and destroying the loader would leave them
    // dangling for any later Vulkan use in the same test binary.
    static vk::detail::DynamicLoader loader;
    try {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(loader);
        vk::raii::Context context;
        const vk::ApplicationInfo app_info("MazeGenerator-tests", 1, "MazeGenerator-tests", 1, vk::ApiVersion13);
        const vk::InstanceCreateInfo create_info({}, &app_info);
        const vk::raii::Instance instance(context, create_info);
        return !instance.enumeratePhysicalDevices().empty();
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace detail

// True when at least one Vulkan physical device is present. The result is
// cached: the probe is a process-wide property, not per-test state.
//
// Convention for device-bound tests:
//   if (!TestSupport::IsGpuDeviceAvailable()) {
//       GTEST_SKIP() << "no Vulkan device available";
//   }
// and register the target with `setup_test_target(<target> LABELS gpu)` so the
// default `ctest` preset (which excludes label `gpu`) stays device-free.
[[nodiscard]] inline bool IsGpuDeviceAvailable() {
    static const bool available = detail::ProbeGpuDevice();
    return available;
}

}  // namespace TestSupport
