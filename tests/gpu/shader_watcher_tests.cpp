#include <gtest/gtest.h>

import std;

import vulkan;

import ShaderReflection;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ShaderManager;
import VulkanEngine.ShaderWatcher;
import test_gpu;

namespace {

struct BareDevice {
    vk::raii::Context context{};
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physical_device{nullptr};
    vk::raii::Device device{nullptr};
};

BareDevice CreateBareDevice() {
    BareDevice bare{};
    const vk::ApplicationInfo app_info("VulkanEngineV5-tests", 1, "VulkanEngineV5-tests", 1, vk::ApiVersion13);
    bare.instance = vk::raii::Instance(bare.context, vk::InstanceCreateInfo({}, &app_info));
    const auto devices = bare.instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        return bare;
    }
    bare.physical_device = devices.front();
    const auto families = bare.physical_device.getQueueFamilyProperties();
    std::uint32_t graphics_family = vk::QueueFamilyIgnored;
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if ((families[i].queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{}) {
            graphics_family = i;
            break;
        }
    }
    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queue_info({}, graphics_family, 1, &priority);
    bare.device = vk::raii::Device(bare.physical_device, vk::DeviceCreateInfo({}, queue_info));
    return bare;
}

// Device-gated: ShaderWatcher needs a ShaderManager, which needs a device.
// Verifies registration-time directory watching: queued before Start(),
// picked up by Refresh(), and de-duplicated.
TEST(GpuShaderWatcherTest, RegistersAndDeduplicatesDirectories) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }

    BareDevice bare = CreateBareDevice();
    ASSERT_NE(*bare.device, nullptr);

    const auto root = std::filesystem::temp_directory_path() / "vkengine-shader-watcher-test";
    const auto cache_dir = root / "cache";
    const auto dir_a = root / "a";
    const auto dir_b = root / "b";
    const auto dir_c = root / "c";
    std::filesystem::create_directories(cache_dir);
    std::filesystem::create_directories(dir_a);
    std::filesystem::create_directories(dir_b);
    std::filesystem::create_directories(dir_c);

    // Default capabilities are sufficient: the watcher test never compiles a
    // shader, it only needs the manager for its registered directory list.
    const VulkanBackend::Vulkan::VulkanCapabilities capabilities{};

    {
        VulkanEngine::ShaderSystem::ShaderManager shaders(bare.device, capabilities, cache_dir.string());
        VulkanEngine::ShaderSystem::ShaderWatcher watcher(shaders);

        // Registered before Start(): queued, not lost.
        EXPECT_TRUE(watcher.AddDirectory(dir_a.string()));
        EXPECT_GE(watcher.GetRegisteredDirectoryCount(), 1u);

        watcher.Start();

        // A shader registered after Start() lives in a directory the watcher has
        // never seen; Refresh() picks it up from the manager.
        (void)shaders.RegisterManual((dir_b / "b.spv").string(),
                                     (dir_b / "b.slang").string(),
                                     ShaderStage::eFragment);
        const std::size_t before_refresh = watcher.GetRegisteredDirectoryCount();
        watcher.Refresh();
        EXPECT_GT(watcher.GetRegisteredDirectoryCount(), before_refresh);

        // Adding the same directory again is a no-op.
        const std::size_t after_refresh = watcher.GetRegisteredDirectoryCount();
        EXPECT_TRUE(watcher.AddDirectory(dir_a.string()));
        EXPECT_EQ(watcher.GetRegisteredDirectoryCount(), after_refresh);

        // A brand new directory is watched immediately once started.
        EXPECT_TRUE(watcher.AddDirectory(dir_c.string()));
        EXPECT_GT(watcher.GetRegisteredDirectoryCount(), after_refresh);

        watcher.Stop();
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

// One source may back several entry-point variant modules. A reload of the path
// must enumerate every module, and a slot must keep its own entry point so a
// reload recompiles the right wrapper instead of clobbering it with "main".
TEST(GpuShaderWatcherTest, OneSourceEnumeratesEveryVariantModule) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }

    BareDevice bare = CreateBareDevice();
    ASSERT_NE(*bare.device, nullptr);

    const auto root = std::filesystem::temp_directory_path() / "vkengine-shader-variant-test";
    const auto cache_dir = root / "cache";
    std::filesystem::create_directories(cache_dir);
    const auto slang_path = (root / "multi.slang").string();

    const VulkanBackend::Vulkan::VulkanCapabilities capabilities{};
    VulkanEngine::ShaderSystem::ShaderManager shaders(bare.device, capabilities, cache_dir.string());

    // Two modules from one source, distinct entry points and fake .spv paths
    // (registration never loads the module).
    const auto main_id = shaders.RegisterManual(
        (root / "multi_main.spv").string(), slang_path, ShaderStage::eVertex, "main");
    const auto uv1_id = shaders.RegisterManual(
        (root / "multi_uv1.spv").string(), slang_path, ShaderStage::eVertex, "main_uv1");

    EXPECT_EQ(shaders.GetSlot(uv1_id).entry_point, "main_uv1");
    EXPECT_EQ(shaders.GetSlot(main_id).entry_point, "main");

    const auto all = shaders.FindAllBySlangPath(slang_path);
    ASSERT_EQ(all.size(), 2u);
    EXPECT_NE(std::ranges::find(all, main_id), all.end());
    EXPECT_NE(std::ranges::find(all, uv1_id), all.end());

    const auto by_name = shaders.FindAllBySlangFilename("multi.slang");
    EXPECT_EQ(by_name.size(), 2u);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

}  // namespace
