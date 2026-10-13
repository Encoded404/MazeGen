#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanEngine.GpuStats;
import test_gpu;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Device-gated check of the premise the per-run GPU-statistics path relies on:
// a graphics run and a compute run each own an independent query, and the
// compute run uses a compute-only pool (a pool enabling graphics statistics
// cannot be used from a compute command pool).
//
// Registered with the `gpu` ctest label, so the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

struct BareDualQueueDevice {
    vk::raii::Context context{};
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physical_device{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue graphics_queue{nullptr};
    vk::raii::Queue compute_queue{nullptr};
    std::uint32_t graphics_family = vk::QueueFamilyIgnored;
    std::uint32_t compute_family = vk::QueueFamilyIgnored;
};

// Returns a device with a graphics family and a distinct compute-only capable
// family. When no such compute family exists, compute_family stays
// QueueFamilyIgnored and no device is created.
BareDualQueueDevice CreateDualQueueDevice() {
    BareDualQueueDevice bare{};
    const vk::ApplicationInfo app_info("VulkanEngineV5-tests", 1, "VulkanEngineV5-tests", 1,
                                       vk::ApiVersion13);
    bare.instance = vk::raii::Instance(bare.context, vk::InstanceCreateInfo({}, &app_info));
    const auto devices = bare.instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        return bare;
    }
    bare.physical_device = devices.front();
    const auto families = bare.physical_device.getQueueFamilyProperties();

    for (std::uint32_t i = 0; i < families.size(); ++i) {
        const bool graphics = (families[i].queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{};
        const bool compute = (families[i].queueFlags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{};
        if (graphics && bare.graphics_family == vk::QueueFamilyIgnored) {
            bare.graphics_family = i;
        }
        if (compute && !graphics && bare.compute_family == vk::QueueFamilyIgnored) {
            bare.compute_family = i;
        }
    }
    if (bare.graphics_family == vk::QueueFamilyIgnored ||
        bare.compute_family == vk::QueueFamilyIgnored) {
        return bare;
    }

    const float priority = 1.0f;
    const std::array<vk::DeviceQueueCreateInfo, 2> queue_infos{
        vk::DeviceQueueCreateInfo({}, bare.graphics_family, 1, &priority),
        vk::DeviceQueueCreateInfo({}, bare.compute_family, 1, &priority),
    };
    bare.device = vk::raii::Device(
        bare.physical_device,
        vk::DeviceCreateInfo({}, static_cast<std::uint32_t>(queue_infos.size()), queue_infos.data()));
    bare.graphics_queue = bare.device.getQueue(bare.graphics_family, 0);
    bare.compute_queue = bare.device.getQueue(bare.compute_family, 0);
    return bare;
}

// Reads one query and reports whether the driver marked it available.
bool QueryIsAvailable(const vk::raii::Device& device, const vk::raii::QueryPool& pool,
                      std::uint32_t slot, std::uint32_t words_per_query) {
    std::vector<std::uint64_t> data(words_per_query, 0);
    const vk::Result result = static_cast<vk::Result>(device.getDispatcher()->vkGetQueryPoolResults(
        static_cast<vk::Device::CType>(*device),
        static_cast<vk::QueryPool::CType>(*pool),
        slot, 1,
        static_cast<vk::DeviceSize>(data.size() * sizeof(std::uint64_t)), data.data(),
        static_cast<vk::DeviceSize>(words_per_query * sizeof(std::uint64_t)),
        static_cast<vk::QueryResultFlags::MaskType>(vk::QueryResultFlagBits::e64 |
                                                    vk::QueryResultFlagBits::eWithAvailability)));
    if (result != vk::Result::eSuccess) {
        return false;
    }
    return static_cast<std::uint32_t>(data[words_per_query - 1]) != 0u;
}

TEST(GpuStatsGpuTest, GraphicsAndComputeRunsOwnIndependentQueries) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    BareDualQueueDevice bare = CreateDualQueueDevice();
    if (*bare.device == nullptr || bare.compute_family == vk::QueueFamilyIgnored) {
        GTEST_SKIP() << "no distinct async-compute queue family";
    }

    using VulkanEngine::GpuStats::kComputeQueryWords;
    using VulkanEngine::GpuStats::kQueryWords;

    constexpr std::uint32_t kQueries = 2;

    // Graphics pool: enables graphics statistics, so it may only be used from a
    // graphics-capable command pool. Compute pool: compute invocations only.
    const vk::QueryPipelineStatisticFlags graphics_flags =
        vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices |
        vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
        vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
        vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
        vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
        vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations |
        vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;
    const vk::QueryPipelineStatisticFlags compute_flags =
        vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;

    const vk::raii::QueryPool graphics_pool(
        bare.device,
        vk::QueryPoolCreateInfo({}, vk::QueryType::ePipelineStatistics, kQueries, graphics_flags));
    const vk::raii::QueryPool compute_pool(
        bare.device,
        vk::QueryPoolCreateInfo({}, vk::QueryType::ePipelineStatistics, kQueries, compute_flags));
    // The default dispatcher is not initialized in this standalone test, so go
    // through the device's dispatcher explicitly.
    auto* dispatcher = bare.device.getDispatcher();
    dispatcher->vkResetQueryPool(static_cast<vk::Device::CType>(*bare.device),
                                 static_cast<vk::QueryPool::CType>(*graphics_pool), 0, kQueries);
    dispatcher->vkResetQueryPool(static_cast<vk::Device::CType>(*bare.device),
                                 static_cast<vk::QueryPool::CType>(*compute_pool), 0, kQueries);

    const vk::raii::CommandPool graphics_command_pool(
        bare.device, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient,
                                               bare.graphics_family));
    const vk::raii::CommandPool compute_command_pool(
        bare.device, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient,
                                               bare.compute_family));
    auto graphics_commands = bare.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(
        *graphics_command_pool, vk::CommandBufferLevel::ePrimary, 1));
    auto compute_commands = bare.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(
        *compute_command_pool, vk::CommandBufferLevel::ePrimary, 1));
    const vk::raii::CommandBuffer& graphics_command = graphics_commands.front();
    const vk::raii::CommandBuffer& compute_command = compute_commands.front();

    // Graphics run owns slot 0 of the graphics pool; compute run owns slot 1 of
    // the compute pool.
    graphics_command.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    graphics_command.beginQuery(*graphics_pool, 0, {});
    graphics_command.endQuery(*graphics_pool, 0);
    graphics_command.end();

    compute_command.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    compute_command.beginQuery(*compute_pool, 1, {});
    compute_command.endQuery(*compute_pool, 1);
    compute_command.end();

    const vk::raii::Semaphore boundary(bare.device, vk::SemaphoreCreateInfo());
    const vk::raii::Fence fence(bare.device, vk::FenceCreateInfo());

    vk::Semaphore signal_handles[1] = {*boundary};
    vk::SubmitInfo graphics_submit{};
    graphics_submit.commandBufferCount = 1;
    graphics_submit.pCommandBuffers = &*graphics_command;
    graphics_submit.signalSemaphoreCount = 1;
    graphics_submit.pSignalSemaphores = signal_handles;
    bare.graphics_queue.submit(graphics_submit, nullptr);

    vk::Semaphore wait_handles[1] = {*boundary};
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eComputeShader;
    vk::SubmitInfo compute_submit{};
    compute_submit.waitSemaphoreCount = 1;
    compute_submit.pWaitSemaphores = wait_handles;
    compute_submit.pWaitDstStageMask = &wait_stage;
    compute_submit.commandBufferCount = 1;
    compute_submit.pCommandBuffers = &*compute_command;
    bare.compute_queue.submit(compute_submit, *fence);
    ASSERT_EQ(bare.device.waitForFences(*fence, vk::True, UINT64_MAX), vk::Result::eSuccess);

    // Both recorded queries completed and are independent of each other.
    EXPECT_TRUE(QueryIsAvailable(bare.device, graphics_pool, 0, kQueryWords));
    EXPECT_TRUE(QueryIsAvailable(bare.device, compute_pool, 1, kComputeQueryWords));
    // A slot that was never begun in a pool reports unavailable.
    EXPECT_FALSE(QueryIsAvailable(bare.device, graphics_pool, 1, kQueryWords));
    EXPECT_FALSE(QueryIsAvailable(bare.device, compute_pool, 0, kComputeQueryWords));

    // The engine's parser accepts a real driver result.
    std::vector<std::uint64_t> graphics_words(kQueryWords, 0);
    const vk::Result read = static_cast<vk::Result>(bare.device.getDispatcher()->vkGetQueryPoolResults(
        static_cast<vk::Device::CType>(*bare.device),
        static_cast<vk::QueryPool::CType>(*graphics_pool),
        0, 1,
        static_cast<vk::DeviceSize>(graphics_words.size() * sizeof(std::uint64_t)),
        graphics_words.data(),
        static_cast<vk::DeviceSize>(kQueryWords * sizeof(std::uint64_t)),
        static_cast<vk::QueryResultFlags::MaskType>(vk::QueryResultFlagBits::e64 |
                                                    vk::QueryResultFlagBits::eWithAvailability)));
    ASSERT_EQ(read, vk::Result::eSuccess);
    VulkanEngine::GpuStats::Counters counters{};
    EXPECT_TRUE(VulkanEngine::GpuStats::TryParseQueryResult(graphics_words.data(), counters));
}

}  // namespace
