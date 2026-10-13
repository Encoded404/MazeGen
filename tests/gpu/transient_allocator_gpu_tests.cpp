#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.TransientAllocator;
import test_gpu;

namespace {

// Minimal IVulkanBootstrap adapter exposing the real device created below.
// Everything the heaps do not need throws, so an accidental dependency is a
// test failure rather than a silent null-handle read.
class RealDeviceBootstrap final : public VulkanBackend::Vulkan::IVulkanBootstrap {
public:
    RealDeviceBootstrap(const vk::raii::PhysicalDevice& physical_device, const vk::raii::Device& device)
        : physical_device_(&physical_device), device_(&device) {}

    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const override { return *physical_device_; }
    [[nodiscard]] const vk::raii::Device& GetDevice() const override { return *device_; }
    [[nodiscard]] std::uint32_t GetFramesInFlight() const override { return 2; }
    [[nodiscard]] std::uint32_t GetRunSlotsPerFrame() const override { return 1; }
    [[nodiscard]] bool IsFrameComplete(std::uint32_t) override { return true; }

    bool CreateInstance(const VulkanBackend::Vulkan::VulkanBootstrapConfig&) override { throw std::runtime_error("n/a"); }
    bool SelectPhysicalDevice() override { throw std::runtime_error("n/a"); }
    bool CreateLogicalDevice(std::uint32_t) override { throw std::runtime_error("n/a"); }
    bool CreateSwapchain(std::uint32_t, VulkanBackend::Vulkan::PresentMode, std::uint32_t&) override { throw std::runtime_error("n/a"); }
    bool GetSwapchainExtent(std::uint32_t&, std::uint32_t&) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Instance& GetInstance() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const override { return 0; }
    [[nodiscard]] const vk::raii::CommandPool& GetCommandPool() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] bool HasAsyncCompute() const override { return false; }
    [[nodiscard]] const vk::raii::Queue& GetComputeQueue() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] std::uint32_t GetComputeQueueFamily() const override { return 0; }
    [[nodiscard]] const vk::raii::CommandPool& GetComputeCommandPool() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t) override { throw std::runtime_error("n/a"); }
    [[nodiscard]] std::span<const std::uint32_t> GetQueueFamilies() const override {
        static const std::array<std::uint32_t, 1> families{0u};
        return families;
    }
    [[nodiscard]] vk::raii::CommandBuffer& GetRunCommandBuffer(bool, std::uint32_t, std::uint32_t) override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t, std::uint32_t) const override { throw std::runtime_error("n/a"); }
    void SetFrameRuns(std::span<const QueueRunSubmit>) override {}
    [[nodiscard]] const vk::raii::Fence& GetInFlightFence(std::uint32_t) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Semaphore& GetRenderFinishedSemaphore(std::uint32_t) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t) override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const VulkanBackend::Vulkan::VulkanCapabilities& GetCapabilities() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const std::string& GetErrorMessage() const override { static const std::string message; return message; }
    [[nodiscard]] bool HasUnmetRequirements() const override { return false; }
    [[nodiscard]] const vk::raii::SwapchainKHR& GetSwapchain() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const std::vector<vk::Image>& GetSwapchainImages() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const std::vector<vk::raii::ImageView>& GetSwapchainImageViews() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] std::vector<bool>& GetSwapchainImageInitializedFlags() override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::SurfaceFormatKHR& GetSurfaceFormat() const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] vk::Format GetDepthFormat() const override { return vk::Format::eUndefined; }
    [[nodiscard]] const vk::raii::ImageView& GetDepthImageView(std::uint32_t) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] const vk::raii::Image& GetDepthImage(std::uint32_t) const override { throw std::runtime_error("n/a"); }
    [[nodiscard]] bool AcquireNextImage(std::uint32_t, std::uint32_t&) override { throw std::runtime_error("n/a"); }
    [[nodiscard]] bool SubmitFrame(std::uint32_t, std::uint32_t, bool) override { throw std::runtime_error("n/a"); }
    [[nodiscard]] bool Present(std::uint32_t) override { throw std::runtime_error("n/a"); }
    void WaitDeviceIdle() override { device_->waitIdle(); }
    void Shutdown() override {}

private:
    const vk::raii::PhysicalDevice* physical_device_ = nullptr;
    const vk::raii::Device* device_ = nullptr;
};

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

vk::ImageCreateInfo ColorImageInfo(std::uint32_t width, std::uint32_t height, bool aliasable) {
    vk::ImageCreateInfo info{};
    info.imageType = vk::ImageType::e2D;
    info.format = vk::Format::eR8G8B8A8Unorm;
    info.extent = vk::Extent3D{width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = vk::SampleCountFlagBits::e1;
    info.tiling = vk::ImageTiling::eOptimal;
    info.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferDst;
    info.sharingMode = vk::SharingMode::eExclusive;
    info.initialLayout = vk::ImageLayout::eUndefined;
    if (aliasable) {
        info.flags |= vk::ImageCreateFlagBits::eAlias;
    }
    return info;
}

TEST(GpuTransientAllocatorTest, ImageHeapBindsSuballocatedImages) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    BareDevice bare = CreateBareDevice();
    ASSERT_NE(*bare.device, nullptr);
    RealDeviceBootstrap bootstrap(bare.physical_device, bare.device);

    VulkanEngine::GpuResources::GpuImageHeap heap;
    ASSERT_TRUE(heap.Initialize(bootstrap, VulkanEngine::GpuResources::ImageHeapConfig{}, "test-image-heap"));
    EXPECT_GE(heap.GetBufferImageGranularity(), 1u);

    const auto a = heap.Allocate(ColorImageInfo(64, 64, /*aliasable=*/true), vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor});
    const auto b = heap.Allocate(ColorImageInfo(64, 64, /*aliasable=*/true), vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor});
    ASSERT_TRUE(a.IsValid());
    ASSERT_TRUE(b.IsValid());
    EXPECT_FALSE(heap.IsDedicated(a.image_index));
    EXPECT_NE(heap.GetImage(a.image_index), nullptr);
    EXPECT_NE(heap.GetImageView(a.image_index), nullptr);
    EXPECT_NE(heap.GetImage(b.image_index), nullptr);

    auto free_a = a;
    auto free_b = b;
    heap.Free(free_a);
    heap.Free(free_b);
    EXPECT_FALSE(free_a.IsValid());
    heap.Shutdown();
}

TEST(GpuTransientAllocatorTest, RuntimePlacesAliasedCopiesPerFifSlot) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    BareDevice bare = CreateBareDevice();
    ASSERT_NE(*bare.device, nullptr);
    RealDeviceBootstrap bootstrap(bare.physical_device, bare.device);

    using VulkanEngine::GpuResources::TransientAllocator;
    using VulkanEngine::GpuResources::TransientKind;

    TransientAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(bootstrap, "test-transients"));

    std::vector<TransientAllocator::Desc> descs(3);

    descs[0].is_image = true;
    descs[0].requirements = {.name = "a", .kind = TransientKind::Image, .heap_key = 0,
                             .first_pass = 0, .last_pass = 1, .aliasable = true};
    descs[0].image = {.format = vk::Format::eR8G8B8A8Unorm, .width = 64, .height = 64,
                      .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                      .samples = vk::SampleCountFlagBits::e1, .aspect = vk::ImageAspectFlagBits::eColor};

    descs[1].is_image = true;
    descs[1].requirements = {.name = "b", .kind = TransientKind::Image, .heap_key = 0,
                             .first_pass = 2, .last_pass = 3, .aliasable = true};
    descs[1].image = descs[0].image;

    descs[2].is_image = false;
    descs[2].requirements = {.name = "scratch", .kind = TransientKind::Buffer, .heap_key = 1,
                             .size = 4096, .alignment = 256, .first_pass = 0, .last_pass = 3,
                             .aliasable = false};
    descs[2].buffer = {.usage = vk::BufferUsageFlagBits::eStorageBuffer,
                       .memory = vk::MemoryPropertyFlagBits::eDeviceLocal};

    allocator.Sync(descs, /*frames_in_flight=*/2, /*current_frame=*/0);

    for (std::uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_NE(allocator.GetImage(0, slot), nullptr);
        EXPECT_NE(allocator.GetImageView(0, slot), nullptr);
        EXPECT_NE(allocator.GetImage(1, slot), nullptr);

        vk::Buffer buffer{};
        vk::DeviceSize offset = 0;
        vk::DeviceSize size = 0;
        ASSERT_TRUE(allocator.GetBuffer(2, slot, buffer, offset, size));
        EXPECT_NE(buffer, nullptr);
        EXPECT_EQ(size, 4096u);
    }

    // a and b have disjoint lifetimes and are aliasable, so memory is reused.
    EXPECT_FALSE(allocator.GetPlan().aliases.empty());

    // Reconfiguring retires the old generation; CollectGarbage frees it once the
    // retiring frame is complete (the adapter reports completion immediately).
    descs[0].image.width = 32;
    allocator.Sync(descs, 2, 1);
    allocator.CollectGarbage(2);
    EXPECT_NE(allocator.GetImage(0, 0), nullptr);

    allocator.Shutdown();
}

}  // namespace
