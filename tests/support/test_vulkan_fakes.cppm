module;

export module test_vulkan_fakes;

import std;

import vulkan;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanShared.RenderGraphTypes;

export namespace TestSupport {

// Deterministic stand-in for the real Vulkan bootstrap backend.
//
// It implements only the configuration/lifecycle surface that budget, frame
// loop, and graph tests need. Every accessor that would hand out a live Vulkan
// handle throws instead of returning a null/dangling raii object, so a test
// that accidentally depends on a device fails loudly rather than dereferencing
// an empty handle. Tests that genuinely need a device must use test_gpu.cppm.
class FakeVulkanBootstrapBackend final : public VulkanBackend::Vulkan::IVulkanBootstrap {
public:
    bool instance_result = true; // NOLINT(misc-non-private-member-variables-in-classes)
    bool physical_device_result = true; // NOLINT(misc-non-private-member-variables-in-classes)
    bool logical_device_result = true; // NOLINT(misc-non-private-member-variables-in-classes)
    bool swapchain_result = true; // NOLINT(misc-non-private-member-variables-in-classes)
    bool shutdown_called = false; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t produced_swapchain_image_count = 3; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t produced_width = 1280; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t produced_height = 720; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t current_frames_in_flight = 2; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool CreateInstance(const VulkanBackend::Vulkan::VulkanBootstrapConfig&) override {
        return instance_result;
    }
    [[nodiscard]] bool SelectPhysicalDevice() override { return physical_device_result; }
    [[nodiscard]] bool CreateLogicalDevice(std::uint32_t frames_in_flight) override {
        current_frames_in_flight = frames_in_flight;
        return logical_device_result;
    }
    [[nodiscard]] bool CreateSwapchain(std::uint32_t, VulkanBackend::Vulkan::PresentMode, std::uint32_t& out_image_count) override {
        out_image_count = produced_swapchain_image_count;
        return swapchain_result;
    }
    [[nodiscard]] bool GetSwapchainExtent(std::uint32_t& out_width, std::uint32_t& out_height) const override {
        out_width = produced_width;
        out_height = produced_height;
        return true;
    }

    // Unimplemented handle accessors for fake: any real device dependency is a test bug.
    [[nodiscard]] const vk::raii::Instance& GetInstance() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Device& GetDevice() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const override { return 0; }
    [[nodiscard]] const vk::raii::CommandPool& GetCommandPool() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] bool HasAsyncCompute() const override { return async_compute; }
    [[nodiscard]] const vk::raii::Queue& GetComputeQueue() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] std::uint32_t GetComputeQueueFamily() const override { return async_compute ? 1u : 0u; }
    [[nodiscard]] const vk::raii::CommandPool& GetComputeCommandPool() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t) override { throw std::runtime_error("Fake"); }
    [[nodiscard]] std::span<const std::uint32_t> GetQueueFamilies() const override {
        static const std::array<std::uint32_t, 1> families{0u};
        return families;
    }
    [[nodiscard]] vk::raii::CommandBuffer& GetRunCommandBuffer(bool, std::uint32_t, std::uint32_t) override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t, std::uint32_t) const override { throw std::runtime_error("Fake"); }
    void SetFrameRuns(std::span<const QueueRunSubmit>) override {}

    bool async_compute = false;
    [[nodiscard]] const vk::raii::Fence& GetInFlightFence(std::uint32_t) const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t) const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Semaphore& GetRenderFinishedSemaphore(std::uint32_t) const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t) override { throw std::runtime_error("Fake"); }

    [[nodiscard]] std::uint32_t GetFramesInFlight() const override { return current_frames_in_flight; }
    [[nodiscard]] std::uint32_t GetRunSlotsPerFrame() const override {
        return VulkanEngine::RenderGraph::kMaxQueueRuns + 1;
    }

    [[nodiscard]] const VulkanBackend::Vulkan::VulkanCapabilities& GetCapabilities() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const std::string& GetErrorMessage() const override { static const std::string dummy; return dummy; }
    [[nodiscard]] bool HasUnmetRequirements() const override { return false; }

    [[nodiscard]] const vk::raii::SwapchainKHR& GetSwapchain() const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const std::vector<vk::Image>& GetSwapchainImages() const override { static const std::vector<vk::Image> dummy; return dummy; }
    [[nodiscard]] const std::vector<vk::raii::ImageView>& GetSwapchainImageViews() const override { static const std::vector<vk::raii::ImageView> dummy; return dummy; }
    [[nodiscard]] std::vector<bool>& GetSwapchainImageInitializedFlags() override { static std::vector<bool> dummy; return dummy; }
    [[nodiscard]] const vk::SurfaceFormatKHR& GetSurfaceFormat() const override { static const vk::SurfaceFormatKHR dummy; return dummy; }
    [[nodiscard]] vk::Format GetDepthFormat() const override { return vk::Format::eUndefined; }
    [[nodiscard]] const vk::raii::ImageView& GetDepthImageView(std::uint32_t) const override { throw std::runtime_error("Fake"); }
    [[nodiscard]] const vk::raii::Image& GetDepthImage(std::uint32_t) const override { throw std::runtime_error("Fake"); }

    [[nodiscard]] bool AcquireNextImage(std::uint32_t, std::uint32_t&) override { return true; }
    [[nodiscard]] bool SubmitFrame(std::uint32_t, std::uint32_t, bool) override { return true; }
    [[nodiscard]] bool Present(std::uint32_t) override { return true; }
    [[nodiscard]] bool IsFrameComplete(std::uint32_t) override { return true; }

    void WaitDeviceIdle() override {}

    void Shutdown() override { shutdown_called = true; }
};

}  // namespace TestSupport
