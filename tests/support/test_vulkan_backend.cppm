module;

export module TestSupport.HeadlessVulkanBackend;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.CommonTypes;

export namespace TestSupport {

// Real headless IVulkanBootstrap for GPU-labelled tests: instance + physical
// device + logical device + one graphics queue + a small command pool, with no
// surface/swapchain. Capability snapshots (formats, descriptor indexing,
// properties) are captured through the production capability builder path so
// tests exercise the same FormatCapabilities the engine sees. Swapchain /
// present / run-list members are unsupported and throw.
class HeadlessVulkanBackend : public VulkanBackend::Vulkan::IVulkanBootstrap {
public:
    HeadlessVulkanBackend() = default;
    ~HeadlessVulkanBackend() override { Shutdown(); }

    HeadlessVulkanBackend(const HeadlessVulkanBackend&) = delete;
    HeadlessVulkanBackend& operator=(const HeadlessVulkanBackend&) = delete;

    // Creates the full stack. Returns false when no Vulkan device exists.
    [[nodiscard]] bool Initialize();

    [[nodiscard]] bool CreateInstance(const VulkanBackend::Vulkan::VulkanBootstrapConfig&) override;
    [[nodiscard]] bool SelectPhysicalDevice() override;
    [[nodiscard]] bool CreateLogicalDevice(std::uint32_t frames_in_flight) override;
    [[nodiscard]] bool CreateSwapchain(std::uint32_t, VulkanBackend::Vulkan::PresentMode, std::uint32_t&) override;
    [[nodiscard]] bool GetSwapchainExtent(std::uint32_t&, std::uint32_t&) const override;

    [[nodiscard]] const vk::raii::Instance& GetInstance() const override { return *instance_; }
    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const override { return *physical_device_; }
    [[nodiscard]] const vk::raii::Device& GetDevice() const override { return *device_; }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const override { return *graphics_queue_; }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const override { return graphics_family_; }
    [[nodiscard]] const vk::raii::CommandPool& GetCommandPool() const override { return *command_pool_; }

    [[nodiscard]] bool HasAsyncCompute() const override { return false; }
    [[nodiscard]] const vk::raii::CommandPool& GetComputeCommandPool() const override { return *command_pool_; }
    [[nodiscard]] const vk::raii::Queue& GetComputeQueue() const override { return *graphics_queue_; }
    [[nodiscard]] std::uint32_t GetComputeQueueFamily() const override { return graphics_family_; }
    [[nodiscard]] vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t) override { return command_buffers_[0]; }
    [[nodiscard]] std::span<const std::uint32_t> GetQueueFamilies() const override { return queue_families_; }

    [[nodiscard]] vk::raii::CommandBuffer& GetRunCommandBuffer(bool, std::uint32_t, std::uint32_t) override { return command_buffers_[0]; }
    [[nodiscard]] const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t, std::uint32_t) const override { return semaphores_[0]; }
    void SetFrameRuns(std::span<const QueueRunSubmit>) override {}

    [[nodiscard]] const VulkanBackend::Vulkan::VulkanCapabilities& GetCapabilities() const override { return capabilities_; }
    [[nodiscard]] const std::string& GetErrorMessage() const override { return error_message_; }
    [[nodiscard]] bool HasUnmetRequirements() const override { return false; }

    [[nodiscard]] const vk::raii::Fence& GetInFlightFence(std::uint32_t) const override { return *fence_; }
    [[nodiscard]] const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t) const override { return semaphores_[0]; }
    [[nodiscard]] const vk::raii::Semaphore& GetRenderFinishedSemaphore(std::uint32_t) const override { return semaphores_[0]; }
    [[nodiscard]] vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t) override { return command_buffers_[0]; }

    [[nodiscard]] std::uint32_t GetFramesInFlight() const override { return frames_in_flight_; }
    [[nodiscard]] std::uint32_t GetRunSlotsPerFrame() const override { return 2; }

    [[nodiscard]] const vk::raii::SwapchainKHR& GetSwapchain() const override;
    [[nodiscard]] const std::vector<vk::Image>& GetSwapchainImages() const override;
    [[nodiscard]] const std::vector<vk::raii::ImageView>& GetSwapchainImageViews() const override;
    [[nodiscard]] std::vector<bool>& GetSwapchainImageInitializedFlags() override;
    [[nodiscard]] const vk::SurfaceFormatKHR& GetSurfaceFormat() const override;
    [[nodiscard]] vk::Format GetDepthFormat() const override;
    [[nodiscard]] const vk::raii::ImageView& GetDepthImageView(std::uint32_t) const override;
    [[nodiscard]] const vk::raii::Image& GetDepthImage(std::uint32_t) const override;

    [[nodiscard]] bool AcquireNextImage(std::uint32_t, std::uint32_t&) override;
    [[nodiscard]] bool SubmitFrame(std::uint32_t, std::uint32_t, bool) override;
    [[nodiscard]] bool Present(std::uint32_t) override;
    [[nodiscard]] bool IsFrameComplete(std::uint32_t) override;

    void WaitDeviceIdle() override;

    void Shutdown() override;

private:
    // Keeps the loader alive; the global dispatcher's function pointers are
    // resolved through it (same pattern as VulkanInstance).
    std::unique_ptr<vk::detail::DynamicLoader> loader_{};
    std::unique_ptr<vk::raii::Context> context_{};
    std::unique_ptr<vk::raii::Instance> instance_{};
    std::unique_ptr<vk::raii::PhysicalDevice> physical_device_{};
    std::unique_ptr<vk::raii::Device> device_{};
    std::unique_ptr<vk::raii::Queue> graphics_queue_{};
    std::unique_ptr<vk::raii::CommandPool> command_pool_{};
    std::vector<vk::raii::CommandBuffer> command_buffers_{};
    std::vector<vk::raii::Semaphore> semaphores_{};
    std::unique_ptr<vk::raii::Fence> fence_{};
    std::uint32_t graphics_family_ = 0;
    std::uint32_t frames_in_flight_ = 1;
    std::vector<std::uint32_t> queue_families_{};
    VulkanBackend::Vulkan::VulkanCapabilities capabilities_{};
    std::string error_message_{};
};

} // namespace TestSupport
