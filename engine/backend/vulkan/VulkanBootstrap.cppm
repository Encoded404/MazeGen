module;

export module VulkanBackend.Vulkan.VulkanBootstrap;

import std;

import vulkan;

export import VulkanBackend.Vulkan.CommonTypes;
export import VulkanBackend.Vulkan.VulkanCapabilities;

export namespace VulkanBackend::Vulkan {

// Pipeline stages the swapchain acquire semaphore must wait at for the frame.
// The acquired image can be written as a colour attachment and/or sampled by a
// shader, so waiting only at colour-attachment output races a fragment/compute
// read of the backbuffer. Wait at all commands until the render graph reports
// the real first-use stage.
[[nodiscard]] inline vk::PipelineStageFlags AcquireWaitStageMask() {
    return vk::PipelineStageFlagBits::eAllCommands;
}

class IVulkanBootstrap {
public:
    virtual ~IVulkanBootstrap() = default;

    [[nodiscard]] virtual bool CreateInstance(const VulkanBootstrapConfig& config) = 0;
    [[nodiscard]] virtual bool SelectPhysicalDevice() = 0;
    [[nodiscard]] virtual bool CreateLogicalDevice(std::uint32_t frames_in_flight) = 0; // Modified
    [[nodiscard]] virtual bool CreateSwapchain(std::uint32_t preferred_image_count, PresentMode present_mode, std::uint32_t& out_image_count) = 0;
    [[nodiscard]] virtual bool GetSwapchainExtent(std::uint32_t& out_width, std::uint32_t& out_height) const = 0;

    // Core low-level access
    [[nodiscard]] virtual const vk::raii::Instance& GetInstance() const = 0;
    [[nodiscard]] virtual const vk::raii::PhysicalDevice& GetPhysicalDevice() const = 0;
    [[nodiscard]] virtual const vk::raii::Device& GetDevice() const = 0;
    [[nodiscard]] virtual const vk::raii::Queue& GetGraphicsQueue() const = 0;
    [[nodiscard]] virtual std::uint32_t GetGraphicsQueueFamily() const = 0;
    [[nodiscard]] virtual const vk::raii::CommandPool& GetCommandPool() const = 0;

    // Async compute queue (aliases graphics when HasAsyncCompute() is false).
    [[nodiscard]] virtual bool HasAsyncCompute() const = 0;
    [[nodiscard]] virtual const vk::raii::Queue& GetComputeQueue() const = 0;
    [[nodiscard]] virtual std::uint32_t GetComputeQueueFamily() const = 0;
    [[nodiscard]] virtual const vk::raii::CommandPool& GetComputeCommandPool() const = 0;
    [[nodiscard]] virtual vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t frame_idx) = 0;
    [[nodiscard]] virtual std::span<const std::uint32_t> GetQueueFamilies() const = 0;

    // A recorded queue run: one command buffer that executes contiguously on
    // either the graphics or the async compute queue.
    struct QueueRunSubmit {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        bool compute = false;
        vk::CommandBuffer command_buffer = nullptr;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    [[nodiscard]] virtual vk::raii::CommandBuffer& GetRunCommandBuffer(bool compute,
                                                                       std::uint32_t frame_idx,
                                                                       std::uint32_t run_slot) = 0;
    [[nodiscard]] virtual const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t frame_idx,
                                                                     std::uint32_t run_slot) const = 0;
    // Run list for the frame about to be submitted; SubmitFrame consumes it.
    virtual void SetFrameRuns(std::span<const QueueRunSubmit> runs) = 0;

    // Capabilities snapshot (device domain) + bootstrap error state
    [[nodiscard]] virtual const VulkanCapabilities& GetCapabilities() const = 0;
    [[nodiscard]] virtual const std::string& GetErrorMessage() const = 0;
    [[nodiscard]] virtual bool HasUnmetRequirements() const = 0;

    // Modified to accept frame_idx
    [[nodiscard]] virtual const vk::raii::Fence& GetInFlightFence(std::uint32_t frame_idx) const = 0;
    [[nodiscard]] virtual const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t frame_idx) const = 0;
    [[nodiscard]] virtual const vk::raii::Semaphore& GetRenderFinishedSemaphore(std::uint32_t frame_idx) const = 0;
    [[nodiscard]] virtual vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t frame_idx) = 0; // Modified

    [[nodiscard]] virtual std::uint32_t GetFramesInFlight() const = 0; // New method
    // Number of per-run command buffers (and cross-queue semaphores) the device
    // budgets per frames-in-flight slot. Slot 0 is reserved for the graphics
    // preamble; the rest hold the frame's queue runs.
    [[nodiscard]] virtual std::uint32_t GetRunSlotsPerFrame() const = 0;

    // Swapchain access
    [[nodiscard]] virtual const vk::raii::SwapchainKHR& GetSwapchain() const = 0;
    [[nodiscard]] virtual const std::vector<vk::Image>& GetSwapchainImages() const = 0;
    [[nodiscard]] virtual const std::vector<vk::raii::ImageView>& GetSwapchainImageViews() const = 0;
    [[nodiscard]] virtual std::vector<bool>& GetSwapchainImageInitializedFlags() = 0;
    [[nodiscard]] virtual const vk::SurfaceFormatKHR& GetSurfaceFormat() const = 0;
    [[nodiscard]] virtual vk::Format GetDepthFormat() const = 0;
    [[nodiscard]] virtual const vk::raii::ImageView& GetDepthImageView(std::uint32_t image_index) const = 0;
    [[nodiscard]] virtual const vk::raii::Image& GetDepthImage(std::uint32_t image_index) const = 0;

    // RenderGraph support
    [[nodiscard]] virtual bool AcquireNextImage(std::uint32_t frame_idx, std::uint32_t& out_image_index) = 0; // Modified
    // Resets the per-slot fence and queues the frame's command buffer for GPU
    // execution. Does NOT present; the image stays app-owned until Present().
    [[nodiscard]] virtual bool SubmitFrame(std::uint32_t frame_idx, std::uint32_t image_index, bool rendering_succeeded) = 0;
    // Queues vkQueuePresentKHR for a previously submitted image. The present
    // waits on that image's render-finished semaphore (signaled by SubmitFrame).
    [[nodiscard]] virtual bool Present(std::uint32_t image_index) = 0;
    // Non-blocking query: has all command work of frame frame_idx completed?
    [[nodiscard]] virtual bool IsFrameComplete(std::uint32_t frame_idx) = 0;

    // Device-scope idle. Takes the process-wide device scope so concurrent
    // callers (parallel teardown tasks, resource destructors) serialize on the
    // VkQueue objects, which vkDeviceWaitIdle accesses and the API requires to
    // be externally synchronized. No-op when the device is not valid.
    virtual void WaitDeviceIdle() = 0;

    virtual void Shutdown() = 0;
};

class VulkanBootstrap {
public:
    explicit VulkanBootstrap(std::shared_ptr<IVulkanBootstrap> backend);

    [[nodiscard]] bool Initialize(const VulkanBootstrapConfig& config);
    void Shutdown();

    [[nodiscard]] VulkanBootstrapState BeginFrame();
    void EndFrame();

    void NotifySwapchainOutOfDate();
    [[nodiscard]] bool RecreateSwapchain();

    // RenderGraph support
    [[nodiscard]] bool AcquireNextImage(std::uint32_t& out_image_index);
    [[nodiscard]] bool SubmitFrame(std::uint32_t image_index, bool rendering_succeeded);
    [[nodiscard]] bool Present(std::uint32_t image_index);
    [[nodiscard]] bool IsFrameComplete(std::uint32_t frame_idx) const;

    // Direct access to backend for custom rendering
    [[nodiscard]] IVulkanBootstrap& GetBackend() { return *backend_; }

    void NotifyDeviceLost();

    [[nodiscard]] bool IsInitialized() const;
    [[nodiscard]] const VulkanBootstrapState& GetSnapshot() const;

private:
    std::shared_ptr<IVulkanBootstrap> backend_{};
    VulkanBootstrapConfig config_{};
    VulkanBootstrapState snapshot_{};
    BootstrapStatus pending_status_ = BootstrapStatus::Ok;
    bool initialized_ = false;
};

}  // namespace VulkanBackend::Vulkan
