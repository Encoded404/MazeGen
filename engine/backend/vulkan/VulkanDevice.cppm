module;

export module VulkanBackend.Vulkan.VulkanDevice;

import std;

import vulkan;

import VulkanBackend.Vulkan.VulkanInstance;
import VulkanBackend.Vulkan.CommonTypes;
import VulkanBackend.Vulkan.VulkanCapabilities;

import VulkanShared.RenderGraphTypes;

export namespace VulkanBackend::Vulkan {

class VulkanDevice {
public:
    // New method for physical device selection (hard-floor check + one-time capability query)
    [[nodiscard]] bool SelectPhysicalDevice(const VulkanInstance& instance);
    // Modified method for logical device creation and resource setup
    [[nodiscard]] bool CreateLogicalDeviceAndResources(std::uint32_t frames_in_flight,
                                                       const VulkanBootstrapConfig& config);
    void Shutdown();

    [[nodiscard]] bool IsValid() const { return device_ != nullptr; }

    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const { return *physical_device_; }
    [[nodiscard]] const vk::raii::Device& GetDevice() const { return *device_; }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const { return graphics_queue_; }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const { return graphics_queue_family_; }
    [[nodiscard]] const vk::raii::CommandPool& GetCommandPool() const { return *command_pool_; }

    // Async compute queue. When no dedicated compute-only family exists this
    // aliases the graphics family (HasAsyncCompute() is false).
    [[nodiscard]] bool HasAsyncCompute() const { return async_compute_available_; }
    [[nodiscard]] const vk::raii::Queue& GetComputeQueue() const { return compute_queue_; }
    [[nodiscard]] std::uint32_t GetComputeQueueFamily() const { return compute_queue_family_; }
    [[nodiscard]] const vk::raii::CommandPool& GetComputeCommandPool() const { return *compute_command_pool_; }
    [[nodiscard]] vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t frame_idx) {
        return compute_command_buffers_[frame_idx % frames_in_flight_];
    }

    // Unique queue families the device created queues for. Resources that may be
    // touched from more than one of them use concurrent sharing.
    [[nodiscard]] std::span<const std::uint32_t> GetQueueFamilies() const { return queue_families_; }
    [[nodiscard]] bool IsQueueFamilySharingSupported() const { return queue_families_.size() > 1; }

    // Multi-queue run recording: each queue run gets its own command buffer, and
    // cross-queue boundaries are ordered with per-run binary semaphores. The cap
    // is shared with the planner so the two cannot drift.
    static constexpr std::uint32_t kMaxQueueRuns = VulkanEngine::RenderGraph::kMaxQueueRuns;
    // One extra slot per frame is reserved for the engine's graphics preamble
    // (scene uploads/descriptors), which is submitted ahead of the graph when
    // the graph's first run is not on the graphics queue.
    static constexpr std::uint32_t kRunSlotsPerFrame = kMaxQueueRuns + 1;
    [[nodiscard]] vk::raii::CommandBuffer& GetRunCommandBuffer(bool compute, std::uint32_t frame_idx,
                                                               std::uint32_t run_slot);
    [[nodiscard]] const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t frame_idx,
                                                             std::uint32_t run_slot) const;

    [[nodiscard]] const vk::raii::Fence& GetInFlightFence(std::uint32_t frame_idx) const { return *in_flight_fences_[frame_idx % frames_in_flight_]; }
    [[nodiscard]] const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t frame_idx) const { return *image_available_semaphores_[frame_idx % frames_in_flight_]; }
    [[nodiscard]] vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t frame_idx) { return command_buffers_[frame_idx % frames_in_flight_]; }

    [[nodiscard]] std::uint32_t GetFramesInFlight() const { return frames_in_flight_; }

    [[nodiscard]] const VulkanCapabilities& GetCapabilities() const { return capabilities_; }

private:
    std::unique_ptr<vk::raii::PhysicalDevice> physical_device_{};
    std::unique_ptr<vk::raii::Device> device_{};
    vk::raii::Queue graphics_queue_ = nullptr;
    std::uint32_t graphics_queue_family_ = 0;

    std::unique_ptr<vk::raii::CommandPool> command_pool_{};
    std::vector<vk::raii::CommandBuffer> command_buffers_{};

    vk::raii::Queue compute_queue_ = nullptr;
    std::uint32_t compute_queue_family_ = 0;
    bool async_compute_available_ = false;
    std::unique_ptr<vk::raii::CommandPool> compute_command_pool_{};
    std::vector<vk::raii::CommandBuffer> compute_command_buffers_{};
    std::vector<std::uint32_t> queue_families_{};
    // Per-run command buffers (frames_in_flight * kMaxQueueRuns) and the binary
    // semaphores used at cross-queue run boundaries.
    std::vector<vk::raii::CommandBuffer> graphics_run_buffers_{};
    std::vector<vk::raii::CommandBuffer> compute_run_buffers_{};
    std::vector<std::unique_ptr<vk::raii::Semaphore>> run_semaphores_{};

    std::vector<std::unique_ptr<vk::raii::Semaphore>> image_available_semaphores_{};
    std::vector<std::unique_ptr<vk::raii::Fence>> in_flight_fences_{};

    std::uint32_t frames_in_flight_ = 0;

    const VulkanInstance* instance_ = nullptr;
    VulkanCapabilities capabilities_{};
    SupportedDeviceState supported_{};
};

} // namespace VulkanBackend::Vulkan
