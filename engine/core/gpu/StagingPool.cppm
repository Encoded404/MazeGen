module;

export module VulkanEngine.GpuResources.StagingPool;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.DeviceBufferHeap;
import VulkanEngine.GpuResources.FrameRing;

export namespace VulkanEngine::GpuResources {

struct StagingPoolConfig {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    // Blocks are allocated exactly at this size; an item larger than
    // `block_size` takes the dedicated-buffer path. This is the only
    // dedicated-allocation threshold.
    std::uint64_t block_size = 64ULL << 20;
    // Hard ceiling on outstanding staged bytes. Allocation fails (back-pressure)
    // rather than growing without bound.
    std::uint64_t capacity_bytes = 256ULL << 20;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct StagingAlloc {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    vk::Buffer buffer = nullptr;
    std::uint64_t offset = 0;
    void* mapped_ptr = nullptr;   // host-visible, host-coherent range start
    std::uint64_t size = 0;
    bool dedicated = false;

    // Internal bookkeeping for free.
    std::uint32_t heap_buffer_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t dedicated_index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] bool IsValid() const { return buffer != nullptr; }
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Per-item, frame-gated staging pool over a host-visible DeviceBufferHeap.
//
// Items are sized to the upload; there are no byte-range cursors and no partial
// uploads. Two lifetimes exist:
//   * Immediate: Allocate -> RecordBufferCopy/RecordBufferToImage ->
//     FlushImmediate() records one command buffer, submits, waits for the GPU
//     fence, and frees. Boot and setup-time uploads (meshes, materials,
//     lighting, shared data) use this.
//   * Frame-gated: the caller records copies with an explicit frame command
//     buffer and calls Retire(alloc, recording_frame); the range is freed when
//     the pool's FrameRing drains, one FIF cycle later. No fence poll.
class StagingPool {
public:
    StagingPool() = default;
    ~StagingPool();

    StagingPool(const StagingPool&) = delete;
    StagingPool& operator=(const StagingPool&) = delete;
    StagingPool(StagingPool&&) = delete;
    StagingPool& operator=(StagingPool&&) = delete;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    const StagingPoolConfig& config = {});
    void Shutdown();

    // Reserves an item-sized range. Returns nullopt on capacity back-pressure.
    // A size above `block_size` takes a dedicated host-visible buffer.
    [[nodiscard]] std::optional<StagingAlloc> Allocate(std::uint64_t size,
                                                       std::uint64_t alignment = 256);

    // ── Immediate path (boot/setup only; blocking) ─────────────────────
    void RecordBufferCopy(const StagingAlloc& alloc, vk::Buffer dst_buffer,
                          std::uint64_t dst_offset);
    void RecordBufferToImage(const StagingAlloc& alloc, vk::Image dst_image,
                             std::span<const vk::BufferImageCopy> regions);
    // Ends+submits the immediate command buffer, waits for the GPU, and frees
    // every immediate allocation. Returns false if the submit/wait failed.
    bool FlushImmediate();

    // ── Frame-gated path (async uploads) ───────────────────────────────
    void RecordBufferCopy(vk::CommandBuffer cmd, const StagingAlloc& alloc,
                          vk::Buffer dst_buffer, std::uint64_t dst_offset);
    void RecordBufferToImage(vk::CommandBuffer cmd, const StagingAlloc& alloc,
                             vk::Image dst_image,
                             std::span<const vk::BufferImageCopy> regions);
    // Enqueues the range for release after `recording_frame`'s ring drains.
    void Retire(const StagingAlloc& alloc, std::uint32_t recording_frame);
    // Frees every range whose recording frame is one FIF cycle old.
    void BeginFrame(std::uint32_t frame_index);

    [[nodiscard]] std::uint64_t OutstandingBytes() const { return outstanding_bytes_; }
    [[nodiscard]] std::uint64_t CapacityBytes() const { return config_.capacity_bytes; }
    [[nodiscard]] const StagingPoolConfig& GetConfig() const { return config_; }
    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }
    VulkanBackend::Vulkan::IVulkanBootstrap* GetBackend() { return backend_; }

private:
    void FreeAlloc(const StagingAlloc& alloc);
    void EnsureImmediateCommandBuffer();
    void ReleaseDedicated(std::uint32_t dedicated_index);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    StagingPoolConfig config_{};

    DeviceBufferHeap heap_{};
    std::vector<std::unique_ptr<GpuBuffer>> dedicated_{};
    std::vector<std::uint32_t> free_dedicated_ids_{};

    std::vector<StagingAlloc> immediate_pending_{};
    std::unique_ptr<vk::raii::CommandPool> immediate_pool_{};
    vk::raii::CommandBuffer immediate_cmd_{nullptr};
    bool immediate_recording_ = false;
    std::unique_ptr<vk::raii::Fence> immediate_fence_{};

    FrameRing<StagingAlloc> ring_{};
    std::uint64_t outstanding_bytes_ = 0;
};

} // namespace VulkanEngine::GpuResources
