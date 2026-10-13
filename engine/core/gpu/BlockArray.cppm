module;

export module VulkanEngine.GpuResources.BlockArray;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;

import VulkanEngine.GpuResources.StagingPool;

export namespace VulkanEngine::GpuResources {

enum class MemoryMode : std::uint8_t {
    HostVisible,   // CPU-mapped, Get() returns direct pointer
    DeviceLocal,   // GPU-resident, upload via UploadEntry() with StagingManager
};

class BlockArray {
public:
    struct Config {
        std::uint32_t entry_size = 0;
        std::uint32_t entries_per_block = 256;
        // Hard ceiling on blocks. EnsureCapacity/AddBlock refuse to grow past
        // this, so a caller can bound an array to its descriptor-set binding
        // array size. Effectively unlimited by default.
        std::uint32_t max_blocks = std::numeric_limits<std::uint32_t>::max();
        vk::BufferUsageFlags extra_usage = {};
        vk::MemoryPropertyFlags memory = vk::MemoryPropertyFlagBits::eHostVisible |
                                          vk::MemoryPropertyFlagBits::eHostCoherent;
        MemoryMode memory_mode = MemoryMode::HostVisible;
    };

    BlockArray() = default;
    ~BlockArray();

    BlockArray(const BlockArray&) = delete;
    BlockArray& operator=(const BlockArray&) = delete;

    BlockArray(BlockArray&&) noexcept;
    BlockArray& operator=(BlockArray&&) noexcept;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend, const Config& cfg);

    void Shutdown();

    // Grows the array to hold at least `count` entries. Returns false without
    // growing when the request would exceed Config::max_blocks or a block
    // allocation fails. Callers whose descriptor-set layout bounds this array
    // MUST check the result: writing a block past the binding array size is
    // invalid descriptor usage.
    [[nodiscard]] bool EnsureCapacity(std::uint32_t count);

    [[nodiscard]] std::uint32_t BlockLimit() const { return cfg_.max_blocks; }

    // For HostVisible memory: returns pointer to entry data.
    // For DeviceLocal memory: asserts and returns nullptr — use UploadEntry() instead.
    void* Get(std::uint32_t index);

    // Upload data to a specific entry. For HostVisible memory, does a direct memcpy.
    // For DeviceLocal memory, uses StagingManager for transfer.
    void UploadEntry(std::uint32_t index, const void* data, std::uint64_t size,
                     StagingPool& staging);

    [[nodiscard]] bool IsDeviceLocal() const { return cfg_.memory_mode == MemoryMode::DeviceLocal; }
    [[nodiscard]] std::uint32_t BlockCount() const { return static_cast<std::uint32_t>(blocks_.size()); }
    [[nodiscard]] vk::Buffer GetBlockArray(std::uint32_t block_index) const;
    [[nodiscard]] std::uint64_t BlockSize() const { return static_cast<std::uint64_t>(cfg_.entries_per_block) * cfg_.entry_size; }
    [[nodiscard]] std::uint32_t EntriesPerBlock() const { return cfg_.entries_per_block; }
    [[nodiscard]] std::uint32_t EntrySize() const { return cfg_.entry_size; }
    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }

private:
    bool AddBlock();

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    Config cfg_{};
    std::vector<GpuBuffer> blocks_;
    std::vector<void*> mappings_;
};

} // namespace VulkanEngine::GpuResources
