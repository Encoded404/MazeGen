module;


export module VulkanEngine.GpuResources.DeviceBufferHeap;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.GpuBuffer;

import VulkanEngine.GpuResources.TlsfAllocator;

#ifndef UINT32_MAX
constexpr std::uint32_t UINT32_MAX =
    std::numeric_limits<std::uint32_t>::max();
#endif

#ifndef UINT64_MAX
constexpr std::uint64_t UINT64_MAX =
    std::numeric_limits<std::uint64_t>::max();
#endif

export namespace VulkanEngine::GpuResources {

struct HeapConfig {
    std::uint64_t block_size = 128ULL << 20; // 128 MiB
    vk::MemoryPropertyFlags memory_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    vk::BufferUsageFlags extra_usage = {};
    std::uint64_t default_alignment = 256ULL; // this is the minimum byte alignment for sub-allocations on the heap, older gpu's usually have 256 bytes, while newer have 16-64
};

struct HeapAllocation {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t buffer_index = UINT32_MAX;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsValid() const { return buffer_index != UINT32_MAX; }
};

class DeviceBufferHeap {
public:
    DeviceBufferHeap() = default;
    ~DeviceBufferHeap();

    DeviceBufferHeap(const DeviceBufferHeap&) = delete;
    DeviceBufferHeap& operator=(const DeviceBufferHeap&) = delete;
    // Movable so the per-frame heap ring can be runtime-sized (frames in flight is
    // configurable). All members (vector<Block>, TlsfAllocator, GpuBuffer) support
    // default move: TLSF arena pointers reference the mapped GPU memory, which is
    // stable across moving the owning objects.
    DeviceBufferHeap(DeviceBufferHeap&&) noexcept = default;
    DeviceBufferHeap& operator=(DeviceBufferHeap&&) noexcept = default;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    const HeapConfig& config = {},
                    const std::string& debug_name = "unnamed");
    void Shutdown();
    void Reset();

    HeapAllocation Allocate(std::uint64_t size, std::uint64_t alignment = 0);
    void Free(HeapAllocation& alloc);

    [[nodiscard]] void* MapBuffer(std::uint32_t block_index, std::uint64_t offset);

    [[nodiscard]] vk::Buffer GetBuffer(std::uint32_t index) const;
    [[nodiscard]] std::uint32_t GetBufferCount() const { return static_cast<std::uint32_t>(blocks_.size()); }
    [[nodiscard]] const HeapConfig& GetConfig() const { return config_; }
    [[nodiscard]] const std::string& GetDebugName() const { return debug_name_; }
    VulkanBackend::Vulkan::IVulkanBootstrap* GetBackend() { return backend_; }
    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }

private:
    struct Block {
        VulkanEngine::GpuResources::GpuBuffer buffer;
        std::uint64_t size;
        TlsfAllocator allocator;
        void* mapped_ptr = nullptr;
    };

    std::uint32_t AddBlock();

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    std::vector<Block> blocks_;
    HeapConfig config_;
    std::string debug_name_ = "unnamed";
};

}
