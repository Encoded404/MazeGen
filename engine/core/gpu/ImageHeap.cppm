module;

export module VulkanEngine.GpuResources.GpuImageHeap;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;

import VulkanEngine.GpuResources.TlsfAllocator;
import VulkanEngine.GpuResources.FrameRing;

export namespace VulkanEngine::GpuResources {

struct ImageHeapConfig {
    std::uint64_t block_size = 64ULL << 20; // 64 MiB
    vk::MemoryPropertyFlags memory_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
};

struct HeapImage {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t image_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;
    bool dedicated = false;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsValid() const {
        return image_index != std::numeric_limits<std::uint32_t>::max();
    }
};

// View parameters for a heap image. The heap derives the view type when
// `view_type` is unset: eCube/eCubeArray for cube-compatible images with
// arrayLayers % 6 == 0, e2DArray for multi-layer, e3D for 3D, else e2D.
// `components` carries the KTXswizzle mapping (a view property, never a
// sampler property).
struct HeapViewDesc {
    HeapViewDesc() = default;
    // Implicit conversion keeps the historical Allocate(info, aspect) call shape.
    HeapViewDesc(vk::ImageAspectFlags aspect_flags) // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
        : aspect(aspect_flags) {}

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::optional<vk::ImageViewType> view_type{};
    vk::ComponentMapping components{};
    vk::ImageAspectFlags aspect{vk::ImageAspectFlagBits::eColor};
    std::uint32_t base_mip_level{0};
    std::uint32_t level_count{vk::RemainingMipLevels};
    std::uint32_t base_array_layer{0};
    std::uint32_t layer_count{vk::RemainingArrayLayers};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Device memory heap that owns sub-allocated VkImages. Mirrors
// DeviceBufferHeap: block memory + TLSF, with images bound at sub-offsets.
// Dedicated allocations are used whenever the driver requires them, and
// sub-allocations are aligned to bufferImageGranularity so an image never
// straddles a linear/non-linear boundary with a neighbouring allocation.
// Image indices are recycled through a free list and every record carries a
// generation, so a stale HeapImage handle cannot read or free a reused slot.
class GpuImageHeap {
public:
    GpuImageHeap() = default;
    ~GpuImageHeap();

    GpuImageHeap(const GpuImageHeap&) = delete;
    GpuImageHeap& operator=(const GpuImageHeap&) = delete;
    GpuImageHeap(GpuImageHeap&&) noexcept = default;
    GpuImageHeap& operator=(GpuImageHeap&&) noexcept = default;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    const ImageHeapConfig& config = {},
                    const std::string& debug_name = "unnamed");
    void Shutdown();

    // Creates `image_info` (initial layout is forced Undefined) and binds it to
    // either a dedicated allocation or a sub-allocated block region, then
    // creates a view from `view_desc`.
    HeapImage Allocate(vk::ImageCreateInfo image_info, const HeapViewDesc& view_desc = {});
    // Immediate free. Only safe when the device is idle (or the image is
    // provably unused); otherwise use Retire for frame-gated destruction.
    void Free(HeapImage& image);
    // Frame-gated retire: the image is freed when the frame that reuses
    // `recording_frame % frames_in_flight` begins, i.e. one full FIF cycle later.
    void Retire(HeapImage image, std::uint32_t recording_frame);
    // Drains the retire ring for the frame being started. Call once per frame.
    void BeginFrame(std::uint32_t frame_index);

    [[nodiscard]] vk::Image GetImage(std::uint32_t image_index) const;
    [[nodiscard]] vk::ImageView GetImageView(std::uint32_t image_index) const;
    [[nodiscard]] bool IsDedicated(std::uint32_t image_index) const;
    // Generation-checked overloads: a stale handle resolves to null.
    [[nodiscard]] vk::Image GetImage(const HeapImage& image) const;
    [[nodiscard]] vk::ImageView GetImageView(const HeapImage& image) const;
    // Device bytes reserved for the image (its memory requirement). Generation
    // checked: a stale handle reports 0.
    [[nodiscard]] std::uint64_t GetAllocationSize(const HeapImage& image) const;
    [[nodiscard]] std::uint64_t GetBufferImageGranularity() const { return buffer_image_granularity_; }
    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }
    [[nodiscard]] const std::string& GetDebugName() const { return debug_name_; }

private:
    struct Block {
        std::unique_ptr<vk::raii::DeviceMemory> memory;
        std::uint64_t size = 0;
        std::uint32_t memory_type_index = 0;
        TlsfAllocator allocator;
    };

    struct ImageRecord {
        std::unique_ptr<vk::raii::Image> image;
        std::unique_ptr<vk::raii::ImageView> view;
        std::unique_ptr<vk::raii::DeviceMemory> dedicated_memory;
        std::uint32_t block_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t generation = 0;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint64_t alignment = 1;
        bool dedicated = false;
    };

    // Creates a fresh device-memory block of the given type (at least min_size).
    [[nodiscard]] std::uint32_t CreateBlock(std::uint32_t memory_type_index, std::uint64_t min_size);
    void FreeRecord(ImageRecord& record);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    ImageHeapConfig config_{};
    std::string debug_name_ = "unnamed";
    std::uint64_t buffer_image_granularity_ = 1;

    std::vector<Block> blocks_{};
    std::vector<ImageRecord> images_{};
    std::vector<std::uint32_t> free_image_indices_{};
    std::uint32_t generation_counter_{0};
    FrameRing<HeapImage> retire_ring_{};
};

} // namespace VulkanEngine::GpuResources
