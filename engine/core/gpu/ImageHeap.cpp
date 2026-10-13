module;

#include <logging/logging_macros.hpp>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_hpp_macros.hpp>

module VulkanEngine.GpuResources.GpuImageHeap;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;

import VulkanEngine.GpuResources.TlsfAllocator;
import VulkanEngine.GpuResources.FrameRing;

namespace VulkanEngine::GpuResources {

namespace {

constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kInvalidOffset = std::numeric_limits<std::uint64_t>::max();

}  // namespace

GpuImageHeap::~GpuImageHeap() {
    Shutdown();
}

bool GpuImageHeap::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                              const ImageHeapConfig& config,
                              const std::string& debug_name) {
    backend_ = &backend;
    config_ = config;
    debug_name_ = debug_name;

    const auto properties = backend.GetPhysicalDevice().getProperties();
    buffer_image_granularity_ = std::max<vk::DeviceSize>(properties.limits.bufferImageGranularity, 1);

    retire_ring_.Initialize(backend.GetFramesInFlight());

    LOGIFACE_LOG(debug, "GpuImageHeap '" + debug_name_ + "' initialized: block_size=" +
                            std::to_string(config_.block_size / (1024ULL * 1024ULL)) +
                            " MB, bufferImageGranularity=" + std::to_string(buffer_image_granularity_));
    return true;
}

void GpuImageHeap::Shutdown() {
    retire_ring_.Flush([this](HeapImage& image, std::uint32_t) {
        if (image.IsValid() && image.image_index < images_.size()) {
            FreeRecord(images_[image.image_index]);
        }
    });
    images_.clear();
    free_image_indices_.clear();
    blocks_.clear();
    backend_ = nullptr;
}

std::uint32_t GpuImageHeap::CreateBlock(std::uint32_t memory_type_index, std::uint64_t min_size) {
    if (!backend_) {
        return kInvalidIndex;
    }

    auto& device = backend_->GetDevice();
    const std::uint64_t size = std::max(config_.block_size, min_size);

    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_type_index;

    auto memory = std::make_unique<vk::raii::DeviceMemory>(device, alloc_info);
    if (!**memory) {
        LOGIFACE_LOG(error, "GpuImageHeap '" + debug_name_ + "': failed to allocate a block");
        return kInvalidIndex;
    }

    Block block;
    block.memory = std::move(memory);
    block.size = size;
    block.memory_type_index = memory_type_index;
    block.allocator.Initialize(size);

    const std::uint32_t index = static_cast<std::uint32_t>(blocks_.size());
    blocks_.push_back(std::move(block));
    return index;
}

namespace {

vk::ImageViewType DeriveViewType(const vk::ImageCreateInfo& info) {
    if (info.imageType == vk::ImageType::e3D) {
        return vk::ImageViewType::e3D;
    }
    const bool cube_compatible =
        (info.flags & vk::ImageCreateFlagBits::eCubeCompatible) != vk::ImageCreateFlags{};
    if (cube_compatible && info.arrayLayers >= 6U && info.arrayLayers % 6U == 0U) {
        return info.arrayLayers == 6U ? vk::ImageViewType::eCube : vk::ImageViewType::eCubeArray;
    }
    if (info.imageType == vk::ImageType::e1D) {
        return info.arrayLayers > 1U ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
    }
    return info.arrayLayers > 1U ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
}

}  // namespace

HeapImage GpuImageHeap::Allocate(vk::ImageCreateInfo image_info, const HeapViewDesc& view_desc) {
    if (!backend_) {
        return {};
    }

    auto& device = backend_->GetDevice();
    const auto& physical_device = backend_->GetPhysicalDevice();

    // Aliasable/sub-allocated images must start Undefined; callers relying on
    // contents-initialization are responsible for an explicit transition.
    image_info.initialLayout = vk::ImageLayout::eUndefined;

    auto image = std::make_unique<vk::raii::Image>(device, image_info);

    vk::MemoryDedicatedRequirements dedicated_requirements{};
    vk::MemoryRequirements2 requirements{};
    requirements.pNext = &dedicated_requirements;
    vk::ImageMemoryRequirementsInfo2 requirements_info{};
    requirements_info.image = static_cast<vk::Image>(**image);
    // The pNext-chained dedicated-requirements query has no enhanced-return
    // overload, so go through the dispatcher with the layout-identical C structs.
    auto* dispatcher = device.getDispatcher();
    dispatcher->vkGetImageMemoryRequirements2(
        static_cast<vk::Device::CType>(*device),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2*>(&requirements_info),
        reinterpret_cast<VkMemoryRequirements2*>(&requirements));

    const vk::MemoryRequirements memory_requirements = requirements.memoryRequirements;

    const auto memory_properties = physical_device.getMemoryProperties();
    std::optional<std::uint32_t> memory_type_index;
    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((memory_requirements.memoryTypeBits & (1u << i)) != 0u &&
            (memory_properties.memoryTypes[i].propertyFlags & config_.memory_flags) ==
                config_.memory_flags) {
            memory_type_index = i;
            break;
        }
    }
    if (!memory_type_index) {
        LOGIFACE_LOG(error, "GpuImageHeap '" + debug_name_ + "': no memory type matches the requested flags");
        return {};
    }

    ImageRecord record{};
    record.size = memory_requirements.size;
    record.alignment = std::max<vk::DeviceSize>(memory_requirements.alignment, buffer_image_granularity_);
    record.dedicated = dedicated_requirements.requiresDedicatedAllocation == vk::True;

    if (record.dedicated) {
        vk::MemoryDedicatedAllocateInfo dedicated_info{};
        dedicated_info.image = static_cast<vk::Image>(**image);

        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = memory_requirements.size;
        alloc_info.memoryTypeIndex = *memory_type_index;
        alloc_info.pNext = &dedicated_info;

        record.dedicated_memory = std::make_unique<vk::raii::DeviceMemory>(device, alloc_info);
        if (!**record.dedicated_memory) {
            return {};
        }
        image->bindMemory(*record.dedicated_memory, 0);
        record.offset = 0;
    } else {
        // Try every block of the matching memory type, then grow a new one.
        bool placed = false;
        for (std::uint32_t block_index = 0; block_index < blocks_.size(); ++block_index) {
            auto& block = blocks_[block_index];
            if (block.memory_type_index != *memory_type_index) {
                continue;
            }
            const std::uint64_t offset = block.allocator.Allocate(record.size, record.alignment);
            if (offset != kInvalidOffset) {
                image->bindMemory(*block.memory, offset);
                record.block_index = block_index;
                record.offset = offset;
                placed = true;
                break;
            }
        }
        if (!placed) {
            const std::uint32_t block_index = CreateBlock(*memory_type_index, record.size);
            if (block_index == kInvalidIndex) {
                return {};
            }
            auto& block = blocks_[block_index];
            const std::uint64_t offset = block.allocator.Allocate(record.size, record.alignment);
            if (offset == kInvalidOffset) {
                return {};
            }
            image->bindMemory(*block.memory, offset);
            record.block_index = block_index;
            record.offset = offset;
        }
    }

    const vk::ImageViewType view_type = view_desc.view_type.value_or(DeriveViewType(image_info));
    vk::ImageViewCreateInfo view_info{};
    view_info.image = static_cast<vk::Image>(**image);
    view_info.viewType = view_type;
    view_info.format = image_info.format;
    view_info.components = view_desc.components;
    view_info.subresourceRange = {view_desc.aspect, view_desc.base_mip_level, view_desc.level_count,
                                   view_desc.base_array_layer, view_desc.layer_count};
    record.view = std::make_unique<vk::raii::ImageView>(device, view_info);

    record.image = std::move(image);
    record.generation = ++generation_counter_;

    std::uint32_t image_index = kInvalidIndex;
    if (!free_image_indices_.empty()) {
        image_index = free_image_indices_.back();
        free_image_indices_.pop_back();
        images_[image_index] = std::move(record);
    } else {
        image_index = static_cast<std::uint32_t>(images_.size());
        images_.push_back(std::move(record));
    }

    return HeapImage{.image_index = image_index,
                     .generation = images_[image_index].generation,
                     .dedicated = images_[image_index].dedicated};
}

void GpuImageHeap::FreeRecord(ImageRecord& record) {
    if (!record.dedicated && record.block_index < blocks_.size()) {
        blocks_[record.block_index].allocator.Free(record.offset, record.size);
    }
    record.image.reset();
    record.view.reset();
    record.dedicated_memory.reset();
    record.block_index = kInvalidIndex;
    record.generation = 0;
}

void GpuImageHeap::Free(HeapImage& image) {
    if (!image.IsValid() || image.image_index >= images_.size()) {
        image = {};
        return;
    }

    auto& record = images_[image.image_index];
    if (record.generation != image.generation) {
        // Stale handle: the slot was already freed and recycled.
        image = {};
        return;
    }

    FreeRecord(record);
    free_image_indices_.push_back(image.image_index);
    image = {};
}

void GpuImageHeap::Retire(HeapImage image, std::uint32_t recording_frame) {
    if (!image.IsValid()) {
        return;
    }
    retire_ring_.Enqueue(image, recording_frame, /*gated=*/false);
}

void GpuImageHeap::BeginFrame(std::uint32_t frame_index) {
    retire_ring_.BeginFrame(
        frame_index,
        [this](HeapImage& image, std::uint32_t) {
            Free(image);
        },
        [](std::uint32_t) { return true; },
        [](const HeapImage&, std::uint32_t) {});
}

vk::Image GpuImageHeap::GetImage(std::uint32_t image_index) const {
    if (image_index >= images_.size() || !images_[image_index].image) {
        return nullptr;
    }
    return static_cast<vk::Image>(**images_[image_index].image);
}

vk::ImageView GpuImageHeap::GetImageView(std::uint32_t image_index) const {
    if (image_index >= images_.size() || !images_[image_index].view) {
        return nullptr;
    }
    return static_cast<vk::ImageView>(**images_[image_index].view);
}

bool GpuImageHeap::IsDedicated(std::uint32_t image_index) const {
    return image_index < images_.size() && images_[image_index].dedicated;
}

vk::Image GpuImageHeap::GetImage(const HeapImage& image) const {
    if (!image.IsValid() || image.image_index >= images_.size() ||
        images_[image.image_index].generation != image.generation) {
        return nullptr;
    }
    return GetImage(image.image_index);
}

vk::ImageView GpuImageHeap::GetImageView(const HeapImage& image) const {
    if (!image.IsValid() || image.image_index >= images_.size() ||
        images_[image.image_index].generation != image.generation) {
        return nullptr;
    }
    return GetImageView(image.image_index);
}

std::uint64_t GpuImageHeap::GetAllocationSize(const HeapImage& image) const {
    if (!image.IsValid() || image.image_index >= images_.size() ||
        images_[image.image_index].generation != image.generation) {
        return 0;
    }
    return images_[image.image_index].size;
}

} // namespace VulkanEngine::GpuResources
