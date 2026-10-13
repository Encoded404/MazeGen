module;

#include <logging/logging_macros.hpp>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_hpp_macros.hpp>

module VulkanEngine.GpuResources.TransientAllocator;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;

namespace VulkanEngine::GpuResources {

namespace {

struct ImageReqs {
    std::uint64_t size = 0;
    std::uint64_t alignment = 1;
    std::uint32_t memory_type_bits = 0;
    bool requires_dedicated = false;
};

std::optional<std::uint32_t> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                            std::uint32_t type_bits,
                                            vk::MemoryPropertyFlags required) {
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0u &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return std::nullopt;
}

ImageReqs QueryImageRequirements(const vk::raii::Device& device, const vk::raii::Image& image) {
    vk::MemoryDedicatedRequirements dedicated{};
    vk::MemoryRequirements2 requirements{};
    requirements.pNext = &dedicated;

    vk::ImageMemoryRequirementsInfo2 info{};
    info.image = static_cast<vk::Image>(*image);

    auto* dispatcher = device.getDispatcher();
    dispatcher->vkGetImageMemoryRequirements2(
        static_cast<vk::Device::CType>(*device),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2*>(&info),
        reinterpret_cast<VkMemoryRequirements2*>(&requirements));

    ImageReqs result{};
    result.size = requirements.memoryRequirements.size;
    result.alignment = std::max<vk::DeviceSize>(requirements.memoryRequirements.alignment, 1);
    result.memory_type_bits = requirements.memoryRequirements.memoryTypeBits;
    result.requires_dedicated = dedicated.requiresDedicatedAllocation == vk::True;
    return result;
}

}  // namespace

TransientAllocator::~TransientAllocator() {
    Shutdown();
}

bool TransientAllocator::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                    const std::string& debug_name) {
    backend_ = &backend;
    debug_name_ = debug_name;
    return true;
}

void TransientAllocator::SetQueueFamilies(std::span<const std::uint32_t> families) {
    queue_families_.assign(families.begin(), families.end());
}

void TransientAllocator::Shutdown() {
    if (current_ != nullptr) {
        ReleaseGeneration(*current_);
        current_.reset();
    }
    for (auto& generation : retired_) {
        if (generation != nullptr) {
            ReleaseGeneration(*generation);
        }
    }
    retired_.clear();
    plan_ = {};
    backend_ = nullptr;
}

void TransientAllocator::ReleaseGeneration(Generation& generation) {
    // Destruction order: views/images first, then memory.
    for (auto& image : generation.images) {
        image.view.reset();
        image.image.reset();
        image.dedicated_memory.reset();
    }
    generation.images.clear();
    for (auto& heap : generation.heaps) {
        heap.buffer.reset();
        heap.memory.reset();
    }
    generation.heaps.clear();
}

std::uint64_t TransientAllocator::HashDescs(const std::vector<Desc>& descs, std::uint32_t frames_in_flight) const {
    std::string serialized = std::to_string(frames_in_flight) + ";";
    for (const auto& desc : descs) {
        const auto& r = desc.requirements;
        serialized += r.name + "," + std::to_string(static_cast<int>(r.kind)) + "," +
                      std::to_string(r.heap_key) + "," + std::to_string(r.size) + "," +
                      std::to_string(r.alignment) + "," + std::to_string(r.first_pass) + "," +
                      std::to_string(r.last_pass) + "," + (r.aliasable ? "1" : "0") + "," +
                      (desc.is_image ? "i" : "b") + "," +
                      std::to_string(static_cast<int>(desc.image.format)) + "," +
                      std::to_string(desc.image.width) + "x" + std::to_string(desc.image.height) + "," +
                      std::to_string(static_cast<std::uint32_t>(desc.image.usage)) + ";";
    }
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char c : serialized) {
        hash ^= static_cast<std::uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool TransientAllocator::BuildGeneration(const std::vector<Desc>& descs,
                                         std::uint32_t frames_in_flight,
                                         Generation& out) {
    if (!backend_ || descs.empty()) {
        return false;
    }

    auto& device = backend_->GetDevice();
    const auto& physical_device = backend_->GetPhysicalDevice();
    const vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();
    const std::uint32_t frames = std::max(1u, frames_in_flight);

    std::vector<TransientRequirements> requirements;
    requirements.reserve(descs.size());
    std::vector<ImageReqs> image_requirements(descs.size());
    std::vector<bool> dedicated(descs.size(), false);
    std::uint32_t next_unique_heap = 0x80000000u;

    // Pass 1: create one unbound image per (image resource, FIF slot) and collect
    // its memory requirements; buffers use their declared size directly.
    for (std::uint32_t d = 0; d < descs.size(); ++d) {
        const auto& desc = descs[d];
        if (!desc.requirements.active) {
            // Index placeholder: keep positions aligned with graph resource
            // indices, but allocate nothing.
            requirements.push_back(desc.requirements);
            continue;
        }
        if (!desc.is_image) {
            requirements.push_back(desc.requirements);
            continue;
        }

        vk::ImageCreateInfo image_info{};
        image_info.imageType = vk::ImageType::e2D;
        image_info.format = desc.image.format;
        image_info.extent = vk::Extent3D{desc.image.width, desc.image.height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = desc.image.samples;
        image_info.tiling = vk::ImageTiling::eOptimal;
        image_info.usage = desc.image.usage;
        if (queue_families_.size() > 1) {
            image_info.sharingMode = vk::SharingMode::eConcurrent;
            image_info.queueFamilyIndexCount = static_cast<std::uint32_t>(queue_families_.size());
            image_info.pQueueFamilyIndices = queue_families_.data();
        } else {
            image_info.sharingMode = vk::SharingMode::eExclusive;
        }
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        if (desc.requirements.aliasable) {
            image_info.flags |= vk::ImageCreateFlagBits::eAlias;
        }

        std::vector<std::uint32_t> slots;
        slots.reserve(frames);
        for (std::uint32_t slot = 0; slot < frames; ++slot) {
            auto image = std::make_unique<vk::raii::Image>(device, image_info);
            if (slot == 0) {
                image_requirements[d] = QueryImageRequirements(device, *image);
            }
            out.images.push_back(OwnedImage{.image = std::move(image)});
            slots.push_back(static_cast<std::uint32_t>(out.images.size() - 1));
        }
        out.image_lookup[d] = std::move(slots);

        TransientRequirements requirement = desc.requirements;
        requirement.size = image_requirements[d].size;
        requirement.alignment = image_requirements[d].alignment;
        if (image_requirements[d].requires_dedicated) {
            // A dedicated image cannot share memory with anything, so give it a
            // private heap (the planner will never alias it).
            dedicated[d] = true;
            requirement.heap_key = next_unique_heap++;
            requirement.aliasable = false;
        }
        requirements.push_back(requirement);
    }

    // Pass 2: pure placement, including alias reuse.
    out.plan = PlanTransientPlacements(requirements, frames);
    out.placements = out.plan.placements;
    if (!out.plan.valid) {
        return false;
    }

    // Pass 3: allocate one device-memory block per heap key, and a buffer for
    // buffer heaps.
    struct HeapInfo {
        std::uint32_t memory_type_bits = std::numeric_limits<std::uint32_t>::max();
        vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        vk::BufferUsageFlags buffer_usage{};
        bool is_buffer = false;
        bool has_image = false;
    };

    std::vector<std::uint32_t> heap_order;
    std::unordered_map<std::uint32_t, HeapInfo> heap_infos;
    for (std::uint32_t d = 0; d < descs.size(); ++d) {
        if (dedicated[d] || !requirements[d].active) {
            continue;
        }
        const std::uint32_t key = requirements[d].heap_key;
        if (heap_infos.find(key) == heap_infos.end()) {
            heap_order.push_back(key);
        }
        auto& heap = heap_infos[key];
        if (descs[d].is_image) {
            heap.has_image = true;
            heap.memory_type_bits &= image_requirements[d].memory_type_bits;
        } else {
            heap.is_buffer = true;
            heap.buffer_usage |= descs[d].buffer.usage;
            heap.flags = descs[d].buffer.memory;
        }
    }

    for (const std::uint32_t key : heap_order) {
        const auto& heap_info = heap_infos[key];
        const std::uint64_t copy_size = out.plan.GetHeapCopySize(key);
        if (copy_size == 0) {
            continue;
        }
        const std::uint64_t total_size = copy_size * frames;

        const auto memory_type = FindMemoryType(memory_properties, heap_info.memory_type_bits, heap_info.flags);
        if (!memory_type) {
            LOGIFACE_LOG(error, "TransientAllocator '" + debug_name_ + "': no memory type for heap " +
                                    std::to_string(key));
            return false;
        }

        Heap heap{};
        heap.heap_key = key;
        heap.total_size = total_size;

        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = total_size;
        alloc_info.memoryTypeIndex = *memory_type;
        heap.memory = std::make_unique<vk::raii::DeviceMemory>(device, alloc_info);

        if (heap_info.is_buffer) {
            vk::BufferCreateInfo buffer_info{};
            buffer_info.size = total_size;
            buffer_info.usage = heap_info.buffer_usage |
                                vk::BufferUsageFlagBits::eTransferSrc |
                                vk::BufferUsageFlagBits::eTransferDst;
            if (queue_families_.size() > 1) {
                buffer_info.sharingMode = vk::SharingMode::eConcurrent;
                buffer_info.queueFamilyIndexCount = static_cast<std::uint32_t>(queue_families_.size());
                buffer_info.pQueueFamilyIndices = queue_families_.data();
            } else {
                buffer_info.sharingMode = vk::SharingMode::eExclusive;
            }
            heap.buffer = std::make_unique<vk::raii::Buffer>(device, buffer_info);
            heap.buffer->bindMemory(*heap.memory, 0);
        }

        out.heaps.push_back(std::move(heap));
    }

    const auto find_heap = [&](std::uint32_t key) -> Heap* {
        for (auto& heap : out.heaps) {
            if (heap.heap_key == key) {
                return &heap;
            }
        }
        return nullptr;
    };

    // Pass 4: bind each image at its planned offset and create its view.
    for (const auto& placement : out.placements) {
        const std::uint32_t d = placement.resource_index;
        if (d >= descs.size() || !descs[d].is_image) {
            continue;
        }
        const auto lookup = out.image_lookup.find(d);
        if (lookup == out.image_lookup.end() || placement.fif_slot >= lookup->second.size()) {
            continue;
        }
        auto& owned = out.images[lookup->second[placement.fif_slot]];

        if (dedicated[d]) {
            const auto memory_type = FindMemoryType(memory_properties,
                                                    image_requirements[d].memory_type_bits,
                                                    vk::MemoryPropertyFlagBits::eDeviceLocal);
            if (!memory_type) {
                return false;
            }
            vk::MemoryAllocateInfo alloc_info{};
            alloc_info.allocationSize = image_requirements[d].size;
            alloc_info.memoryTypeIndex = *memory_type;
            owned.dedicated_memory = std::make_unique<vk::raii::DeviceMemory>(device, alloc_info);
            owned.image->bindMemory(*owned.dedicated_memory, 0);
        } else {
            Heap* heap = find_heap(placement.heap_key);
            if (heap == nullptr || !heap->memory) {
                return false;
            }
            owned.image->bindMemory(*heap->memory, placement.offset);
        }

        vk::ImageViewCreateInfo view_info{};
        view_info.image = static_cast<vk::Image>(**owned.image);
        view_info.viewType = vk::ImageViewType::e2D;
        view_info.format = descs[d].image.format;
        view_info.subresourceRange = {descs[d].image.aspect, 0, 1, 0, 1};
        owned.view = std::make_unique<vk::raii::ImageView>(device, view_info);
    }

    return true;
}

void TransientAllocator::Sync(const std::vector<Desc>& descs,
                              std::uint32_t frames_in_flight,
                              std::uint32_t current_frame) {
    if (!backend_) {
        return;
    }

    const std::uint64_t hash = HashDescs(descs, frames_in_flight);
    if (current_ != nullptr && current_->hash == hash) {
        return;
    }

    if (current_ != nullptr) {
        current_->retire_frame = current_frame;
        retired_.push_back(std::move(current_));
    }

    if (descs.empty()) {
        plan_ = {};
        return;
    }

    auto generation = std::make_unique<Generation>();
    generation->hash = hash;
    if (!BuildGeneration(descs, frames_in_flight, *generation)) {
        LOGIFACE_LOG(error, "TransientAllocator '" + debug_name_ + "': failed to build a transient generation");
        return;
    }

    plan_ = generation->plan;
    current_ = std::move(generation);
}

void TransientAllocator::CollectGarbage(std::uint32_t current_frame) {
    if (!backend_) {
        return;
    }
    for (auto it = retired_.begin(); it != retired_.end();) {
        Generation& generation = **it;
        // Free only once the frame that last used this generation has completed;
        // a same-frame retire cannot be retired yet.
        if (current_frame != generation.retire_frame &&
            backend_->IsFrameComplete(generation.retire_frame)) {
            ReleaseGeneration(generation);
            it = retired_.erase(it);
        } else {
            ++it;
        }
    }
}

vk::Image TransientAllocator::GetImage(std::uint32_t resource_index, std::uint32_t fif_slot) const {
    if (current_ == nullptr) {
        return nullptr;
    }
    const auto lookup = current_->image_lookup.find(resource_index);
    if (lookup == current_->image_lookup.end() || fif_slot >= lookup->second.size()) {
        return nullptr;
    }
    const auto& owned = current_->images[lookup->second[fif_slot]];
    return owned.image ? static_cast<vk::Image>(**owned.image) : nullptr;
}

vk::ImageView TransientAllocator::GetImageView(std::uint32_t resource_index, std::uint32_t fif_slot) const {
    if (current_ == nullptr) {
        return nullptr;
    }
    const auto lookup = current_->image_lookup.find(resource_index);
    if (lookup == current_->image_lookup.end() || fif_slot >= lookup->second.size()) {
        return nullptr;
    }
    const auto& owned = current_->images[lookup->second[fif_slot]];
    return owned.view ? static_cast<vk::ImageView>(**owned.view) : nullptr;
}

bool TransientAllocator::GetBuffer(std::uint32_t resource_index, std::uint32_t fif_slot,
                                   vk::Buffer& out_buffer, vk::DeviceSize& out_offset,
                                   vk::DeviceSize& out_size) const {
    if (current_ == nullptr) {
        return false;
    }
    for (const auto& placement : current_->placements) {
        if (placement.resource_index != resource_index || placement.fif_slot != fif_slot) {
            continue;
        }
        for (const auto& heap : current_->heaps) {
            if (heap.heap_key == placement.heap_key && heap.buffer) {
                out_buffer = static_cast<vk::Buffer>(**heap.buffer);
                out_offset = placement.offset;
                out_size = placement.size;
                return true;
            }
        }
    }
    return false;
}

}  // namespace VulkanEngine::GpuResources
