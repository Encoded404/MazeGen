module;

#include <logging/logging_macros.hpp>

module VulkanEngine.GpuResources.DeviceBufferHeap;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;

namespace VulkanEngine::GpuResources {

DeviceBufferHeap::~DeviceBufferHeap() {
    Shutdown();
}

bool DeviceBufferHeap::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                   const HeapConfig& config,
                                   const std::string& debug_name) {
    backend_ = &backend;
    config_ = config;
    debug_name_ = debug_name;
    if (config_.default_alignment == 0ULL) {
        config_.default_alignment = 256ULL;
    }

    LOGIFACE_LOG(debug, "DeviceBufferHeap '" + debug_name_ + "' initialized: block_size=" +
                 std::to_string(config_.block_size / (1024ULL * 1024ULL)) + " MB");
    return true;
}

void DeviceBufferHeap::Reset() {
    for (auto& block : blocks_) {
        block.allocator.Reset();
    }
}

void DeviceBufferHeap::Shutdown() {
    for (auto& block : blocks_) {
        if (block.mapped_ptr) {
            block.buffer.Unmap();
            block.mapped_ptr = nullptr;
        }
    }
    blocks_.clear();
    backend_ = nullptr;
}

void* DeviceBufferHeap::MapBuffer(std::uint32_t block_index, std::uint64_t offset) {
    if (block_index >= blocks_.size()) return nullptr;
    auto& block = blocks_[block_index];
    if (block.mapped_ptr) {
        return static_cast<char*>(block.mapped_ptr) + offset;
    }
    return nullptr;
}

uint32_t DeviceBufferHeap::AddBlock() {
    if (!backend_) return UINT32_MAX;

    const std::uint32_t index = static_cast<std::uint32_t>(blocks_.size());

    const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer |
                                       vk::BufferUsageFlagBits::eTransferDst |
                                       vk::BufferUsageFlagBits::eTransferSrc |
                                       config_.extra_usage;

    GpuBuffer buffer = GpuBuffer::Create(
        *backend_, config_.block_size, usage,
        config_.memory_flags);

    if (!buffer.IsValid()) {
        LOGIFACE_LOG(error, "DeviceBufferHeap: failed to allocate block " + std::to_string(index));
        return UINT32_MAX;
    }

    Block block;
    block.buffer = std::move(buffer);
    block.size = config_.block_size;
    block.allocator.Initialize(config_.block_size);
    block.mapped_ptr = nullptr;

    if (config_.memory_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        block.mapped_ptr = block.buffer.Map(0, block.size);
    }

    blocks_.push_back(std::move(block));

    LOGIFACE_LOG(debug, "DeviceBufferHeap '" + debug_name_ + "': added block " + std::to_string(index) +
                 " (" + std::to_string(config_.block_size / (1024ULL * 1024ULL)) + " MB)");
    return index;
}

HeapAllocation DeviceBufferHeap::Allocate(uint64_t size, std::uint64_t alignment) {
    if (size == 0) return {};
    if (alignment == 0) alignment = config_.default_alignment;

    for (std::uint32_t bi = 0; bi < static_cast<std::uint32_t>(blocks_.size()); ++bi) {
        auto& block = blocks_[bi];
        const std::uint64_t offset = block.allocator.Allocate(size, alignment);
        if (offset != UINT64_MAX) {
            HeapAllocation alloc{};
            alloc.buffer_index = bi;
            alloc.offset = offset;
            alloc.size = size;
            return alloc;
        }
    }

    // No space found — add a new block
    const std::uint32_t new_bi = AddBlock();
    if (new_bi == UINT32_MAX) return {};

    auto& new_block = blocks_[new_bi];
    const std::uint64_t offset = new_block.allocator.Allocate(size, alignment);
    if (offset == UINT64_MAX) return {};

    HeapAllocation alloc{};
    alloc.buffer_index = new_bi;
    alloc.offset = offset;
    alloc.size = size;
    return alloc;
}

void DeviceBufferHeap::Free(HeapAllocation& alloc) {
    if (alloc.buffer_index >= blocks_.size()) return;

    auto& block = blocks_[alloc.buffer_index];
    block.allocator.Free(alloc.offset, alloc.size);

    alloc = {};
}

vk::Buffer DeviceBufferHeap::GetBuffer(std::uint32_t index) const {
    if (index >= blocks_.size()) return nullptr;
    return *blocks_[index].buffer.GetBuffer();
}

} // namespace VulkanEngine::GpuResources
