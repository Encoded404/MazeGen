module;
#include <cassert>

module VulkanEngine.GpuResources.BlockArray;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.StagingPool;

namespace VulkanEngine::GpuResources {

BlockArray::~BlockArray() {
    Shutdown();
}

BlockArray::BlockArray(BlockArray&&) noexcept = default;
BlockArray& BlockArray::operator=(BlockArray&&) noexcept = default;

bool BlockArray::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend, const Config& cfg) {
    if (cfg.entry_size == 0) return false;
    if (cfg.entries_per_block == 0) return false;
    backend_ = &backend;
    cfg_ = cfg;
    return true;
}

void BlockArray::Shutdown() {
    blocks_.clear();
    mappings_.clear();
    backend_ = nullptr;
}

bool BlockArray::AddBlock() {
    if (!backend_) return false;
    if (blocks_.size() >= cfg_.max_blocks) return false;

    const std::uint64_t block_bytes = BlockSize();
    if (block_bytes == 0) return false;

    // For device-local blocks, add TransferDst usage for staging uploads.
    // Host-visible blocks use direct CPU mapping instead.
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (cfg_.memory_mode == MemoryMode::DeviceLocal) {
        usage |= vk::BufferUsageFlagBits::eTransferDst;
    }
    usage |= cfg_.extra_usage;

    GpuBuffer buffer = GpuBuffer::Create(*backend_, block_bytes, usage, cfg_.memory);

    if (!buffer.IsValid()) return false;

    void* mapping = nullptr; // NOLINT
    if (cfg_.memory_mode == MemoryMode::HostVisible && (cfg_.memory & vk::MemoryPropertyFlagBits::eHostVisible)) {
        mapping = buffer.Map(0, block_bytes);
    }

    blocks_.push_back(std::move(buffer));
    mappings_.push_back(mapping);
    return true;
}

bool BlockArray::EnsureCapacity(std::uint32_t count) {
    if (!backend_) return false;

    const std::uint32_t needed_blocks = (count + cfg_.entries_per_block - 1) / cfg_.entries_per_block;
    if (needed_blocks > cfg_.max_blocks) return false;
    while (blocks_.size() < needed_blocks) {
        if (!AddBlock()) return false;
    }
    // count == 0 is satisfied by an empty array; it is not a failure.
    return true;
}

void* BlockArray::Get(std::uint32_t index) {
    const std::uint32_t block_idx = index / cfg_.entries_per_block;
    const std::uint32_t local_idx = index % cfg_.entries_per_block;

    if (block_idx >= blocks_.size()) return nullptr;
    if (!mappings_[block_idx]) {
        // Device-local blocks have no mapping — caller must use UploadEntry() instead
        assert(!"Get() called on device-local BlockArray — use UploadEntry() instead");
        return nullptr;
    }

    return static_cast<char*>(mappings_[block_idx]) + static_cast<uint64_t>(local_idx) * cfg_.entry_size;
}

vk::Buffer BlockArray::GetBlockArray(std::uint32_t block_index) const {
    if (block_index >= blocks_.size()) return nullptr;
    return *blocks_[block_index].GetBuffer();
}

void BlockArray::UploadEntry(std::uint32_t index, const void* data, std::uint64_t size,
                              StagingPool& staging) {
    const std::uint32_t block_idx = index / cfg_.entries_per_block;
    const std::uint32_t local_idx = index % cfg_.entries_per_block;
    assert(block_idx < blocks_.size());
    assert(size <= cfg_.entry_size);

    const std::uint64_t dst_offset = static_cast<std::uint64_t>(block_idx) * BlockSize()
                                   + static_cast<std::uint64_t>(local_idx) * cfg_.entry_size;

    if (cfg_.memory_mode == MemoryMode::DeviceLocal) {
        // Staging upload path: allocate from the staging pool, memcpy, record
        // copy. The caller flushes once for the whole batch.
        auto slice = staging.Allocate(static_cast<std::uint64_t>(size), 256);
        if (!slice.has_value()) return;
        std::memcpy(slice->mapped_ptr, data, static_cast<std::size_t>(size));
        staging.RecordBufferCopy(*slice, *blocks_[block_idx].GetBuffer(), dst_offset);
    } else {
        // Host-visible path: direct memcpy to mapped memory
        void* ptr = Get(index);
        if (ptr) {
            std::memcpy(ptr, data, static_cast<std::size_t>(size));
        }
    }
}

} // namespace VulkanEngine::GpuResources
