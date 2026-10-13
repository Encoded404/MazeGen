module;

#include <logging/logging_macros.hpp>

module VulkanEngine.GpuResources.StagingPool;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.DeviceBufferHeap;
import VulkanEngine.GpuResources.FrameRing;
import VulkanShared.DeviceScope;

namespace VulkanEngine::GpuResources {

namespace {
constexpr std::uint64_t kWaitForever = std::numeric_limits<std::uint64_t>::max();
}

StagingPool::~StagingPool() {
    Shutdown();
}

bool StagingPool::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                             const StagingPoolConfig& config) {
    if (config.block_size == 0 || config.capacity_bytes == 0) {
        return false;
    }
    backend_ = &backend;
    config_ = config;
    if (config_.capacity_bytes < config_.block_size) {
        config_.capacity_bytes = config_.block_size;
    }

    HeapConfig heap_config{};
    heap_config.block_size = config_.block_size;
    heap_config.memory_flags =
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    heap_config.extra_usage = vk::BufferUsageFlagBits::eTransferSrc;
    heap_config.default_alignment = 256;
    if (!heap_.Initialize(backend, heap_config, "staging-pool")) {
        return false;
    }

    vk::CommandPoolCreateInfo pool_info{};
    pool_info.queueFamilyIndex = backend.GetGraphicsQueueFamily();
    pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    immediate_pool_ = std::make_unique<vk::raii::CommandPool>(backend.GetDevice(), pool_info);
    VulkanBackend::Vulkan::SetVulkanObjectName(backend.GetDevice(), *immediate_pool_, "staging-immediate-pool");

    vk::CommandBufferAllocateInfo alloc_info{};
    alloc_info.commandPool = **immediate_pool_;
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandBufferCount = 1;
    auto buffers = backend.GetDevice().allocateCommandBuffers(alloc_info);
    if (buffers.empty()) {
        return false;
    }
    immediate_cmd_ = std::move(buffers[0]);

    vk::FenceCreateInfo fence_info{};
    fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
    immediate_fence_ = std::make_unique<vk::raii::Fence>(backend.GetDevice(), fence_info);

    ring_.Initialize(backend.GetFramesInFlight());

    LOGIFACE_LOG(debug, "StagingPool initialized: block " +
                            std::to_string(config_.block_size / (1024ULL * 1024ULL)) +
                            " MB, capacity " +
                            std::to_string(config_.capacity_bytes / (1024ULL * 1024ULL)) + " MB");
    return true;
}

void StagingPool::Shutdown() {
    if (backend_) {
        try {
            FlushImmediate();
            if (immediate_fence_) {
                static_cast<void>(backend_->GetDevice().waitForFences(
                    **immediate_fence_, vk::True, kWaitForever));
            }
        } catch (const std::exception& err) {
            LOGIFACE_LOG(error, std::string("StagingPool: error during shutdown: ") + err.what());
        }
    }
    // Free every outstanding range so the heap can be destroyed cleanly.
    for (auto& alloc : immediate_pending_) {
        FreeAlloc(alloc);
    }
    immediate_pending_.clear();
    ring_.Flush([this](StagingAlloc& alloc, std::uint32_t) { FreeAlloc(alloc); });
    ring_.Initialize(1);

    immediate_cmd_ = vk::raii::CommandBuffer(nullptr);
    immediate_pool_.reset();
    immediate_fence_.reset();
    immediate_recording_ = false;
    dedicated_.clear();
    free_dedicated_ids_.clear();
    heap_.Shutdown();
    backend_ = nullptr;
    outstanding_bytes_ = 0;
}

std::optional<StagingAlloc> StagingPool::Allocate(std::uint64_t size, std::uint64_t alignment) {
    if (!backend_ || size == 0) {
        return std::nullopt;
    }
    // Back-pressure: a hard ceiling on outstanding staged bytes.
    if (outstanding_bytes_ + size > config_.capacity_bytes) {
        return std::nullopt;
    }

    if (size > config_.block_size) {
        // Dedicated host-visible buffer, size-classed by reuse; never split.
        auto buffer = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            *backend_, size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        if (!buffer->IsValid()) {
            return std::nullopt;
        }
        void* mapped = buffer->Map(0, size);
        if (mapped == nullptr) {
            return std::nullopt;
        }
        StagingAlloc alloc{};
        alloc.buffer = *buffer->GetBuffer();
        alloc.offset = 0;
        alloc.mapped_ptr = mapped;
        alloc.size = size;
        alloc.dedicated = true;
        alloc.dedicated_index = static_cast<std::uint32_t>(dedicated_.size());
        dedicated_.push_back(std::move(buffer));
        outstanding_bytes_ += size;
        return alloc;
    }

    HeapAllocation heap_alloc = heap_.Allocate(size, alignment);
    if (!heap_alloc.IsValid()) {
        return std::nullopt;
    }
    void* mapped = heap_.MapBuffer(heap_alloc.buffer_index, heap_alloc.offset);
    if (mapped == nullptr) {
        heap_.Free(heap_alloc);
        return std::nullopt;
    }
    StagingAlloc alloc{};
    alloc.buffer = heap_.GetBuffer(heap_alloc.buffer_index);
    alloc.offset = heap_alloc.offset;
    alloc.mapped_ptr = mapped;
    alloc.size = size;
    alloc.dedicated = false;
    alloc.heap_buffer_index = heap_alloc.buffer_index;
    outstanding_bytes_ += size;
    return alloc;
}

void StagingPool::FreeAlloc(const StagingAlloc& alloc) {
    if (alloc.size > outstanding_bytes_) {
        outstanding_bytes_ = 0;
    } else {
        outstanding_bytes_ -= alloc.size;
    }
    if (alloc.dedicated) {
        ReleaseDedicated(alloc.dedicated_index);
        return;
    }
    if (alloc.heap_buffer_index != std::numeric_limits<std::uint32_t>::max()) {
        HeapAllocation heap_alloc{};
        heap_alloc.buffer_index = alloc.heap_buffer_index;
        heap_alloc.offset = alloc.offset;
        heap_alloc.size = alloc.size;
        heap_.Free(heap_alloc);
    }
}

void StagingPool::ReleaseDedicated(std::uint32_t dedicated_index) {
    if (dedicated_index >= dedicated_.size()) {
        return;
    }
    dedicated_[dedicated_index].reset();
    free_dedicated_ids_.push_back(dedicated_index);
}

void StagingPool::EnsureImmediateCommandBuffer() {
    if (immediate_recording_) {
        return;
    }
    if (immediate_fence_) {
        static_cast<void>(backend_->GetDevice().waitForFences(
            **immediate_fence_, vk::True, kWaitForever));
        backend_->GetDevice().resetFences(**immediate_fence_);
    }
    immediate_cmd_.reset({});
    immediate_cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    immediate_recording_ = true;
}

void StagingPool::RecordBufferCopy(const StagingAlloc& alloc, vk::Buffer dst_buffer,
                                   std::uint64_t dst_offset) {
    if (!alloc.IsValid()) {
        return;
    }
    EnsureImmediateCommandBuffer();
    vk::BufferCopy region{};
    region.srcOffset = alloc.offset;
    region.dstOffset = dst_offset;
    region.size = alloc.size;
    immediate_cmd_.copyBuffer(alloc.buffer, dst_buffer, region);
    immediate_pending_.push_back(alloc);
}

void StagingPool::RecordBufferToImage(const StagingAlloc& alloc, vk::Image dst_image,
                                      std::span<const vk::BufferImageCopy> regions) {
    if (!alloc.IsValid() || regions.empty()) {
        return;
    }
    EnsureImmediateCommandBuffer();
    std::vector<vk::BufferImageCopy> adjusted(regions.begin(), regions.end());
    for (auto& region : adjusted) {
        region.bufferOffset += alloc.offset;
    }
    immediate_cmd_.copyBufferToImage(alloc.buffer, dst_image,
                                     vk::ImageLayout::eTransferDstOptimal, adjusted);
    immediate_pending_.push_back(alloc);
}

bool StagingPool::FlushImmediate() {
    if (!backend_ || !immediate_recording_) {
        return true;
    }
    immediate_recording_ = false;
    immediate_cmd_.end();

    vk::SubmitInfo submit{};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &*immediate_cmd_;
    {
        VulkanShared::DeviceScopeGuard guard;
        backend_->GetGraphicsQueue().submit(submit, **immediate_fence_);
        const auto result =
            backend_->GetDevice().waitForFences(**immediate_fence_, vk::True, kWaitForever);
            if (result != vk::Result::eSuccess) {
            LOGIFACE_LOG(error, "StagingPool: immediate fence wait failed");
            return false;
        }
    }

    for (auto& alloc : immediate_pending_) {
        FreeAlloc(alloc);
    }
    immediate_pending_.clear();
    return true;
}

void StagingPool::RecordBufferCopy(vk::CommandBuffer cmd, const StagingAlloc& alloc,
                                   vk::Buffer dst_buffer, std::uint64_t dst_offset) {
    if (!alloc.IsValid()) {
        return;
    }
    vk::BufferCopy region{};
    region.srcOffset = alloc.offset;
    region.dstOffset = dst_offset;
    region.size = alloc.size;
    cmd.copyBuffer(alloc.buffer, dst_buffer, region);
}

void StagingPool::RecordBufferToImage(vk::CommandBuffer cmd, const StagingAlloc& alloc,
                                      vk::Image dst_image,
                                      std::span<const vk::BufferImageCopy> regions) {
    if (!alloc.IsValid() || regions.empty()) {
        return;
    }
    std::vector<vk::BufferImageCopy> adjusted(regions.begin(), regions.end());
    for (auto& region : adjusted) {
        region.bufferOffset += alloc.offset;
    }
    cmd.copyBufferToImage(alloc.buffer, dst_image,
                          vk::ImageLayout::eTransferDstOptimal, adjusted);
}

void StagingPool::Retire(const StagingAlloc& alloc, std::uint32_t recording_frame) {
    ring_.Enqueue(alloc, recording_frame, /*gated=*/false);
}

void StagingPool::BeginFrame(std::uint32_t frame_index) {
    ring_.BeginFrame(
        frame_index,
        [this](StagingAlloc& alloc, std::uint32_t) { FreeAlloc(alloc); },
        [](std::uint32_t) { return true; },
        [](const StagingAlloc&, std::uint32_t) {});
}

} // namespace VulkanEngine::GpuResources
