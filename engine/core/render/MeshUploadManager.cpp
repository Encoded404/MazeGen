module;

#include <logging/logging_macros.hpp>

module VulkanEngine.MeshManager;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuResources.DeviceBufferHeap;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.StandardMeshPipeline;

namespace VulkanEngine {

MeshManager::~MeshManager() {
    Shutdown();
}

bool MeshManager::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                              VulkanEngine::GpuResources::DeviceBufferHeap* vertex_heap,
                              VulkanEngine::GpuResources::DeviceBufferHeap* index_heap,
                              VulkanEngine::GpuResources::DeviceBufferHeap* uv_heap,
                              VulkanEngine::GpuResources::StagingPool* staging_mgr,
                              VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_vertex_heaps,
                              VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_index_heaps,
                              std::uint32_t frames_in_flight) {
    if (!vertex_heap || !index_heap || !uv_heap || !staging_mgr || !dynamic_vertex_heaps || !dynamic_index_heaps) {
        LOGIFACE_LOG(error, "MeshManager: null dependencies");
        return false;
    }
    backend_ = &backend;
    vertex_heap_ = vertex_heap;
    index_heap_ = index_heap;
    uv_heap_ = uv_heap;
    staging_mgr_ = staging_mgr;
    dynamic_vertex_heaps_ = dynamic_vertex_heaps;
    dynamic_index_heaps_ = dynamic_index_heaps;
    frames_in_flight_ = frames_in_flight;
    return true;
}

void MeshManager::Shutdown() {
    entries_.clear();
    free_handles_.clear();
    deferred_free_queue_.clear();
    backend_ = nullptr;
}

MeshManager::Handle MeshManager::UploadPersistent(
    const VulkanEngine::GpuResources::MeshData& data) {
    Handle handle{};

    if (!backend_ || data.vertices.empty() || data.indices.empty()) {
        LOGIFACE_LOG(warn, "MeshManager::UploadPersistent: empty data");
        return handle;
    }

    const std::uint64_t vertex_data_size = data.vertices.size() *
        sizeof(VulkanEngine::StandardMeshPipeline::Vertex);
    const std::uint64_t index_data_size = data.indices.size() * sizeof(std::uint32_t);

    // Vertex allocations must be aligned to a multiple of the vertex stride:
    // baseVertex in expand.slang is computed as offset / sizeof(Vertex) with
    // integer division, so any offset that isn't a multiple of the stride
    // truncates and every vertex of the mesh reads shifted bytes (squished
    // geometry with far-flung vertices). 24 divides 240; use 240 so that the
    // common 256-multiples elsewhere don't matter but the stride always does.
    // 240 = LCM(24, 16) — also satisfies any 16B requirement, and divides 720/960.
    constexpr std::uint64_t kVertexAlignment = 240ULL;
    constexpr std::uint64_t alignment = 256ULL;
    auto vertex_alloc = vertex_heap_->Allocate(vertex_data_size, kVertexAlignment);
    if (!vertex_alloc.IsValid()) {
        LOGIFACE_LOG(error, "MeshManager::UploadPersistent: vertex heap allocation failed");
        return handle;
    }

    auto index_alloc = index_heap_->Allocate(index_data_size, alignment);
    if (!index_alloc.IsValid()) {
        vertex_heap_->Free(vertex_alloc);
        LOGIFACE_LOG(error, "MeshManager::UploadPersistent: index heap allocation failed");
        return handle;
    }

    // Optional out-of-line UV1. Allocated from the UV heap in its own element
    // space; the region base re-bases absolute vertex indices.
    const bool has_uv1 = !data.uv_extra[0].empty() &&
                         data.uv_extra[0].size() >= data.vertices.size();
    VulkanEngine::GpuResources::HeapAllocation uv_alloc{};
    std::uint32_t uv_region_base = 0;
    if (has_uv1) {
        const std::uint64_t uv_bytes = data.uv_extra[0].size() * sizeof(MeshVertexVec2);
        uv_alloc = uv_heap_->Allocate(uv_bytes, 16ULL);
        if (!uv_alloc.IsValid()) {
            vertex_heap_->Free(vertex_alloc);
            index_heap_->Free(index_alloc);
            LOGIFACE_LOG(error, "MeshManager::UploadPersistent: UV heap allocation failed");
            return handle;
        }
        uv_region_base = static_cast<std::uint32_t>(uv_alloc.offset / sizeof(MeshVertexVec2));
    }

    {
        auto slice = staging_mgr_->Allocate(vertex_data_size);
        if (!slice.has_value()) {
            vertex_heap_->Free(vertex_alloc);
            index_heap_->Free(index_alloc);
            if (uv_alloc.IsValid()) uv_heap_->Free(uv_alloc);
            LOGIFACE_LOG(error, "MeshManager::UploadPersistent: vertex staging allocation failed");
            return handle;
        }
        std::memcpy(slice->mapped_ptr, data.vertices.data(), vertex_data_size);
        staging_mgr_->RecordBufferCopy(*slice,
            vertex_heap_->GetBuffer(vertex_alloc.buffer_index), vertex_alloc.offset);
    }

    {
        auto slice = staging_mgr_->Allocate(index_data_size);
        if (!slice.has_value()) {
            vertex_heap_->Free(vertex_alloc);
            index_heap_->Free(index_alloc);
            if (uv_alloc.IsValid()) uv_heap_->Free(uv_alloc);
            LOGIFACE_LOG(error, "MeshManager::UploadPersistent: index staging allocation failed");
            return handle;
        }
        std::memcpy(slice->mapped_ptr, data.indices.data(), index_data_size);
        staging_mgr_->RecordBufferCopy(*slice,
            index_heap_->GetBuffer(index_alloc.buffer_index), index_alloc.offset);
    }

    if (uv_alloc.IsValid()) {
        const std::uint64_t uv_bytes = data.uv_extra[0].size() * sizeof(MeshVertexVec2);
        auto slice = staging_mgr_->Allocate(uv_bytes);
        if (!slice.has_value()) {
            vertex_heap_->Free(vertex_alloc);
            index_heap_->Free(index_alloc);
            uv_heap_->Free(uv_alloc);
            LOGIFACE_LOG(error, "MeshManager::UploadPersistent: UV staging allocation failed");
            return handle;
        }
        std::memcpy(slice->mapped_ptr, data.uv_extra[0].data(), uv_bytes);
        staging_mgr_->RecordBufferCopy(*slice,
            uv_heap_->GetBuffer(uv_alloc.buffer_index), uv_alloc.offset);
    }

    staging_mgr_->FlushImmediate();

    std::uint32_t handle_id;
    if (!free_handles_.empty()) {
        handle_id = free_handles_.back();
        free_handles_.pop_back();
        entries_[handle_id] = {};
    } else {
        handle_id = static_cast<std::uint32_t>(entries_.size());
        entries_.emplace_back();
    }

    MeshEntry& entry = entries_[handle_id];
    entry.strategy = Strategy::Persistent;
    entry.info.streamed_vertex_alloc.resize(frames_in_flight_);
    entry.info.streamed_index_alloc.resize(frames_in_flight_);
    entry.info.vertex_allocation = vertex_alloc;
    entry.info.index_allocation = index_alloc;
    entry.info.uv_allocation = uv_alloc;
    entry.info.vertex_buffer_index = vertex_alloc.buffer_index;
    entry.info.index_buffer_index = index_alloc.buffer_index;
    entry.info.uv_buffer_index = uv_alloc.buffer_index;
    entry.info.uv_region_base = uv_region_base;
    entry.info.has_uv1 = has_uv1;
    auto normalized = data;
    VulkanEngine::GpuResources::EnsureSubmeshBounds(normalized);
    entry.info.sub_meshes = std::move(normalized.sub_meshes);

    handle.id = handle_id;
    return handle;
}

MeshManager::Handle MeshManager::RegisterStreamed(
    const VulkanEngine::GpuResources::MeshData& initial_data) {
    Handle handle{};

    if (!dynamic_vertex_heaps_ || !dynamic_index_heaps_) return handle;

    const std::uint64_t vertex_bytes = initial_data.vertices.size() *
        sizeof(VulkanEngine::StandardMeshPipeline::Vertex);
    const std::uint64_t index_bytes = initial_data.indices.size() * sizeof(std::uint32_t);

    if (vertex_bytes == 0 || index_bytes == 0) {
        LOGIFACE_LOG(warn, "RegisterStreamed: empty mesh data (" +
                     std::to_string(initial_data.vertices.size()) + " verts, " +
                     std::to_string(initial_data.indices.size()) + " idxs)");
        return handle;
    }

    LOGIFACE_LOG(trace, "RegisterStreamed: " + std::to_string(initial_data.vertices.size()) +
                 " verts (" + std::to_string(vertex_bytes) + " bytes), " +
                 std::to_string(initial_data.indices.size()) + " idxs (" +
                 std::to_string(index_bytes) + " bytes), " +
                 std::to_string(initial_data.sub_meshes.size()) + " submeshes, " +
                 std::to_string(frames_in_flight_) + " FIFs");

    // Vertex allocations must be aligned to a multiple of sizeof(Vertex) (24):
    // see the note in UploadPersistent. 240 = LCM(24, 16).
    constexpr std::uint64_t kVertexAlignment = 240ULL;
    constexpr std::uint64_t alignment = 256ULL;

    auto normalized = initial_data;
    VulkanEngine::GpuResources::EnsureSubmeshBounds(normalized);
    GpuMeshInfo info{};
    info.streamed_vertex_alloc.resize(frames_in_flight_);
    info.streamed_index_alloc.resize(frames_in_flight_);
    info.sub_meshes = std::move(normalized.sub_meshes);

    for (std::uint32_t fif = 0; fif < frames_in_flight_; ++fif) {
        auto& vtx_alloc = info.streamed_vertex_alloc[fif];
        auto& idx_alloc = info.streamed_index_alloc[fif];

        vtx_alloc = dynamic_vertex_heaps_[fif].Allocate(vertex_bytes, kVertexAlignment);
        if (!vtx_alloc.IsValid()) {
            for (std::uint32_t j = 0; j < fif; ++j) {
                if (info.streamed_vertex_alloc[j].IsValid())
                    dynamic_vertex_heaps_[j].Free(info.streamed_vertex_alloc[j]);
                if (info.streamed_index_alloc[j].IsValid())
                    dynamic_index_heaps_[j].Free(info.streamed_index_alloc[j]);
            }
            LOGIFACE_LOG(error, "MeshManager::RegisterStreamed: vertex allocation failed for FIF " +
                         std::to_string(fif));
            return handle;
        }

        idx_alloc = dynamic_index_heaps_[fif].Allocate(index_bytes, alignment);
        if (!idx_alloc.IsValid()) {
            dynamic_vertex_heaps_[fif].Free(vtx_alloc);
            for (std::uint32_t j = 0; j < fif; ++j) {
                if (info.streamed_vertex_alloc[j].IsValid())
                    dynamic_vertex_heaps_[j].Free(info.streamed_vertex_alloc[j]);
                if (info.streamed_index_alloc[j].IsValid())
                    dynamic_index_heaps_[j].Free(info.streamed_index_alloc[j]);
            }
            LOGIFACE_LOG(error, "MeshManager::RegisterStreamed: index allocation failed for FIF " +
                         std::to_string(fif));
            return handle;
        }

        LOGIFACE_LOG(trace, "RegisterStreamed: FIF " + std::to_string(fif) +
                     " vtx_alloc buf_idx=" + std::to_string(vtx_alloc.buffer_index) +
                     " offset=" + std::to_string(vtx_alloc.offset) +
                     " size=" + std::to_string(vtx_alloc.size) +
                     " idx_alloc buf_idx=" + std::to_string(idx_alloc.buffer_index) +
                     " offset=" + std::to_string(idx_alloc.offset) +
                     " size=" + std::to_string(idx_alloc.size));

        void* vptr = dynamic_vertex_heaps_[fif].MapBuffer(
            vtx_alloc.buffer_index, vtx_alloc.offset);
        void* iptr = dynamic_index_heaps_[fif].MapBuffer(
            idx_alloc.buffer_index, idx_alloc.offset);
        if (vptr) std::memcpy(vptr, initial_data.vertices.data(), vertex_bytes);
        if (iptr) std::memcpy(iptr, initial_data.indices.data(), index_bytes);
    }

    std::uint32_t handle_id;
    if (!free_handles_.empty()) {
        handle_id = free_handles_.back();
        free_handles_.pop_back();
        entries_[handle_id] = {};
    } else {
        handle_id = static_cast<std::uint32_t>(entries_.size());
        entries_.emplace_back();
    }

    MeshEntry& entry = entries_[handle_id];
    entry.strategy = Strategy::Streamed;
    entry.info = std::move(info);

    handle.id = handle_id;
    return handle;
}

void MeshManager::UpdateStreamed(Handle handle,
                                  const VulkanEngine::GpuResources::MeshData& data,
                                  std::uint32_t frame_index) {
    if (!handle.IsValid() || handle.id >= entries_.size()) {
        if (LOGIFACE_ENABLED(debug) &&
            reported_update_issues_.insert(handle.id).second) {
            LOGIFACE_LOG(debug, "UpdateStreamed: invalid handle (id=" +
                         std::to_string(handle.id) + "), skipping");
        }
        return;
    }

    const auto& entry = entries_[handle.id];
    if (entry.strategy != Strategy::Streamed) {
        if (LOGIFACE_ENABLED(debug) &&
            reported_update_issues_.insert(handle.id).second) {
            LOGIFACE_LOG(debug, "UpdateStreamed: handle " + std::to_string(handle.id) +
                         " is not Streamed, skipping");
        }
        return;
    }

    const std::uint32_t fif = frame_index % frames_in_flight_;
    const auto& vtx_alloc = entry.info.streamed_vertex_alloc[fif];
    const auto& idx_alloc = entry.info.streamed_index_alloc[fif];

    if (!vtx_alloc.IsValid() || !idx_alloc.IsValid()) {
        if (LOGIFACE_ENABLED(debug) &&
            reported_update_issues_.insert(handle.id).second) {
            LOGIFACE_LOG(debug, "UpdateStreamed: FIF " + std::to_string(fif) +
                         " streamed alloc invalid for handle " + std::to_string(handle.id) +
                         ", skipping");
        }
        return;
    }

    if (LOGIFACE_ENABLED(debug)) {
        reported_update_issues_.erase(handle.id);
    }

    const std::uint64_t vertex_bytes = data.vertices.size() *
        sizeof(VulkanEngine::StandardMeshPipeline::Vertex);
    const std::uint64_t index_bytes = data.indices.size() * sizeof(std::uint32_t);

    LOGIFACE_LOG(trace, "UpdateStreamed: handle=" + std::to_string(handle.id) +
                 " fif=" + std::to_string(fif) +
                 " vtx buf_idx=" + std::to_string(vtx_alloc.buffer_index) +
                 " offset=" + std::to_string(vtx_alloc.offset) +
                 " bytes=" + std::to_string(vertex_bytes) +
                 " idx buf_idx=" + std::to_string(idx_alloc.buffer_index) +
                 " offset=" + std::to_string(idx_alloc.offset) +
                 " bytes=" + std::to_string(index_bytes));

    void* vptr = dynamic_vertex_heaps_[fif].MapBuffer(
        vtx_alloc.buffer_index, vtx_alloc.offset);
    if (!vptr) {
        LOGIFACE_LOG(warn, "UpdateStreamed: MapBuffer returned null for vertex heap " +
                     std::to_string(fif) + " block " + std::to_string(vtx_alloc.buffer_index));
    } else {
        std::memcpy(vptr, data.vertices.data(), vertex_bytes);
    }

    void* iptr = dynamic_index_heaps_[fif].MapBuffer(
        idx_alloc.buffer_index, idx_alloc.offset);
    if (!iptr) {
        LOGIFACE_LOG(warn, "UpdateStreamed: MapBuffer returned null for index heap " +
                     std::to_string(fif) + " block " + std::to_string(idx_alloc.buffer_index));
    } else {
        std::memcpy(iptr, data.indices.data(), index_bytes);
    }
}

void MeshManager::Remove(Handle handle) {
    if (!handle.IsValid() || handle.id >= entries_.size()) return;

    MeshEntry& entry = entries_[handle.id];
    const std::int32_t frames_wait = static_cast<std::int32_t>(frames_in_flight_);

    if (entry.strategy == Strategy::Persistent) {
        if (entry.info.vertex_allocation.IsValid()) {
            deferred_free_queue_.push_back({
                entry.info.vertex_allocation, vertex_heap_, frames_wait
            });
        }
        if (entry.info.index_allocation.IsValid()) {
            deferred_free_queue_.push_back({
                entry.info.index_allocation, index_heap_, frames_wait
            });
        }
        if (entry.info.uv_allocation.IsValid()) {
            deferred_free_queue_.push_back({
                entry.info.uv_allocation, uv_heap_, frames_wait
            });
        }
    } else if (entry.strategy == Strategy::Streamed) {
        for (std::uint32_t fif = 0; fif < frames_in_flight_; ++fif) {
            auto& valloc = entry.info.streamed_vertex_alloc[fif];
            auto& ialloc = entry.info.streamed_index_alloc[fif];
            if (valloc.IsValid()) {
                deferred_free_queue_.push_back({
                    valloc, &dynamic_vertex_heaps_[fif], frames_wait
                });
            }
            if (ialloc.IsValid()) {
                deferred_free_queue_.push_back({
                    ialloc, &dynamic_index_heaps_[fif], frames_wait
                });
            }
        }
    }

    entry = {};
    free_handles_.push_back(handle.id);
}

void MeshManager::EndFrame(std::uint32_t) {
    auto it = deferred_free_queue_.begin();
    while (it != deferred_free_queue_.end()) {
        it->frames_remaining--;
        if (it->frames_remaining <= 0) {
            if (it->heap && it->allocation.IsValid()) {
                auto alloc = it->allocation;
                it->heap->Free(alloc);
            }
            it = deferred_free_queue_.erase(it);
        } else {
            ++it;
        }
    }
}

const MeshManager::GpuMeshInfo* MeshManager::GetMeshInfo(Handle handle) const {
    if (!handle.IsValid() || handle.id >= entries_.size()) return nullptr;
    return &entries_[handle.id].info;
}

vk::Buffer MeshManager::GetUvBuffer(std::uint32_t buffer_index) const {
    if (!uv_heap_) return nullptr;
    return uv_heap_->GetBuffer(buffer_index);
}

vk::Buffer MeshManager::GetDynamicVertexBuffer(std::uint32_t fif_index, std::uint32_t buffer_index) const {
    if (fif_index >= frames_in_flight_) return nullptr;
    return dynamic_vertex_heaps_[fif_index].GetBuffer(buffer_index);
}

vk::Buffer MeshManager::GetDynamicIndexBuffer(std::uint32_t fif_index, std::uint32_t buffer_index) const {
    if (fif_index >= frames_in_flight_) return nullptr;
    return dynamic_index_heaps_[fif_index].GetBuffer(buffer_index);
}

uint64_t MeshManager::GetDynamicVertexBlockSize(std::uint32_t fif_index) const {
    if (fif_index >= frames_in_flight_) return 0;
    return dynamic_vertex_heaps_[fif_index].GetConfig().block_size;
}

uint64_t MeshManager::GetDynamicIndexBlockSize(std::uint32_t fif_index) const {
    if (fif_index >= frames_in_flight_) return 0;
    return dynamic_index_heaps_[fif_index].GetConfig().block_size;
}

uint32_t MeshManager::GetDynamicVertexBlockCount(std::uint32_t fif_index) const {
    if (fif_index >= frames_in_flight_) return 0;
    return dynamic_vertex_heaps_[fif_index].GetBufferCount();
}

uint32_t MeshManager::GetDynamicIndexBlockCount(std::uint32_t fif_index) const {
    if (fif_index >= frames_in_flight_) return 0;
    return dynamic_index_heaps_[fif_index].GetBufferCount();
}

} // namespace VulkanEngine
