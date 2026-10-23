module;

export module VulkanEngine.MeshManager;

export import VulkanEngine.GpuResources.DeviceBufferHeap;
export import VulkanEngine.GpuResources.StagingPool;
export import VulkanEngine.GpuResources.MeshData;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.Mesh.MeshTypes;

import std;

import vulkan;

constexpr std::uint32_t UINT32_MAX =
    std::numeric_limits<std::uint32_t>::max();

export namespace VulkanEngine {

class MeshManager {
public:
    static constexpr std::uint32_t INVALID_HANDLE = UINT32_MAX;

    enum class Strategy : std::uint8_t { Persistent, Streamed };

    struct GpuMeshInfo {
        VulkanEngine::GpuResources::HeapAllocation vertex_allocation{};
        VulkanEngine::GpuResources::HeapAllocation index_allocation{};
        // Out-of-line UV set 1 allocation (invalid when the mesh has no UV1).
        VulkanEngine::GpuResources::HeapAllocation uv_allocation{};
        std::uint32_t vertex_buffer_index = 0;
        std::uint32_t index_buffer_index = 0;
        std::uint32_t uv_buffer_index = 0;
        // Element offset of the mesh's UV1 region in its heap block. The UV
        // heap is a separate element space, so this re-bases an absolute vertex
        // index: uv = uvBuffers[uv_buffer_index][uv_region_base + local].
        std::uint32_t uv_region_base = 0;
        bool has_uv1 = false;
        // Sized to the runtime frames in flight (see Initialize/RegisterStreamed);
        // one mirrored allocation per FIF slot so each in-flight frame writes and
        // reads an independent copy of the streamed geometry.
        std::vector<VulkanEngine::GpuResources::HeapAllocation> streamed_vertex_alloc;
        std::vector<VulkanEngine::GpuResources::HeapAllocation> streamed_index_alloc;
        std::vector<SubMesh> sub_meshes;
    };

    struct Handle {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        std::uint32_t id = INVALID_HANDLE;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
        [[nodiscard]] bool IsValid() const { return id != INVALID_HANDLE; }
    };

    MeshManager() = default;
    ~MeshManager();

    MeshManager(const MeshManager&) = delete;
    MeshManager& operator=(const MeshManager&) = delete;
    MeshManager(MeshManager&&) = delete;
    MeshManager& operator=(MeshManager&&) = delete;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    VulkanEngine::GpuResources::DeviceBufferHeap* vertex_heap,
                    VulkanEngine::GpuResources::DeviceBufferHeap* index_heap,
                    VulkanEngine::GpuResources::DeviceBufferHeap* uv_heap,
                    VulkanEngine::GpuResources::StagingPool* staging_mgr,
                    VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_vertex_heaps,
                    VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_index_heaps,
                    std::uint32_t frames_in_flight);
    void Shutdown();

    Handle UploadPersistent(const VulkanEngine::GpuResources::MeshData& data);

    Handle RegisterStreamed(const VulkanEngine::GpuResources::MeshData& initial_data);

    void UpdateStreamed(Handle handle, const VulkanEngine::GpuResources::MeshData& data,
                        std::uint32_t frame_index);

    void Remove(Handle handle);

    void EndFrame(std::uint32_t frame_index);

    [[nodiscard]] const GpuMeshInfo* GetMeshInfo(Handle handle) const;

    [[nodiscard]] vk::Buffer GetUvBuffer(std::uint32_t buffer_index) const;

    [[nodiscard]] vk::Buffer GetDynamicVertexBuffer(std::uint32_t fif_index, std::uint32_t buffer_index) const;
    [[nodiscard]] vk::Buffer GetDynamicIndexBuffer(std::uint32_t fif_index, std::uint32_t buffer_index) const;
    [[nodiscard]] std::uint64_t GetDynamicVertexBlockSize(std::uint32_t fif_index) const;
    [[nodiscard]] std::uint64_t GetDynamicIndexBlockSize(std::uint32_t fif_index) const;
    [[nodiscard]] std::uint32_t GetDynamicVertexBlockCount(std::uint32_t fif_index) const;
    [[nodiscard]] std::uint32_t GetDynamicIndexBlockCount(std::uint32_t fif_index) const;

    [[nodiscard]] std::uint32_t GetFramesInFlight() const { return frames_in_flight_; }

    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }

private:
    struct MeshEntry {
        Strategy strategy = Strategy::Persistent;
        GpuMeshInfo info{};
    };

    struct DeferredFree {
        VulkanEngine::GpuResources::HeapAllocation allocation;
        VulkanEngine::GpuResources::DeviceBufferHeap* heap = nullptr;
        std::int32_t frames_remaining = 0;
    };

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    VulkanEngine::GpuResources::DeviceBufferHeap* vertex_heap_ = nullptr;
    VulkanEngine::GpuResources::DeviceBufferHeap* index_heap_ = nullptr;
    VulkanEngine::GpuResources::DeviceBufferHeap* uv_heap_ = nullptr;
    VulkanEngine::GpuResources::StagingPool* staging_mgr_ = nullptr;
    VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_vertex_heaps_ = nullptr;
    VulkanEngine::GpuResources::DeviceBufferHeap* dynamic_index_heaps_ = nullptr;

    std::vector<MeshEntry> entries_;
    std::vector<std::uint32_t> free_handles_;
    std::vector<DeferredFree> deferred_free_queue_;
    std::uint32_t frames_in_flight_ = 3;
    // UpdateStreamed runs once per dynamic mesh per frame; a broken handle
    // would otherwise repeat the same diagnostic forever. Log once per handle
    // and re-arm when an update succeeds.
    std::unordered_set<std::uint32_t> reported_update_issues_{};
};

} // namespace VulkanEngine
