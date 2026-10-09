module;


export module VulkanEngine.MeshRenderSystem;

import std;

export import VulkanEngine.ECS.ComponentRegistry;
export import VulkanEngine.Components.Transform;
export import VulkanEngine.Components.MeshReference;
export import VulkanEngine.Components.DynamicMesh;
export import VulkanEngine.Components.MaterialOverride;
export import VulkanEngine.Components.TransformOverride;
export import VulkanEngine.MeshRegistry;
export import VulkanEngine.MeshManager;
export import VulkanEngine.SceneRenderer;
export import VulkanEngine.SceneLimits;
export import VulkanEngine.GpuResources.DeviceBufferHeap;
export import VulkanEngine.Mesh.MeshTypes;
export import VulkanEngine.MaterialManager;
export import VulkanEngine.TechniqueManager;
export import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
export import VulkanEngine.BindlessManager;

export namespace VulkanEngine {

class MeshRenderSystem {
public:
    MeshRenderSystem() = default;

    MeshRenderSystem(const MeshRenderSystem&) = delete;
    MeshRenderSystem& operator=(const MeshRenderSystem&) = delete;

    static constexpr std::uint32_t EVICTION_TIMEOUT_FRAMES = 120;

    void ProcessFrame(ComponentRegistry& registry,
                      MeshRegistry& mesh_registry,
                      MeshManager& mesh_mgr,
                      SceneRenderer::SceneRenderer& renderer,
                      GpuResources::DeviceBufferHeap& vtx_heap,
                      GpuResources::DeviceBufferHeap& idx_heap,
                      MaterialManager::MaterialManager& material_mgr,
                      std::uint32_t frame_index);

private:
    struct DrawEntity {
        Entity* entity = nullptr;
        const Components::Transform* transform = nullptr;
        const Components::MeshReference* mesh_ref = nullptr;
    };

    struct DynamicEntity {
        Entity* entity = nullptr;
        const Components::Transform* transform = nullptr;
        const Components::DynamicMesh* dyn_mesh = nullptr;
    };

    // Gather scratch, reused across frames. The vectors are cleared (not
    // freed) each frame, so their capacity persists at the historical peak and
    // a steady scene performs no allocation. Growing is amortized: the hint is
    // last frame's count, and the reserve target doubles the current capacity,
    // so a rising scene does not reserve once per frame. This replaces a
    // per-frame reserve of the full entity ceiling.
    template<typename T>
    static void EnsureGatherReserve(std::vector<T>& scratch, const std::size_t last_count) {
        const std::size_t hint =
            std::max<std::size_t>(last_count, SceneLimits::kInitialGatherCapacity);
        if (scratch.capacity() < hint) {
            scratch.reserve(std::max(hint, scratch.capacity() + scratch.capacity() / 2));
        }
    }

    std::vector<DrawEntity> static_ents_{};
    std::vector<DynamicEntity> dyn_ents_{};
    std::array<std::uint32_t, SceneRenderer::SceneRenderer::MAX_DRAW_GROUPS> tech_submeshes_{};
    std::size_t last_static_count_ = 0;
    std::size_t last_dyn_count_ = 0;
};

}

