module;

export module VulkanEngine.EngineContext;

import std;
import vulkan;

import VulkanShared.Jobs;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.BindlessManager;
import VulkanEngine.TextureUploader;
import VulkanEngine.TextureResidency;
import VulkanEngine.TextureReloader;
import VulkanEngine.TextureWatcher;
import VulkanEngine.Text.TextSystem;
import VulkanEngine.Text.FontWatcher;
import VulkanEngine.Text.FontReloader;
import VulkanEngine.SceneRenderer;
import VulkanEngine.TechniqueManager;
import VulkanEngine.Renderer;
import VulkanEngine.MeshManager;
import VulkanEngine.MeshRenderSystem;
import VulkanEngine.MeshRegistry;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;
import VulkanEngine.ImGui;
import VulkanBackend.ImGui;
import VulkanEngine.GpuResources;
import VulkanEngine.DefaultTextureFactory;
import VulkanEngine.StandardMeshPipeline;
import VulkanEngine.MaterialManager;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.GplPolicy;
import VulkanEngine.ShaderWatcher;
import VulkanEngine.ShaderRegistration;

#ifdef VKENGINE_PHYSICAL_CAMERA
import VulkanEngine.PhysicalCameraSystem;
#endif

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.World;
#endif

export namespace VulkanEngine {

struct GameConfig {
    StandardMeshPipeline::PipelineConfig pipeline_config{
        .depth_test_enable = true,
        .depth_write_enable = true,
        .depth_compare_op = vk::CompareOp::eLessOrEqual
    };
    Renderer::RendererConfig renderer_config{};
    // Advanced setting: indexed-drawing compaction/draw shape. Fixed at
    // renderer init; see SceneRenderer::DrawMode / Reinitialize.
    SceneRenderer::DrawMode draw_mode = SceneRenderer::DrawMode::CID;
    std::uint64_t geometry_buffer_size_mb = 128;
    // Requested bindless combined-image-sampler capacity. Clamped at boot by the
    // device limits minus the other update-after-bind pools' descriptor counts.
    std::uint32_t bindless_capacity = 65536;
    // Whole-texture residency budget in bytes. 0 means "use the
    // VK_EXT_memory_budget heap budget when present, otherwise disable
    // eviction". Set it to force a budget (or a smaller one) on any device.
    std::uint64_t texture_memory_budget_bytes = 0;
    // Shared compute-pool worker count — the engine's single thread budget for
    // everything compute-hungry (ECS updates, decode, compiles, physics). The
    // default picks `hardware_concurrency - 1`; `0` collapses every parallel
    // path onto the calling thread, which is the bit-identical deterministic
    // debugging mode.
    unsigned int job_workers = VulkanShared::JobBudget::kAuto;
    bool enable_imgui = true;
    std::string shader_data_dir;
    // Empty means "the resolved per-user cache root" (see GameEngine::Setup).
    // Never give this a CWD-relative default: a pipeline cache must survive
    // relaunches and a user-data reset, which a path relative to the working
    // directory does not.
    std::string shader_cache_dir;
    // Directory for the cooked collision-shape blobs (`PhysicsWorldConfig::
    // shape_cache_directory`). Empty — the default — disables the disk cache and
    // every shape is cooked at load. A cooked shape is a cache and never primary
    // data: the blobs are keyed by Jolt version and shape hash, so a Jolt upgrade
    // or a corrupt file is a miss that re-cooks, never a load failure. Point it at
    // a per-user cache root, for the same reason as `shader_cache_dir`.
    std::string physics_shape_cache_dir;
    VulkanEngine::ShaderSystem::GplPolicy gpl_policy = VulkanEngine::ShaderSystem::GplPolicy::Auto;
    VulkanEngine::ShaderSystem::GplStructurePolicy gpl_structure = VulkanEngine::ShaderSystem::GplStructurePolicy::Auto;
#ifdef VKENGINE_PHYSICAL_CAMERA
    bool enable_physical_camera = true;
#endif
};

struct EngineContext {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)

    // GPU resources (constructed in order). The dynamic FIFO heap rings are
    // runtime-sized: EngineBootstrap::Initialize resizes them to the device's
    // configured frames in flight (backend.GetFramesInFlight()).
    GpuResources::StagingPool staging_pool;
    GpuResources::DeviceBufferHeap vertex_heap;
    GpuResources::DeviceBufferHeap index_heap;
    // Out-of-line UV set 1 (optional). Same block-growth model as the vertex
    // heap; a mesh that authors no out-of-line set allocates nothing here.
    GpuResources::DeviceBufferHeap uv_heap;
    // Sub-allocated images (asset textures, camera streams/targets). Must
    // outlive bindless_mgr in shutdown (textures free their heap images).
    GpuResources::GpuImageHeap image_heap;
    // Shared, owning VkSampler cache. Must outlive every texture that samples
    // through it (bindless slots, camera streams/targets).
    GpuResources::SamplerCache sampler_cache;
    std::vector<GpuResources::DeviceBufferHeap> dynamic_vertex_heaps;
    std::vector<GpuResources::DeviceBufferHeap> dynamic_index_heaps;

    // Rendering subsystems
    std::unique_ptr<BindlessManager::BindlessManager> bindless_mgr;
    // Async texture uploader (render/bindless); constructed after bindless and
    // the staging pool, before anything that uploads a texture.
    std::unique_ptr<Textures::TextureUploader> texture_uploader;
    // Whole-texture residency under a device memory budget (optional
    // VK_EXT_memory_budget). Constructed after the uploader and material
    // manager; inert when no budget is available.
    std::unique_ptr<Textures::TextureResidency> texture_residency;
    // Texture hot reload: watches source image directories and swaps bindings
    // frame-safely. Gated by VKENGINE_HOT_RELOAD.
    std::unique_ptr<TextureSystem::TextureWatcher> texture_watcher;
    std::unique_ptr<Textures::TextureReloader> texture_reloader;

    // Text pipeline: font registry, shaping cache, hinted-glyph rasterizer and
    // the pure glyph atlas, plus an optional GPU layer added once the device
    // resources exist. The device-free core is constructed unconditionally;
    // InitializeGpu() adds the layer.
    std::unique_ptr<Text::TextSystem> text_system;
    // Font hot reload: watches source font files and rebuilds faces frame-safely
    // (new UniqueId, cache invalidation, ring-drained page retirement). Gated by
    // VKENGINE_HOT_RELOAD like the texture hot-reload pair.
    std::unique_ptr<Text::FontWatcher> font_watcher;
    std::unique_ptr<Text::FontReloader> font_reloader;
    std::unique_ptr<SceneRenderer::SceneRenderer> scene_renderer;
    std::unique_ptr<TechniqueManager::TechniqueManager> technique_mgr;
    std::unique_ptr<Renderer::Renderer> renderer;
    std::unique_ptr<MeshManager> mesh_manager;
    MeshRenderSystem mesh_render_system;
    MeshRegistry mesh_registry;
    VulkanEngine::ComponentRegistry component_registry{};

    // Assets
    ResourceManager resource_manager;
    std::shared_ptr<TextureResource> missing_texture;
    ResourceHandle<TextureResource> fallback_handle;

    // UI
    std::unique_ptr<ImGui::ImGuiSystem> imgui_system;
    std::shared_ptr<VulkanBackend::ImGui::IImGuiBackend> imgui_backend;

    // Material manager (no singleton — owned by context)
    VulkanEngine::MaterialManager::MaterialManager material_mgr{};

    // Shader system
    std::unique_ptr<ShaderSystem::ShaderManager> shader_manager;
    std::unique_ptr<ShaderSystem::PipelineFactory> pipeline_factory;
    std::unique_ptr<ShaderSystem::ShaderWatcher> shader_watcher;
    EngineShaderIds shader_ids{};

#ifdef VKENGINE_PHYSICAL_CAMERA
    // Webcam capture + compositing (inert until a camera is opened)
    std::unique_ptr<PhysicalCamera::PhysicalCameraSystem> physical_camera;
#endif

#ifdef VKENGINE_PHYSICS
    // The physics world (Jolt behind the Physics facade). Constructed by
    // GameEngine::Setup, not by EngineBootstrap, because its frame contract is
    // registered against the engine's ECS schedule. Null when physics is not
    // compiled in.
    std::unique_ptr<Physics::World> physics_world;
#endif

    // Convenience accessors
    auto& GetBindlessManager() { return *bindless_mgr; }
    auto& GetSceneRenderer() { return *scene_renderer; }
    auto& GetTechniqueManager() { return *technique_mgr; }
    auto& GetRenderer() { return *renderer; }
    auto& GetMeshManager() { return *mesh_manager; }
    auto& GetMeshRenderSystem() { return mesh_render_system; }
    auto& GetMeshRegistry() { return mesh_registry; }
    auto& GetComponentRegistry() { return component_registry; }
    auto& GetMaterialManager() { return material_mgr; }
    auto& GetResourceManager() { return resource_manager; }
    auto& GetShaderManager() { return *shader_manager; }
    auto& GetPipelineFactory() { return *pipeline_factory; }
    auto& GetShaderIds() { return shader_ids; }
    auto& GetTextSystem() { return *text_system; }
    Text::FontWatcher* GetFontWatcher() { return font_watcher.get(); }
    Text::FontReloader* GetFontReloader() { return font_reloader.get(); }

    GpuResources::DeviceBufferHeap& GetVertexHeap() { return vertex_heap; }
    GpuResources::DeviceBufferHeap& GetIndexHeap() { return index_heap; }
    GpuResources::DeviceBufferHeap& GetUvHeap() { return uv_heap; }
    GpuResources::GpuImageHeap& GetImageHeap() { return image_heap; }
    GpuResources::SamplerCache& GetSamplerCache() { return sampler_cache; }
    GpuResources::StagingPool& GetStagingPool() { return staging_pool; }
    auto& GetDynamicVertexHeaps() { return dynamic_vertex_heaps; }
    auto& GetDynamicIndexHeaps() { return dynamic_index_heaps; }
    Textures::TextureResidency* GetTextureResidency() { return texture_residency.get(); }

    ImGui::ImGuiSystem* GetImGuiSystem() { return imgui_system.get(); }
    VulkanBackend::ImGui::IImGuiBackend* GetImGuiBackend() { return imgui_backend.get(); }

#ifdef VKENGINE_PHYSICAL_CAMERA
    PhysicalCamera::PhysicalCameraSystem* GetPhysicalCameraSystem() { return physical_camera.get(); }
#endif

    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

} // namespace VulkanEngine
