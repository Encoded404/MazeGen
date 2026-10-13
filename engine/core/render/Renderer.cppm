module;

#include <glm/glm.hpp> //NOLINT(misc-include-cleaner)

export module VulkanEngine.Renderer;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.ECS.ComponentRegistry;
export import VulkanEngine.RenderPipeline;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.SceneRenderer;
export import VulkanEngine.TechniqueManager;
export import VulkanEngine.BindlessManager;
export import VulkanEngine.TextureUploader;
export import VulkanEngine.Components.Camera;
export import VulkanEngine.GpuResources;
export import VulkanEngine.ImGui;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.Render.Passes.ExpandPass;
import VulkanEngine.Render.Passes.OccluderSelectPass;
import VulkanEngine.Render.Passes.OccluderPrePass;
import VulkanEngine.Render.Passes.DepthPrePass;
import VulkanEngine.Render.Passes.HiZPass;
import VulkanEngine.Render.Passes.PreCullPass;
import VulkanEngine.Render.Passes.OcclusionPass;
import VulkanEngine.Render.Passes.CollectPass;
import VulkanEngine.Render.Passes.MainPass;
import VulkanEngine.Render.Passes.ImGuiPass;
import VulkanEngine.Render.Passes.TextPass;
// Re-exported because RendererConfig names SceneRenderer::TextBackend.
export import VulkanEngine.Render.Passes.WorldTextPass;
import VulkanEngine.ShaderRegistration;

#ifdef VKENGINE_PHYSICAL_CAMERA
import VulkanEngine.PhysicalCameraSystem;
#endif

export namespace VulkanEngine::Renderer {

struct RendererConfig {
    bool enable_imgui = true;
    // Registers the screen-space text overlay built-in. It draws nothing (and
    // costs one empty pass) until a caller queues a run through
    // Renderer::GetTextPass().
    bool enable_text = true;
    // Which rasterizer the depth-tested world-text built-in draws with. The MSDF
    // backend is the default; Slug selects the analytic hb-gpu backend, which
    // must have its blobs queued through WorldTextPass::QueueSlugRun instead.
    VulkanEngine::SceneRenderer::TextBackend text_backend =
        VulkanEngine::SceneRenderer::kDefaultTextBackend;
    glm::vec4 clear_color{0.1f, 0.1f, 0.1f, 1.0f};
    vk::ClearDepthStencilValue clear_depth_stencil{1.0f, 0};
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // `shader_ids` carries the registered engine shader handles the built-in
    // pass pipelines are built from. When null (or when the text overlay
    // shaders are not registered) the text built-in is not added.
    bool Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                    const RendererConfig& config,
                    VulkanEngine::SceneRenderer::SceneRenderer& scene_renderer,
                    ShaderSystem::ShaderManager* shader_manager = nullptr,
                    ShaderSystem::PipelineFactory* pipeline_factory = nullptr,
                    const VulkanEngine::EngineShaderIds* shader_ids = nullptr);

    // The text overlay pass, or null when it was not registered. A caller queues
    // shaped runs here for the frame; the queue is consumed by the frame that
    // draws it.
    [[nodiscard]] VulkanEngine::SceneRenderer::TextPass* GetTextPass() const {
        return text_pass_;
    }

    // The depth-tested world-space text pass, or null when it was not registered
    // (the same enable_text + registered-shader condition as the overlay). A
    // caller queues a Text component plus its Transform here for the frame; the
    // queue is consumed by the frame that draws it.
    [[nodiscard]] VulkanEngine::SceneRenderer::WorldTextPass* GetWorldTextPass() const {
        return world_text_pass_;
    }

    // Engine-standard set layouts (0-4) used to build custom-pass pipelines.
    void SetEngineDescriptorSetLayouts(std::array<vk::DescriptorSetLayout, 5> layouts);

    [[nodiscard]] VulkanEngine::RenderPipeline::RenderPipeline& GetRenderPipeline();
    [[nodiscard]] const VulkanEngine::RenderPipeline::RenderPipeline& GetRenderPipeline() const;

    void Shutdown();

    void RenderFrame(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                     VulkanEngine::ComponentRegistry& registry,
                     const VulkanEngine::Components::Camera& camera,
                     VulkanEngine::TechniqueManager::TechniqueManager& technique_mgr,
                     VulkanEngine::BindlessManager::BindlessManager& bindless_mgr,
                     VulkanEngine::SceneRenderer::SceneRenderer& scene_renderer,
                     VulkanEngine::ImGui::ImGuiSystem* imgui,
                     std::uint32_t image_index,
                     VulkanEngine::Textures::TextureUploader* texture_uploader = nullptr
#ifdef VKENGINE_PHYSICAL_CAMERA
                     , VulkanEngine::PhysicalCamera::PhysicalCameraSystem* physical_cameras = nullptr
#endif
                     );

private:
    VulkanBackend::Vulkan::VulkanBootstrap* bootstrap_ = nullptr;
    std::unique_ptr<VulkanEngine::RenderPipeline::RenderPipeline> pipeline_{};

    // Pass classes
    VulkanEngine::SceneRenderer::SceneRenderer* scene_renderer_ = nullptr;
    // Owned by the render pipeline once registered; non-owning here so the
    // instance-buffer resolver can find the pass for the frame being resolved.
    VulkanEngine::SceneRenderer::TextPass* text_pass_ = nullptr;
    VulkanEngine::SceneRenderer::WorldTextPass* world_text_pass_ = nullptr;

    std::uint32_t frame_counter_ = 0;
    std::uint32_t last_swapchain_image_count_ = 0;
    std::unique_ptr<vk::raii::Sampler> default_sampler_{};


    static constexpr vk::QueryPipelineStatisticFlags GPU_STATS_FLAGS =
        vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices |
        vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
        vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
        vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
        vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
        vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations |
        vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;

    // Compute-queue runs may only use a pool whose enabled statistics are not
    // graphics operations, so they get their own compute-only pool.
    static constexpr vk::QueryPipelineStatisticFlags GPU_STATS_COMPUTE_FLAGS =
        vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;
    // Graphics runs use the full-statistics pool; compute runs use a pool that
    // enables only the compute counter (a pool with graphics counters cannot be
    // used from a compute command pool).
    std::unique_ptr<vk::raii::QueryPool> gpu_stats_pool_{};
    std::unique_ptr<vk::raii::QueryPool> gpu_stats_compute_pool_{};

    // One query per queue run per frames-in-flight slot. The slot for a run is
    // (ring_index * stats_run_slots_per_frame_) + run_slot, so a frame's results
    // are read only when that ring is reused (its prior work is fence-complete).
    std::uint32_t stats_frames_in_flight_ = 0;
    std::uint32_t stats_run_slots_per_frame_ = 0;
    std::vector<std::uint32_t> stats_ring_run_counts_{};   // runs recorded last use
    std::vector<std::uint64_t> stats_ring_frame_ids_{};    // frame id those runs belong to
    std::vector<std::uint8_t> stats_ring_run_kinds_{};     // 1 = compute, 0 = graphics
};

} // namespace VulkanEngine::Renderer
