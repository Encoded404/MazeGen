module;
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp> //NOLINT(misc-include-cleaner)

#include <logging/logging_macros.hpp>

module VulkanEngine.Renderer;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.RenderGraph;
import VulkanEngine.RenderPipeline;
import VulkanEngine.SceneRenderer;
import VulkanEngine.TechniqueManager;
import VulkanEngine.BindlessManager;
import VulkanEngine.Components.Camera;
import VulkanEngine.GpuResources;
import VulkanEngine.GpuStats;
import VulkanEngine.ImGui;

namespace VulkanEngine::Renderer {

Renderer::~Renderer() {
    Shutdown();
}

void Renderer::SetEngineDescriptorSetLayouts(std::array<vk::DescriptorSetLayout, 5> layouts) {
    if (pipeline_) {
        pipeline_->SetEngineDescriptorSetLayouts(layouts);
    }
}

VulkanEngine::RenderPipeline::RenderPipeline& Renderer::GetRenderPipeline() {
    return *pipeline_;
}

const VulkanEngine::RenderPipeline::RenderPipeline& Renderer::GetRenderPipeline() const {
    return *pipeline_;
}

bool Renderer::Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                  const RendererConfig& config,
                                  VulkanEngine::SceneRenderer::SceneRenderer& scene_renderer,
                                  ShaderSystem::ShaderManager* shader_manager,
                                  ShaderSystem::PipelineFactory* pipeline_factory,
                                  const VulkanEngine::EngineShaderIds* shader_ids) {
    bootstrap_ = &bootstrap;
    scene_renderer_ = &scene_renderer;

    pipeline_ = std::make_unique<VulkanEngine::RenderPipeline::RenderPipeline>();
    pipeline_->Initialize(bootstrap, shader_manager, pipeline_factory);

    // Import the engine-owned resources the built-in passes declare against.
    // Logical buffers ("scene-buffers", etc.) are hazard-only: they carry no
    // Vulkan handle, so the graph uses them purely for ordering.
    (void)pipeline_->ImportBuffer("scene-buffers");
    (void)pipeline_->ImportBuffer("draw-indirect");
    (void)pipeline_->ImportBuffer("depth-indirect");
    (void)pipeline_->ImportBuffer("occluder-indirect");
    // The text overlay's per-frame glyph instance buffer. It lives in the pass
    // (one host-visible slot per frames-in-flight) and is resolved back into the
    // graph through the buffer resolver registered below.
    (void)pipeline_->ImportBuffer("text-instances");
    auto backbuffer = pipeline_->ImportBackbuffer();
    (void)pipeline_->ImportDepthBuffer();

    // "hiz-image" is the resolution-dependent Hi-Z ring owned by the scene
    // renderer; its slot follows the frame counter.
    pipeline_->RegisterResourceResolver("hiz-image",
        [this](std::uint32_t) { return scene_renderer_ ? scene_renderer_->GetHizImage(frame_counter_) : nullptr; },
        [this](std::uint32_t) { return scene_renderer_ ? scene_renderer_->GetHizFullView(frame_counter_) : nullptr; },
        vk::Format::eR32Sfloat);

    pipeline_->SetFinalState(
        backbuffer,
        VulkanEngine::RenderGraph::ResourceState::ImageState(
            VulkanEngine::RenderGraph::PipelineStageIntent::BottomOfPipe,
            VulkanEngine::RenderGraph::AccessIntent::None,
            VulkanEngine::RenderGraph::QueueType::Graphics,
            VulkanEngine::RenderGraph::ImageLayoutIntent::Present));

    // Capture the render extent for RegisterPass()'s PassSetupContext.
    std::uint32_t init_width = 0;
    std::uint32_t init_height = 0;
    (void)bootstrap.GetBackend().GetSwapchainExtent(init_width, init_height);
    pipeline_->SetRenderExtent(init_width, init_height);
    pipeline_->SetFramesInFlight(bootstrap.GetSnapshot().frames_in_flight);
    pipeline_->SetQueueFamilies(bootstrap.GetBackend().GetQueueFamilies());
    pipeline_->SetAsyncComputeAvailable(bootstrap.GetBackend().HasAsyncCompute());
    LOGIFACE_LOG(info, "Renderer: async compute " +
                           std::string(bootstrap.GetBackend().HasAsyncCompute() ? "available" : "unavailable") +
                           ", queue families=" +
                           std::to_string(bootstrap.GetBackend().GetQueueFamilies().size()));

    // Register every built-in through the same engine-managed path as app
    // passes: the pipeline constructs the PassSetupContext and calls Setup().
    bool builtins_ok = true;
    const auto register_builtin = [this, &builtins_ok](
        std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass> pass)
        -> VulkanEngine::RenderGraph::PassHandle {
        auto result = pipeline_->RegisterPass(std::move(pass));
        if (!result.has_value()) {
            LOGIFACE_LOG(error, "Renderer: built-in pass registration failed: " +
                                    result.error().message);
            builtins_ok = false;
            return {};
        }
        return *result;
    };

    const auto expand_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::ExpandPass>(scene_renderer));
    const auto occluder_select_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::OccluderSelectPass>(scene_renderer));
    const auto occluder_prepass_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::OccluderPrePass>(
            scene_renderer, config.clear_depth_stencil));
    const auto hiz_pre_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::HiZPass>(scene_renderer, "hiz-gen-pre"));
    const auto pre_cull_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::PreCullPass>(scene_renderer));
    const auto depth_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::DepthPrePass>(scene_renderer));
    const auto hiz_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::HiZPass>(scene_renderer, "hiz-gen"));
    const auto occlusion_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::OcclusionPass>(scene_renderer));
    const auto collect_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::CollectPass>(scene_renderer));
    const auto main_handle = register_builtin(
        std::make_unique<VulkanEngine::SceneRenderer::MainPass>(scene_renderer, config.clear_color));

    VulkanEngine::RenderGraph::PassHandle imgui_handle{};
    if (config.enable_imgui) {
        imgui_handle = register_builtin(
            std::make_unique<VulkanEngine::SceneRenderer::ImGuiPass>(&bootstrap));
    }

    // Screen-space text sits above the scene and below the ImGui overlay, so an
    // app can order an effect between the two. It needs the registered engine
    // shader handles; without them there is no pipeline to build.
    VulkanEngine::RenderGraph::PassHandle text_handle{};
    VulkanEngine::RenderGraph::PassHandle world_text_handle{};
    if (config.enable_text && shader_ids != nullptr) {
        auto text_pass = std::make_unique<VulkanEngine::SceneRenderer::TextPass>(
            &bootstrap.GetBackend(), shader_ids->ui_text_vert, shader_ids->ui_text_frag);
        text_pass_ = text_pass.get();
        // The resolver reads the same ring slot Execute writes, keyed by the
        // renderer's frame counter (the frame context's frame_index).
        pipeline_->RegisterBufferResolver(
            "text-instances", [this](std::uint32_t) -> vk::Buffer {
                return text_pass_ != nullptr ? text_pass_->InstanceBufferForFrame(frame_counter_)
                                             : vk::Buffer{};
            });
        text_handle = register_builtin(std::move(text_pass));

        // World text is depth-tested scene content: it draws after the main
        // pass, so scene geometry occludes it, and before the screen-space
        // overlay, which must never be hidden by scene depth. The configured
        // backend picks which shader pair (and instance layout) the pass owns.
        const bool slug_text =
            config.text_backend == VulkanEngine::SceneRenderer::TextBackend::Slug;
        auto world_text_pass = std::make_unique<VulkanEngine::SceneRenderer::WorldTextPass>(
            &bootstrap.GetBackend(), config.text_backend,
            slug_text ? shader_ids->slug_text_vert : shader_ids->world_text_vert,
            slug_text ? shader_ids->slug_text_frag : shader_ids->world_text_frag);
        world_text_pass_ = world_text_pass.get();
        pipeline_->RegisterBufferResolver(
            "world-text-instances", [this](std::uint32_t) -> vk::Buffer {
                return world_text_pass_ != nullptr
                           ? world_text_pass_->InstanceBufferForFrame(frame_counter_)
                           : vk::Buffer{};
            });
        if (slug_text) {
            // The Slug fragment reads the pass-owned blob storage buffer; the
            // pass grows it (retiring the old buffer frame-gated), so the
            // resolver returns whatever buffer is current for this frame.
            pipeline_->RegisterBufferResolver(
                "world-text-blobs", [this](std::uint32_t) -> vk::Buffer {
                    auto* blobs = world_text_pass_ != nullptr ? world_text_pass_->BlobBuffer()
                                                              : nullptr;
                    return blobs != nullptr ? blobs->Buffer() : vk::Buffer{};
                });
        }
        world_text_handle = register_builtin(std::move(world_text_pass));
    }

    // Explicit ordering ensures correct pipeline:
    // expand → occluder-select → occluder-prepass → hiz-gen-pre → pre-cull →
    // depth-prepass → hiz-gen (full) → occlusion → collect → main →
    // world text (depth-tested) → screen-space text → ImGui.
    pipeline_->AddDependency(expand_handle, occluder_select_handle);
    pipeline_->AddDependency(occluder_select_handle, occluder_prepass_handle);
    pipeline_->AddDependency(occluder_prepass_handle, hiz_pre_handle);
    pipeline_->AddDependency(hiz_pre_handle, pre_cull_handle);
    pipeline_->AddDependency(pre_cull_handle, depth_handle);
    pipeline_->AddDependency(depth_handle, hiz_handle);
    pipeline_->AddDependency(hiz_handle, occlusion_handle);
    pipeline_->AddDependency(occlusion_handle, collect_handle);
    pipeline_->AddDependency(collect_handle, main_handle);
    if (world_text_handle.IsValid()) {
        // main pass -> depth-tested world text -> screen-space overlay -> ImGui.
        pipeline_->AddDependency(main_handle, world_text_handle);
        if (text_handle.IsValid()) {
            pipeline_->AddDependency(world_text_handle, text_handle);
        } else if (imgui_handle.IsValid()) {
            pipeline_->AddDependency(world_text_handle, imgui_handle);
        }
    }
    if (text_handle.IsValid()) {
        if (!world_text_handle.IsValid()) {
            pipeline_->AddDependency(main_handle, text_handle);
        }
        if (imgui_handle.IsValid()) {
            pipeline_->AddDependency(text_handle, imgui_handle);
        }
    }

    // Expose every built-in as an ordering anchor before app passes apply.
    pipeline_->SetBuiltinHandles(std::array<
        VulkanEngine::RenderGraph::PassHandle,
        VulkanEngine::PipelinePass::kBuiltinPassCount>{
        expand_handle,
        occluder_select_handle,
        occluder_prepass_handle,
        hiz_pre_handle,
        pre_cull_handle,
        depth_handle,
        hiz_handle,
        occlusion_handle,
        collect_handle,
        main_handle,
        world_text_handle,
        text_handle,
        imgui_handle,
    });

    pipeline_->Compile();
    if (!builtins_ok || !pipeline_->IsCompiled()) return false;

    {
        auto& device = bootstrap.GetBackend().GetDevice();
        stats_frames_in_flight_ =
            std::max<std::uint32_t>(bootstrap.GetBackend().GetFramesInFlight(), 1);
        stats_run_slots_per_frame_ =
            std::max<std::uint32_t>(bootstrap.GetBackend().GetRunSlotsPerFrame(), 1);
        const std::uint32_t query_count =
            VulkanEngine::GpuStats::QueryPoolSize(stats_frames_in_flight_, stats_run_slots_per_frame_);
        vk::QueryPoolCreateInfo qp_info{};
        qp_info.queryType = vk::QueryType::ePipelineStatistics;
        qp_info.pipelineStatistics = GPU_STATS_FLAGS;
        qp_info.queryCount = query_count;
        gpu_stats_pool_ = std::make_unique<vk::raii::QueryPool>(device, qp_info);
        VulkanBackend::Vulkan::SetVulkanObjectName(device, *gpu_stats_pool_, "gpu-stats-pool");

        vk::QueryPoolCreateInfo compute_qp_info{};
        compute_qp_info.queryType = vk::QueryType::ePipelineStatistics;
        compute_qp_info.pipelineStatistics = GPU_STATS_COMPUTE_FLAGS;
        compute_qp_info.queryCount = query_count;
        gpu_stats_compute_pool_ = std::make_unique<vk::raii::QueryPool>(device, compute_qp_info);
        VulkanBackend::Vulkan::SetVulkanObjectName(device, *gpu_stats_compute_pool_,
                                                   "gpu-stats-compute-pool");

        const vk::Device raw_device = *device;
        raw_device.resetQueryPool(*gpu_stats_pool_, 0, query_count);
        raw_device.resetQueryPool(*gpu_stats_compute_pool_, 0, query_count);
        stats_ring_run_counts_.assign(stats_frames_in_flight_, 0);
        stats_ring_frame_ids_.assign(stats_frames_in_flight_, 0);
        stats_ring_run_kinds_.assign(
            static_cast<std::size_t>(stats_frames_in_flight_) * stats_run_slots_per_frame_, 0);
    }

    // Engine-owned sampler for app-pass sampled/combined-image bindings.
    {
        auto& device = bootstrap.GetBackend().GetDevice();
        vk::SamplerCreateInfo sampler_info{};
        sampler_info.magFilter = vk::Filter::eLinear;
        sampler_info.minFilter = vk::Filter::eLinear;
        sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.maxLod = vk::LodClampNone;
        default_sampler_ = std::make_unique<vk::raii::Sampler>(device, sampler_info);
    }

    LOGIFACE_LOG(info, "Renderer initialized with full render-graph pipeline");
    return true;
}

void Renderer::Shutdown() {
    gpu_stats_pool_.reset();
    gpu_stats_compute_pool_.reset();
    stats_ring_run_counts_.clear();
    stats_ring_frame_ids_.clear();
    stats_ring_run_kinds_.clear();
    default_sampler_.reset();
    if (pipeline_) {
        pipeline_->Shutdown();
        pipeline_.reset();
    }
    // The pipeline owned the pass; the non-owning resolver pointer must not
    // outlive it.
    text_pass_ = nullptr;
    world_text_pass_ = nullptr;
    bootstrap_ = nullptr;
}

void Renderer::RenderFrame(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                   VulkanEngine::ComponentRegistry& registry,
                                   const VulkanEngine::Components::Camera& camera,
                                   VulkanEngine::TechniqueManager::TechniqueManager& technique_mgr,
                                   VulkanEngine::BindlessManager::BindlessManager& bindless_mgr,
                                   VulkanEngine::SceneRenderer::SceneRenderer& scene_renderer,
                                   VulkanEngine::ImGui::ImGuiSystem* imgui,
                                   std::uint32_t image_index,
                                   VulkanEngine::Textures::TextureUploader* texture_uploader
#ifdef VKENGINE_PHYSICAL_CAMERA
                                   , VulkanEngine::PhysicalCamera::PhysicalCameraSystem* physical_cameras
#endif
                                   ) {
    if (!pipeline_) return;

    // Read the extent first so a resize is queued before ApplyChanges drains it
    // (and so relative transients are sized against the current extent on the
    // very first compile).
    std::uint32_t width = 0, height = 0;
    (void)bootstrap.GetBackend().GetSwapchainExtent(width, height);
    pipeline_->SetRenderExtent(width, height);

    // Rebuild the graph at a frame boundary if passes were added/removed/enabled
    // since the last frame. Removals are freed once their last frame completes.
    pipeline_->ApplyChanges();
    if (!pipeline_->IsCompiled()) return;

    LOGIFACE_LOG(trace, "RenderFrame frame=" + std::to_string(frame_counter_) +
                 " img=" + std::to_string(image_index) + " w=" + std::to_string(width) +
                 " h=" + std::to_string(height));

    // The swapchain extent may have changed (window resize). Re-create the
    // resolution-dependent Hi-Z ring at the new size before this frame binds its
    // descriptors or dispatches any pass; it is a no-op when unchanged.
    scene_renderer.EnsureRenderExtent(width, height);

    if (imgui && imgui->IsInitialized()) {
        imgui->NewFrame();
    }

    auto& backend = bootstrap.GetBackend();
    const std::uint32_t sc_count = bootstrap.GetSnapshot().swapchain_image_count;
    if (sc_count != last_swapchain_image_count_) {
        last_swapchain_image_count_ = sc_count;
        // Swapchain images (and depth views) were recreated: drop recorded
        // imported layouts so the next use starts from Undefined, and re-resolve
        // every view. Idempotent when only the count is first observed.
        pipeline_->OnSwapchainRecreated(sc_count);
        if (imgui && imgui->IsInitialized()) {
            imgui->OnSwapchainRecreated(sc_count,
                static_cast<vk::Format>(bootstrap.GetBackend().GetSurfaceFormat().format));
        }
    }

    const std::uint32_t frame_idx = bootstrap.GetSnapshot().frame_index;
    const std::uint32_t frames_in_flight = std::max<std::uint32_t>(backend.GetFramesInFlight(), 1);
    // Bootstrap's frame_index is the authoritative counter; the ring index used
    // by every per-FIF resource is its remainder. The device indexes command
    // buffers the same way, so this must match.
    const std::uint32_t ring_index = frame_idx % frames_in_flight;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const glm::mat4 view = camera.GetViewMatrix();
    const glm::mat4 proj = camera.GetProjectionMatrix(aspect);
    const glm::mat4 view_proj = proj * view;
    const auto& depth_view = backend.GetDepthImageView(image_index);

    // Fully populated per-frame context shared by built-in and app passes.
    VulkanEngine::PipelinePass::FrameContext frame{};
    frame.render_extent = vk::Extent2D{width, height};
    frame.frame_index = frame_counter_;
    frame.ring_index = ring_index;
    frame.swapchain_image_index = image_index;
    frame.view = view;
    frame.proj = proj;
    frame.view_proj = view_proj;
    frame.bindless_textures = { bindless_mgr.GetDescriptorSet() };
    frame.submesh_vertices = { scene_renderer.GetFrameSubmeshVertexEntriesSet(frame_counter_) };
    frame.raw_vertex_buffers = { scene_renderer.GetFrameVertexBuffersSet(frame_counter_) };
    frame.indirection_data = { scene_renderer.GetFrameIndirectionSet(frame_counter_) };
    frame.scene_uniforms = { scene_renderer.GetSceneUniformSet() };
    frame.depth_pyramid = { scene_renderer.GetHizImage(frame_counter_),
                            scene_renderer.GetHizFullView(frame_counter_) };
    frame.depth_buffer = { *depth_view };
    frame.techniques = &technique_mgr;
    frame.bindless = &bindless_mgr;
    frame.registry = &registry;
    frame.imgui = imgui;
    frame.default_sampler = default_sampler_ ? static_cast<vk::Sampler>(**default_sampler_) : nullptr;
    frame.entity_count = scene_renderer.GetCurrentEntityCount();
    frame.render_width = width;
    frame.render_height = height;

    VulkanEngine::PipelinePass::RenderFrameData frame_data{};
    frame_data.frame = frame;

    // Resolve resources and build this frame's barrier/queue-run plan.
    pipeline_->BeginFrame(&frame_data, image_index, ring_index);
    const auto& runs = pipeline_->GetQueueRuns();
    const bool single_graphics_run =
        runs.runs.size() <= 1 &&
        (runs.runs.empty() || runs.runs[0].queue == VulkanEngine::RenderGraph::QueueType::Graphics);

    // GPU gather/upload/descriptor work is recorded once, ahead of the graph.
    const auto record_prep = [&](vk::CommandBuffer target) {
        // Bind actual depth to Hi-Z descriptor before hiz-gen pass executes
        scene_renderer.UpdateHizDepthBinding(frame_counter_, *depth_view);

        // Initialize Hi-Z on first frame
        scene_renderer.InitializeHizFirstFrame(target);

        // Upload technique PipelineFlags and enable/disable the depth filter
        // pass before PrepareCompute retargets the depth indirection set.
        scene_renderer.UpdateTechniqueFlags(technique_mgr);

        // Record this frame's staged async texture uploads (no wait, no frame-0
        // command buffer): per-subresource copies plus the bindless commit,
        // which publishes one FIF cycle later.
        if (texture_uploader != nullptr) {
            (void)texture_uploader->RecordUploads(
                target, frame_counter_, VulkanEngine::Textures::TextureUploader::kRecordBudget());
        }

        // CPU gather + upload + descriptor writes for all passes
        scene_renderer.PrepareCompute(target, registry, view, proj, width, height, frame_counter_);

#ifdef VKENGINE_PHYSICAL_CAMERA
        // PhysicalCamera uploads + compositing run before the scene graph so
        // scene passes sampling a camera target see this frame's content.
        if (physical_cameras != nullptr) {
            physical_cameras->Execute(target, frame_counter_);
        }
#endif
    };

    // ── Per-run GPU statistics ──
    //
    // One pipeline-statistics query per queue run, in the ring slot for this
    // frames-in-flight. A query cannot span queues, so measuring the whole frame
    // requires one query per run; the previous use of this ring is read once its
    // fence-complete work is replaced.
    const auto run_query_slot = [&](std::uint32_t run_slot) {
        return VulkanEngine::GpuStats::QuerySlot(ring_index, run_slot, stats_frames_in_flight_,
                                                 stats_run_slots_per_frame_);
    };
    const auto read_previous_run_stats = [&]() {
        if (!gpu_stats_pool_ || ring_index >= stats_ring_run_counts_.size() ||
            stats_ring_run_counts_[ring_index] == 0) {
            return;
        }
        const std::uint32_t first_slot = ring_index * stats_run_slots_per_frame_;
        const std::uint32_t run_count = stats_ring_run_counts_[ring_index];
        const std::uint64_t recorded_frame = stats_ring_frame_ids_[ring_index];

        std::vector<VulkanEngine::GpuStats::Counters> per_run;
        per_run.reserve(run_count);

        // Read each used run individually from its queue's pool. Reading a whole
        // range at once would return VK_NOT_READY as soon as it covered a query
        // that was never begun in that pool (a run of the other queue).
        const auto& device = backend.GetDevice();
        auto* dev_dispatcher = device.getDispatcher();
        for (std::uint32_t run = 0; run < run_count; ++run) {
            const bool compute = stats_ring_run_kinds_[first_slot + run] != 0;
            const vk::raii::QueryPool* pool =
                compute ? gpu_stats_compute_pool_.get() : gpu_stats_pool_.get();
            if (pool == nullptr) {
                continue;
            }
            const std::uint32_t words_per_query =
                compute ? VulkanEngine::GpuStats::kComputeQueryWords
                        : VulkanEngine::GpuStats::kQueryWords;
            std::array<std::uint64_t, VulkanEngine::GpuStats::kQueryWords> words{};
            const vk::Result qr = static_cast<vk::Result>(dev_dispatcher->vkGetQueryPoolResults(
                static_cast<vk::Device::CType>(*device),
                static_cast<vk::QueryPool::CType>(**pool),
                first_slot + run, 1,
                static_cast<vk::DeviceSize>(words_per_query * sizeof(std::uint64_t)), words.data(),
                static_cast<vk::DeviceSize>(words_per_query * sizeof(std::uint64_t)),
                static_cast<vk::QueryResultFlags::MaskType>(vk::QueryResultFlagBits::e64 |
                                                            vk::QueryResultFlagBits::eWithAvailability)));
            if (qr != vk::Result::eSuccess) {
                continue;
            }
            VulkanEngine::GpuStats::Counters counters{};
            const bool parsed =
                compute ? VulkanEngine::GpuStats::TryParseComputeQueryResult(words.data(), counters)
                        : VulkanEngine::GpuStats::TryParseQueryResult(words.data(), counters);
            if (parsed) {
                per_run.push_back(counters);
            }
        }

        if (!per_run.empty()) {
            const auto total = VulkanEngine::GpuStats::Accumulate(per_run);
            LOGIFACE_LOG(trace,
                "GPU frame=" + std::to_string(recorded_frame) +
                " runs=" + std::to_string(per_run.size()) + "/" + std::to_string(run_count) +
                " IA_verts=" + std::to_string(total.input_assembly_vertices) +
                " IA_prims=" + std::to_string(total.input_assembly_primitives) +
                " VS_invoc=" + std::to_string(total.vertex_shader_invocations) +
                " clip_invoc=" + std::to_string(total.clipping_invocations) +
                " clip_prims=" + std::to_string(total.clipping_primitives) +
                " FS_invoc=" + std::to_string(total.fragment_shader_invocations) +
                " CS_invoc=" + std::to_string(total.compute_shader_invocations));
        }

        // This ring's prior submission is fence-complete before RenderFrame runs,
        // so its queries can be reset host-side for reuse.
        const vk::Device raw_device = *backend.GetDevice();
        raw_device.resetQueryPool(**gpu_stats_pool_, first_slot, stats_run_slots_per_frame_);
        if (gpu_stats_compute_pool_) {
            raw_device.resetQueryPool(**gpu_stats_compute_pool_, first_slot,
                                      stats_run_slots_per_frame_);
        }
        stats_ring_run_counts_[ring_index] = 0;
    };

    const auto begin_run_stats = [&](vk::CommandBuffer target, std::uint32_t run_slot, bool compute) {
        const vk::raii::QueryPool* pool =
            compute ? gpu_stats_compute_pool_.get() : gpu_stats_pool_.get();
        if (pool != nullptr) {
            target.beginQuery(**pool, run_query_slot(run_slot), {});
        }
        if (gpu_stats_pool_ && ring_index < stats_ring_run_counts_.size()) {
            stats_ring_run_kinds_[run_query_slot(run_slot)] = compute ? 1 : 0;
        }
    };
    const auto end_run_stats = [&](vk::CommandBuffer target, std::uint32_t run_slot, bool compute) {
        const vk::raii::QueryPool* pool =
            compute ? gpu_stats_compute_pool_.get() : gpu_stats_pool_.get();
        if (pool != nullptr) {
            target.endQuery(**pool, run_query_slot(run_slot));
        }
    };
    const auto note_run_slot = [&](std::uint32_t run_slot) {
        if (gpu_stats_pool_ && ring_index < stats_ring_run_counts_.size() &&
            run_slot + 1 > stats_ring_run_counts_[ring_index]) {
            stats_ring_run_counts_[ring_index] = run_slot + 1;
            stats_ring_frame_ids_[ring_index] = frame_counter_;
        }
    };

    read_previous_run_stats();

    if (single_graphics_run) {
        // Existing single-queue fast path: one command buffer, no run list.
        auto& cmd = backend.GetCommandBuffer(frame_idx);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        begin_run_stats(cmd, 0, /*compute=*/false);
        record_prep(cmd);
        pipeline_->RecordRun(0, cmd);
        end_run_stats(cmd, 0, /*compute=*/false);
        note_run_slot(0);
        cmd.end();
    } else {
        // Multi-queue path: one command buffer per queue run, submitted in order.
        LOGIFACE_LOG(debug, "Renderer: multi-queue frame with " +
                                std::to_string(runs.runs.size()) + " runs");
        std::vector<VulkanBackend::Vulkan::IVulkanBootstrap::QueueRunSubmit> submits;
        submits.reserve(runs.runs.size() + 1);

        // The engine's scene prep (uploads, descriptor writes, Hi-Z init,
        // physical-camera compositing) is graphics work. When the graph's first
        // run is on the graphics queue it is folded into that run; otherwise it
        // is submitted as a dedicated graphics preamble run before the graph.
        std::uint32_t slot_base = 0;
        if (!runs.StartsWithGraphics()) {
            slot_base = 1;
            auto& preamble_cmd = backend.GetRunCommandBuffer(false, frame_idx, 0);
            preamble_cmd.reset({});
            preamble_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            begin_run_stats(preamble_cmd, 0, /*compute=*/false);
            record_prep(preamble_cmd);
            end_run_stats(preamble_cmd, 0, /*compute=*/false);
            note_run_slot(0);
            preamble_cmd.end();
            submits.push_back({.compute = false, .command_buffer = *preamble_cmd});
        }

        for (std::uint32_t i = 0; i < runs.runs.size(); ++i) {
            const bool compute = runs.runs[i].queue != VulkanEngine::RenderGraph::QueueType::Graphics;
            const std::uint32_t run_slot = i + slot_base;
            auto& run_cmd = backend.GetRunCommandBuffer(compute, frame_idx, run_slot);
            run_cmd.reset({});
            run_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            begin_run_stats(run_cmd, run_slot, compute);
            if (i == 0 && slot_base == 0) {
                // Run 0 is graphics: fold prep into it.
                record_prep(run_cmd);
            }
            pipeline_->RecordRun(i, run_cmd);
            end_run_stats(run_cmd, run_slot, compute);
            note_run_slot(run_slot);
            run_cmd.end();
            submits.push_back({.compute = compute, .command_buffer = *run_cmd});
        }
        backend.SetFrameRuns(submits);
    }
    pipeline_->EndFrame();

    frame_counter_++;
}

} // namespace VulkanEngine::Renderer
