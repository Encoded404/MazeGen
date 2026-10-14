module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

#include <logging/logging_macros.hpp>

module VulkanEngine.SceneRenderer;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.PipelineCompiler;
import VulkanEngine.PipelineFactory;
import VulkanEngine.PipelineTemplates;

namespace VulkanEngine::SceneRenderer {
    namespace {
        struct ExpandPC { glm::mat4 vp; std::uint32_t cnt; std::uint32_t p0; std::uint32_t p1; };
        // projInfo = (|proj[0][0]|, |proj[1][1]|, proj[2][2], 1 if perspective else 0)
        struct OccPC { std::uint32_t cnt; std::uint32_t refineLevel; std::uint32_t hizWidth; std::uint32_t hizHeight; glm::vec4 projInfo; };
        struct HiZPC { std::uint32_t bl; std::uint32_t sw; std::uint32_t sh; std::uint32_t tc; };
        struct CollectPC { std::uint32_t cnt; std::uint32_t p0; std::uint32_t mt; std::uint32_t pass; };
        struct WritePC { std::uint32_t cnt; std::uint32_t p0; std::uint32_t techniqueCount; std::uint32_t p1; };
        struct OccluderSelectPC { std::uint32_t cnt; std::uint32_t minAreaPx; std::uint32_t screenW; std::uint32_t screenH; };
        struct PreCullPC { std::uint32_t cnt; std::uint32_t hizW; std::uint32_t hizH; std::uint32_t p0; glm::vec4 projInfo; };

        // One pipeline-layout creation shape for every scene pipeline: set
        // layouts (borrowed views into the owning raii objects) plus an
        // optional compute push-constant range. The layouts stay named members
        // owned by the frame code that binds them; only creation collapses.
        std::unique_ptr<vk::raii::PipelineLayout> MakePipelineLayout(
                const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                std::string_view layout_name,
                std::span<const vk::DescriptorSetLayout> set_layouts,
                std::uint32_t push_constant_size = 0) {
            const auto& dev = be.GetDevice();
            vk::PipelineLayoutCreateInfo li{};
            li.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
            li.pSetLayouts = set_layouts.data();
            vk::PushConstantRange pr{};
            if (push_constant_size > 0) {
                pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
                pr.size = push_constant_size;
                li.pushConstantRangeCount = 1;
                li.pPushConstantRanges = &pr;
            }
            auto layout = std::make_unique<vk::raii::PipelineLayout>(dev, li);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, *layout, std::string(layout_name));
            return layout;
        }
    } // anonymous namespace

bool SceneRenderer::CreateScenePipelines(VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                          ShaderSystem::ShaderManager& shader_mgr,
                                          ShaderSystem::PipelineFactory& pipeline_factory,
                                          const EngineShaderIds& shader_ids) {
    LOGIFACE_LOG(debug, "Creating scene pipelines...");
    const auto& dev = be.GetDevice();

    // ── Hi-Z set layout + pool (bindings sized by MAX_HIZ_MIPS) ──
    {
        std::array<vk::DescriptorSetLayoutBinding, 3> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eSampledImage;
        bs[0].descriptorCount = 1;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        bs[1].binding = 1;
        bs[1].descriptorType = vk::DescriptorType::eSampler;
        bs[1].descriptorCount = 1;
        bs[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
        bs[2].binding = 2;
        bs[2].descriptorType = vk::DescriptorType::eStorageImage;
        bs[2].descriptorCount = MAX_HIZ_MIPS;
        bs[2].stageFlags = vk::ShaderStageFlagBits::eCompute;
        hiz_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{{}, static_cast<std::uint32_t>(bs.size()), bs.data()});
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_images = frames_in_flight_ * MAX_HIZ_MIPS;
        pc.max_sampled_images = frames_in_flight_;
        pc.max_samplers = frames_in_flight_;
        hiz_pool_ = GpuResources::DescriptorPool::Create(be, pc);
    }

    // ── Pipeline layouts ──
    {
        const std::array<vk::DescriptorSetLayout, 2> expand_sets{
            **expand_layout_, **index_buffers_layout_};
        expand_pipeline_layout_ =
            MakePipelineLayout(be, "expand-pipeline-layout", expand_sets, sizeof(ExpandPC));

        const std::array<vk::DescriptorSetLayout, 1> occlusion_sets{**occlusion_layout_};
        occlusion_pipeline_layout_ =
            MakePipelineLayout(be, "occlusion-pipeline-layout", occlusion_sets, sizeof(OccPC));

        const std::array<vk::DescriptorSetLayout, 1> occluder_select_sets{**occluder_select_layout_};
        occluder_select_pipeline_layout_ =
            MakePipelineLayout(be, "occluder-select-pipeline-layout", occluder_select_sets, sizeof(OccluderSelectPC));

        // Pre-cull shares the occlusion set (+ survivor compaction bindings 6-8).
        pre_cull_pipeline_layout_ =
            MakePipelineLayout(be, "pre-cull-pipeline-layout", occlusion_sets, sizeof(PreCullPC));

        const std::array<vk::DescriptorSetLayout, 1> collect_counts{**collect_layout_};
        collect_pipeline_layout_ =
            MakePipelineLayout(be, "collect-count-pipeline-layout", collect_counts, sizeof(CollectPC));

        const std::array<vk::DescriptorSetLayout, 1> collect_writes{**collect_write_layout_};
        collect_write_pipeline_layout_ =
            MakePipelineLayout(be, "collect-write-pipeline-layout", collect_writes, sizeof(WritePC));

        const std::array<vk::DescriptorSetLayout, 1> hiz_sets{**hiz_layout_};
        hiz_pipeline_layout_ =
            MakePipelineLayout(be, "hiz-gen-pipeline-layout", hiz_sets, sizeof(HiZPC));

        const std::array<vk::DescriptorSetLayout, 4> depth_sets{
            **empty_layout_, **submesh_vertex_layout_, **vertex_buffers_layout_, **indirection_layout_};
        depth_pipeline_layout_ = MakePipelineLayout(be, "depth-prepass-pipeline-layout", depth_sets);
    }

    // ── Compute pipeline table ──
    // One row per built-in compute pass. Creation and the draw-mode rebuild
    // (RebuildCompactionPipelines) cover the same set via this table shape.
    struct ComputeRow {
        std::string_view pipeline_name;
        ShaderSystem::ShaderId shader;
        vk::PipelineLayout layout;
        bool draw_mode_specialized;
        ShaderSystem::PipelineSlot& slot;
        std::optional<ShaderSystem::ComputePipelineDesc>& desc_out;
    };
    std::vector<ComputeRow> rows;
    rows.reserve(8);
    rows.push_back({
        .pipeline_name = "expand", .shader = shader_ids.expand_comp,
        .layout = **expand_pipeline_layout_, .draw_mode_specialized = true,
        .slot = expand_slot_, .desc_out = expand_desc_});
    rows.push_back({
        .pipeline_name = "occlusion", .shader = shader_ids.occlusion_cull_comp,
        .layout = **occlusion_pipeline_layout_, .draw_mode_specialized = false,
        .slot = occlusion_slot_, .desc_out = occlusion_desc_});
    rows.push_back({
        .pipeline_name = "occluder-select", .shader = shader_ids.occluder_select_comp,
        .layout = **occluder_select_pipeline_layout_, .draw_mode_specialized = true,
        .slot = occluder_select_slot_, .desc_out = occluder_select_desc_});
    rows.push_back({
        .pipeline_name = "pre-cull", .shader = shader_ids.pre_cull_comp,
        .layout = **pre_cull_pipeline_layout_, .draw_mode_specialized = true,
        .slot = pre_cull_slot_, .desc_out = pre_cull_desc_});
    rows.push_back({
        .pipeline_name = "collect-count", .shader = shader_ids.collect_count_compact_comp,
        .layout = **collect_pipeline_layout_, .draw_mode_specialized = true,
        .slot = collect_count_slot_, .desc_out = collect_count_desc_});
    rows.push_back({
        .pipeline_name = "collect-write", .shader = shader_ids.collect_write_comp,
        .layout = **collect_write_pipeline_layout_, .draw_mode_specialized = false,
        .slot = collect_write_slot_, .desc_out = collect_write_desc_});
    rows.push_back({
        .pipeline_name = "hiz-gen", .shader = shader_ids.hiz_gen_comp,
        .layout = **hiz_pipeline_layout_, .draw_mode_specialized = false,
        .slot = hiz_slot_, .desc_out = hiz_desc_});

    bool ok = true;

    // ── Depth-prepass graphics pipeline desc ──
    // Depth-only pass: no color attachments, so the blend state must not
    // declare any (the empty blend_attachments template yields that). The
    // vertex stage is draw-mode-specialized like the compute cullers.
    ShaderSystem::GraphicsPipelineTemplate depth_tmpl{
        .debug_name = "depth-prepass",
        .input_assembly = vk::PipelineInputAssemblyStateCreateInfo(
            {}, vk::PrimitiveTopology::eTriangleList),
        .rasterization = vk::PipelineRasterizationStateCreateInfo(
            {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eFront,
            vk::FrontFace::eClockwise, false, 0, 0, 0, 1.0f),
        .multisample = vk::PipelineMultisampleStateCreateInfo(
            {}, vk::SampleCountFlagBits::e1),
        .depth_stencil = vk::PipelineDepthStencilStateCreateInfo(
            {}, true, true, vk::CompareOp::eLess),
    };
    auto depth_desc = ShaderSystem::MakeGraphicsDesc(
        depth_tmpl, shader_ids.depth_indir_vert, shader_ids.depth_prepass_frag,
        **depth_pipeline_layout_, {}, be.GetDepthFormat());
    ApplyDrawModeSpec(depth_desc, ShaderSystem::DrawModeSpecValue(draw_mode_));
    depth_desc_ = depth_desc;

    // ── Compile the whole window in one submission ──
    // One batched driver call for all seven compute pipelines plus the depth
    // pipeline (per-path batching inside the factory: compute is always one
    // call; graphics batching applies on the monolithic path). Every delivery
    // that fails records the error and fails initialization — the previous
    // per-helper code swallowed these.
    ShaderSystem::PipelineFactoryExecutor executor{pipeline_factory};
    ShaderSystem::PipelineCompiler compiler{executor};
    for (auto& row : rows) {
        auto desc = ShaderSystem::MakeComputeDesc(
            ShaderSystem::ComputePipelineTemplate{.debug_name = row.pipeline_name},
            row.shader, row.layout);
        if (row.draw_mode_specialized) {
            ApplyDrawModeSpec(desc, ShaderSystem::DrawModeSpecValue(draw_mode_));
        }
        row.desc_out = desc;
        auto& slot = row.slot;
        const auto name = row.pipeline_name;
        compiler.SubmitCompute(std::move(desc), name,
            [this, &slot, &ok, name](ShaderSystem::PipelineOutcome&& outcome) {
                if (!outcome.has_value()) {
                    LOGIFACE_LOG(error, std::format(
                        "SceneRenderer: {} pipeline creation failed: {}",
                        name, outcome.error().message));
                    ok = false;
                    return;
                }
                slot.Swap(std::move(*outcome), 0);
                VulkanBackend::Vulkan::SetVulkanObjectName(
                    backend_->GetDevice(), slot.Get(), std::format("{}-pipeline", name));
            });
    }
    {
        auto& slot = depth_slot_;
        compiler.SubmitGraphics(std::move(depth_desc), "depth-prepass",
            [this, &slot, &ok](ShaderSystem::PipelineOutcome&& outcome) {
                if (!outcome.has_value()) {
                    LOGIFACE_LOG(error, std::format(
                        "SceneRenderer: depth-prepass pipeline creation failed: {}",
                        outcome.error().message));
                    ok = false;
                    return;
                }
                slot.Swap(std::move(*outcome), 0);
                VulkanBackend::Vulkan::SetVulkanObjectName(
                    backend_->GetDevice(), slot.Get(), "depth-prepass-pipeline");
            });
    }
    compiler.Flush(shader_mgr);

    LOGIFACE_LOG(debug, ok ? "Scene pipelines created"
                           : "Scene pipeline creation failed");
    return ok;
}

bool SceneRenderer::RebuildCompactionPipelines() {
    if (!backend_ || !shader_mgr_ || !pipeline_factory_) return false;
    // Rebuild only the pipeline objects: the existing pipeline layouts are
    // reused so pipelines still in a retire ring keep valid layouts. The
    // draw-mode spec value is updated on each stored desc first.
    //
    // Technique pipelines also carry the draw-mode spec, but TechniqueManager
    // owns them. The caller of Reinitialize re-specializes them
    // (GameEngine::SetDrawMode); this function covers only SceneRenderer's own
    // pipelines. Occlusion is absent: it reads no kCompactionMode constant, so
    // its pipeline is mode-independent.
    struct RebuildRow {
        ShaderSystem::PipelineSlot SceneRenderer::* slot;
        std::optional<ShaderSystem::ComputePipelineDesc> SceneRenderer::* desc;
    };
    static constexpr std::array kRebuildRows{
        RebuildRow{&SceneRenderer::expand_slot_, &SceneRenderer::expand_desc_},
        RebuildRow{&SceneRenderer::occluder_select_slot_, &SceneRenderer::occluder_select_desc_},
        RebuildRow{&SceneRenderer::pre_cull_slot_, &SceneRenderer::pre_cull_desc_},
        RebuildRow{&SceneRenderer::collect_count_slot_, &SceneRenderer::collect_count_desc_},
        // collect_write is CID-only and has no specialization constant, but its
        // pipeline still rebuilds so a shader edit swaps it too.
        RebuildRow{&SceneRenderer::collect_write_slot_, &SceneRenderer::collect_write_desc_},
    };

    const auto rebuild = [this](ShaderSystem::PipelineSlot& slot, auto& desc) {
        if (!desc) return true;
        using Desc = std::remove_cvref_t<decltype(*desc)>;
        std::expected<ShaderSystem::PipelineProduct, ShaderSystem::PipelineError> result =
            std::unexpected(ShaderSystem::PipelineError{});
        if constexpr (std::same_as<Desc, ShaderSystem::ComputePipelineDesc>) {
            result = pipeline_factory_->CreateCompute(*desc, *shader_mgr_);
        } else {
            result = pipeline_factory_->CreateGraphics(*desc, *shader_mgr_);
        }
        if (!result.has_value()) {
            LOGIFACE_LOG(error, "SceneRenderer: failed to rebuild mode-specialized pipeline");
            return false;
        }
        slot.Swap(std::move(result.value()), 0);
        return true;
    };

    bool ok = true;
    for (auto& row : kRebuildRows) {
        auto& desc = this->*row.desc;
        if (!desc) continue;
        ApplyDrawModeSpec(*desc, ShaderSystem::DrawModeSpecValue(draw_mode_));
        ok = rebuild(this->*row.slot, desc) && ok;
    }
    if (depth_desc_) {
        ApplyDrawModeSpec(*depth_desc_, ShaderSystem::DrawModeSpecValue(draw_mode_));
        ok = rebuild(depth_slot_, depth_desc_) && ok;
    }
    return ok;
}

} // namespace VulkanEngine::SceneRenderer