module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

#include <logging/logging_macros.hpp>

module VulkanEngine.SceneRenderer;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanDebugUtils;

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

        // kCompactionMode specialization constant (constant_id 0):
        //   0 = CID (4 B compact index copies)
        //   1 = MID (20 B DrawIndexedIndirectCommand emission)
        // Applied to both compute pipelines and the vertex-stage graphics
        // pipelines (draw-mode-specialized vertex shaders).
        template <typename Desc>
        void SetDrawModeSpec(Desc& desc, DrawMode mode) {
            const std::uint32_t value =
                (mode == DrawMode::MID) ? 1u : 0u;
            desc.spec_entries = {
                vk::SpecializationMapEntry(0, 0, sizeof(std::uint32_t))
            };
            desc.spec_data.resize(sizeof(std::uint32_t));
            std::memcpy(desc.spec_data.data(), &value, sizeof(value));
        }
    } // anonymous namespace

bool SceneRenderer::CreateExpandPipeline(const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                          ShaderSystem::ShaderManager& shader_mgr,
                                          ShaderSystem::PipelineFactory& pipeline_factory,
                                          ShaderSystem::ShaderId shader_id) {
    LOGIFACE_LOG(debug, "Creating expand pipeline...");
    const auto& dev = be.GetDevice();
    vk::PushConstantRange pr{};
    pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pr.size = sizeof(ExpandPC);
    std::array<vk::DescriptorSetLayout, 2> sl{ *expand_layout_, *index_buffers_layout_ };
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = static_cast<std::uint32_t>(sl.size());
    li.pSetLayouts = sl.data();
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    expand_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *expand_pipeline_layout_, "expand-pipeline-layout");

    ShaderSystem::ComputePipelineDesc desc{};
    desc.shader = shader_id;
    desc.layout = *expand_pipeline_layout_;
    SetDrawModeSpec(desc, draw_mode_);
    expand_desc_ = desc;
    auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
    if (result.has_value()) {
        expand_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, expand_slot_.Get(), "expand-pipeline");
    }
    LOGIFACE_LOG(debug, "Expand pipeline created");
    return true;
}

bool SceneRenderer::CreateDepthPipeline(VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                         ShaderSystem::ShaderManager& shader_mgr,
                                         ShaderSystem::PipelineFactory& pipeline_factory,
                                         ShaderSystem::ShaderId vert_id,
                                         ShaderSystem::ShaderId frag_id,
                                         const vk::PipelineRasterizationStateCreateInfo& rs) {
    LOGIFACE_LOG(debug, "Creating depth pipeline...");
    const auto& dev = be.GetDevice();
    std::array<vk::DescriptorSetLayout, 4> sl{
        *empty_layout_, *submesh_vertex_layout_, *vertex_buffers_layout_, *indirection_layout_
    };
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = static_cast<std::uint32_t>(sl.size());
    li.pSetLayouts = sl.data();
    depth_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *depth_pipeline_layout_, "depth-prepass-pipeline-layout");

    ShaderSystem::GraphicsPipelineDesc desc{};
    desc.vertex_shader = vert_id;
    desc.fragment_shader = frag_id;
    desc.vertex_input = vk::PipelineVertexInputStateCreateInfo({}, 0, nullptr, 0, nullptr);
    desc.input_assembly = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);
    desc.viewport = vk::PipelineViewportStateCreateInfo({}, 1, nullptr, 1, nullptr);
    desc.rasterization = rs;
    desc.multisample = vk::PipelineMultisampleStateCreateInfo({}, vk::SampleCountFlagBits::e1);
    desc.depth_stencil = vk::PipelineDepthStencilStateCreateInfo({}, true, true, vk::CompareOp::eLess);
    // Depth-only pass: no color attachments, so the blend state must not declare any.
    desc.color_blend = vk::PipelineColorBlendStateCreateInfo({}, false, vk::LogicOp::eCopy, 0, nullptr);
    desc.dynamic_states = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    desc.layout = *depth_pipeline_layout_;
    desc.depth_format = be.GetDepthFormat();
    SetDrawModeSpec(desc, draw_mode_);
    depth_desc_ = desc;

    auto result = pipeline_factory.CreateGraphics(desc, shader_mgr);
    if (result.has_value()) {
        depth_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, depth_slot_.Get(), "depth-prepass-pipeline");
    }
    LOGIFACE_LOG(debug, "Depth pipeline created");
    return true;
}

bool SceneRenderer::CreateHiZPipeline(VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                       ShaderSystem::ShaderManager& shader_mgr,
                                       ShaderSystem::PipelineFactory& pipeline_factory,
                                       ShaderSystem::ShaderId shader_id) {
    LOGIFACE_LOG(debug, "Creating HIZ pipeline...");
    const auto& dev = be.GetDevice();
    vk::PushConstantRange pr{};
    pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pr.size = sizeof(HiZPC);
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
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = 1;
    li.pSetLayouts = &**hiz_layout_;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    hiz_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *hiz_pipeline_layout_, "hiz-gen-pipeline-layout");

    ShaderSystem::ComputePipelineDesc desc{};
    desc.shader = shader_id;
    desc.layout = *hiz_pipeline_layout_;
    hiz_desc_ = desc;
    auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
    if (result.has_value()) {
        hiz_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, hiz_slot_.Get(), "hiz-gen-pipeline");
    }
    LOGIFACE_LOG(debug, "HiZ pipeline created");
    return true;
}

bool SceneRenderer::CreateOcclusionPipeline(const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                              ShaderSystem::ShaderManager& shader_mgr,
                                              ShaderSystem::PipelineFactory& pipeline_factory,
                                              ShaderSystem::ShaderId shader_id) {
    LOGIFACE_LOG(debug, "Creating occlusion pipeline...");
    const auto& dev = be.GetDevice();
    vk::PushConstantRange pr{};
    pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pr.size = sizeof(OccPC);
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = 1;
    li.pSetLayouts = &**occlusion_layout_;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    occlusion_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *occlusion_pipeline_layout_, "occlusion-pipeline-layout");

    ShaderSystem::ComputePipelineDesc desc{};
    desc.shader = shader_id;
    desc.layout = *occlusion_pipeline_layout_;
    occlusion_desc_ = desc;
    auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
    if (result.has_value()) {
        occlusion_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, occlusion_slot_.Get(), "occlusion-pipeline");
    }
    LOGIFACE_LOG(debug, "Occlusion pipeline created");
    return true;
}

bool SceneRenderer::CreateOccluderSelectPipeline(const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                                  ShaderSystem::ShaderManager& shader_mgr,
                                                  ShaderSystem::PipelineFactory& pipeline_factory,
                                                  ShaderSystem::ShaderId shader_id) {
    LOGIFACE_LOG(debug, "Creating occluder-select pipeline...");
    const auto& dev = be.GetDevice();
    vk::PushConstantRange pr{};
    pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pr.size = sizeof(OccluderSelectPC);
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = 1;
    li.pSetLayouts = &**occluder_select_layout_;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    occluder_select_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *occluder_select_pipeline_layout_, "occluder-select-pipeline-layout");

    ShaderSystem::ComputePipelineDesc desc{};
    desc.shader = shader_id;
    desc.layout = *occluder_select_pipeline_layout_;
    SetDrawModeSpec(desc, draw_mode_);
    occluder_select_desc_ = desc;
    auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
    if (result.has_value()) {
        occluder_select_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, occluder_select_slot_.Get(), "occluder-select-pipeline");
    }
    LOGIFACE_LOG(debug, "Occluder-select pipeline created");
    return true;
}

bool SceneRenderer::CreatePreCullPipeline(const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                           ShaderSystem::ShaderManager& shader_mgr,
                                           ShaderSystem::PipelineFactory& pipeline_factory,
                                           ShaderSystem::ShaderId shader_id) {
    LOGIFACE_LOG(debug, "Creating pre-cull pipeline...");
    const auto& dev = be.GetDevice();
    vk::PushConstantRange pr{};
    pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pr.size = sizeof(PreCullPC);
    vk::PipelineLayoutCreateInfo li{};
    li.setLayoutCount = 1;
    li.pSetLayouts = &**occlusion_layout_; // shares the occlusion set (+ survivor compaction bindings 6-8)
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    pre_cull_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *pre_cull_pipeline_layout_, "pre-cull-pipeline-layout");

    ShaderSystem::ComputePipelineDesc desc{};
    desc.shader = shader_id;
    desc.layout = *pre_cull_pipeline_layout_;
    SetDrawModeSpec(desc, draw_mode_);
    pre_cull_desc_ = desc;
    auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
    if (result.has_value()) {
        pre_cull_slot_.Swap(std::move(result.value()), 0);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, pre_cull_slot_.Get(), "pre-cull-pipeline");
    }
    LOGIFACE_LOG(debug, "Pre-cull pipeline created");
    return true;
}

bool SceneRenderer::CreateCollectPipelines(const VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                           ShaderSystem::ShaderManager& shader_mgr,
                                           ShaderSystem::PipelineFactory& pipeline_factory,
                                           ShaderSystem::ShaderId count_id,
                                           ShaderSystem::ShaderId write_id) {
    LOGIFACE_LOG(debug, "Creating collect pipelines...");
    const auto& dev = be.GetDevice();

    {
        vk::PushConstantRange pr{};
        pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pr.size = sizeof(CollectPC);
        vk::PipelineLayoutCreateInfo li{};
        li.setLayoutCount = 1;
        li.pSetLayouts = &**collect_layout_;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pr;
        collect_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *collect_pipeline_layout_, "collect-count-pipeline-layout");

        ShaderSystem::ComputePipelineDesc desc{};
        desc.shader = count_id;
        desc.layout = *collect_pipeline_layout_;
        SetDrawModeSpec(desc, draw_mode_);
        collect_count_desc_ = desc;
        auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
        if (result.has_value()) {
            collect_count_slot_.Swap(std::move(result.value()), 0);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, collect_count_slot_.Get(), "collect-count-pipeline");
        }
    }

    {
        vk::PushConstantRange pr{};
        pr.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pr.size = sizeof(WritePC);
        vk::PipelineLayoutCreateInfo li{};
        li.setLayoutCount = 1;
        li.pSetLayouts = &**collect_write_layout_;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pr;
        collect_write_pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(dev, li);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *collect_write_pipeline_layout_, "collect-write-pipeline-layout");

        ShaderSystem::ComputePipelineDesc desc{};
        desc.shader = write_id;
        desc.layout = *collect_write_pipeline_layout_;
        collect_write_desc_ = desc;
        auto result = pipeline_factory.CreateCompute(desc, shader_mgr);
        if (result.has_value()) {
            collect_write_slot_.Swap(std::move(result.value()), 0);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, collect_write_slot_.Get(), "collect-write-pipeline");
        }
    }

    LOGIFACE_LOG(debug, "Collect pipelines created");
    return true;
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
    // pipelines.
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

    if (expand_desc_)         SetDrawModeSpec(*expand_desc_, draw_mode_);
    if (occluder_select_desc_) SetDrawModeSpec(*occluder_select_desc_, draw_mode_);
    if (pre_cull_desc_)       SetDrawModeSpec(*pre_cull_desc_, draw_mode_);
    if (collect_count_desc_)  SetDrawModeSpec(*collect_count_desc_, draw_mode_);
    if (depth_desc_)          SetDrawModeSpec(*depth_desc_, draw_mode_);
    // collect_write is CID-only and has no specialization constant.

    bool ok = true;
    ok = rebuild(expand_slot_, expand_desc_) && ok;
    ok = rebuild(occluder_select_slot_, occluder_select_desc_) && ok;
    ok = rebuild(pre_cull_slot_, pre_cull_desc_) && ok;
    ok = rebuild(collect_count_slot_, collect_count_desc_) && ok;
    ok = rebuild(collect_write_slot_, collect_write_desc_) && ok;
    ok = rebuild(depth_slot_, depth_desc_) && ok;
    return ok;
}

} // namespace VulkanEngine::SceneRenderer
