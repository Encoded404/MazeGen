module;

#include <logging/logging_macros.hpp>

module VulkanEngine.StandardMeshPipeline;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.GpuResources;

namespace VulkanEngine::StandardMeshPipeline {

GraphicsPipeline::GraphicsPipeline() = default;
GraphicsPipeline::~GraphicsPipeline() = default;

void GraphicsPipeline::Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                  const std::vector<std::uint32_t>& vertex_spirv,
                                  const std::vector<std::uint32_t>& fragment_spirv,
                                  const PipelineConfig& config,
                                  vk::DescriptorSetLayout* bindless_layout,
                                  vk::DescriptorSetLayout* object_data_layout,
                                  vk::DescriptorSetLayout* raw_vertex_layout,
                                  vk::DescriptorSetLayout* indirection_layout) {
    bootstrap_ = &bootstrap;
    config_ = config;
    object_data_layout_ = object_data_layout;
    raw_vertex_layout_ = raw_vertex_layout;
    indirection_layout_ = indirection_layout;

    if (bindless_layout) {
        external_layout_ = bindless_layout;
    } else {
        CreateDescriptorSetLayout(bootstrap);
        CreateDescriptorPool(bootstrap);
    }
    CreatePipeline(bootstrap, vertex_spirv, fragment_spirv, config);
}

void GraphicsPipeline::Shutdown() {
    if (bootstrap_) {
        bootstrap_->GetBackend().WaitDeviceIdle();
    }
    pipeline_.reset();
    pipeline_layout_.reset();
    pool_.reset();
    descriptor_set_layout_.reset();
    bootstrap_ = nullptr;
}

vk::PipelineLayout* GraphicsPipeline::GetPipelineLayout() {
    return pipeline_layout_ ? const_cast<vk::PipelineLayout*>(&**pipeline_layout_) : nullptr;
}

const vk::PipelineLayout* GraphicsPipeline::GetPipelineLayout() const {
    return pipeline_layout_ ? &**pipeline_layout_ : nullptr;
}

vk::raii::Pipeline* GraphicsPipeline::GetPipeline() {
    return pipeline_.get();
}

const vk::raii::Pipeline* GraphicsPipeline::GetPipeline() const {
    return pipeline_.get();
}

vk::DescriptorSetLayout* GraphicsPipeline::GetDescriptorSetLayout() {
    if (external_layout_) return external_layout_;
    return descriptor_set_layout_ ? const_cast<vk::DescriptorSetLayout*>(&**descriptor_set_layout_) : nullptr;
}

const vk::DescriptorSetLayout* GraphicsPipeline::GetDescriptorSetLayout() const {
    if (external_layout_) return external_layout_;
    return descriptor_set_layout_ ? &**descriptor_set_layout_ : nullptr;
}

void GraphicsPipeline::CreateDescriptorSetLayout(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap) {
    LOGIFACE_LOG(trace, "entering CreateDescriptorSetLayout");

    const auto& device = bootstrap.GetBackend().GetDevice();

    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}
    };

    const vk::DescriptorSetLayoutCreateInfo info({}, bindings);
    descriptor_set_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(device, info);
    VulkanBackend::Vulkan::SetVulkanObjectName(device, *descriptor_set_layout_, "standard-mesh-layout");

    LOGIFACE_LOG(trace, "leaving CreateDescriptorSetLayout successfully");
}

void GraphicsPipeline::CreateDescriptorPool(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap) {
    LOGIFACE_LOG(trace, "entering CreateDescriptorPool");

    VulkanEngine::GpuResources::DescriptorPoolConfig config{};
    config.max_sets = 100;
    config.max_combined_image_samplers = 100;
    pool_ = VulkanEngine::GpuResources::DescriptorPool::Create(bootstrap.GetBackend(), config);
    pool_->SetDebugName(bootstrap.GetBackend().GetDevice(), "standard-mesh-pool");

    LOGIFACE_LOG(trace, "leaving CreateDescriptorPool successfully");
}

void GraphicsPipeline::CreatePipeline(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                      const std::vector<std::uint32_t>& vertex_spirv,
                                      const std::vector<std::uint32_t>& fragment_spirv,
                                      const PipelineConfig& config) {
    LOGIFACE_LOG(trace, "entering CreatePipeline");

    const auto& device = bootstrap.GetBackend().GetDevice();

    const vk::ShaderModuleCreateInfo vert_info({}, vertex_spirv.size() * sizeof(std::uint32_t), vertex_spirv.data());
    const vk::raii::ShaderModule vert_module(device, vert_info);

    const vk::ShaderModuleCreateInfo frag_info({}, fragment_spirv.size() * sizeof(std::uint32_t), fragment_spirv.data());
    const vk::raii::ShaderModule frag_module(device, frag_info);

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vert_module, "main"),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *frag_module, "main")
    };

    const vk::DescriptorSetLayout tex_layout = external_layout_ ? *external_layout_ : **descriptor_set_layout_;
    std::vector<vk::DescriptorSetLayout> set_layouts = { tex_layout };
    if (object_data_layout_) {
        set_layouts.push_back(*object_data_layout_);     // set 1
    }
    if (raw_vertex_layout_) {
        set_layouts.push_back(*raw_vertex_layout_);      // set 2
    }
    if (indirection_layout_) {
        set_layouts.push_back(*indirection_layout_);      // set 3
    }

    constexpr std::uint32_t push_constant_size = 64;
    constexpr vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eVertex, 0, push_constant_size);

    vk::PipelineLayoutCreateInfo layout_info{};
    layout_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
    layout_info.pSetLayouts = set_layouts.data();

    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(device, layout_info);
    VulkanBackend::Vulkan::SetVulkanObjectName(device, *pipeline_layout_, "standard-mesh-pipeline-layout");

    // When indirection layout is present, use zero vertex bindings (SSBO pulling)
    const bool use_vertex_pulling = (indirection_layout_ != nullptr);
    vk::PipelineVertexInputStateCreateInfo vertex_input({}, 0, nullptr, 0, nullptr);
    if (!use_vertex_pulling) {
        constexpr std::array<vk::VertexInputBindingDescription, 1> vertex_bindings = {
            vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex)
        };
        // Note: explicit offsets are used instead of offsetof() because
        // offsetof is a C preprocessor macro and is not available through
        // C++20 module imports (import std; does not export macros).
        // Struct Vertex layout: px(0-11), nx(12-23), u(24-31) — each float = 4 bytes.
        constexpr std::array<vk::VertexInputAttributeDescription, 3> vertex_attributes = {
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat,  0},   // Vertex::px
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, 12},   // Vertex::nx
            vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32Sfloat,    24},   // Vertex::u
        };
        vertex_input = vk::PipelineVertexInputStateCreateInfo({}, vertex_bindings, vertex_attributes);
    }
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, config.primitive_topology);
    constexpr vk::PipelineViewportStateCreateInfo viewport_state({}, 1, nullptr, 1, nullptr);
    const vk::PipelineRasterizationStateCreateInfo rasterization({}, false, false, config.polygon_mode, config.cull_mode, config.front_face, false, 0, 0, 0, config.line_width);
    const vk::PipelineMultisampleStateCreateInfo multisample({}, config.sample_count);
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil({}, config.depth_test_enable, config.depth_write_enable, config.depth_compare_op);

    const vk::PipelineColorBlendAttachmentState color_blend_attachment(
        config.blend_enable,
        config.src_color_blend_factor, config.dst_color_blend_factor, config.color_blend_op,
        config.src_alpha_blend_factor, config.dst_alpha_blend_factor, config.alpha_blend_op,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    const vk::PipelineColorBlendStateCreateInfo color_blend({}, false, vk::LogicOp::eCopy, color_blend_attachment);

    std::array<vk::DynamicState, 2> dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic_state({}, dynamic_states);

    const vk::Format format = bootstrap.GetBackend().GetSurfaceFormat().format;
    vk::PipelineRenderingCreateInfo rendering_info{};
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &format;
    rendering_info.depthAttachmentFormat = bootstrap.GetBackend().GetDepthFormat();
    rendering_info.stencilAttachmentFormat = vk::Format::eUndefined;

    vk::GraphicsPipelineCreateInfo pipeline_info({}, stages, &vertex_input, &input_assembly, nullptr, &viewport_state, &rasterization, &multisample, &depth_stencil, &color_blend, &dynamic_state, *pipeline_layout_, nullptr, 0, {}, 0);
    pipeline_info.setPNext(&rendering_info);

    pipeline_ = std::make_unique<vk::raii::Pipeline>(device, nullptr, pipeline_info);
    VulkanBackend::Vulkan::SetVulkanObjectName(device, *pipeline_, "standard-mesh-pipeline");

    LOGIFACE_LOG(trace, "leaving CreatePipeline successfully");
}

void GraphicsPipeline::CreatePipelineWithLayout(
        VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
        const std::vector<std::uint32_t>& vertex_spirv,
        const std::vector<std::uint32_t>& fragment_spirv,
        const PipelineConfig& config,
        [[maybe_unused]] const std::vector<vk::DescriptorSetLayout>& set_layouts,
        vk::PipelineLayout external_pipeline_layout) {
        LOGIFACE_LOG(trace, "entering CreatePipelineWithLayout");

        const auto& device = bootstrap.GetBackend().GetDevice();

        const vk::ShaderModuleCreateInfo vert_info({}, vertex_spirv.size() * sizeof(std::uint32_t), vertex_spirv.data());
        const vk::raii::ShaderModule vert_module(device, vert_info);

        const vk::ShaderModuleCreateInfo frag_info({}, fragment_spirv.size() * sizeof(std::uint32_t), fragment_spirv.data());
        const vk::raii::ShaderModule frag_module(device, frag_info);

        std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vert_module, "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *frag_module, "main")
        };

        // Use the pre-built layout — do NOT create our own
        // The external_pipeline_layout is owned by BaseTechnique

        // When indirection layout is present, use zero vertex bindings (SSBO pulling)
        const bool use_vertex_pulling = true; // Use SSBO pulling for technique pipelines
        vk::PipelineVertexInputStateCreateInfo vertex_input({}, 0, nullptr, 0, nullptr);
        if (!use_vertex_pulling) {
            constexpr std::array<vk::VertexInputBindingDescription, 1> vertex_bindings = {
                vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex)
            };
            constexpr std::array<vk::VertexInputAttributeDescription, 3> vertex_attributes = {
                vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat,  0},
                vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, 12},
                vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32Sfloat,    24},
            };
            vertex_input = vk::PipelineVertexInputStateCreateInfo({}, vertex_bindings, vertex_attributes);
        }
        const vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, config.primitive_topology);
        constexpr vk::PipelineViewportStateCreateInfo viewport_state({}, 1, nullptr, 1, nullptr);
        const vk::PipelineRasterizationStateCreateInfo rasterization({}, false, false, config.polygon_mode, config.cull_mode, config.front_face, false, 0, 0, 0, config.line_width);
        const vk::PipelineMultisampleStateCreateInfo multisample({}, config.sample_count);
        const vk::PipelineDepthStencilStateCreateInfo depth_stencil({}, config.depth_test_enable, config.depth_write_enable, config.depth_compare_op);

        const vk::PipelineColorBlendAttachmentState color_blend_attachment(
            config.blend_enable,
            config.src_color_blend_factor, config.dst_color_blend_factor, config.color_blend_op,
            config.src_alpha_blend_factor, config.dst_alpha_blend_factor, config.alpha_blend_op,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        const vk::PipelineColorBlendStateCreateInfo color_blend({}, false, vk::LogicOp::eCopy, color_blend_attachment);

        std::array<vk::DynamicState, 2> dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        const vk::PipelineDynamicStateCreateInfo dynamic_state({}, dynamic_states);

        const vk::Format format = bootstrap.GetBackend().GetSurfaceFormat().format;
        vk::PipelineRenderingCreateInfo rendering_info{};
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachmentFormats = &format;
        rendering_info.depthAttachmentFormat = bootstrap.GetBackend().GetDepthFormat();
        rendering_info.stencilAttachmentFormat = vk::Format::eUndefined;

        vk::GraphicsPipelineCreateInfo pipeline_info({}, stages, &vertex_input, &input_assembly, nullptr, &viewport_state, &rasterization, &multisample, &depth_stencil, &color_blend, &dynamic_state, external_pipeline_layout, nullptr, 0, {}, 0);
        pipeline_info.setPNext(&rendering_info);

        pipeline_ = std::make_unique<vk::raii::Pipeline>(device, nullptr, pipeline_info);
        VulkanBackend::Vulkan::SetVulkanObjectName(device, *pipeline_, "technique-pipeline");

        LOGIFACE_LOG(trace, "leaving CreatePipelineWithLayout successfully");
    }

} // namespace VulkanEngine::StandardMeshPipeline
