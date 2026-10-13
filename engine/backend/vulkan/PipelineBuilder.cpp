module;

module VulkanBackend.Vulkan.PipelineUtils;

import std;

import vulkan;

namespace VulkanBackend::Vulkan {

vk::Viewport PipelineUtils::CreateViewport(float width,
                                         float height,
                                         float min_depth,
                                         float max_depth,
                                         float x,
                                         float y) {
    vk::Viewport viewport{};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = min_depth;
    viewport.maxDepth = max_depth;
    return viewport;
}

vk::Rect2D PipelineUtils::CreateScissor(std::uint32_t width, std::uint32_t height, std::int32_t offset_x, std::int32_t offset_y) {
    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{offset_x, offset_y};
    scissor.extent = vk::Extent2D{width, height};
    return scissor;
}

vk::PipelineColorBlendAttachmentState PipelineUtils::CreateDefaultColorBlendAttachment() {
    vk::PipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                vk::ColorComponentFlagBits::eG |
                                vk::ColorComponentFlagBits::eB |
                                vk::ColorComponentFlagBits::eA;
    attachment.blendEnable = vk::False;
    return attachment;
}

vk::PipelineColorBlendAttachmentState PipelineUtils::CreateAlphaBlendAttachment() {
    vk::PipelineColorBlendAttachmentState attachment = CreateDefaultColorBlendAttachment();
    attachment.blendEnable = vk::True;
    attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    attachment.colorBlendOp = vk::BlendOp::eAdd;
    attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    attachment.alphaBlendOp = vk::BlendOp::eAdd;
    return attachment;
}

vk::PipelineColorBlendAttachmentState PipelineUtils::CreateAdditiveBlendAttachment() {
    vk::PipelineColorBlendAttachmentState attachment = CreateDefaultColorBlendAttachment();
    attachment.blendEnable = vk::True;
    attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    attachment.dstColorBlendFactor = vk::BlendFactor::eOne;
    attachment.colorBlendOp = vk::BlendOp::eAdd;
    attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    attachment.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    attachment.alphaBlendOp = vk::BlendOp::eAdd;
    return attachment;
}

vk::PipelineColorBlendAttachmentState PipelineUtils::CreateNoBlendAttachment() {
    vk::PipelineColorBlendAttachmentState attachment = CreateDefaultColorBlendAttachment();
    attachment.colorWriteMask = vk::ColorComponentFlags{};
    return attachment;
}

vk::PipelineDepthStencilStateCreateInfo PipelineUtils::CreateDefaultDepthStencilState() {
    vk::PipelineDepthStencilStateCreateInfo info{};
    info.depthTestEnable = vk::True;
    info.depthWriteEnable = vk::True;
    info.depthCompareOp = vk::CompareOp::eLess;
    info.depthBoundsTestEnable = vk::False;
    info.stencilTestEnable = vk::False;
    info.minDepthBounds = 0.0f;
    info.maxDepthBounds = 1.0f;
    return info;
}

vk::PipelineDepthStencilStateCreateInfo PipelineUtils::CreateDepthTestOnlyState() {
    vk::PipelineDepthStencilStateCreateInfo info = CreateDefaultDepthStencilState();
    info.depthWriteEnable = vk::False;
    return info;
}

vk::PipelineDepthStencilStateCreateInfo PipelineUtils::CreateNoDepthTestState() {
    vk::PipelineDepthStencilStateCreateInfo info = CreateDefaultDepthStencilState();
    info.depthTestEnable = vk::False;
    info.depthWriteEnable = vk::False;
    return info;
}

vk::PipelineRasterizationStateCreateInfo PipelineUtils::CreateDefaultRasterizationState() {
    vk::PipelineRasterizationStateCreateInfo info{};
    info.depthClampEnable = vk::False;
    info.rasterizerDiscardEnable = vk::False;
    info.polygonMode = vk::PolygonMode::eFill;
    info.cullMode = vk::CullModeFlagBits::eFront;
    info.frontFace = vk::FrontFace::eClockwise;
    info.depthBiasEnable = vk::False;
    info.lineWidth = 1.0f;
    return info;
}

vk::PipelineRasterizationStateCreateInfo PipelineUtils::CreateWireframeRasterizationState() {
    vk::PipelineRasterizationStateCreateInfo info = CreateDefaultRasterizationState();
    info.polygonMode = vk::PolygonMode::eLine;
    return info;
}

vk::PipelineRasterizationStateCreateInfo PipelineUtils::CreateNoCullRasterizationState() {
    vk::PipelineRasterizationStateCreateInfo info = CreateDefaultRasterizationState();
    info.cullMode = vk::CullModeFlagBits::eNone;
    return info;
}

vk::PipelineMultisampleStateCreateInfo PipelineUtils::CreateDefaultMultisampleState() {
    vk::PipelineMultisampleStateCreateInfo info{};
    info.rasterizationSamples = vk::SampleCountFlagBits::e1;
    info.sampleShadingEnable = vk::False;
    return info;
}

vk::PipelineMultisampleStateCreateInfo PipelineUtils::CreateMsaaMultisampleState(vk::SampleCountFlagBits samples) {
    vk::PipelineMultisampleStateCreateInfo info = CreateDefaultMultisampleState();
    info.rasterizationSamples = samples;
    return info;
}

vk::VertexInputBindingDescription PipelineUtils::CreateVertexInputBinding(std::uint32_t binding,
                                                                        std::uint32_t stride,
                                                                        vk::VertexInputRate input_rate) {
    vk::VertexInputBindingDescription desc{};
    desc.binding = binding;
    desc.stride = stride;
    desc.inputRate = input_rate;
    return desc;
}

vk::VertexInputAttributeDescription PipelineUtils::CreateVertexInputAttribute(std::uint32_t location,
                                                                            std::uint32_t binding,
                                                                            vk::Format format,
                                                                            std::uint32_t offset) {
    vk::VertexInputAttributeDescription desc{};
    desc.location = location;
    desc.binding = binding;
    desc.format = format;
    desc.offset = offset;
    return desc;
}

vk::PipelineShaderStageCreateInfo PipelineUtils::CreateShaderStageCreateInfo(vk::ShaderStageFlagBits stage,
                                                                           vk::ShaderModule module,
                                                                           const char* entry_point,
                                                                           const vk::SpecializationInfo* specialization_info) {
    vk::PipelineShaderStageCreateInfo info{};
    info.stage = stage;
    info.module = module;
    info.pName = entry_point ? entry_point : "main";
    info.pSpecializationInfo = specialization_info;
    return info;
}

vk::PipelineInputAssemblyStateCreateInfo PipelineUtils::CreateInputAssemblyState(vk::PrimitiveTopology topology,
                                                                               vk::Bool32 primitive_restart_enable) {
    vk::PipelineInputAssemblyStateCreateInfo info{};
    info.topology = topology;
    info.primitiveRestartEnable = primitive_restart_enable;
    return info;
}

vk::PipelineTessellationStateCreateInfo PipelineUtils::CreateTessellationState(std::uint32_t patch_control_points) {
    vk::PipelineTessellationStateCreateInfo info{};
    info.patchControlPoints = patch_control_points;
    return info;
}

vk::PipelineViewportStateCreateInfo PipelineUtils::CreateViewportState(const std::vector<vk::Viewport>& viewports,
                                                                 const std::vector<vk::Rect2D>& scissors) {
    vk::PipelineViewportStateCreateInfo info{};
    info.viewportCount = static_cast<std::uint32_t>(viewports.size());
    info.pViewports = viewports.empty() ? nullptr : viewports.data();
    info.scissorCount = static_cast<std::uint32_t>(scissors.size());
    info.pScissors = scissors.empty() ? nullptr : scissors.data();
    return info;
}

vk::PipelineViewportStateCreateInfo PipelineUtils::CreateDynamicViewportState(std::uint32_t viewport_count,
                                                                            std::uint32_t scissor_count) {
    vk::PipelineViewportStateCreateInfo info{};
    info.viewportCount = viewport_count;
    info.pViewports = nullptr;
    info.scissorCount = scissor_count;
    info.pScissors = nullptr;
    return info;
}

vk::PipelineColorBlendStateCreateInfo PipelineUtils::CreateColorBlendState(const std::vector<vk::PipelineColorBlendAttachmentState>& attachments,
                                                                         vk::Bool32 logic_op_enable,
                                                                         vk::LogicOp logic_op) {
    vk::PipelineColorBlendStateCreateInfo info{};
    info.logicOpEnable = logic_op_enable;
    info.logicOp = logic_op;
    info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    info.pAttachments = attachments.empty() ? nullptr : attachments.data();
    return info;
}

vk::PipelineDynamicStateCreateInfo PipelineUtils::CreateDynamicState(const std::vector<vk::DynamicState>& dynamic_states) {
    vk::PipelineDynamicStateCreateInfo info{};
    info.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    info.pDynamicStates = dynamic_states.empty() ? nullptr : dynamic_states.data();
    return info;
}

std::vector<vk::DynamicState> PipelineUtils::GetBasicDynamicStates() {
    return {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
}

std::vector<vk::DynamicState> PipelineUtils::GetExtendedDynamicStates() {
    return {vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eLineWidth,
            vk::DynamicState::eDepthBias,
            vk::DynamicState::eBlendConstants};
}

} // namespace VulkanBackend::Vulkan
