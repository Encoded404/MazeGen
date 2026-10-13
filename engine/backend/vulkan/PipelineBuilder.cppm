module;

export module VulkanBackend.Vulkan.PipelineUtils;

import std;
import std.compat;

import vulkan;

export namespace VulkanBackend::Vulkan {

class PipelineUtils {
public:
    static vk::Viewport CreateViewport(float width,
                                     float height,
                                     float min_depth = 0.0f,
                                     float max_depth = 1.0f,
                                     float x = 0.0f,
                                     float y = 0.0f);

    static vk::Rect2D CreateScissor(std::uint32_t width, std::uint32_t height, std::int32_t offset_x = 0, std::int32_t offset_y = 0);

    static vk::PipelineColorBlendAttachmentState CreateDefaultColorBlendAttachment();
    static vk::PipelineColorBlendAttachmentState CreateAlphaBlendAttachment();
    static vk::PipelineColorBlendAttachmentState CreateAdditiveBlendAttachment();
    static vk::PipelineColorBlendAttachmentState CreateNoBlendAttachment();

    static vk::PipelineDepthStencilStateCreateInfo CreateDefaultDepthStencilState();
    static vk::PipelineDepthStencilStateCreateInfo CreateDepthTestOnlyState();
    static vk::PipelineDepthStencilStateCreateInfo CreateNoDepthTestState();

    static vk::PipelineRasterizationStateCreateInfo CreateDefaultRasterizationState();
    static vk::PipelineRasterizationStateCreateInfo CreateWireframeRasterizationState();
    static vk::PipelineRasterizationStateCreateInfo CreateNoCullRasterizationState();

    static vk::PipelineMultisampleStateCreateInfo CreateDefaultMultisampleState();
    static vk::PipelineMultisampleStateCreateInfo CreateMsaaMultisampleState(vk::SampleCountFlagBits samples);

    static vk::VertexInputBindingDescription CreateVertexInputBinding(std::uint32_t binding,
                                                                    std::uint32_t stride,
                                                                    vk::VertexInputRate input_rate);

    static vk::VertexInputAttributeDescription CreateVertexInputAttribute(std::uint32_t location,
                                                                        std::uint32_t binding,
                                                                        vk::Format format,
                                                                        std::uint32_t offset);

    static vk::PipelineShaderStageCreateInfo CreateShaderStageCreateInfo(vk::ShaderStageFlagBits stage,
                                                                       vk::ShaderModule module,
                                                                       const char* entry_point = nullptr,
                                                                       const vk::SpecializationInfo* specialization_info = nullptr);

    static vk::PipelineInputAssemblyStateCreateInfo CreateInputAssemblyState(vk::PrimitiveTopology topology,
                                                                           vk::Bool32 primitive_restart_enable = vk::False);

    static vk::PipelineTessellationStateCreateInfo CreateTessellationState(std::uint32_t patch_control_points = 0);

    static vk::PipelineViewportStateCreateInfo CreateViewportState(const std::vector<vk::Viewport>& viewports,
                                                                 const std::vector<vk::Rect2D>& scissors);

    static vk::PipelineViewportStateCreateInfo CreateDynamicViewportState(std::uint32_t viewport_count, std::uint32_t scissor_count);

    static vk::PipelineColorBlendStateCreateInfo CreateColorBlendState(const std::vector<vk::PipelineColorBlendAttachmentState>& attachments,
                                                                     vk::Bool32 logic_op_enable = vk::False,
                                                                     vk::LogicOp logic_op = vk::LogicOp::eClear);

    static vk::PipelineDynamicStateCreateInfo CreateDynamicState(const std::vector<vk::DynamicState>& dynamic_states);

    static std::vector<vk::DynamicState> GetBasicDynamicStates();
    static std::vector<vk::DynamicState> GetExtendedDynamicStates();
};

} // namespace VulkanBackend::Vulkan
