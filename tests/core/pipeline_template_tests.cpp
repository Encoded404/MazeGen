#include <gtest/gtest.h>

import std;
import std.compat;

import vulkan;

import VulkanEngine.DrawMode;
import VulkanEngine.PipelineFactory;
import VulkanEngine.PipelineTemplates;
import VulkanEngine.ShaderManager;

namespace {

using namespace VulkanEngine::ShaderSystem;

// ShaderId is a plain uint16 alias with 0 as the invalid sentinel (see
// ShaderManager.cppm). Tests use nonzero ids; the builders only assert on
// validity, they never touch a device.
constexpr ShaderId kVert = 11;
constexpr ShaderId kFrag = 12;
constexpr ShaderId kComp = 13;

vk::PipelineLayout NullLayout() {
    return vk::PipelineLayout{}; // null handle, no device needed
}

// ── Draw-mode spec constant ─────────────────────────────────────────────────

TEST(DrawModeSpecValue, MapsEnumToCompactionConstant) {
    // constant_id 0 payload: 0 = CID, 1 = MID (kCompactionMode contract).
    EXPECT_EQ(DrawModeSpecValue(VulkanEngine::SceneRenderer::DrawMode::CID), 0u);
    EXPECT_EQ(DrawModeSpecValue(VulkanEngine::SceneRenderer::DrawMode::MID), 1u);
}

TEST(ApplyDrawModeSpec, WritesConstantIdZeroPayloadOnComputeDesc) {
    ComputePipelineDesc desc{};
    ApplyDrawModeSpec(desc, 1u);
    ASSERT_EQ(desc.spec_entries.size(), 1u);
    EXPECT_EQ(desc.spec_entries[0].constantID, 0u);
    EXPECT_EQ(desc.spec_entries[0].offset, 0u);
    EXPECT_EQ(desc.spec_entries[0].size, sizeof(std::uint32_t));
    ASSERT_EQ(desc.spec_data.size(), sizeof(std::uint32_t));
    std::uint32_t value = 0;
    std::memcpy(&value, desc.spec_data.data(), sizeof(value));
    EXPECT_EQ(value, 1u);
}

TEST(ApplyDrawModeSpec, WritesConstantIdZeroPayloadOnGraphicsDesc) {
    GraphicsPipelineDesc desc{};
    ApplyDrawModeSpec(desc, 0u);
    ASSERT_EQ(desc.spec_entries.size(), 1u);
    EXPECT_EQ(desc.spec_entries[0].constantID, 0u);
    ASSERT_EQ(desc.spec_data.size(), sizeof(std::uint32_t));
    std::uint32_t value = 0xDEADBEEFu;
    std::memcpy(&value, desc.spec_data.data(), sizeof(value));
    EXPECT_EQ(value, 0u);
}

// ── Compute builder ──────────────────────────────────────────────────────────

TEST(MakeComputeDesc, CarriesShaderLayoutAndTemplateEntryPoint) {
    const ComputePipelineTemplate tmpl{
        .debug_name = "test-compute",
        .push_constant_size = 16,
        .entry_point = "main",
    };
    const auto desc = MakeComputeDesc(tmpl, kComp, NullLayout());
    EXPECT_EQ(desc.shader, kComp);
    EXPECT_EQ(desc.layout, NullLayout());
    EXPECT_EQ(desc.entry_point, "main");
    // No spec set by the builder: draw-mode specialization is an explicit
    // ApplyDrawModeSpec call, never implicit.
    EXPECT_TRUE(desc.spec_entries.empty());
    EXPECT_TRUE(desc.spec_data.empty());
}

TEST(MakeComputeDesc, HonorsNonDefaultEntryPoint) {
    const ComputePipelineTemplate tmpl{
        .debug_name = "uv1-compute",
        .push_constant_size = 16,
        .entry_point = "main",
    };
    const auto desc = MakeComputeDesc(tmpl, kComp, NullLayout());
    EXPECT_EQ(desc.entry_point, "main");
    // A caller that needs a named wrapper entry point sets it on the template.
    const ComputePipelineTemplate named{
        .debug_name = "named",
        .push_constant_size = 16,
        .entry_point = "main_uv1",
    };
    EXPECT_EQ(MakeComputeDesc(named, kComp, NullLayout()).entry_point, "main_uv1");
}

// ── Graphics builder ─────────────────────────────────────────────────────────

GraphicsPipelineTemplate DepthOnlyTemplate() {
    GraphicsPipelineTemplate tmpl{
        .debug_name = "depth-only",
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
    return tmpl;
}

TEST(MakeGraphicsDesc, CopiesTemplateStateAndFormats) {
    const auto tmpl = DepthOnlyTemplate();
    const std::array formats{vk::Format::eB8G8R8A8Unorm};
    const auto desc = MakeGraphicsDesc(tmpl, kVert, kFrag, NullLayout(),
                                       formats, vk::Format::eD32Sfloat);
    EXPECT_EQ(desc.vertex_shader, kVert);
    EXPECT_EQ(desc.fragment_shader, kFrag);
    EXPECT_EQ(desc.vertex_entry_point, "main");
    EXPECT_EQ(desc.fragment_entry_point, "main");
    EXPECT_EQ(desc.input_assembly.topology, vk::PrimitiveTopology::eTriangleList);
    EXPECT_EQ(desc.rasterization.cullMode, vk::CullModeFlagBits::eFront);
    EXPECT_EQ(desc.depth_stencil.depthTestEnable, true);
    ASSERT_EQ(desc.color_formats.size(), 1u);
    EXPECT_EQ(desc.color_formats[0], vk::Format::eB8G8R8A8Unorm);
    EXPECT_EQ(desc.depth_format, vk::Format::eD32Sfloat);
    EXPECT_EQ(desc.stencil_format, vk::Format::eUndefined);
    // Builder-fixed engine defaults: pull-mode vertices + dynamic viewport/scissor.
    EXPECT_EQ(desc.vertex_input.vertexAttributeDescriptionCount, 0u);
    EXPECT_EQ(desc.vertex_input.vertexBindingDescriptionCount, 0u);
    EXPECT_EQ(desc.viewport.viewportCount, 1u);
    ASSERT_EQ(desc.dynamic_states.size(), 2u);
    EXPECT_EQ(desc.dynamic_states[0], vk::DynamicState::eViewport);
    EXPECT_EQ(desc.dynamic_states[1], vk::DynamicState::eScissor);
}

TEST(MakeGraphicsDesc, EmptyBlendAttachmentsLeaveBlendDisabledAndUnset) {
    // The desc contract: attachments are OWNED by the desc; color_blend's
    // attachmentCount/pAttachments stay unset and are rebound by the factory
    // at each creation entry (see BindColorBlendAttachments). Depth-only
    // passes build with no attachments at all.
    const auto desc = MakeGraphicsDesc(DepthOnlyTemplate(), kVert, kFrag,
                                       NullLayout(), {}, vk::Format::eD32Sfloat);
    EXPECT_TRUE(desc.color_blend_attachments.empty());
    EXPECT_EQ(desc.color_blend.attachmentCount, 0u);
    EXPECT_EQ(desc.color_blend.pAttachments, nullptr);
    EXPECT_EQ(desc.color_blend.logicOp, vk::LogicOp::eCopy);
}

TEST(MakeGraphicsDesc, OwnedBlendAttachmentsOutliveTemplateScope) {
    // The desc must own a copy: build inside a scope, drop the template, and
    // the attachments vector must still carry the state.
    std::optional<GraphicsPipelineDesc> desc;
    {
        auto tmpl = DepthOnlyTemplate();
        tmpl.blend_attachments = {
            vk::PipelineColorBlendAttachmentState(
                true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
                vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
                vk::BlendOp::eAdd,
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        };
        const std::array formats{vk::Format::eB8G8R8A8Unorm};
        desc = MakeGraphicsDesc(tmpl, kVert, kFrag, NullLayout(), formats,
                                vk::Format::eUndefined);
    }
    ASSERT_EQ(desc->color_blend_attachments.size(), 1u);
    EXPECT_EQ(desc->color_blend_attachments[0].blendEnable, true);
    EXPECT_EQ(desc->color_blend_attachments[0].srcColorBlendFactor,
              vk::BlendFactor::eSrcAlpha);
    // color_blend still unset: the factory rebinds from the owned vector.
    EXPECT_EQ(desc->color_blend.attachmentCount, 0u);
}

TEST(MakeGraphicsDesc, EntryPointsAndDynamicStatesOverridable) {
    GraphicsPipelineTemplate tmpl = DepthOnlyTemplate();
    tmpl.vertex_entry_point = "main_uv1";
    tmpl.fragment_entry_point = "main_uv1";
    tmpl.dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor,
                           vk::DynamicState::eLineWidth};
    const auto desc = MakeGraphicsDesc(tmpl, kVert, kFrag, NullLayout(), {},
                                       vk::Format::eD32Sfloat);
    EXPECT_EQ(desc.vertex_entry_point, "main_uv1");
    EXPECT_EQ(desc.fragment_entry_point, "main_uv1");
    ASSERT_EQ(desc.dynamic_states.size(), 3u);
    EXPECT_EQ(desc.dynamic_states[2], vk::DynamicState::eLineWidth);
    // The UV1 render-state variants also bind real vertex buffers; the
    // builder leaves vertex_input defaulted so the caller can replace it.
    EXPECT_EQ(desc.vertex_input.vertexBindingDescriptionCount, 0u);
}

} // namespace