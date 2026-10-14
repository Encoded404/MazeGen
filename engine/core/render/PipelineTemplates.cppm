module;

#include <cassert>
#include <cstring>

export module VulkanEngine.PipelineTemplates;

import std;
import vulkan;

import VulkanEngine.DrawMode;
import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderManager;

export namespace VulkanEngine::ShaderSystem {

// ── Draw-mode specialization constant ──────────────────────────────────────
//
// kCompactionMode (constant_id 0): 0 = CID (4 B compact index copies),
// 1 = MID (20 B DrawIndexedIndirectCommand emission). Applied to the compute
// culling pipelines and to the draw-mode-specialized *vertex* stage of the
// graphics pipelines; the fragment stage declares no constant, so the spec
// map is vertex-only for graphics descs.
//
// One writer keeps pipeline creation and the runtime draw-mode switch
// identical. DrawModeSpecValue converts the enum to the constant's payload;
// ApplyDrawModeSpec embeds it into either desc kind.

[[nodiscard]] constexpr std::uint32_t DrawModeSpecValue(
    VulkanEngine::SceneRenderer::DrawMode mode) noexcept {
    return mode == VulkanEngine::SceneRenderer::DrawMode::MID ? 1u : 0u;
}

template <typename Desc>
void ApplyDrawModeSpec(Desc& desc, std::uint32_t draw_mode_value) {
    desc.spec_entries = {
        vk::SpecializationMapEntry(0, 0, sizeof(std::uint32_t))
    };
    desc.spec_data.resize(sizeof(std::uint32_t));
    std::memcpy(desc.spec_data.data(), &draw_mode_value, sizeof(draw_mode_value));
}

// ── Declarative pipeline templates ─────────────────────────────────────────
//
// A template carries everything about a built-in pipeline that does not vary
// per shader id, layout instance, or swapchain-dependent format. The builders
// below turn a template + the varying parts into a PipelineFactory desc. The
// layout (descriptor sets + push-constant range) remains owned by the
// subsystem that binds it; the builder only references it.

struct ComputePipelineTemplate {
    std::string_view debug_name;
    std::string      entry_point{"main"};
    // NOTE: the pipeline layout (descriptor sets + push-constant range) is not
    // template state — it is owned by the subsystem that binds it and is
    // referenced by the built desc. The builder only reads the entry point.
};

struct GraphicsPipelineTemplate {
    std::string_view debug_name;
    vk::PipelineInputAssemblyStateCreateInfo  input_assembly;
    vk::PipelineRasterizationStateCreateInfo  rasterization;
    vk::PipelineMultisampleStateCreateInfo    multisample;
    vk::PipelineDepthStencilStateCreateInfo   depth_stencil;
    // Owned attachment states. Empty means color blend stays disabled with
    // zero attachments (depth-only or colorless passes). The builder copies
    // this into the desc's owned vector; PipelineFactory rebinds the
    // color_blend pointer at each creation entry, so no pointer is stored.
    std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments;
    // Entry points: the UV1 interface variants name out-of-line wrappers.
    std::string vertex_entry_point{"main"};
    std::string fragment_entry_point{"main"};
    // Pull-mode vertices plus dynamic viewport/scissor is the engine default;
    // a technique that binds real vertex buffers overrides this after building.
    std::vector<vk::DynamicState> dynamic_states = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor
    };
};

[[nodiscard]] inline ComputePipelineDesc MakeComputeDesc(
    const ComputePipelineTemplate& tmpl,
    ShaderId shader,
    vk::PipelineLayout layout) {
    ComputePipelineDesc desc{};
    desc.shader = shader;
    desc.entry_point = tmpl.entry_point;
    desc.layout = layout;
    assert(shader != 0); // ShaderId 0 is the invalid sentinel
    return desc;
}

// Formats are swapchain/render-target dependent, so they are per-call state,
// not template state: the caller reads them from the bootstrap/backend at
// creation time (and re-reads them on swapchain resize if ever needed).
[[nodiscard]] inline GraphicsPipelineDesc MakeGraphicsDesc(
    const GraphicsPipelineTemplate& tmpl,
    ShaderId vertex_shader,
    ShaderId fragment_shader,
    vk::PipelineLayout layout,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    vk::Format stencil_format = vk::Format::eUndefined) {
    assert(vertex_shader != 0);   // ShaderId 0 is the invalid sentinel
    assert(fragment_shader != 0);
    GraphicsPipelineDesc desc{};
    desc.vertex_shader = vertex_shader;
    desc.fragment_shader = fragment_shader;
    desc.vertex_entry_point = tmpl.vertex_entry_point;
    desc.fragment_entry_point = tmpl.fragment_entry_point;
    desc.vertex_input = vk::PipelineVertexInputStateCreateInfo({}, 0, nullptr, 0, nullptr);
    desc.input_assembly = tmpl.input_assembly;
    desc.viewport = vk::PipelineViewportStateCreateInfo({}, 1, nullptr, 1, nullptr);
    desc.rasterization = tmpl.rasterization;
    desc.multisample = tmpl.multisample;
    desc.depth_stencil = tmpl.depth_stencil;
    desc.color_blend = vk::PipelineColorBlendStateCreateInfo({}, false, vk::LogicOp::eCopy, 0, nullptr);
    desc.color_blend_attachments = tmpl.blend_attachments;
    desc.dynamic_states = tmpl.dynamic_states;
    desc.layout = layout;
    desc.color_formats.assign(color_formats.begin(), color_formats.end());
    desc.depth_format = depth_format;
    desc.stencil_format = stencil_format;
    return desc;
}

} // namespace VulkanEngine::ShaderSystem