module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.ExpandPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

ExpandPass::ExpandPass(SceneRenderer& sr) : scene_renderer_(sr) {}
ExpandPass::~ExpandPass() = default;

void ExpandPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto draw_indirect = ctx.ImportBuffer("draw-indirect");
    ctx.AddWrite(scene_buffers);
    ctx.AddWrite(draw_indirect);
}

void ExpandPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                          vk::CommandBuffer cmd) {
    const std::uint32_t cnt = scene_renderer_.GetCurrentEntityCount();
    if (cnt == 0) return;
    scene_renderer_.DispatchExpand(cmd, cnt, ctx.view_proj,
                                    ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
