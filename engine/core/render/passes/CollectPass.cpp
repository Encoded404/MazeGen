module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.CollectPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

CollectPass::CollectPass(SceneRenderer& sr) : scene_renderer_(sr) {}
CollectPass::~CollectPass() = default;

void CollectPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto draw_indirect = ctx.ImportBuffer("draw-indirect");
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(scene_buffers);
    ctx.AddWrite(draw_indirect);
}

void CollectPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                           vk::CommandBuffer cmd) {
    scene_renderer_.DispatchCollect(cmd, ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
