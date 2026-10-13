module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.PreCullPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

PreCullPass::PreCullPass(SceneRenderer& sr) : scene_renderer_(sr) {}
PreCullPass::~PreCullPass() = default;

void PreCullPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto hiz_image = ctx.ImportImage("hiz-image");
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto depth_indirect = ctx.ImportBuffer("depth-indirect");
    ctx.AddRead(hiz_image, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(scene_buffers);
    ctx.AddWrite(depth_indirect);
}

void PreCullPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                           vk::CommandBuffer cmd) {
    scene_renderer_.DispatchPreCull(cmd, ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
