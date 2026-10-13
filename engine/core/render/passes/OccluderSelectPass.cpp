module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.OccluderSelectPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

OccluderSelectPass::OccluderSelectPass(SceneRenderer& sr) : scene_renderer_(sr) {}
OccluderSelectPass::~OccluderSelectPass() = default;

void OccluderSelectPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto occluder_indirect = ctx.ImportBuffer("occluder-indirect");
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(scene_buffers);
    ctx.AddWrite(occluder_indirect);
}

void OccluderSelectPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                                  vk::CommandBuffer cmd) {
    scene_renderer_.DispatchOccluderSelect(cmd,
        ctx.render_extent.width, ctx.render_extent.height,
        ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
