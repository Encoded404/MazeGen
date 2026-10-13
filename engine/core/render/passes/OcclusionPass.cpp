module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.OcclusionPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

OcclusionPass::OcclusionPass(SceneRenderer& sr) : scene_renderer_(sr) {}
OcclusionPass::~OcclusionPass() = default;

void OcclusionPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto hiz_image = ctx.ImportImage("hiz-image");
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    ctx.AddRead(hiz_image, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(scene_buffers);
}

void OcclusionPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                             vk::CommandBuffer cmd) {
    scene_renderer_.DispatchOcclusion(cmd, ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
