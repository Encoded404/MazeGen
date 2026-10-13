module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.HiZPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

HiZPass::HiZPass(SceneRenderer& sr, std::string name)
    : scene_renderer_(sr), name_(std::move(name)) {}
HiZPass::~HiZPass() = default;

void HiZPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto depth_buffer = ctx.ReadDepthBuffer();
    auto hiz_image = ctx.ImportImage("hiz-image");
    ctx.AddRead(depth_buffer, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(hiz_image);
}

void HiZPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                       vk::CommandBuffer cmd) {
    scene_renderer_.DispatchHiZGen(cmd,
        ctx.render_extent.width, ctx.render_extent.height,
        ctx.frame_index, ctx.swapchain_image_index);
}

} // namespace VulkanEngine::SceneRenderer
