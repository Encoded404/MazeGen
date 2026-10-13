module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.OccluderPrePass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

OccluderPrePass::OccluderPrePass(SceneRenderer& sr, vk::ClearDepthStencilValue clear_depth)
    : scene_renderer_(sr), clear_depth_(clear_depth) {}
OccluderPrePass::~OccluderPrePass() = default;

void OccluderPrePass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto occluder_indirect = ctx.ImportBuffer("occluder-indirect");
    auto depth_buffer = ctx.ReadDepthBuffer();
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::IndexInput,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(occluder_indirect, VulkanEngine::RenderGraph::PipelineStageIntent::IndirectDraw,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(occluder_indirect, VulkanEngine::RenderGraph::PipelineStageIntent::IndexInput,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(depth_buffer);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = true;
    VulkanEngine::RenderGraph::AttachmentInfo depth_attach{};
    depth_attach.resource = depth_buffer;
    depth_attach.load_op = vk::AttachmentLoadOp::eClear;
    depth_attach.store_op = vk::AttachmentStoreOp::eStore;
    depth_attach.clear_depth = clear_depth_;
    setup.depth_attachment = depth_attach;
    ctx.SetPassAttachments(setup);
}

void OccluderPrePass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                                vk::CommandBuffer cmd) {
    scene_renderer_.OccluderPrepass(cmd,
        ctx.render_extent.width, ctx.render_extent.height,
        ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
