module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.MainPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

namespace VulkanEngine::SceneRenderer {

MainPass::MainPass(SceneRenderer& sr, glm::vec4 clear_color)
    : scene_renderer_(sr), clear_color_(clear_color) {}
MainPass::~MainPass() = default;

void MainPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto scene_buffers = ctx.ImportBuffer("scene-buffers");
    auto draw_indirect = ctx.ImportBuffer("draw-indirect");
    auto backbuffer = ctx.ReadBackbuffer();
    auto depth_buffer = ctx.ReadDepthBuffer();
    ctx.AddRead(scene_buffers, VulkanEngine::RenderGraph::PipelineStageIntent::IndexInput,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(draw_indirect, VulkanEngine::RenderGraph::PipelineStageIntent::IndirectDraw,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(draw_indirect, VulkanEngine::RenderGraph::PipelineStageIntent::IndexInput,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(backbuffer);
    ctx.AddWrite(depth_buffer);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = true;

    VulkanEngine::RenderGraph::AttachmentInfo color_attach{};
    color_attach.resource = backbuffer;
    color_attach.load_op = vk::AttachmentLoadOp::eClear;
    color_attach.store_op = vk::AttachmentStoreOp::eStore;
    color_attach.clear_color = vk::ClearColorValue(std::array<float, 4>{
        clear_color_.r, clear_color_.g, clear_color_.b, clear_color_.a});
    setup.color_attachments.push_back(color_attach);

    VulkanEngine::RenderGraph::AttachmentInfo depth_attach{};
    depth_attach.resource = depth_buffer;
    depth_attach.load_op = vk::AttachmentLoadOp::eLoad;
    depth_attach.store_op = vk::AttachmentStoreOp::eStore;
    setup.depth_attachment = depth_attach;

    ctx.SetPassAttachments(setup);
}

void MainPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                       vk::CommandBuffer cmd) {
    if (ctx.registry == nullptr || ctx.techniques == nullptr || ctx.bindless == nullptr) {
        return;
    }
    scene_renderer_.Render(cmd, *ctx.registry, *ctx.techniques, *ctx.bindless,
                           ctx.proj, ctx.view,
                           ctx.render_width, ctx.render_height, ctx.frame_index);
}

} // namespace VulkanEngine::SceneRenderer
