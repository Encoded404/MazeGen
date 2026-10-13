module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.ImGuiPass;

import std;
import logiface;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.ImGui;

namespace VulkanEngine::SceneRenderer {

ImGuiPass::ImGuiPass(VulkanBackend::Vulkan::VulkanBootstrap* bootstrap)
    : bootstrap_(bootstrap) {}
ImGuiPass::~ImGuiPass() = default;

void ImGuiPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    auto backbuffer = ctx.ReadBackbuffer();
    ctx.AddWrite(backbuffer);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = false;
    VulkanEngine::RenderGraph::AttachmentInfo color_attach{};
    color_attach.resource = backbuffer;
    color_attach.load_op = vk::AttachmentLoadOp::eLoad;
    color_attach.store_op = vk::AttachmentStoreOp::eStore;
    setup.color_attachments.push_back(color_attach);
    ctx.SetPassAttachments(setup);
}

void ImGuiPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                        vk::CommandBuffer cmd) {
    if (ctx.imgui == nullptr || bootstrap_ == nullptr || !ctx.imgui->IsInitialized()) {
        return;
    }
    auto& backend = bootstrap_->GetBackend();
    const auto& views = backend.GetSwapchainImageViews();
    if (ctx.swapchain_image_index >= views.size()) {
        return;
    }
    ctx.imgui->RenderDrawData(cmd, *views[ctx.swapchain_image_index],
                              ctx.render_width, ctx.render_height);
}

} // namespace VulkanEngine::SceneRenderer
