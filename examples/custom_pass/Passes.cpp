module;

module Examples.CustomPass.Passes;

import std;
import vulkan;
import VulkanEngine.PipelinePass;

namespace Examples::CustomPass {

namespace {

[[nodiscard]] DescriptorDecl CombinedImage(std::uint32_t binding, vk::ShaderStageFlags stages) {
    DescriptorDecl decl{};
    decl.set = kAppSet;
    decl.binding = binding;
    decl.kind = DescriptorKind::SampledImage;
    decl.descriptor_type = vk::DescriptorType::eCombinedImageSampler;
    decl.stage_flags = stages;
    decl.count = 1;
    return decl;
}

[[nodiscard]] DescriptorDecl StorageBuffer(std::uint32_t binding, vk::ShaderStageFlags stages) {
    DescriptorDecl decl{};
    decl.set = kAppSet;
    decl.binding = binding;
    decl.kind = DescriptorKind::Shared;
    decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
    decl.stage_flags = stages;
    decl.count = 1;
    return decl;
}

void BindFullscreenTriangle(const FrameContext& ctx, vk::CommandBuffer cmd,
                            vk::PipelineBindPoint bind_point) {
    cmd.bindPipeline(bind_point, ctx.pass_pipeline);
    cmd.bindDescriptorSets(bind_point, ctx.pipeline_layout, ctx.first_app_descriptor_set,
                           ctx.app_descriptor_sets, {});
    if (bind_point == vk::PipelineBindPoint::eGraphics) {
        // Each queue run gets its own command buffer, so dynamic viewport/scissor
        // state must be set per pass rather than relying on an earlier pass.
        cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(ctx.render_width),
                                        static_cast<float>(ctx.render_height), 0.0f, 1.0f});
        cmd.setScissor(0, vk::Rect2D{{0, 0}, ctx.render_extent});
    }
}

} // namespace

// ── CapturePass ──

CapturePass::CapturePass(std::uint64_t vertex_shader, std::uint64_t fragment_shader)
    : vertex_shader_(vertex_shader), fragment_shader_(fragment_shader) {}
CapturePass::~CapturePass() = default;

void CapturePass::Setup(PassSetupContext& ctx) {
    const auto backbuffer = ctx.ReadBackbuffer();
    const auto scene_color = ctx.CreateTransientImage(SceneColorDesc());

    ctx.AddRead(backbuffer, PipelineStageIntent::FragmentShader, AccessIntent::Read);
    ctx.AddWrite(scene_color);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = true;
    VulkanEngine::RenderGraph::AttachmentInfo color{};
    color.resource = scene_color;
    color.load_op = vk::AttachmentLoadOp::eClear;
    color.store_op = vk::AttachmentStoreOp::eStore;
    color.clear_color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    setup.color_attachments.push_back(color);
    ctx.SetPassAttachments(setup);

    ctx.DeclareBindings({CombinedImage(0, vk::ShaderStageFlagBits::eFragment)});
    ctx.BindResource(kAppSet, 0, backbuffer);
    ctx.RequestGraphicsPipeline(vertex_shader_, fragment_shader_);
}

void CapturePass::Execute(const FrameContext& ctx, vk::CommandBuffer cmd) {
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr) {
        return;
    }
    BindFullscreenTriangle(ctx, cmd, vk::PipelineBindPoint::eGraphics);
    cmd.draw(3, 1, 0, 0);
}

// ── ExposurePass ──

ExposurePass::ExposurePass(std::uint64_t compute_shader, bool use_async_queue)
    : compute_shader_(compute_shader), use_async_queue_(use_async_queue) {}
ExposurePass::~ExposurePass() = default;

void ExposurePass::Setup(PassSetupContext& ctx) {
    const auto scene_color = ctx.CreateTransientImage(SceneColorDesc());
    const auto exposure = ctx.CreateTransientBuffer(ExposureBufferDesc());

    ctx.AddRead(scene_color, PipelineStageIntent::ComputeShader, AccessIntent::Read);
    ctx.AddWrite(exposure);

    ctx.DeclareBindings({
        CombinedImage(0, vk::ShaderStageFlagBits::eCompute),
        StorageBuffer(1, vk::ShaderStageFlagBits::eCompute),
    });
    ctx.BindResource(kAppSet, 0, scene_color);
    ctx.BindResource(kAppSet, 1, exposure);
    ctx.RequestComputePipeline(compute_shader_);

    // Run truly async when the device exposes a dedicated compute queue. The
    // transients it touches are created with concurrent sharing by the engine.
    if (use_async_queue_) {
        ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
    }
}

void ExposurePass::Execute(const FrameContext& ctx, vk::CommandBuffer cmd) {
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr) {
        return;
    }
    BindFullscreenTriangle(ctx, cmd, vk::PipelineBindPoint::eCompute);
    cmd.dispatch(1, 1, 1);
}

// ── ToneMapPass ──

ToneMapPass::ToneMapPass(std::uint64_t vertex_shader, std::uint64_t fragment_shader)
    : vertex_shader_(vertex_shader), fragment_shader_(fragment_shader) {}
ToneMapPass::~ToneMapPass() = default;

void ToneMapPass::Setup(PassSetupContext& ctx) {
    const auto scene_color = ctx.CreateTransientImage(SceneColorDesc());
    const auto exposure = ctx.CreateTransientBuffer(ExposureBufferDesc());
    const auto backbuffer = ctx.ReadBackbuffer();

    ctx.AddRead(scene_color, PipelineStageIntent::FragmentShader, AccessIntent::Read);
    ctx.AddRead(exposure, PipelineStageIntent::FragmentShader, AccessIntent::Read);
    ctx.AddWrite(backbuffer);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = true;
    VulkanEngine::RenderGraph::AttachmentInfo color{};
    color.resource = backbuffer;
    color.load_op = vk::AttachmentLoadOp::eClear;
    color.store_op = vk::AttachmentStoreOp::eStore;
    color.clear_color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    setup.color_attachments.push_back(color);
    ctx.SetPassAttachments(setup);

    ctx.DeclareBindings({
        CombinedImage(0, vk::ShaderStageFlagBits::eFragment),
        StorageBuffer(1, vk::ShaderStageFlagBits::eFragment),
    });
    ctx.BindResource(kAppSet, 0, scene_color);
    ctx.BindResource(kAppSet, 1, exposure);
    ctx.RequestGraphicsPipeline(vertex_shader_, fragment_shader_);

    // The tonemap must run before the ImGui overlay writes the backbuffer.
    ctx.RunBefore(BuiltinPass::ImGui);
}

void ToneMapPass::Execute(const FrameContext& ctx, vk::CommandBuffer cmd) {
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr) {
        return;
    }
    BindFullscreenTriangle(ctx, cmd, vk::PipelineBindPoint::eGraphics);
    cmd.draw(3, 1, 0, 0);
}

} // namespace Examples::CustomPass
