module;

module MazeGenerator.MazeRenderPass;

import std;
import vulkan;

import VulkanEngine.PipelinePass;
import VulkanEngine.RenderGraph;

namespace MazeGenerator {

namespace {

constexpr std::uint32_t kAppSet = 5;

[[nodiscard]] VulkanEngine::Render::DescriptorDecl StorageBuffer(std::uint32_t binding,
                                                                 vk::ShaderStageFlags stages) {
    VulkanEngine::Render::DescriptorDecl decl{};
    decl.set = kAppSet;
    decl.binding = binding;
    decl.kind = VulkanEngine::Render::DescriptorKind::Shared;
    decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
    decl.stage_flags = stages;
    decl.count = 1;
    return decl;
}

} // namespace

MazeRenderPass::MazeRenderPass(std::uint64_t vertex_shader, std::uint64_t fragment_shader)
    : vertex_shader_(vertex_shader), fragment_shader_(fragment_shader) {}
MazeRenderPass::~MazeRenderPass() = default;

void MazeRenderPass::Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) {
    const auto maze_bits = ctx.ImportBuffer("maze-bits");
    const auto maze_density = ctx.ImportBuffer("maze-density");
    const auto backbuffer = ctx.ReadBackbuffer();

    ctx.AddRead(maze_bits, VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddRead(maze_density, VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);
    ctx.AddWrite(backbuffer);

    VulkanEngine::RenderGraph::PassAttachmentSetup attachments{};
    attachments.auto_begin_rendering = true;
    VulkanEngine::RenderGraph::AttachmentInfo color{};
    color.resource = backbuffer;
    color.load_op = vk::AttachmentLoadOp::eClear;
    color.store_op = vk::AttachmentStoreOp::eStore;
    color.clear_color = vk::ClearColorValue(std::array<float, 4>{0.03f, 0.03f, 0.05f, 1.0f});
    attachments.color_attachments.push_back(color);
    ctx.SetPassAttachments(attachments);

    ctx.DeclareBindings({
        StorageBuffer(0, vk::ShaderStageFlagBits::eFragment),
        StorageBuffer(1, vk::ShaderStageFlagBits::eFragment),
    });
    ctx.BindResource(kAppSet, 0, maze_bits);
    ctx.BindResource(kAppSet, 1, maze_density);

    ctx.RequestGraphicsPipeline(vertex_shader_, fragment_shader_);
    ctx.DeclarePushConstants<MazeConstants>(vk::ShaderStageFlagBits::eFragment);
    ctx.RunAfter(VulkanEngine::PipelinePass::BuiltinPass::MainPass);
}

void MazeRenderPass::Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                             vk::CommandBuffer cmd) {
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr) {
        return;
    }

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, ctx.pass_pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ctx.pipeline_layout,
                           ctx.first_app_descriptor_set, ctx.app_descriptor_sets, {});
    cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(ctx.render_width),
                                    static_cast<float>(ctx.render_height), 0.0f, 1.0f});
    cmd.setScissor(0, vk::Rect2D{{0, 0}, ctx.render_extent});
    static_cast<void>(ctx.SetPushConstants(cmd, constants_));
    cmd.draw(3, 1, 0, 0);
}

} // namespace MazeGenerator
