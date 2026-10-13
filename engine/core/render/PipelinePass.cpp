module;

module VulkanEngine.PipelinePass;

import std;
import std.compat;

import vulkan;

import VulkanEngine.RenderGraph;
import VulkanBackend.Vulkan.VulkanBootstrap;

namespace VulkanEngine::PipelinePass {

// ── PassSetupContext implementation ──

PassSetupContext::PassSetupContext(IResourceRegistry& registry,
                                   std::uint32_t render_width,
                                   std::uint32_t render_height)
    : registry_(&registry), render_width_(render_width), render_height_(render_height) {}

void PassSetupContext::RunBefore(BuiltinPass pass) {
    before_builtin_passes_.push_back(pass);
}

void PassSetupContext::RunAfter(BuiltinPass pass) {
    after_builtin_passes_.push_back(pass);
}

void PassSetupContext::RequestGraphicsPipeline(std::uint64_t vertex_shader,
                                               std::uint64_t fragment_shader,
                                               std::vector<vk::Format> color_formats,
                                               vk::Format depth_format) {
    pipeline_request_.kind = PassPipelineKind::Graphics;
    pipeline_request_.vertex_shader = vertex_shader;
    pipeline_request_.fragment_shader = fragment_shader;
    pipeline_request_.color_formats = std::move(color_formats);
    pipeline_request_.depth_format = depth_format;
}

void PassSetupContext::RequestComputePipeline(std::uint64_t compute_shader) {
    pipeline_request_.kind = PassPipelineKind::Compute;
    pipeline_request_.compute_shader = compute_shader;
}

void PassSetupContext::DeclareBindings(std::vector<VulkanEngine::Render::DescriptorDecl> bindings) {
    for (auto& binding : bindings) {
        declared_bindings_.push_back(std::move(binding));
    }
}

void PassSetupContext::BindResource(std::uint32_t set, std::uint32_t binding,
                                    VulkanEngine::RenderGraph::ResourceHandle resource) {
    binding_assignments_.push_back(BindingAssignment{
        .set = set,
        .binding = binding,
        .resource = resource,
    });
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::ReadDepthBuffer() {
    return registry_->ImportDepthBuffer();
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::ReadBackbuffer() {
    return registry_->ImportBackbuffer();
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::ImportImage(std::string_view name) {
    return registry_->ImportImage(std::string(name));
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::ImportBuffer(std::string_view name) {
    return registry_->ImportBuffer(std::string(name));
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::CreateTransientImage(
    const TransientImageDesc& desc) {
    return registry_->CreateTransientImage(desc);
}

VulkanEngine::RenderGraph::ResourceHandle PassSetupContext::CreateTransientBuffer(
    const TransientBufferDesc& desc) {
    return registry_->CreateTransientBuffer(desc);
}

void PassSetupContext::AddRead(VulkanEngine::RenderGraph::ResourceHandle res,
                                VulkanEngine::RenderGraph::PipelineStageIntent stage,
                                VulkanEngine::RenderGraph::AccessIntent access) {
    read_resources_.push_back(res);
    read_stages_.push_back(stage);
    read_accesses_.push_back(access);
}

void PassSetupContext::AddWrite(VulkanEngine::RenderGraph::ResourceHandle res) {
    write_resources_.push_back(res);
}

void PassSetupContext::SetPassAttachments(VulkanEngine::RenderGraph::PassAttachmentSetup setup) {
    attachment_setup_ = std::move(setup);
}

std::uint32_t PassSetupContext::GetRenderWidth() const {
    return render_width_;
}

std::uint32_t PassSetupContext::GetRenderHeight() const {
    return render_height_;
}

// ── PassResource/FrameContext are header-inline ──

} // namespace VulkanEngine::PipelinePass
