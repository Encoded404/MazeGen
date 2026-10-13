module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.TextPass;

import std;
import std.compat;

import logiface;
import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.PipelinePass;
import VulkanEngine.ResourceSystem;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.Shaping;

namespace VulkanEngine::SceneRenderer {

namespace {

// The bindless slot a page the GPU side has not created yet resolves to. Slot 0
// is the manager's permanent fallback, so the descriptor is always sampleable.
constexpr std::uint32_t kFallbackSlot = VulkanEngine::BindlessManager::kFallbackSlot;

} // namespace

void BuildTextInstances(VulkanEngine::Text::GlyphRasterizer& rasterizer,
                        const PageSlotLookup& page_slot, const TextRunRequest& request,
                        std::vector<TextInstance>& out) {
    if (request.face == nullptr || request.run == nullptr || request.pixel_size <= 0.0f) {
        return;
    }
    const VulkanEngine::Text::AtlasConfig& config = rasterizer.Atlas().Config();
    if (config.page_width == 0 || config.page_height == 0) {
        return;
    }
    const float scale = request.face->ScaleForSize(request.pixel_size);
    const float inv_page_width = 1.0f / static_cast<float>(config.page_width);
    const float inv_page_height = 1.0f / static_cast<float>(config.page_height);

    // The pen advances by the run's own scaled advances; only the placement is
    // rounded, so a long run does not accumulate a rounding error.
    float pen_x = request.x;
    for (const VulkanEngine::Text::ShapedGlyph& glyph : request.run->glyphs) {
        const float glyph_x = std::round(pen_x + glyph.offset_x * scale);
        const float glyph_y = std::round(request.y - glyph.offset_y * scale);
        auto bitmap = rasterizer.Get(*request.face, glyph.glyph_id, request.pixel_size,
                                     request.hinting);
        const auto slot = rasterizer.Rasterize(*request.face, glyph.glyph_id,
                                               request.pixel_size, request.hinting);
        if (bitmap != nullptr && slot.has_value() && bitmap->width > 0 && bitmap->height > 0) {
            const std::uint32_t inner_x = slot->rect.x + config.padding;
            const std::uint32_t inner_y = slot->rect.y + config.padding;

            TextInstance instance{};
            instance.position[0] = glyph_x + static_cast<float>(bitmap->left);
            instance.position[1] = glyph_y + static_cast<float>(bitmap->top);
            instance.size[0] = static_cast<float>(bitmap->width);
            instance.size[1] = static_cast<float>(bitmap->height);
            instance.uv_min[0] = static_cast<float>(inner_x) * inv_page_width;
            instance.uv_min[1] = static_cast<float>(inner_y) * inv_page_height;
            instance.uv_max[0] =
                static_cast<float>(inner_x + bitmap->width) * inv_page_width;
            instance.uv_max[1] =
                static_cast<float>(inner_y + bitmap->height) * inv_page_height;
            instance.color[0] = request.color[0];
            instance.color[1] = request.color[1];
            instance.color[2] = request.color[2];
            instance.color[3] = request.color[3];
            instance.page = page_slot ? page_slot(slot->page) : kFallbackSlot;
            out.push_back(instance);
        }
        pen_x += glyph.advance_x * scale;
    }
}

TextPass::TextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap,
                   std::uint64_t vertex_shader, std::uint64_t fragment_shader)
    : bootstrap_(bootstrap), vertex_shader_(vertex_shader), fragment_shader_(fragment_shader) {
    frames_in_flight_ = bootstrap_ != nullptr ? std::max(bootstrap_->GetFramesInFlight(), 1U) : 1U;
    if (bootstrap_ == nullptr) {
        return;
    }
    // Host-visible and coherent: Execute writes the frame's instances straight
    // into the slot its command buffer will read, with no staging copy.
    instance_buffers_.reserve(frames_in_flight_);
    for (std::uint32_t slot = 0; slot < frames_in_flight_; ++slot) {
        instance_buffers_.push_back(GpuBuffer::Create(
            *bootstrap_, static_cast<std::uint64_t>(kMaxInstances) * sizeof(TextInstance),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    }
}

TextPass::~TextPass() = default;

void TextPass::Setup(PassSetupContext& ctx) {
    // Text composites over the scene: the main pass has already begun rendering
    // and filled the backbuffer, so this pass continues that rendering area and
    // loads rather than clears it.
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

    // The instance buffer is engine-provided through the renderer's buffer
    // resolver; the pass only declares that it reads it in the vertex stage.
    const auto instances = ctx.ImportBuffer("text-instances");
    ctx.AddRead(instances, VulkanEngine::RenderGraph::PipelineStageIntent::VertexShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);

    VulkanEngine::Render::DescriptorDecl decl{};
    decl.set = VulkanEngine::Render::kFirstAppDescriptorSet;
    decl.binding = 0;
    decl.kind = VulkanEngine::Render::DescriptorKind::Shared;
    decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
    decl.stage_flags = vk::ShaderStageFlagBits::eVertex;
    decl.count = 1;
    ctx.DeclareBindings({decl});
    ctx.BindResource(VulkanEngine::Render::kFirstAppDescriptorSet, 0, instances);

    ctx.DeclarePushConstants<std::array<float, 2>>(vk::ShaderStageFlagBits::eVertex);

    // Straight-alpha source-over: the fragment emits non-premultiplied colour
    // with the atlas coverage in alpha, so colour is srcAlpha over
    // oneMinusSrcAlpha and alpha is one over oneMinusSrcAlpha.
    VulkanEngine::PipelinePass::PassBlendState blend{};
    blend.enable = true;
    ctx.SetBlendState(blend);

    ctx.RequestGraphicsPipeline(vertex_shader_, fragment_shader_);
}

void TextPass::QueueRun(VulkanEngine::Text::GlyphRasterizer& rasterizer,
                        const VulkanEngine::Text::GlyphAtlasGpu& atlas,
                        const TextRunRequest& request) {
    const PageSlotLookup page_slot = [&atlas](std::uint32_t page) {
        const auto handle = atlas.PageHandle(page);
        return handle.IsValid() ? handle.slot : kFallbackSlot;
    };
    BuildTextInstances(rasterizer, page_slot, request, pending_);
}

vk::Buffer TextPass::InstanceBufferForFrame(std::uint32_t frame_index) const {
    const std::uint32_t ring = RingForFrame(frame_index);
    if (ring >= instance_buffers_.size() || !instance_buffers_[ring].IsValid()) {
        return vk::Buffer{};
    }
    return static_cast<vk::Buffer>(*instance_buffers_[ring].GetBuffer());
}

void TextPass::Execute(const FrameContext& ctx, vk::CommandBuffer cmd) {
    if (pending_.empty()) {
        return; // nothing queued: draw nothing, and open no rendering area
    }
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr || bootstrap_ == nullptr) {
        return;
    }
    const auto& views = bootstrap_->GetSwapchainImageViews();
    if (ctx.swapchain_image_index >= views.size()) {
        return;
    }
    const std::uint32_t ring = RingForFrame(ctx.frame_index);
    if (ring >= instance_buffers_.size() || !instance_buffers_[ring].IsValid()) {
        return;
    }
    const std::uint32_t count = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(pending_.size()), kMaxInstances);
    if (static_cast<std::uint32_t>(pending_.size()) > kMaxInstances) {
        LOGIFACE_LOG(warn, "TextPass: dropping " +
                               std::to_string(pending_.size() - kMaxInstances) +
                               " glyph instances over the per-frame budget");
    }
    instance_buffers_[ring].UploadAt(pending_.data(),
                                     static_cast<std::uint64_t>(count) * sizeof(TextInstance), 0);

    // The atlas pages the queued instances sample are copied first, so the draw
    // below is ordered after the transfer in this same command buffer. This is
    // the only place the pass reaches outside its own data, and it is a hook:
    // the pass still owns no atlas and no face.
    if (pre_record_) {
        pre_record_(cmd, ctx.frame_index);
    }

    // Setup() declares auto_begin_rendering = false, like the ImGui overlay:
    // the engine's executor only opens a dynamic-rendering scope around passes
    // that ask for one, so a pass that issues its own draws owns that scope. The
    // declared eLoad attachment is what tells the graph the target content is
    // carried in, and this is where that load actually happens.
    vk::RenderingAttachmentInfo color{};
    color.imageView = *views[ctx.swapchain_image_index];
    color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    color.loadOp = vk::AttachmentLoadOp::eLoad;
    color.storeOp = vk::AttachmentStoreOp::eStore;
    vk::RenderingInfo rendering{};
    rendering.renderArea = vk::Rect2D{{0, 0}, ctx.render_extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    cmd.beginRendering(rendering);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, ctx.pass_pipeline);
    // The fragment samples the glyph's atlas page from the engine's bindless
    // array, which is set 0 of every engine-built pass layout, and the vertex
    // reads the pass's own instance buffer from its app set (5). Both are bound
    // explicitly: the engine prefixes each pass pipeline layout with its five set
    // layouts, and a pass that statically uses one has to bind it.
    const std::array<vk::DescriptorSet, 1> bindless{ctx.bindless_textures.handle};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ctx.pipeline_layout, 0, bindless, {});
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ctx.pipeline_layout,
                           ctx.first_app_descriptor_set, ctx.app_descriptor_sets, {});
    const std::array<float, 2> screen_size{static_cast<float>(ctx.render_width),
                                           static_cast<float>(ctx.render_height)};
    static_cast<void>(ctx.SetPushConstants(cmd, screen_size));
    // Each queue run records into its own command buffer, so the dynamic state
    // cannot be inherited from the scene passes.
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(ctx.render_width),
                                    static_cast<float>(ctx.render_height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D{{0, 0}, ctx.render_extent});
    cmd.draw(6, count, 0, 0);
    cmd.endRendering();

    // The queue is consumed by the frame that draws it.
    pending_.clear();
}

} // namespace VulkanEngine::SceneRenderer
