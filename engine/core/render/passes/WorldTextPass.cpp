module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/matrix_transform.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <logging/logging_macros.hpp>

module VulkanEngine.Render.Passes.WorldTextPass;

import std;
import std.compat;

import logiface;
import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuBuffer;
import VulkanEngine.PipelinePass;
import VulkanEngine.ResourceSystem;
import VulkanEngine.Components.Text;
import VulkanEngine.Components.Transform;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GpuTextBlobBuffer;
import VulkanEngine.Text.Layout;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;

namespace VulkanEngine::SceneRenderer {

namespace {

// The bindless slot a page the GPU side has not created yet resolves to. Slot 0
// is the manager's permanent fallback, so the descriptor is always sampleable.
constexpr std::uint32_t kFallbackSlot = VulkanEngine::BindlessManager::kFallbackSlot;

[[nodiscard]] VulkanEngine::Text::TextAlign ToLayoutAlign(
    VulkanEngine::Components::TextAlign align) noexcept {
    switch (align) {
        case VulkanEngine::Components::TextAlign::Center:
            return VulkanEngine::Text::TextAlign::Center;
        case VulkanEngine::Components::TextAlign::Right:
            return VulkanEngine::Text::TextAlign::Right;
        case VulkanEngine::Components::TextAlign::Left:
            break;
    }
    return VulkanEngine::Text::TextAlign::Left;
}

} // namespace

glm::mat4 WorldModelMatrix(const VulkanEngine::Components::Transform& transform) {
    return glm::translate(glm::mat4(1.0f), transform.position) *
           glm::mat4_cast(transform.rotation) *
           glm::scale(glm::mat4(1.0f), transform.scale);
}

void BuildWorldTextInstances(VulkanEngine::Text::MsdfGenerator& generator,
                             const PageSlotLookup& page_slot,
                             const VulkanEngine::Components::Text& component,
                             const glm::mat4& model, const VulkanEngine::Text::FontFace& face,
                             const VulkanEngine::Text::ShapedRun& run,
                             std::vector<WorldTextInstance>& out,
                             const VulkanEngine::Text::MsdfConfig& config) {
    if (component.content.empty() || !(component.world_height > 0.0f) || run.Empty()) {
        return;
    }
    if (!(config.field_pixel_size > 0.0f)) {
        return;
    }
    const VulkanEngine::Text::AtlasConfig& atlas_config = generator.Atlas().Config();
    if (atlas_config.page_width == 0 || atlas_config.page_height == 0) {
        return;
    }

    // Layout runs in local units: pass the component's em size as the layout's
    // pixel size, so wrapped line widths, alignment offsets and baseline
    // advances all come out in the same units the model matrix places.
    VulkanEngine::Text::LayoutOptions options{};
    options.pixel_size = component.world_height;
    options.max_width =
        component.wrap == VulkanEngine::Components::TextWrap::Word ? component.max_width : 0.0f;
    options.align = ToLayoutAlign(component.align);
    const VulkanEngine::Text::TextLayout layout = VulkanEngine::Text::LayoutText(face, run, options);

    // Two scales meet here: field texels become local units at the ratio of the
    // component's em size to the size the field was generated at, while the
    // run's design-unit advances and offsets go straight to local units through
    // the face's size scale.
    const float field_to_local = component.world_height / config.field_pixel_size;
    const float design_to_local = face.ScaleForSize(component.world_height);
    const float inv_page_width = 1.0f / static_cast<float>(atlas_config.page_width);
    const float inv_page_height = 1.0f / static_cast<float>(atlas_config.page_height);

    for (const VulkanEngine::Text::LayoutLine& line : layout.lines) {
        float pen_x = line.offset_x;
        const float pen_y = line.advance_y;
        const std::size_t last = std::min(line.first_glyph + line.glyph_count, run.glyphs.size());
        for (std::size_t index = line.first_glyph; index < last; ++index) {
            const VulkanEngine::Text::ShapedGlyph& glyph = run.glyphs[index];
            const auto slot = generator.Generate(face, glyph.glyph_id, config);
            const auto field = generator.Get(face, glyph.glyph_id, config);
            const float glyph_x = pen_x + glyph.offset_x * design_to_local;
            // y-up local frame. The layout measures a block downward from its
            // top, so a baseline sits at -advance_y; HarfBuzz's glyph offsets are
            // already y-up and add directly. The screen-space builder works in a
            // y-down frame because screens are y-down, but the engine's world is
            // y-up (the camera's up is (0,1,0)), so reusing that frame here
            // rendered every glyph vertically mirrored.
            const float baseline_y = -pen_y + glyph.offset_y * design_to_local;

            if (slot.has_value() && field != nullptr && !field->Empty()) {
                // The field's own top-left in local units: it already includes
                // the range padding, so its corner is left/top of the ink box.
                // `field->top` is y-down, so in this y-up frame the bitmap top
                // sits *above* the baseline by its magnitude.
                const float local_left = glyph_x + field->left * field_to_local;
                const float local_top = baseline_y - field->top * field_to_local;
                const float width = static_cast<float>(field->width) * field_to_local;
                const float height = static_cast<float>(field->height) * field_to_local;

                const std::uint32_t inner_x = slot->rect.x + atlas_config.padding;
                const std::uint32_t inner_y = slot->rect.y + atlas_config.padding;

                // The model matrix turns the two local edge vectors into world
                // vectors; the shader adds them to origin so the quad follows any
                // rotation and non-uniform scale. `origin` is the field's top-left
                // and `up` runs to its bottom-left, which in this y-up frame is
                // -y, matching the atlas's top-down row order.
                const glm::vec4 origin = model * glm::vec4(local_left, local_top, 0.0f, 1.0f);
                const glm::vec4 right = model * glm::vec4(width, 0.0f, 0.0f, 0.0f);
                const glm::vec4 up = model * glm::vec4(0.0f, -height, 0.0f, 0.0f);

                WorldTextInstance instance{};
                instance.origin[0] = origin.x;
                instance.origin[1] = origin.y;
                instance.origin[2] = origin.z;
                instance.right[0] = right.x;
                instance.right[1] = right.y;
                instance.right[2] = right.z;
                instance.up[0] = up.x;
                instance.up[1] = up.y;
                instance.up[2] = up.z;
                instance.uv_min[0] = static_cast<float>(inner_x) * inv_page_width;
                instance.uv_min[1] = static_cast<float>(inner_y) * inv_page_height;
                instance.uv_max[0] =
                    static_cast<float>(inner_x + field->width) * inv_page_width;
                instance.uv_max[1] =
                    static_cast<float>(inner_y + field->height) * inv_page_height;
                for (std::size_t channel = 0; channel < 4; ++channel) {
                    instance.color[channel] = component.color[channel];
                }
                instance.px_range = field->range;
                instance.page = page_slot ? page_slot(slot->page) : kFallbackSlot;
                out.push_back(instance);
            }
            pen_x += glyph.advance_x * design_to_local;
        }
    }
}

void BuildWorldTextInstances(VulkanEngine::Text::MsdfGenerator& generator,
                             const PageSlotLookup& page_slot,
                             const VulkanEngine::Components::Text& component,
                             const VulkanEngine::Components::Transform& transform,
                             const VulkanEngine::Text::FontFace& face,
                             const VulkanEngine::Text::ShapedRun& run,
                             std::vector<WorldTextInstance>& out,
                             const VulkanEngine::Text::MsdfConfig& config) {
    BuildWorldTextInstances(generator, page_slot, component, WorldModelMatrix(transform), face, run,
                            out, config);
}

void BuildSlugTextInstances(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                            const BlobUploader& upload,
                            const VulkanEngine::Components::Text& component,
                            const glm::mat4& model, const VulkanEngine::Text::FontFace& face,
                            const VulkanEngine::Text::ShapedRun& run,
                            std::vector<SlugTextInstance>& out) {
    if (component.content.empty() || !(component.world_height > 0.0f) || run.Empty()) {
        return;
    }

    // The layout is identical to the MSDF builder's: local units, the
    // component's em size as the layout pixel size, and the same y-up frame.
    VulkanEngine::Text::LayoutOptions options{};
    options.pixel_size = component.world_height;
    options.max_width =
        component.wrap == VulkanEngine::Components::TextWrap::Word ? component.max_width : 0.0f;
    options.align = ToLayoutAlign(component.align);
    const VulkanEngine::Text::TextLayout layout = VulkanEngine::Text::LayoutText(face, run, options);

    // Design units to local units. The blob's own coordinates are design units,
    // so this is the only scale between the encoded outline and the world.
    const float design_to_local = face.ScaleForSize(component.world_height);

    for (const VulkanEngine::Text::LayoutLine& line : layout.lines) {
        float pen_x = line.offset_x;
        const float pen_y = line.advance_y;
        const std::size_t last = std::min(line.first_glyph + line.glyph_count, run.glyphs.size());
        for (std::size_t index = line.first_glyph; index < last; ++index) {
            const VulkanEngine::Text::ShapedGlyph& glyph = run.glyphs[index];
            const auto blob = encoder.Get(face, glyph.glyph_id);
            const float glyph_x = pen_x + glyph.offset_x * design_to_local;
            // Same y-up local frame the MSDF builder documents: the baseline sits
            // at -advance_y and HarfBuzz's offsets are already y-up.
            const float baseline_y = -pen_y + glyph.offset_y * design_to_local;

            if (blob != nullptr && !blob->Empty()) {
                const auto element_offset = upload(blob->bytes);
                if (element_offset.has_value()) {
                    // The ink box in design units, y-up: min_y is the bottom
                    // (y_bearing plus the negative height), max_y the top.
                    const float em_min_x = static_cast<float>(blob->x_bearing);
                    const float em_min_y =
                        static_cast<float>(blob->y_bearing + blob->height);
                    const float em_max_x =
                        static_cast<float>(blob->x_bearing + blob->width);
                    const float em_max_y = static_cast<float>(blob->y_bearing);

                    const float local_min_x = glyph_x + em_min_x * design_to_local;
                    const float local_min_y = baseline_y + em_min_y * design_to_local;
                    const float local_max_x = glyph_x + em_max_x * design_to_local;
                    const float local_max_y = baseline_y + em_max_y * design_to_local;

                    // The world basis of the em box: its minimum corner's world
                    // position and one world vector per em axis. The shader
                    // rebuilds the em-to-clip matrix from these, so any rotation
                    // or non-uniform scale carries into the quad.
                    const glm::vec4 origin =
                        model * glm::vec4(local_min_x, local_min_y, 0.0f, 1.0f);
                    const glm::vec4 right =
                        model * glm::vec4(local_max_x - local_min_x, 0.0f, 0.0f, 0.0f);
                    const glm::vec4 up =
                        model * glm::vec4(0.0f, local_max_y - local_min_y, 0.0f, 0.0f);

                    SlugTextInstance instance{};
                    instance.origin[0] = origin.x;
                    instance.origin[1] = origin.y;
                    instance.origin[2] = origin.z;
                    instance.right[0] = right.x;
                    instance.right[1] = right.y;
                    instance.right[2] = right.z;
                    instance.up[0] = up.x;
                    instance.up[1] = up.y;
                    instance.up[2] = up.z;
                    instance.em_min[0] = em_min_x;
                    instance.em_min[1] = em_min_y;
                    instance.em_max[0] = em_max_x;
                    instance.em_max[1] = em_max_y;
                    for (std::size_t channel = 0; channel < 4; ++channel) {
                        instance.color[channel] = component.color[channel];
                    }
                    instance.blob_offset = static_cast<std::uint32_t>(*element_offset);
                    out.push_back(instance);
                }
            }
            pen_x += glyph.advance_x * design_to_local;
        }
    }
}

void BuildSlugTextInstances(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                            const BlobUploader& upload,
                            const VulkanEngine::Components::Text& component,
                            const VulkanEngine::Components::Transform& transform,
                            const VulkanEngine::Text::FontFace& face,
                            const VulkanEngine::Text::ShapedRun& run,
                            std::vector<SlugTextInstance>& out) {
    BuildSlugTextInstances(encoder, upload, component, WorldModelMatrix(transform), face, run,
                           out);
}

WorldTextPass::WorldTextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap,
                             std::uint64_t vertex_shader, std::uint64_t fragment_shader)
    : WorldTextPass(bootstrap, kDefaultTextBackend, vertex_shader, fragment_shader) {}

WorldTextPass::WorldTextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap,
                             TextBackend backend, std::uint64_t vertex_shader,
                             std::uint64_t fragment_shader)
    : bootstrap_(bootstrap),
      vertex_shader_(vertex_shader),
      fragment_shader_(fragment_shader),
      backend_(backend) {
    frames_in_flight_ = bootstrap_ != nullptr ? std::max(bootstrap_->GetFramesInFlight(), 1U) : 1U;
    if (bootstrap_ == nullptr) {
        return;
    }
    // The Slug backend needs the persistent blob storage buffer the instance
    // offsets index. It lives on the pass because the pass is what draws the
    // quads; the encoder (the design-unit cache) is the reusable half.
    if (backend_ == TextBackend::Slug && !blobs_.Initialize(*bootstrap_)) {
        return;
    }
    // Host-visible and coherent: Execute writes the frame's instances straight
    // into the slot its command buffer will read, with no staging copy. The
    // stride follows the backend's instance layout.
    const std::uint64_t stride =
        backend_ == TextBackend::Slug ? sizeof(SlugTextInstance) : sizeof(WorldTextInstance);
    instance_buffers_.reserve(frames_in_flight_);
    for (std::uint32_t slot = 0; slot < frames_in_flight_; ++slot) {
        instance_buffers_.push_back(GpuBuffer::Create(
            *bootstrap_, static_cast<std::uint64_t>(kMaxInstances) * stride,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    }
}

WorldTextPass::~WorldTextPass() = default;

void WorldTextPass::Setup(PassSetupContext& ctx) {
    // World text composites into the scene: the main pass has already filled the
    // backbuffer and the depth buffer, so this pass continues that rendering area
    // and loads rather than clears both.
    auto backbuffer = ctx.ReadBackbuffer();
    auto depth_buffer = ctx.ReadDepthBuffer();
    ctx.AddWrite(backbuffer);
    // Read-only depth: the pass tests against the scene depth but never writes,
    // so the graph transitions the buffer to the read-only layout and every
    // later pass still sees the depth the main pass produced.
    ctx.AddRead(depth_buffer, VulkanEngine::RenderGraph::PipelineStageIntent::DepthAttachment,
                VulkanEngine::RenderGraph::AccessIntent::Read);

    VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
    setup.auto_begin_rendering = false;
    VulkanEngine::RenderGraph::AttachmentInfo color_attach{};
    color_attach.resource = backbuffer;
    color_attach.load_op = vk::AttachmentLoadOp::eLoad;
    color_attach.store_op = vk::AttachmentStoreOp::eStore;
    setup.color_attachments.push_back(color_attach);
    VulkanEngine::RenderGraph::AttachmentInfo depth_attach{};
    depth_attach.resource = depth_buffer;
    depth_attach.load_op = vk::AttachmentLoadOp::eLoad;
    depth_attach.store_op = vk::AttachmentStoreOp::eStore;
    setup.depth_attachment = depth_attach;
    ctx.SetPassAttachments(setup);

    // The instance buffer is engine-provided through the renderer's buffer
    // resolver; the pass only declares that it reads it in the vertex stage.
    const auto instances = ctx.ImportBuffer("world-text-instances");
    ctx.AddRead(instances, VulkanEngine::RenderGraph::PipelineStageIntent::VertexShader,
                VulkanEngine::RenderGraph::AccessIntent::Read);

    VulkanEngine::Render::DescriptorDecl decl{};
    decl.set = VulkanEngine::Render::kFirstAppDescriptorSet;
    decl.binding = 0;
    decl.kind = VulkanEngine::Render::DescriptorKind::Shared;
    decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
    decl.stage_flags = vk::ShaderStageFlagBits::eVertex;
    decl.count = 1;
    std::vector<VulkanEngine::Render::DescriptorDecl> decls{decl};

    if (backend_ == TextBackend::Slug) {
        // The Slug fragment reads the pass-owned blob storage buffer at (5, 1);
        // the instance buffer above carries the flat offsets into it.
        const auto blobs = ctx.ImportBuffer("world-text-blobs");
        ctx.AddRead(blobs, VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader,
                    VulkanEngine::RenderGraph::AccessIntent::Read);
        VulkanEngine::Render::DescriptorDecl blob_decl{};
        blob_decl.set = VulkanEngine::Render::kFirstAppDescriptorSet;
        blob_decl.binding = 1;
        blob_decl.kind = VulkanEngine::Render::DescriptorKind::Shared;
        blob_decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
        blob_decl.stage_flags = vk::ShaderStageFlagBits::eFragment;
        blob_decl.count = 1;
        decls.push_back(blob_decl);
        ctx.DeclareBindings(std::move(decls));
        ctx.BindResource(VulkanEngine::Render::kFirstAppDescriptorSet, 1, blobs);
    } else {
        ctx.DeclareBindings(std::move(decls));
    }
    ctx.BindResource(VulkanEngine::Render::kFirstAppDescriptorSet, 0, instances);

    // The camera matrix goes out in the vertex stage. The MSDF backend also
    // needs the atlas size for its range conversion; the Slug backend needs the
    // viewport size for the half-pixel edge dilation. Both blocks are 80 bytes.
    if (backend_ == TextBackend::Slug) {
        ctx.DeclarePushConstants<SlugTextConstants>(vk::ShaderStageFlagBits::eVertex |
                                                    vk::ShaderStageFlagBits::eFragment);
    } else {
        ctx.DeclarePushConstants<WorldTextConstants>(vk::ShaderStageFlagBits::eVertex |
                                                     vk::ShaderStageFlagBits::eFragment);
    }

    // Straight-alpha source-over: the fragment emits non-premultiplied colour
    // with reconstructed coverage in alpha.
    VulkanEngine::PipelinePass::PassBlendState blend{};
    blend.enable = true;
    ctx.SetBlendState(blend);

    // Depth-tested, depth-write off, the engine's less-or-equal: scene geometry
    // occludes world text, and world text never hides geometry drawn after it.
    VulkanEngine::PipelinePass::PassDepthState depth{};
    depth.test_enable = true;
    depth.write_enable = false;
    depth.compare_op = vk::CompareOp::eLessOrEqual;
    ctx.SetDepthState(depth);

    ctx.RequestGraphicsPipeline(vertex_shader_, fragment_shader_);
}

void WorldTextPass::QueueRun(VulkanEngine::Text::MsdfGenerator& generator,
                             const VulkanEngine::Text::GlyphAtlasGpu& atlas,
                             const VulkanEngine::Components::Text& component,
                             const VulkanEngine::Components::Transform& transform,
                             const VulkanEngine::Text::FontFace& face,
                             const VulkanEngine::Text::ShapedRun& run,
                             const VulkanEngine::Text::MsdfConfig& config) {
    if (backend_ != TextBackend::Msdf) {
        // Routed elsewhere: a Slug pass has no MSDF pipeline, and reinterpreting
        // MSDF instances through the Slug layout would be silent corruption.
        return;
    }
    const PageSlotLookup page_slot = [&atlas](std::uint32_t page) {
        const auto handle = atlas.PageHandle(page);
        return handle.IsValid() ? handle.slot : kFallbackSlot;
    };
    const VulkanEngine::Text::AtlasConfig& atlas_config = generator.Atlas().Config();
    atlas_width_ = static_cast<float>(atlas_config.page_width);
    atlas_height_ = static_cast<float>(atlas_config.page_height);
    BuildWorldTextInstances(generator, page_slot, component, WorldModelMatrix(transform), face, run,
                            pending_, config);
}

void WorldTextPass::QueueSlugRun(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                                 const VulkanEngine::Components::Text& component,
                                 const VulkanEngine::Components::Transform& transform,
                                 const VulkanEngine::Text::FontFace& face,
                                 const VulkanEngine::Text::ShapedRun& run,
                                 std::uint32_t recording_frame) {
    if (backend_ != TextBackend::Slug || !blobs_.IsValid()) {
        return;
    }
    // The builder stays device-free: it asks this callback where each blob went
    // and stores the element offset the shader will add to the buffer base.
    const BlobUploader upload =
        [this, recording_frame](std::span<const std::byte> bytes) -> std::optional<std::uint64_t> {
        const auto offset = blobs_.Store(bytes, recording_frame);
        if (!offset.has_value()) {
            return std::nullopt;
        }
        return blobs_.ElementOffset(*offset);
    };
    BuildSlugTextInstances(encoder, upload, component, WorldModelMatrix(transform), face, run,
                           pending_slug_);
}

vk::Buffer WorldTextPass::InstanceBufferForFrame(std::uint32_t frame_index) const {
    const std::uint32_t ring = RingForFrame(frame_index);
    if (ring >= instance_buffers_.size() || !instance_buffers_[ring].IsValid()) {
        return vk::Buffer{};
    }
    return static_cast<vk::Buffer>(*instance_buffers_[ring].GetBuffer());
}

void WorldTextPass::Execute(const FrameContext& ctx, vk::CommandBuffer cmd) {
    const bool slug = backend_ == TextBackend::Slug;
    if (slug) {
        // Once-per-frame drain of buffers a growth retired; the device is idle
        // for the slot this frame reuses, so the release is now provably safe.
        blobs_.Collect(ctx.frame_index);
    }
    const std::size_t queued = slug ? pending_slug_.size() : pending_.size();
    if (queued == 0) {
        return; // nothing queued: draw nothing, and open no rendering area
    }
    if (ctx.pass_pipeline == nullptr || ctx.pipeline_layout == nullptr || bootstrap_ == nullptr) {
        return;
    }
    // The pass is depth-tested by construction; without a depth target there is
    // no correct rendering scope to open, so draw nothing rather than an
    // invalid one.
    if (ctx.depth_buffer.view == nullptr) {
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
    const std::uint32_t count =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(queued), kMaxInstances);
    if (queued > kMaxInstances) {
        LOGIFACE_LOG(warn, "WorldTextPass: dropping " + std::to_string(queued - kMaxInstances) +
                               " glyph instances over the per-frame budget");
    }
    if (slug) {
        instance_buffers_[ring].UploadAt(
            pending_slug_.data(), static_cast<std::uint64_t>(count) * sizeof(SlugTextInstance), 0);
    } else {
        instance_buffers_[ring].UploadAt(
            pending_.data(), static_cast<std::uint64_t>(count) * sizeof(WorldTextInstance), 0);

        // The MSDF atlas pages the queued instances sample are copied first, so
        // the draw below is ordered after the transfer in this same command
        // buffer. The Slug backend has no atlas: its blobs are already in the
        // buffer the pass owns.
        if (pre_record_) {
            pre_record_(cmd, ctx.frame_index);
        }
    }

    // Setup() declares auto_begin_rendering = false, like the screen-space text
    // overlay: the engine's executor only opens a dynamic-rendering scope around
    // passes that ask for one, so a pass that issues its own draws owns that
    // scope. The depth attachment is bound read-only, matching the layout the
    // graph transitions it to for this pass.
    vk::RenderingAttachmentInfo color{};
    color.imageView = *views[ctx.swapchain_image_index];
    color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    color.loadOp = vk::AttachmentLoadOp::eLoad;
    color.storeOp = vk::AttachmentStoreOp::eStore;
    vk::RenderingAttachmentInfo depth{};
    depth.imageView = ctx.depth_buffer.view;
    depth.imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal;
    depth.loadOp = vk::AttachmentLoadOp::eLoad;
    depth.storeOp = vk::AttachmentStoreOp::eStore;
    vk::RenderingInfo rendering{};
    rendering.renderArea = vk::Rect2D{{0, 0}, ctx.render_extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    rendering.pDepthAttachment = &depth;
    cmd.beginRendering(rendering);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, ctx.pass_pipeline);
    // Engine set 0 is the bindless array the MSDF fragment samples its atlas page
    // from; the Slug fragment uses none of the engine sets. Both backends bind it
    // anyway because the engine prefixes every pass pipeline layout with its five
    // set layouts, and binding the app set (5) alone would leave the pipeline's
    // statically used set 0 unbound.
    const std::array<vk::DescriptorSet, 1> bindless{ctx.bindless_textures.handle};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ctx.pipeline_layout, 0, bindless, {});
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ctx.pipeline_layout,
                           ctx.first_app_descriptor_set, ctx.app_descriptor_sets, {});
    if (slug) {
        SlugTextConstants constants{};
        // Raw glm memory: the engine's shaders read matrices in the row-vector
        // convention mul(float4(p, 1), m), which is what this layout provides.
        std::memcpy(constants.view_proj, &ctx.view_proj, sizeof(constants.view_proj));
        // The viewport size is what hb_gpu_dilate turns into half a screen pixel
        // of edge dilation.
        constants.viewport[0] = static_cast<float>(ctx.render_width);
        constants.viewport[1] = static_cast<float>(ctx.render_height);
        static_cast<void>(ctx.SetPushConstants(cmd, constants));
    } else {
        WorldTextConstants constants{};
        std::memcpy(constants.view_proj, &ctx.view_proj, sizeof(constants.view_proj));
        constants.atlas_size[0] = atlas_width_;
        constants.atlas_size[1] = atlas_height_;
        static_cast<void>(ctx.SetPushConstants(cmd, constants));
    }
    // Each queue run records into its own command buffer, so the dynamic state
    // cannot be inherited from the scene passes.
    //
    // World text is scene content: it is placed in the y-up world and projected
    // by the camera's view_proj, so it draws through the same y-mirroring viewport
    // the main and depth passes use. The screen-space overlay's unflipped viewport
    // would flip every glyph vertically (upside down), which is the bug this
    // shared definition exists to prevent.
    cmd.setViewport(0, VulkanEngine::PipelinePass::SceneViewport(ctx.render_width,
                                                                 ctx.render_height));
    cmd.setScissor(0, vk::Rect2D{{0, 0}, ctx.render_extent});
    cmd.draw(6, count, 0, 0);
    cmd.endRendering();

    // The queue is consumed by the frame that draws it.
    pending_.clear();
    pending_slug_.clear();
}

} // namespace VulkanEngine::SceneRenderer
