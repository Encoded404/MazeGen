module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/matrix_transform.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Render.Passes.WorldTextPass;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.PipelinePass;
import VulkanEngine.Components.Text;
import VulkanEngine.Components.Transform;
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GpuTextBlobBuffer;
import VulkanEngine.Text.Layout;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;

export namespace VulkanEngine::SceneRenderer {

using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::PipelinePass::FrameContext;
using VulkanEngine::PipelinePass::IPipelinePass;
using VulkanEngine::PipelinePass::PassSetupContext;

// Which rasterizer the world-text pass draws with. The MSDF backend samples a
// generated distance field from a bindless atlas page and is the default; the
// Slug backend evaluates the encoded outline analytically. The selector lives
// here because a pass owns exactly one graphics pipeline, so the choice is made
// when the pass is constructed (RendererConfig::text_backend) rather than per
// glyph.
enum class TextBackend : std::uint8_t {
    Msdf,
    Slug,
};

inline constexpr TextBackend kDefaultTextBackend = TextBackend::Msdf;

// One world-space glyph quad. Mirrors engine/core/shaders/world_text.slang
// exactly, C data layout, 80 bytes: a world-space basis (origin plus one vector
// per uv axis, so the quad follows any rotation and scale), the field's uv rect,
// colour, the distance range the field was generated with, and the bindless page
// slot.
struct WorldTextInstance {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    float origin[3]{0.0f, 0.0f, 0.0f}; // world position of the uv (0,0) corner
    float right[3]{1.0f, 0.0f, 0.0f};  // world-space vector to the uv (1,0) corner
    float up[3]{0.0f, 1.0f, 0.0f};     // world-space vector to the uv (0,1) corner
    float uv_min[2]{0.0f, 0.0f};       // inner (field) region of the atlas page
    float uv_max[2]{1.0f, 1.0f};
    float color[4]{1.0f, 1.0f, 1.0f, 1.0f}; // straight alpha
    float px_range = 4.0f;             // the range the field was generated with
    std::uint32_t page = 0;            // bindless slot of the MSDF page
    std::uint32_t padding = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};
static_assert(sizeof(WorldTextInstance) == 80,
              "WorldTextInstance must mirror the world_text shader layout");

// Camera and atlas size pushed for one draw. The matrix is the engine's
// row-vector convention: the shader multiplies it on the right of a row vector,
// mul(float4(p, 1), m), exactly like every other engine shader.
struct WorldTextConstants {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    float view_proj[16]{};
    float atlas_size[2]{1.0f, 1.0f};
    float padding[2]{0.0f, 0.0f};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};
static_assert(sizeof(WorldTextConstants) == 80,
              "WorldTextConstants must mirror the world_text push constant block");

// One world-space Slug glyph quad. Mirrors engine/core/shaders/slug_text.slang
// exactly, C data layout, 76 bytes: the em box's world basis (the world position
// of its minimum corner plus one vector per em axis, so the quad follows any
// rotation and scale), the box in em/design units, colour, and the flat offset
// of the glyph's blob in the blob storage buffer. There is no uv rect, no atlas
// page and no distance range -- those are the MSDF backend's fields.
struct SlugTextInstance {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    float origin[3]{0.0f, 0.0f, 0.0f}; // world position of the em (min_x, min_y) corner
    float right[3]{1.0f, 0.0f, 0.0f};  // world vector from min_x to max_x
    float up[3]{0.0f, 1.0f, 0.0f};     // world vector from min_y to max_y
    float em_min[2]{0.0f, 0.0f};       // ink box minimum, in design units
    float em_max[2]{1.0f, 1.0f};       // ink box maximum, in design units
    float color[4]{1.0f, 1.0f, 1.0f, 1.0f}; // straight alpha
    // Index of the glyph's first 8-byte unit in GpuTextBlobBuffer.
    std::uint32_t blob_offset = 0;
    std::uint32_t padding = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};
static_assert(sizeof(SlugTextInstance) == 76,
              "SlugTextInstance must mirror the slug_text shader layout");

// Camera and viewport pushed for one Slug draw. The matrix is the engine's
// row-vector convention like every other pass; the viewport size is what
// hb_gpu_dilate turns the half-pixel edge dilation into screen pixels.
struct SlugTextConstants {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    float view_proj[16]{};
    float viewport[2]{1.0f, 1.0f};
    float padding[2]{0.0f, 0.0f};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};
static_assert(sizeof(SlugTextConstants) == 80,
              "SlugTextConstants must mirror the slug_text push constant block");

// Reserves a glyph blob in the GPU buffer and returns the element (8-byte unit)
// offset the shader will read it at, or nullopt when the blob cannot be placed.
// Kept as a callback so the instance builder is device-free: a test can supply a
// CPU allocator and assert the offsets a run of glyphs produces.
using BlobUploader = std::function<std::optional<std::uint64_t>(std::span<const std::byte>)>;

// Maps an atlas page index to the bindless slot its page image lives in. TextPass
// has the same alias for the FreeType atlas; the two passes own different page
// formats, so neither borrows the other's pass just for the alias.
using PageSlotLookup = std::function<std::uint32_t(std::uint32_t)>;

// The local-to-world matrix of a Transform: translate * rotation * scale,
// column-vector glm math, which is what the engine uses everywhere else.
[[nodiscard]] glm::mat4 WorldModelMatrix(const VulkanEngine::Components::Transform& transform);

// Turns one Text component plus its world transform into world-space glyph quads
// using the MSDF store.
//
// The component is the source of truth for the string, the em size, the colour
// and the alignment/wrap; `face` and `run` are the caller's resolution of
// `component.font_id` and `component.content` (the component names a font and a
// string, not a face or a shaped run), so this stays a pure transform of shaped
// glyphs into world space. Layout runs in *local* units with the component's
// world_height as the pixel size, so a line's wrap width, alignment offsets and
// advances are already world units and the model matrix places the whole block.
//
// Each drawable glyph contributes one quad: the field's top-left corner in local
// space, the two world-space edge vectors from the model matrix, the atlas slot's
// inner region as the uv rect, and the range the field was generated with. A
// glyph with no field (a space, a missing glyph, a size the atlas cannot hold)
// advances the pen and contributes no quad.
void BuildWorldTextInstances(VulkanEngine::Text::MsdfGenerator& generator,
                             const PageSlotLookup& page_slot,
                             const VulkanEngine::Components::Text& component,
                             const glm::mat4& model, const VulkanEngine::Text::FontFace& face,
                             const VulkanEngine::Text::ShapedRun& run,
                             std::vector<WorldTextInstance>& out,
                             const VulkanEngine::Text::MsdfConfig& config = {});

// Convenience overload: takes the entity's Transform and builds its matrix.
void BuildWorldTextInstances(VulkanEngine::Text::MsdfGenerator& generator,
                             const PageSlotLookup& page_slot,
                             const VulkanEngine::Components::Text& component,
                             const VulkanEngine::Components::Transform& transform,
                             const VulkanEngine::Text::FontFace& face,
                             const VulkanEngine::Text::ShapedRun& run,
                             std::vector<WorldTextInstance>& out,
                             const VulkanEngine::Text::MsdfConfig& config = {});

// Turns one Text component plus its world transform into Slug glyph quads.
//
// Same layout and same local-to-world placement as the MSDF builder -- including
// the y-up local frame -- but each drawable glyph resolves to an encoded
// hb-gpu blob instead of a distance field. The blob is in font design units and
// size independent, so `encoder` is asked once per glyph with no size and the
// instance carries the blob's flat buffer offset instead of an atlas uv rect;
// `upload` places the bytes and hands back that offset. A glyph with no ink (a
// space) advances the pen and contributes no quad, exactly like an empty field.
void BuildSlugTextInstances(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                            const BlobUploader& upload,
                            const VulkanEngine::Components::Text& component,
                            const glm::mat4& model, const VulkanEngine::Text::FontFace& face,
                            const VulkanEngine::Text::ShapedRun& run,
                            std::vector<SlugTextInstance>& out);

// Convenience overload: takes the entity's Transform and builds its matrix.
void BuildSlugTextInstances(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                            const BlobUploader& upload,
                            const VulkanEngine::Components::Text& component,
                            const VulkanEngine::Components::Transform& transform,
                            const VulkanEngine::Text::FontFace& face,
                            const VulkanEngine::Text::ShapedRun& run,
                            std::vector<SlugTextInstance>& out);

// Depth-tested world-space text as an engine-owned render pass.
//
// The pass is the world-space counterpart of TextPass and follows its shape: a
// caller queues a component for the frame, the batch builder turns it into world
// glyph instances, and Execute uploads those instances into the frame's per-FIF
// host-visible storage buffer and draws six vertices per glyph. There is no
// vertex buffer -- the quad corner comes from SV_VertexID and the per-glyph data
// from the instance buffer -- and the atlas page is sampled through the bindless
// array, so the pass owns no image.
//
// Setup declares a write on the backbuffer *and* a read of the scene's depth
// buffer, with auto_begin_rendering = false like the screen-space overlay: the
// executor only opens a dynamic-rendering scope around passes that ask for one,
// so this pass opens its own scope, loads the colour the main pass left behind,
// and attaches the depth buffer read-only. The engine-owned pipeline has depth
// test on, depth write off and less-or-equal compare, which is what makes scene
// geometry occlude text while the text never hides geometry drawn after it. The
// blend is straight-alpha source-over, the same composite the overlay uses,
// because the fragment emits non-premultiplied colour with reconstructed coverage
// in alpha.
//
// This is deliberately not a font manager: it holds no faces, no generator and no
// atlas. The caller supplies the generator and the uploaded GlyphAtlasGpu per
// queue call, exactly as TextPass takes the rasterizer and its atlas.
class WorldTextPass final : public IPipelinePass {
public:
    // Glyph quads one frame can draw. A batch that would exceed it is truncated.
    static constexpr std::uint32_t kMaxInstances = 4096;

    WorldTextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap, std::uint64_t vertex_shader,
                  std::uint64_t fragment_shader);
    // Backend-selecting form: `backend` decides which shader pair (and which
    // instance layout) Setup declares. The 3-argument form above is the MSDF
    // default, so every existing caller is unchanged.
    WorldTextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap, TextBackend backend,
                  std::uint64_t vertex_shader, std::uint64_t fragment_shader);
    ~WorldTextPass() override;

    WorldTextPass(const WorldTextPass&) = delete;
    WorldTextPass& operator=(const WorldTextPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "world-text"; }

    // The rasterizer this pass draws with.
    [[nodiscard]] TextBackend Backend() const noexcept { return backend_; }
    // Routing predicates: a queue call for the other backend is a no-op rather
    // than a second layout silently reinterpreted through the wrong struct.
    [[nodiscard]] bool AcceptsMsdfRuns() const noexcept { return backend_ == TextBackend::Msdf; }
    [[nodiscard]] bool AcceptsSlugRuns() const noexcept { return backend_ == TextBackend::Slug; }

    void Setup(PassSetupContext& ctx) override;
    void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) override;

    // ── Per-frame queue ────────────────────────────────────────────────
    // Builds the world-space instances for one component and appends them to
    // this frame's queue. `generator` resolves (and packs) the MSDF atlas slots;
    // `atlas` resolves the page an instance samples to its bindless slot. A page
    // with no GPU image yet falls back to the bindless fallback slot, so a
    // component queued before its upload draws the placeholder rather than
    // reading an unwritten descriptor.
    void QueueRun(VulkanEngine::Text::MsdfGenerator& generator,
                  const VulkanEngine::Text::GlyphAtlasGpu& atlas,
                  const VulkanEngine::Components::Text& component,
                  const VulkanEngine::Components::Transform& transform,
                  const VulkanEngine::Text::FontFace& face,
                  const VulkanEngine::Text::ShapedRun& run,
                  const VulkanEngine::Text::MsdfConfig& config = {});

    // Slug counterpart of QueueRun. `encoder` owns the hb-gpu blob encoder (the
    // size-independent, design-unit cache); this pass owns the blob storage
    // buffer the offsets point into, so it places each newly seen glyph's blob
    // and records its element offset. `recording_frame` is the frame the caller
    // is about to record: it is what a buffer retired by a growth inside Store()
    // is released against. A no-op on an MSDF pass.
    void QueueSlugRun(VulkanEngine::Text::GlyphBlobEncoder& encoder,
                      const VulkanEngine::Components::Text& component,
                      const VulkanEngine::Components::Transform& transform,
                      const VulkanEngine::Text::FontFace& face,
                      const VulkanEngine::Text::ShapedRun& run,
                      std::uint32_t recording_frame);

    // Drops everything queued. Execute() consumes the queue.
    void ClearQueue() {
        pending_.clear();
        pending_slug_.clear();
    }

    // Installed by the owning system: called once per frame that actually draws,
    // after the pipeline/layout checks and before the glyph draw is recorded,
    // with this frame's command buffer and frame index. The MSDF atlas pages are
    // copied here so the transfer is ordered before the draw in one command
    // buffer.
    void SetPreRecordHook(std::function<void(vk::CommandBuffer, std::uint32_t)> hook) {
        pre_record_ = std::move(hook);
    }

    [[nodiscard]] std::uint32_t QueuedInstanceCount() const {
        return static_cast<std::uint32_t>(pending_.size());
    }
    [[nodiscard]] std::span<const WorldTextInstance> QueuedInstances() const { return pending_; }
    [[nodiscard]] std::uint32_t QueuedSlugInstanceCount() const {
        return static_cast<std::uint32_t>(pending_slug_.size());
    }
    [[nodiscard]] std::span<const SlugTextInstance> QueuedSlugInstances() const {
        return pending_slug_;
    }

    // The pass-owned blob storage buffer the Slug instances offset into.
    // Diagnostics and tests; the renderer's "world-text-blobs" resolver returns
    // its buffer to the descriptor.
    [[nodiscard]] VulkanEngine::Text::GpuTextBlobBuffer* BlobBuffer() noexcept {
        return backend_ == TextBackend::Slug ? &blobs_ : nullptr;
    }

    // The instance buffer the pass binds for `frame_index`'s ring slot. The
    // renderer registers this as the resolver for the imported
    // "world-text-instances" resource.
    [[nodiscard]] vk::Buffer InstanceBufferForFrame(std::uint32_t frame_index) const;

    [[nodiscard]] std::uint32_t FramesInFlight() const { return frames_in_flight_; }

private:
    [[nodiscard]] std::uint32_t RingForFrame(std::uint32_t frame_index) const {
        return frames_in_flight_ == 0 ? 0 : frame_index % frames_in_flight_;
    }

    VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap_ = nullptr;
    std::uint64_t vertex_shader_ = 0;
    std::uint64_t fragment_shader_ = 0;
    TextBackend backend_ = kDefaultTextBackend;
    std::uint32_t frames_in_flight_ = 1;
    // Called before the draw when set; see SetPreRecordHook.
    std::function<void(vk::CommandBuffer, std::uint32_t)> pre_record_{};
    // One host-visible instance buffer per frames-in-flight slot, sized for this
    // backend's instance layout.
    std::vector<GpuBuffer> instance_buffers_{};
    std::vector<WorldTextInstance> pending_{};
    std::vector<SlugTextInstance> pending_slug_{};
    // The Slug backend's blob storage buffer; unused (and uninitialized) for
    // MSDF.
    VulkanEngine::Text::GpuTextBlobBuffer blobs_{};
    // Atlas page dimensions of the queued frame, for the shader's range
    // conversion. Kept with the queue because the pass owns no atlas.
    float atlas_width_ = 1.0f;
    float atlas_height_ = 1.0f;
};

} // namespace VulkanEngine::SceneRenderer
