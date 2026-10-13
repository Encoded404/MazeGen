module;

export module VulkanEngine.Render.Passes.TextPass;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.PipelinePass;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.Shaping;

export namespace VulkanEngine::SceneRenderer {

using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::PipelinePass::FrameContext;
using VulkanEngine::PipelinePass::IPipelinePass;
using VulkanEngine::PipelinePass::PassSetupContext;

// One glyph quad. Mirrors engine/core/shaders/ui_text.slang exactly, C data
// layout, 64 bytes: position/size/uv rect/colour, then the bindless page slot.
struct TextInstance {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    float position[2]{0.0f, 0.0f};   // screen-space pixel top-left of the ink box
    float size[2]{0.0f, 0.0f};       // ink box size in pixels
    float uv_min[2]{0.0f, 0.0f};     // inner region of the atlas page, normalised
    float uv_max[2]{0.0f, 0.0f};
    float color[4]{1.0f, 1.0f, 1.0f, 1.0f}; // straight alpha
    std::uint32_t page = 0;          // bindless slot of the atlas page
    std::uint32_t padding[3]{0U, 0U, 0U};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};
static_assert(sizeof(TextInstance) == 64, "TextInstance must mirror the ui_text shader layout");

// A screen-space run to lay out: a shaped run in design units, the face it was
// shaped with, the pixel size to lay it out at, a pen origin and an ink colour.
struct TextRunRequest {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    const VulkanEngine::Text::FontFace* face = nullptr;
    const VulkanEngine::Text::ShapedRun* run = nullptr;
    float pixel_size = 0.0f;
    // Pen origin in pixels with a top-left origin and y growing down. The first
    // glyph's pen is (x, y) and y is the baseline.
    float x = 0.0f;
    float y = 0.0f;
    float color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    VulkanEngine::Text::GlyphHinting hinting = VulkanEngine::Text::kDefaultGlyphHinting;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Maps an atlas page index to the bindless slot its page image lives in. TextPass
// takes this from the GlyphAtlasGpu the caller uploaded into; a device-free test
// can supply its own.
using PageSlotLookup = std::function<std::uint32_t(std::uint32_t)>;

// Lays `request.run` out at `request.pixel_size` and appends one instance per
// drawable glyph to `out`.
//
// Advances come from the shaped run scaled once by face.ScaleForSize(), so the
// glyphs and the pen move together. The pen is accumulated unrounded but each
// glyph's placement is rounded to whole pixels: hinting snaps the bitmap to the
// pixel grid, and drawing it at a fractional offset would resample that grid
// away. Glyphs are resolved through `rasterizer`, which also packs them into its
// atlas; a glyph with no bitmap (a space, a missing glyph, an inkless glyph, a
// size the atlas cannot hold) advances the pen and contributes no quad. The uv
// rect is the atlas slot's inner region, which is the ink box and nothing of the
// padding gutter.
void BuildTextInstances(VulkanEngine::Text::GlyphRasterizer& rasterizer,
                        const PageSlotLookup& page_slot, const TextRunRequest& request,
                        std::vector<TextInstance>& out);

// Screen-space text drawn over the scene as an engine-owned render pass.
//
// The pass is the integration point between the device-free text stack and the
// render graph: a caller queues shaped runs for a frame, the batch builder turns
// them into glyph instances, and Execute uploads those instances into the
// frame's per-FIF host-visible storage buffer and draws six vertices per glyph.
// There is no vertex buffer -- the quad corner comes from SV_VertexID and the
// per-glyph data from the instance buffer -- and the atlas page is sampled
// through the bindless array, so the pass owns no image.
//
// Setup declares a write on the backbuffer with auto_begin_rendering = false and
// load_op = eLoad, exactly like the ImGui overlay. The engine's executor only
// opens a dynamic-rendering scope around passes that ask for one, so like the
// overlay this pass opens its own scope for the glyph draw and loads the scene
// the main pass left behind rather than clearing it. The engine-owned pipeline
// is built with straight-alpha source-over blending (srcAlpha /
// oneMinusSrcAlpha for colour, one / oneMinusSrcAlpha for alpha), which is the
// correct composite for a non-premultiplied coverage value: the fragment emits
// the instance colour with the atlas's R8 coverage folded into alpha, and the
// destination shows through everywhere the coverage is partial.
//
// This is deliberately not a font manager: it holds no faces and no atlas. The
// caller supplies the rasterizer and the uploaded GlyphAtlasGpu per run; the
// engine-level TextSystem that owns them is a later commit.
class TextPass final : public IPipelinePass {
public:
    // Glyph quads one frame can draw. A run that would exceed it is truncated.
    static constexpr std::uint32_t kMaxInstances = 4096;

    TextPass(VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap, std::uint64_t vertex_shader,
             std::uint64_t fragment_shader);
    ~TextPass() override;

    TextPass(const TextPass&) = delete;
    TextPass& operator=(const TextPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "text-overlay"; }

    void Setup(PassSetupContext& ctx) override;
    void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) override;

    // ── Per-frame queue ────────────────────────────────────────────────
    // Appends the instances for one run to this frame's queue. `rasterizer`
    // resolves the glyphs (and packs their atlas slots); `atlas` resolves the
    // page an instance samples to its bindless slot. A page with no GPU image
    // yet falls back to the bindless fallback slot, so a run queued before its
    // upload draws the placeholder rather than reading an unwritten descriptor.
    void QueueRun(VulkanEngine::Text::GlyphRasterizer& rasterizer,
                  const VulkanEngine::Text::GlyphAtlasGpu& atlas,
                  const TextRunRequest& request);

    // Drops everything queued. Execute() consumes the queue, so this is only
    // needed to discard a queued frame.
    void ClearQueue() { pending_.clear(); }

    // Installed by the owning text system: called once per frame that actually
    // draws, after the pipeline/layout checks and before the glyph draw is
    // recorded, with this frame's command buffer and frame index. The text
    // system uses it to copy the dirty glyph-atlas pages the queued instances
    // sample, so the transfer is ordered before the draw in one command buffer.
    // A pass with no hook (a device-free test, or a caller driving the pass
    // directly) simply draws what was queued.
    void SetPreRecordHook(std::function<void(vk::CommandBuffer, std::uint32_t)> hook) {
        pre_record_ = std::move(hook);
    }

    [[nodiscard]] std::uint32_t QueuedInstanceCount() const {
        return static_cast<std::uint32_t>(pending_.size());
    }
    [[nodiscard]] std::span<const TextInstance> QueuedInstances() const { return pending_; }

    // The instance buffer the pass binds for `frame_index`'s ring slot. The
    // renderer registers this as the resolver for the imported "text-instances"
    // resource, so the descriptor is rewritten from the resolved buffer each
    // frame while Execute writes that same slot.
    [[nodiscard]] vk::Buffer InstanceBufferForFrame(std::uint32_t frame_index) const;

    [[nodiscard]] std::uint32_t FramesInFlight() const { return frames_in_flight_; }

private:
    [[nodiscard]] std::uint32_t RingForFrame(std::uint32_t frame_index) const {
        return frames_in_flight_ == 0 ? 0 : frame_index % frames_in_flight_;
    }

    VulkanBackend::Vulkan::IVulkanBootstrap* bootstrap_ = nullptr;
    std::uint64_t vertex_shader_ = 0;
    std::uint64_t fragment_shader_ = 0;
    std::uint32_t frames_in_flight_ = 1;
    // Called before the draw when set; see SetPreRecordHook.
    std::function<void(vk::CommandBuffer, std::uint32_t)> pre_record_{};
    // One host-visible instance buffer per frames-in-flight slot; the frame
    // recorded against a slot is GPU-complete before that slot is reused.
    std::vector<GpuBuffer> instance_buffers_{};
    std::vector<TextInstance> pending_{};
};

} // namespace VulkanEngine::SceneRenderer
