module;

export module VulkanEngine.Text.TextSystem;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.Layout;
import VulkanEngine.Text.Shaping;
import VulkanEngine.Render.Passes.TextPass;

export namespace VulkanEngine::Text {

// The application-facing owner of the whole text pipeline.
//
// One object owns the four layers that used to be separate objects a caller had
// to wire together by hand:
//
//   * a font registry, which opens each (resource, face index) exactly once and
//     hands out shared_ptr<const FontFace>. A face must stay alive for as long
//     as the system: the shaping cache keys runs by FontFace::UniqueId while
//     borrowing the face, and the FreeType pool caches FT_Faces that own a
//     reference to it, so a face freed under either would be a use-after-free.
//   * one ShapingCache (shaped runs; size-independent, cached per face identity
//     and resource version),
//   * one GlyphRasterizer, which owns the hinted-bitmap LRU and the pure,
//     device-free GlyphAtlas,
//   * an optional GPU layer (GlyphAtlasGpu) added once a device exists, which
//     keeps one R8 page image per atlas page and uploads the dirty rectangles.
//
// Everything is injected, never fetched: the resource manager (for loading by
// path), the device resources (for the GPU layer) and the render pass the
// submission path queues into all arrive through setters. There is no global
// and no singleton, and the device-free core is fully constructible without a
// device: faces, shaping, layout, rasterization and the pure atlas all work
// headless, and submission simply queues nothing until a pass is attached.
//
// The single submission entry point is SubmitScreenText(). It shapes, wraps and
// aligns, resolves each glyph through the rasterizer, and hands per-line runs to
// the attached TextPass. The engine attaches the pass right after the renderer
// builds its built-ins (GameEngine::InitRenderer), which is the one seam that
// gives exactly one entry point without the text layer owning any render-graph
// state.
class TextSystem {
public:
    struct Config {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        AtlasConfig atlas{};
        std::size_t shaping_cache_entries = 512;
        std::size_t glyph_cache_entries = 1024;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    TextSystem();
    explicit TextSystem(Config config);
    ~TextSystem();

    TextSystem(const TextSystem&) = delete;
    TextSystem& operator=(const TextSystem&) = delete;
    TextSystem(TextSystem&&) = delete;
    TextSystem& operator=(TextSystem&&) = delete;

    // Enables LoadFontFromPath(). Injected, not fetched; without it the registry
    // still works by resource id and the path loader reports failure.
    void SetResourceManager(ResourceManager& resources) noexcept { resources_ = &resources; }

    // Adds the GPU layer. Safe to call again (idempotent); returns false when the
    // device resources are not valid, in which case the system stays device-free.
    [[nodiscard]] bool InitializeGpu(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                     GpuResources::GpuImageHeap& heap,
                                     GpuResources::StagingPool& staging,
                                     BindlessManager::BindlessManager& bindless,
                                     GpuResources::SamplerCache* sampler_cache = nullptr);

    // Releases the GPU pages (the device must be idle) and drops every cached
    // face and cache entry. Safe to call without a GPU layer.
    void Shutdown();

    // ── Font registry ──────────────────────────────────────────────────
    // Opens (or returns) the face for (id, face_index). `resource` must outlive
    // the registration. `source` is the file changed to reload this font; it is
    // optional for a font loaded from memory.
    [[nodiscard]] std::shared_ptr<const FontFace> LoadFont(const ResourceId& id,
                                                           FontResource& resource,
                                                           std::uint32_t face_index = 0,
                                                           const std::filesystem::path& source = {});

    // Loads through the injected ResourceManager and registers the result. The
    // resource id is the path the manager assigns. Returns nullptr without a
    // resource manager or when the file is not a readable font.
    [[nodiscard]] std::shared_ptr<const FontFace> LoadFontFromPath(
        const std::filesystem::path& path, std::uint32_t face_index = 0);

    [[nodiscard]] std::shared_ptr<const FontFace> GetFace(const ResourceId& id,
                                                          std::uint32_t face_index = 0) const;
    [[nodiscard]] std::filesystem::path SourcePath(const ResourceId& id,
                                                   std::uint32_t face_index = 0) const;
    [[nodiscard]] std::size_t FontCount() const noexcept { return fonts_.size(); }

    // ── Font hot reload ────────────────────────────────────────────────
    // Re-reads the registered font from `path` and, only when that read and the
    // container check both succeed, rebuilds the face and invalidates everything
    // derived from the old one. A missing or corrupt file returns false and
    // leaves the working payload, version and face exactly as they were.
    [[nodiscard]] bool ReloadFont(const ResourceId& id, const std::filesystem::path& path,
                                  std::uint32_t frame_index, std::uint32_t face_index = 0);

    // Drops every cache entry derived from `face_id` and resets the CPU atlas,
    // then retires the GPU pages through the bindless ring drain for
    // `frame_index`. The retired pages stay bound until the drain, so a frame
    // that already recorded text against them keeps sampling a live image.
    void InvalidateFace(std::uint64_t face_id, std::uint32_t frame_index);

    // ── Display scale ───────────────────────────────────────────────────
    // Physical pixels per logical point that screen text is authored against.
    // The engine pushes this from the platform once per frame, so a caller never
    // reads a window to submit text. 1.0 -- the default, and what a device-free
    // caller gets -- means "unscaled" and leaves every coordinate untouched.
    void SetDisplayScale(float display_scale) noexcept;
    [[nodiscard]] float GetDisplayScale() const noexcept { return display_scale_; }

    // ── Submission ─────────────────────────────────────────────────────
    // THE application-facing entry point: shapes `text` with `face`, wraps and
    // aligns it inside `layout`, resolves every glyph (which packs the atlas),
    // makes sure the GPU pages those glyphs landed on exist, and queues the
    // resulting instances into the attached TextPass for this frame.
    //
    // `point_size` and the pen origin `x_points`/`y_points` are in logical points
    // with a top-left origin -- the space the window is sized in and input
    // arrives in -- and are converted to the pass's pixel space through the
    // display scale, so the same call gives the same apparent size on every
    // display. `layout.pixel_size` is ignored in favour of `point_size` so the
    // size cannot be given twice; `layout.max_width` is a distance in points and
    // is scaled with the origin and the size.
    void SubmitScreenText(const FontFace& face, std::string_view text, float point_size,
                          float x_points, float y_points,
                          const std::array<float, 4>& color = {1.0f, 1.0f, 1.0f, 1.0f},
                          const LayoutOptions& layout = {}, GlyphHinting hinting = kDefaultGlyphHinting,
                          const ShapeOptions& shaping = {});

    // Convenience overload: looks the face up in the registry first. An unknown
    // id queues nothing.
    void SubmitScreenText(const ResourceId& id, std::string_view text, float point_size,
                          float x_points, float y_points,
                          const std::array<float, 4>& color = {1.0f, 1.0f, 1.0f, 1.0f},
                          const LayoutOptions& layout = {}, GlyphHinting hinting = kDefaultGlyphHinting,
                          std::uint32_t face_index = 0, const ShapeOptions& shaping = {});

    // Attaches the render pass the submission path queues into. Called by the
    // engine after the renderer registers its built-ins; until then submit is a
    // no-op, because there is nowhere for a frame's text to go. The pass is
    // borrowed, and the previous one (if any) has its hook cleared.
    void SetTextPass(SceneRenderer::TextPass* pass);
    [[nodiscard]] SceneRenderer::TextPass* GetTextPass() const noexcept { return pass_; }

    // Copies the dirty glyph-atlas pages into `cmd`. The installed hook calls
    // this once per drawn frame, before the draw, so the transfer is ordered
    // ahead of the sampling. Returns the number of pages written.
    std::uint32_t UploadAtlas(vk::CommandBuffer cmd, std::uint32_t frame_index);

    // ── Accessors (tests and diagnostics) ──────────────────────────────
    [[nodiscard]] ShapingCache& GetShapingCache() noexcept { return shaping_cache_; }
    [[nodiscard]] GlyphRasterizer& GetRasterizer() noexcept { return rasterizer_; }
    [[nodiscard]] GlyphAtlasGpu* GetGlyphAtlasGpu() noexcept {
        return gpu_initialized_ ? &gpu_atlas_ : nullptr;
    }
    [[nodiscard]] bool HasGpuLayer() const noexcept { return gpu_initialized_; }

private:
    struct RegistryKey {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        std::string id;
        std::uint32_t face_index = 0;
        // NOLINTEND(misc-non-private-member-variables-in-classes)

        [[nodiscard]] bool operator==(const RegistryKey& other) const noexcept {
            return face_index == other.face_index && id == other.id;
        }
    };

    struct RegistryKeyHash {
        [[nodiscard]] std::size_t operator()(const RegistryKey& key) const noexcept;
    };

    struct Entry {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        ResourceId id;
        // Non-owning: the ResourceManager (or the caller) owns the resource, and
        // the face it produced owns its own copy of the bytes.
        FontResource* resource = nullptr;
        std::filesystem::path source{};
        std::uint32_t face_index = 0;
        std::shared_ptr<const FontFace> face{};
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    [[nodiscard]] Entry* Find(const ResourceId& id, std::uint32_t face_index);
    [[nodiscard]] const Entry* Find(const ResourceId& id, std::uint32_t face_index) const;

    // Shapes/queues one line of an already-laid-out run. `line_run` must outlive
    // the call (the pass copies the instances it builds).
    void QueueLine(const FontFace& face, const ShapedRun& line_run, float pixel_size, float pen_x,
                   float pen_y, const std::array<float, 4>& color, GlyphHinting hinting);

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    Config config_;
    ResourceManager* resources_ = nullptr;
    ShapingCache shaping_cache_;
    GlyphRasterizer rasterizer_;
    GlyphAtlasGpu gpu_atlas_{};
    bool gpu_initialized_ = false;
    SceneRenderer::TextPass* pass_ = nullptr;
    // Logical points -> physical pixels. 1.0 until the engine pushes a platform
    // scale, which keeps every device-free caller (and every test) unscaled.
    float display_scale_ = 1.0f;
    std::unordered_map<RegistryKey, Entry, RegistryKeyHash> fonts_{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

} // namespace VulkanEngine::Text
