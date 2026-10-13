module;

#include <logging/logging_macros.hpp>

module VulkanEngine.Text.TextSystem;

import std;
import std.compat;

import logiface;
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

namespace VulkanEngine::Text {

namespace {

std::size_t MixHash(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + static_cast<std::size_t>(0x9E3779B97F4A7C15ULL) + (seed << 6U) +
                   (seed >> 2U));
}

} // namespace

std::size_t TextSystem::RegistryKeyHash::operator()(const RegistryKey& key) const noexcept {
    return MixHash(std::hash<std::string>{}(key.id),
                   std::hash<std::uint32_t>{}(key.face_index));
}

TextSystem::TextSystem() : TextSystem(Config{}) {}

TextSystem::TextSystem(Config config)
    : config_(config),
      shaping_cache_(config.shaping_cache_entries),
      rasterizer_(config.atlas, config.glyph_cache_entries) {}

TextSystem::~TextSystem() {
    Shutdown();
}

bool TextSystem::InitializeGpu(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                               GpuResources::GpuImageHeap& heap,
                               GpuResources::StagingPool& staging,
                               BindlessManager::BindlessManager& bindless,
                               GpuResources::SamplerCache* sampler_cache) {
    if (gpu_initialized_) {
        return true;
    }
    if (!gpu_atlas_.Initialize(backend, heap, staging, bindless, sampler_cache)) {
        LOGIFACE_LOG(warn, "TextSystem: GPU glyph atlas unavailable; text stays CPU-only");
        return false;
    }
    // The atlas stores rectangles, not pixels, so a full-page upload resolves
    // each live rectangle's bitmap through the rasterizer that packed it.
    gpu_atlas_.SetGlyphSource(
        [this](std::uint64_t key) { return rasterizer_.GlyphForAtlasKey(key); });
    gpu_initialized_ = true;
    return true;
}

void TextSystem::Shutdown() {
    // The renderer owns the pass and is torn down before this system in the
    // engine, so the borrowed pointer is only dropped here -- reaching through it
    // would be a use-after-free. The pass's hook is a lambda over `this`; both
    // die with the renderer's pass, which is the ordering the engine enforces.
    pass_ = nullptr;
    if (gpu_initialized_) {
        gpu_atlas_.Shutdown();
        gpu_initialized_ = false;
    }
    fonts_.clear();
    shaping_cache_.Clear();
    rasterizer_.Reset();
}

TextSystem::Entry* TextSystem::Find(const ResourceId& id, std::uint32_t face_index) {
    const auto found = fonts_.find(RegistryKey{id.value, face_index});
    return found != fonts_.end() ? &found->second : nullptr;
}

const TextSystem::Entry* TextSystem::Find(const ResourceId& id, std::uint32_t face_index) const {
    const auto found = fonts_.find(RegistryKey{id.value, face_index});
    return found != fonts_.end() ? &found->second : nullptr;
}

std::shared_ptr<const FontFace> TextSystem::LoadFont(const ResourceId& id, FontResource& resource,
                                                     std::uint32_t face_index,
                                                     const std::filesystem::path& source) {
    if (Entry* existing = Find(id, face_index); existing != nullptr) {
        if (!source.empty()) {
            existing->source = source;
        }
        return existing->face;
    }

    auto face = FontFace::Create(resource, face_index);
    if (face == nullptr) {
        LOGIFACE_LOG(warn, "TextSystem: could not open face " + std::to_string(face_index) +
                               " of font '" + id.value + "'");
        return nullptr;
    }

    Entry entry{};
    entry.id = id;
    entry.resource = &resource;
    entry.source = source;
    entry.face_index = face_index;
    entry.face = face;
    fonts_.emplace(RegistryKey{id.value, face_index}, std::move(entry));
    return face;
}

std::shared_ptr<const FontFace> TextSystem::LoadFontFromPath(const std::filesystem::path& path,
                                                             std::uint32_t face_index) {
    if (resources_ == nullptr) {
        LOGIFACE_LOG(warn, "TextSystem: no resource manager; cannot load font '" + path.string() + "'");
        return nullptr;
    }
    auto handle = resources_->LoadFromFile<FontResource>(path, ResourceManager::LoadSpeed::Instant);
    if (!handle.IsValid()) {
        LOGIFACE_LOG(warn, "TextSystem: font file not loaded '" + path.string() + "'");
        return nullptr;
    }
    FontResource* resource = handle.Get();
    if (resource == nullptr) {
        return nullptr;
    }
    return LoadFont(handle.GetId(), *resource, face_index, path);
}

std::shared_ptr<const FontFace> TextSystem::GetFace(const ResourceId& id,
                                                    std::uint32_t face_index) const {
    const Entry* entry = Find(id, face_index);
    return entry != nullptr ? entry->face : nullptr;
}

std::filesystem::path TextSystem::SourcePath(const ResourceId& id, std::uint32_t face_index) const {
    const Entry* entry = Find(id, face_index);
    return entry != nullptr ? entry->source : std::filesystem::path{};
}

bool TextSystem::ReloadFont(const ResourceId& id, const std::filesystem::path& path,
                            std::uint32_t frame_index, std::uint32_t face_index) {
    Entry* entry = Find(id, face_index);
    if (entry == nullptr || entry->resource == nullptr) {
        LOGIFACE_LOG(warn, "TextSystem: reload of unregistered font '" + id.value + "'");
        return false;
    }
    // A rejected read leaves the resource's bytes and version untouched, so the
    // working face stays exactly what it was and nothing derived from it is
    // invalidated.
    if (!entry->resource->ReloadFromPath(path)) {
        return false;
    }

    // Only a successful re-read reaches here, so the new face is built against
    // the new payload. Its UniqueId is fresh, which is what makes every cache
    // keyed on the old face unreachable.
    auto face = FontFace::Create(*entry->resource, entry->face_index);
    if (face == nullptr) {
        LOGIFACE_LOG(warn, "TextSystem: reloaded font '" + id.value +
                               "' opened no face; keeping the previous one");
        return false;
    }

    const std::uint64_t old_face_id = entry->face != nullptr ? entry->face->UniqueId() : 0;
    entry->face = face;
    entry->source = path;
    if (old_face_id != 0) {
        InvalidateFace(old_face_id, frame_index);
    }
    return true;
}

void TextSystem::InvalidateFace(std::uint64_t face_id, std::uint32_t frame_index) {
    shaping_cache_.InvalidateFace(face_id);
    // The atlas pages are rebuilt from zero because the GPU side retires them
    // wholesale below: a page image must never be rewritten while a recorded
    // frame may still sample it, so the reload takes fresh pages rather than
    // repainting the live ones.
    rasterizer_.Reset();
    if (gpu_initialized_) {
        // Frame-gated through the bindless ring: the old page descriptors stay
        // bound until the drain proves no recorded frame can still sample them.
        gpu_atlas_.ReleaseAll(frame_index);
    }
}

void TextSystem::SetDisplayScale(float display_scale) noexcept {
    display_scale_ = display_scale;
}

void TextSystem::SubmitScreenText(const FontFace& face, std::string_view text, float point_size,
                                  float x_points, float y_points, const std::array<float, 4>& color,
                                  const LayoutOptions& layout, GlyphHinting hinting,
                                  const ShapeOptions& shaping) {
    if (pass_ == nullptr) {
        // Nowhere for a frame's text to go. The engine attaches the pass right
        // after InitRenderer, and submission is a per-frame call, so this is the
        // pre-renderer case only.
        return;
    }

    // Points -> pixels once, at the boundary. Everything below -- the rasterizer,
    // the layout, the queued instances -- is in the pass's pixel space, and the
    // atlas cache key is the pixel size, so a scale change is a different (and
    // correctly hinted) request rather than a resampled one.
    const ScreenTextRequest request =
        ToPhysicalPixels(point_size, x_points, y_points, layout.max_width, display_scale_);
    const float pixel_size = request.pixel_size;
    const float x = request.x;
    const float y = request.y;

    if (text.empty() || !(pixel_size > 0.0f)) {
        return;
    }

    const std::shared_ptr<const ShapedRun> run = shaping_cache_.Shape(face, text, shaping);
    if (run == nullptr || run->Empty()) {
        return;
    }

    LayoutOptions options = layout;
    // The request's pixel size is the layout's; taking it twice invites a caller
    // to disagree with itself.
    options.pixel_size = pixel_size;
    // max_width arrived in points and was scaled with the origin; a caller that
    // asked for no wrapping still gets no wrapping.
    options.max_width = request.max_width;
    const TextLayout laid_out = LayoutText(face, *run, options);

    // One sub-run per laid-out line: the batch builder walks a whole run from its
    // first glyph, and a line is a contiguous range of the run's glyphs whose
    // clusters still index the same source text.
    std::vector<ShapedRun> line_runs;
    line_runs.reserve(laid_out.lines.size());
    for (const LayoutLine& line : laid_out.lines) {
        ShapedRun line_run{};
        line_run.text = run->text;
        const auto first = run->glyphs.begin() + static_cast<std::ptrdiff_t>(line.first_glyph);
        line_run.glyphs.assign(first, first + static_cast<std::ptrdiff_t>(line.glyph_count));
        // Resolve before queueing: Rasterize() is what packs the atlas, so the
        // pages exist by the time the GPU layer creates their images and the
        // instances capture real bindless slots instead of the fallback.
        for (const ShapedGlyph& glyph : line_run.glyphs) {
            (void)rasterizer_.Rasterize(face, glyph.glyph_id, pixel_size, hinting);
        }
        line_runs.push_back(std::move(line_run));
    }

    if (gpu_initialized_) {
        (void)gpu_atlas_.EnsurePages(rasterizer_.Atlas());
    }

    for (std::size_t i = 0; i < line_runs.size(); ++i) {
        const LayoutLine& line = laid_out.lines[i];
        QueueLine(face, line_runs[i], pixel_size, x + line.offset_x, y + line.advance_y, color,
                  hinting);
    }
}

void TextSystem::SubmitScreenText(const ResourceId& id, std::string_view text, float point_size,
                                  float x_points, float y_points, const std::array<float, 4>& color,
                                  const LayoutOptions& layout, GlyphHinting hinting,
                                  std::uint32_t face_index, const ShapeOptions& shaping) {
    const std::shared_ptr<const FontFace> face = GetFace(id, face_index);
    if (face == nullptr) {
        return;
    }
    SubmitScreenText(*face, text, point_size, x_points, y_points, color, layout, hinting, shaping);
}

void TextSystem::QueueLine(const FontFace& face, const ShapedRun& line_run, float pixel_size,
                           float pen_x, float pen_y, const std::array<float, 4>& color,
                           GlyphHinting hinting) {
    if (pass_ == nullptr) {
        return;
    }
    SceneRenderer::TextRunRequest request{};
    request.face = &face;
    request.run = &line_run;
    request.pixel_size = pixel_size;
    request.x = pen_x;
    request.y = pen_y;
    request.color[0] = color[0];
    request.color[1] = color[1];
    request.color[2] = color[2];
    request.color[3] = color[3];
    request.hinting = hinting;
    pass_->QueueRun(rasterizer_, gpu_atlas_, request);
}

void TextSystem::SetTextPass(SceneRenderer::TextPass* pass) {
    if (pass_ != nullptr && pass_ != pass) {
        pass_->SetPreRecordHook(nullptr);
    }
    pass_ = pass;
    if (pass_ != nullptr) {
        pass_->SetPreRecordHook([this](vk::CommandBuffer cmd, std::uint32_t frame_index) {
            (void)UploadAtlas(cmd, frame_index);
        });
    }
}

std::uint32_t TextSystem::UploadAtlas(vk::CommandBuffer cmd, std::uint32_t frame_index) {
    if (!gpu_initialized_) {
        return 0;
    }
    return gpu_atlas_.UploadDirty(rasterizer_.MutableAtlas(), cmd, frame_index);
}

} // namespace VulkanEngine::Text
