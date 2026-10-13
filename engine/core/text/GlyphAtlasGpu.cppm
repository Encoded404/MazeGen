module;

export module VulkanEngine.Text.GlyphAtlasGpu;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.BindlessManager;
import VulkanEngine.BindlessManager.TextureSlot;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuTexture;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.GlyphRaster;

export namespace VulkanEngine::Text {

// The GPU side of a GlyphAtlas: one persistent page image per CPU page, each
// referenced by a stable bindless slot, plus the dirty-rect uploads that keep
// the image in step with the bitmaps.
//
// The split follows the atlas's own design. GlyphAtlas answers *where* a bitmap
// goes and *what changed*; it stores rectangles and never pixels. This class
// owns the other half: it creates a page image the first time the atlas has a
// page, hands out the bindless slot a shader samples, and copies each dirty
// region's bytes out of the rasterizer's bitmaps into that image.
//
// Pages are created with GpuTexture::CreateStream, the same persistent
// eTransferDst | eSampled image the physical-camera path uploads into: the page
// is written in place for the life of the atlas instead of being recreated per
// frame, and the sampled side is a single UNORM format with a linear sampler so
// the bilinear edge the atlas's padding gutter exists for works.
//
// One uploader design serves both text pipelines through a small seam. The page
// format chooses the image format (R8_UNORM for the FreeType coverage page,
// RGBA8_UNORM for the MSDF page) and how many bytes a texel costs, and a byte
// source resolves an atlas key to the bytes that belong in its rectangle. The
// existing SetGlyphSource A8 path is that seam with the default format: it wraps
// a RasterGlyph's coverage as the bytes. The MSDF pipeline calls SetPageFormat
// with Rgba8 and SetByteSource with a resolver that returns the generated four
// channels. The seam exists so there is exactly one uploader rather than two:
// page creation, page growth, whole-page recomposition after an eviction,
// precise dirty regions, staging retirement and frame-gated slot release all
// stay in one place, and only the format and the byte lookup differ. Duplicating
// that machinery for MSDF would be a second copy of the part most likely to rot.
//
// The format is a property of the uploader, not of one page: this class already
// tracks exactly one GlyphAtlas, and the A8 and MSDF pipelines keep separate
// atlases anyway (different key spaces and different cache owners), so each of
// them owns its own uploader. Two formats in one instance would make the byte
// stride ambiguous for no gain.
//
// SetPageFormat must be called before the first page is created (Initialize is
// not a page creation, but the first EnsurePages/UploadDirty is): a page image's
// format is fixed at creation and cannot be changed by the uploader afterwards.
//
// Lifetime mirrors the bindless manager's own rules rather than adding a second
// release path. A page that the atlas has dropped (Reset()) is released through
// BindlessManager::ReleaseSlot(handle, recording_frame), which keeps the
// descriptor valid until the FIF-drain proves no recorded frame can still
// sample it -- exactly the discipline TextureReloader and TextureResidency use
// for a slot swap. Nothing here calls ReleaseSlotImmediate.
class GlyphAtlasGpu {
public:
    // Resolves an atlas key to the bitmap that lives in that rectangle. The
    // atlas is pixel-free, so an uploader that has to rewrite a whole page (a
    // fresh or evicted page is entirely dirty) needs this to find the pixels of
    // every live entry. Returning nullptr leaves that rectangle zeroed.
    using GlyphSource = std::function<std::shared_ptr<const RasterGlyph>(std::uint64_t)>;

    // Resolves an atlas key to the pixels that belong in its rectangle, in the
    // page's own format. This is the format-agnostic half of the seam: the MSDF
    // store hands back four channels per texel, and the A8 path is expressed
    // through SetGlyphSource below.
    using ByteSource = std::function<std::shared_ptr<const AtlasBitmap>(std::uint64_t)>;

    GlyphAtlasGpu() = default;
    ~GlyphAtlasGpu();

    GlyphAtlasGpu(const GlyphAtlasGpu&) = delete;
    GlyphAtlasGpu& operator=(const GlyphAtlasGpu&) = delete;

    // `sampler_cache` deduplicates the VkSampler the pages share; it may be null,
    // in which case each page owns its private sampler.
    [[nodiscard]] bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                  GpuResources::GpuImageHeap& heap,
                                  GpuResources::StagingPool& staging,
                                  BindlessManager::BindlessManager& bindless,
                                  GpuResources::SamplerCache* sampler_cache = nullptr);
    void Shutdown();

    // The page format every page image is created with. Defaults to A8, the
    // FreeType coverage page, so existing callers are unchanged. Must be set
    // before the first page is created; see the class comment.
    void SetPageFormat(AtlasPageFormat format) { page_format_ = format; }
    [[nodiscard]] AtlasPageFormat PageFormat() const noexcept { return page_format_; }

    // The bitmap source used by a full-page upload. Must be set before the first
    // UploadDirty that can dirty a whole page.
    void SetGlyphSource(GlyphSource source);

    // The format-agnostic pixel source. Setting it supersedes a glyph source;
    // the two exist so the A8 path keeps returning its RasterGlyph owner without
    // the uploader repacking coverage on every upload.
    void SetByteSource(ByteSource source) { byte_source_ = std::move(source); }

    // Creates the image and bindless slot for every atlas page that does not
    // have one yet, without recording anything. A caller that builds glyph
    // instances *before* the frame's command buffer exists -- the text system
    // shapes and resolves during the app's update, then the pass records later --
    // uses this so each instance captures the real page slot instead of the
    // fallback; UploadDirty then writes the dirty bytes into those images before
    // the draw in the same command buffer. Returns true when every page has an
    // image; a heap or bindless-capacity failure leaves the rest for a later
    // frame to retry, exactly as UploadDirty does.
    [[nodiscard]] bool EnsurePages(const GlyphAtlas& atlas);

    [[nodiscard]] bool IsValid() const { return backend_ != nullptr && bindless_ != nullptr; }

    // Pages that currently have a GPU image. Tracks GlyphAtlas::PageCount().
    [[nodiscard]] std::size_t PageCount() const { return pages_.size(); }
    [[nodiscard]] bool HasPage(std::uint32_t page) const {
        return page < pages_.size() && pages_[page].handle.IsValid();
    }

    // The bindless slot page `page` is sampled through. An invalid handle for a
    // page the atlas does not have (or before Initialize).
    [[nodiscard]] BindlessManager::TextureHandle PageHandle(std::uint32_t page) const {
        return page < pages_.size() ? pages_[page].handle : BindlessManager::TextureHandle{};
    }

    // The page image itself. Diagnostic/test access; shaders go through the
    // bindless slot, never a raw image view.
    [[nodiscard]] vk::Image PageImage(std::uint32_t page) const {
        return page < pages_.size() ? pages_[page].image : vk::Image{};
    }
    [[nodiscard]] vk::ImageLayout PageLayout(std::uint32_t page) const {
        return page < pages_.size() ? pages_[page].layout : vk::ImageLayout::eUndefined;
    }

    // Creates a GPU page for every new atlas page, then stages and records each
    // page's dirty region into `cmd` and clears it. Returns the number of pages
    // written.
    //
    // A page whose dirty region is the whole page (a fresh page, or one an
    // eviction invalidated) is uploaded in full: every live rectangle is
    // resolved through the glyph source and composed into a zeroed page-sized
    // staging buffer, so an evicted page's stale texels are gone. A precise
    // region uploads each intersecting live glyph's inner rectangle on its own,
    // so it can never overwrite a neighbouring glyph that sits inside the
    // region's bounding box but was not itself rewritten.
    //
    // `frame_index` is the recording frame: it is what a page retired here is
    // released against.
    std::uint32_t UploadDirty(GlyphAtlas& atlas, vk::CommandBuffer cmd,
                              std::uint32_t frame_index);

    // Releases every GPU page (shutdown, or an atlas Reset()). The slots are
    // frame-gated exactly like UploadDirty's retires.
    void ReleaseAll(std::uint32_t frame_index);

private:
    struct Page {
        BindlessManager::TextureHandle handle{};
        // Raw handles captured before the GpuTexture moves into the bindless
        // slot. They stay valid until the slot's deferred destroy applies, which
        // is after this page record is gone.
        vk::Image image = nullptr;
        vk::ImageView view = nullptr;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };

    // Creates the image and bindless slot for `page` and appends the record.
    [[nodiscard]] bool CreatePage(const AtlasConfig& config, std::uint32_t page);
    // Releases pages above `count`, frame-gated.
    void TrimPages(std::size_t count, std::uint32_t frame_index);
    void Transition(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from,
                    vk::ImageLayout to);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    GpuResources::GpuImageHeap* heap_ = nullptr;
    GpuResources::StagingPool* staging_ = nullptr;
    BindlessManager::BindlessManager* bindless_ = nullptr;
    GpuResources::SamplerCache* sampler_cache_ = nullptr;

    GlyphSource glyph_source_{};
    ByteSource byte_source_{};
    AtlasPageFormat page_format_ = AtlasPageFormat::A8;
    std::vector<Page> pages_{};
};

} // namespace VulkanEngine::Text
