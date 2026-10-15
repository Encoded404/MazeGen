module;

#include <ft2build.h>
#include FT_FREETYPE_H

module VulkanEngine.Text.GlyphRaster;

import std;
import std.compat;

import FileLoader.Types;
import VulkanShared.Jobs;

import VulkanEngine.Text.Font;
import VulkanEngine.Text.Atlas;

namespace VulkanEngine::Text {

namespace {

// boost-style hash_combine, same as the shaping cache's. The hash only chooses a
// bucket; equality is what decides whether two keys are the same entry.
[[nodiscard]] std::size_t MixHash(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + static_cast<std::size_t>(0x9E3779B97F4A7C15ULL) + (seed << 6U) +
                   (seed >> 2U));
}

// 72 dpi is the scale at which a 26.6 point size is a pixel size, so the 26.6
// value is the requested size in 1/64 px and a fractional quantized size (16.125
// px, say) reaches FreeType exactly instead of being rounded to 16.
constexpr FT_UInt kFreeTypeDpi = 72;

[[nodiscard]] FT_Int32 LoadFlags(GlyphHinting hinting) noexcept {
    switch (hinting) {
        case GlyphHinting::Native:
            // No target flag beyond the default: FreeType runs the font's own
            // hinting programs when it has them.
            return FT_LOAD_TARGET_NORMAL;
        case GlyphHinting::Light:
            // FORCE_AUTOHINT makes "the light autohinter" true even for a font
            // that ships native instructions; the target flag is what makes the
            // autohinter light rather than its aggressive default.
            return FT_LOAD_TARGET_LIGHT | FT_LOAD_FORCE_AUTOHINT;
        case GlyphHinting::None:
            return FT_LOAD_NO_HINTING;
    }
    return FT_LOAD_TARGET_NORMAL;
}

// Renders one glyph through a leased library slot. The lease covers the whole
// FreeType interaction -- set size, load, render, copy out -- because the FT_Face
// holds the size and the glyph slot those calls mutate.
[[nodiscard]] std::shared_ptr<const RasterGlyph> RasterizeGlyph(FtLibraryPool& pool,
                                                                const FontFace& face,
                                                                std::uint32_t glyph_id,
                                                                float pixel_size,
                                                                GlyphHinting hinting) {
    FtLibraryPool::Lease lease = pool.Acquire();
    if (!lease.IsValid()) {
        return nullptr;
    }
    FT_Face ft_face = lease.Face(face);
    if (ft_face == nullptr) {
        return nullptr;
    }

    const auto size_26_6 =
        static_cast<FT_F26Dot6>(std::lround(static_cast<double>(pixel_size) * 64.0));
    if (FT_Set_Char_Size(ft_face, size_26_6, size_26_6, kFreeTypeDpi, kFreeTypeDpi) != 0) {
        return nullptr;
    }

    // A glyph id past the face's glyph count (or any other load failure) is a
    // caller error, not a crash: report "no bitmap" and let the caller fall back
    // to .notdef.
    if (FT_Load_Glyph(ft_face, static_cast<FT_UInt>(glyph_id), LoadFlags(hinting)) != 0) {
        return nullptr;
    }
    // Render explicitly with FT_RENDER_MODE_NORMAL, which is what produces the
    // 8-bit antialiased coverage this interface promises (FT_LOAD_RENDER would
    // pick the target's render mode, which is not necessarily normal).
    if (FT_Render_Glyph(ft_face->glyph, FT_RENDER_MODE_NORMAL) != 0) {
        return nullptr;
    }

    const FT_GlyphSlot slot = ft_face->glyph;
    auto glyph = std::make_shared<RasterGlyph>();
    glyph->glyph_id = glyph_id;
    glyph->advance_x = static_cast<float>(slot->advance.x) / 64.0f;
    glyph->left = static_cast<std::int32_t>(slot->bitmap_left);
    // Y-down: the bitmap's top row sits bitmap_top pixels above the baseline,
    // which is negative in a frame where y grows downward.
    glyph->top = -static_cast<std::int32_t>(slot->bitmap_top);

    const FT_Bitmap& bitmap = slot->bitmap;
    const bool is_gray = bitmap.pixel_mode == FT_PIXEL_MODE_GRAY;
    const bool has_pixels = bitmap.buffer != nullptr && bitmap.width > 0 && bitmap.rows > 0;
    // A space, a zero-width glyph or anything FreeType handed back in a mode
    // other than greyscale stays an empty bitmap rather than a crash or a
    // misread buffer. pitch is the row stride and is normally positive; a
    // negative stride means bottom-up rows, which this path does not produce.
    if (is_gray && has_pixels && bitmap.pitch > 0) {
        glyph->width = bitmap.width;
        glyph->height = bitmap.rows;
        glyph->coverage.resize(static_cast<std::size_t>(bitmap.width) *
                               static_cast<std::size_t>(bitmap.rows));
        for (unsigned int row = 0; row < bitmap.rows; ++row) {
            const std::uint8_t* source =
                bitmap.buffer + (static_cast<std::ptrdiff_t>(row) * bitmap.pitch);
            std::memcpy(glyph->coverage.data() +
                            (static_cast<std::size_t>(row) * bitmap.width),
                        source, bitmap.width);
        }
    }
    return glyph;
}

} // namespace

float QuantizePixelSize(float pixel_size) noexcept {
    if (!(pixel_size > 0.0f)) {
        // Zero, negative or NaN: "no size" rather than a bogus eighth.
        return 0.0f;
    }
    return std::round(pixel_size * 8.0f) / 8.0f;
}

std::size_t FtLibraryPool::FaceKeyHash::operator()(const FaceKey& key) const noexcept {
    // Identity, not address: see FaceKey.
    return std::hash<std::uint64_t>{}(key.face_id);
}

FtLibraryPool::FtLibraryPool()
    : FtLibraryPool(std::max<std::size_t>(1, VulkanShared::JobScheduler::Global().WorkerCount())) {}

FtLibraryPool::FtLibraryPool(std::size_t slot_count) {
    slots_.resize(std::max<std::size_t>(slot_count, 1));
    for (Slot& slot : slots_) {
        // A library that fails to initialise leaves a null handle; Face() then
        // reports failure instead of dereferencing it. FT_Init_FreeType fails
        // only on allocation failure, so this is a guard, not a branch.
        if (FT_Init_FreeType(&slot.library) != 0) {
            slot.library = nullptr;
        }
    }
}

FtLibraryPool::~FtLibraryPool() {
    for (Slot& slot : slots_) {
        // Faces first: each points into the buffer its owning FontFace keeps
        // alive, and into the library, so both must outlive the face.
        for (auto& [key, cached] : slot.faces) {
            if (cached.face != nullptr) {
                FT_Done_Face(cached.face);
            }
        }
        slot.faces.clear();
        if (slot.library != nullptr) {
            FT_Done_FreeType(slot.library);
        }
    }
    slots_.clear();
}

std::size_t FtLibraryPool::FaceCount() const {
    const std::scoped_lock lock(mutex_);
    std::size_t total = 0;
    for (const Slot& slot : slots_) {
        total += slot.faces.size();
    }
    return total;
}

FtLibraryPool::Lease FtLibraryPool::Acquire() {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this] {
        return std::ranges::any_of(slots_, [](const Slot& slot) { return !slot.in_use; });
    });
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].in_use) {
            slots_[i].in_use = true;
            return Lease(this, i);
        }
    }
    return {};
}

FtLibraryPool::Lease::Lease(FtLibraryPool* pool, std::size_t index) noexcept
    : pool_(pool), index_(index) {}

FtLibraryPool::Lease::Lease(Lease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)), index_(other.index_) {}

FtLibraryPool::Lease& FtLibraryPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        Release();
        pool_ = std::exchange(other.pool_, nullptr);
        index_ = other.index_;
    }
    return *this;
}

FtLibraryPool::Lease::~Lease() {
    Release();
}

void FtLibraryPool::Lease::Release() noexcept {
    if (pool_ == nullptr) {
        return;
    }
    {
        const std::scoped_lock lock(pool_->mutex_);
        pool_->slots_[index_].in_use = false;
    }
    pool_->available_.notify_one();
    pool_ = nullptr;
}

FT_Face FtLibraryPool::Lease::Face(const FontFace& font) {
    if (pool_ == nullptr) {
        return nullptr;
    }
    FtLibraryPool& pool = *pool_;
    const FileLoader::ByteSpan bytes = font.Bytes();
    if (bytes.empty() || pool.slots_[index_].library == nullptr) {
        return nullptr;
    }

    const FaceKey key{font.UniqueId()};
    {
        const std::scoped_lock lock(pool.mutex_);
        const auto found = pool.slots_[index_].faces.find(key);
        if (found != pool.slots_[index_].faces.end()) {
            return found->second.face;
        }
    }

    // Create the face outside the pool lock: FT_New_Memory_Face parses the whole
    // sfnt and is the expensive part. This slot is exclusively this lease's, so
    // nothing else can be touching its map while we do it.
    FT_Face created = nullptr;
    if (FT_New_Memory_Face(pool.slots_[index_].library,
                           reinterpret_cast<const FT_Byte*>(bytes.data()),
                           static_cast<FT_Long>(bytes.size()),
                           static_cast<FT_Long>(font.FaceIndex()), &created) != 0) {
        return nullptr;
    }

    const std::scoped_lock lock(pool.mutex_);
    auto& faces = pool.slots_[index_].faces;
    // A second Face() call on the same font within one lease is not a race but
    // is still redundant: keep the first face and drop the duplicate.
    if (const auto found = faces.find(key); found != faces.end()) {
        FT_Done_Face(created);
        return found->second.face;
    }
    // shared_from_this() is what keeps the sfnt buffer alive for the cached
    // face's whole life; the face cannot outlive it.
    faces.emplace(key, CachedFace{created, font.shared_from_this()});
    return created;
}

GlyphRasterizer::GlyphRasterizer(AtlasConfig atlas_config, std::size_t max_cached_glyphs)
    : atlas_(atlas_config), max_entries_(max_cached_glyphs) {}

std::size_t GlyphRasterizer::KeyHash::operator()(const Key& key) const noexcept {
    std::size_t hash = std::hash<std::uint64_t>{}(key.face_id);
    hash = MixHash(hash, std::hash<std::uint32_t>{}(key.resource_version));
    hash = MixHash(hash, std::hash<std::uint32_t>{}(key.glyph_id));
    // The bit pattern, not the numeric value: the size is an exact multiple of
    // 1/8, so equal requests produce equal bits and hashing them is stable.
    hash = MixHash(hash, std::hash<std::uint32_t>{}(std::bit_cast<std::uint32_t>(key.pixel_size)));
    hash = MixHash(hash, std::hash<int>{}(static_cast<int>(key.hinting)));
    return hash;
}

std::optional<GlyphSlot> GlyphRasterizer::Rasterize(const FontFace& face, std::uint32_t glyph_id,
                                                    float pixel_size, GlyphHinting hinting) {
    return GetOrRasterize(face, glyph_id, pixel_size, hinting).slot;
}

std::shared_ptr<const RasterGlyph> GlyphRasterizer::Get(const FontFace& face,
                                                        std::uint32_t glyph_id, float pixel_size,
                                                        GlyphHinting hinting) {
    return GetOrRasterize(face, glyph_id, pixel_size, hinting).glyph;
}

std::size_t GlyphRasterizer::Size() const {
    const std::scoped_lock lock(mutex_);
    return entries_.size();
}

std::uint64_t GlyphRasterizer::HitCount() const {
    const std::scoped_lock lock(mutex_);
    return hits_;
}

void GlyphRasterizer::Clear() {
    const std::scoped_lock lock(mutex_);
    for (const Entry& entry : entries_) {
        if (entry.in_atlas) {
            // The key was reserved by this entry, so the erase always finds it.
            static_cast<void>(atlas_.Erase(entry.atlas_key));
        }
    }
    entries_.clear();
    index_.clear();
}

void GlyphRasterizer::Reset() {
    const std::scoped_lock lock(mutex_);
    entries_.clear();
    index_.clear();
    // Drop the pages too, so the next rasterization starts from an empty atlas
    // whose fresh pages are reported entirely dirty.
    atlas_.Reset();
}

std::shared_ptr<const RasterGlyph> GlyphRasterizer::GlyphForAtlasKey(
    std::uint64_t atlas_key) const {
    const std::scoped_lock lock(mutex_);
    for (const Entry& entry : entries_) {
        if (entry.in_atlas && entry.atlas_key == atlas_key) {
            return entry.glyph;
        }
    }
    return nullptr;
}

GlyphRasterizer::Result GlyphRasterizer::GetOrRasterize(const FontFace& face,
                                                        std::uint32_t glyph_id, float pixel_size,
                                                        GlyphHinting hinting) {
    const float quantized = QuantizePixelSize(pixel_size);
    if (quantized <= 0.0f) {
        return {};
    }
    const Key key{face.UniqueId(), face.ResourceVersion(), glyph_id, quantized, hinting};

    {
        const std::scoped_lock lock(mutex_);
        const auto found = index_.find(key);
        if (found != index_.end()) {
            ++hits_;
            entries_.splice(entries_.begin(), entries_, found->second);
            return Result{found->second->glyph, found->second->slot};
        }
    }

    // Rasterize without the lock: it is the expensive part, and holding the
    // mutex across it would serialize the workers the pool exists to feed. Two
    // threads racing on one key simply render it twice and the loser reuses the
    // winner's entry below.
    auto glyph = RasterizeGlyph(pool_, face, glyph_id, quantized, hinting);
    if (glyph == nullptr) {
        return {};
    }

    const std::scoped_lock lock(mutex_);
    if (const auto existing = index_.find(key); existing != index_.end()) {
        ++hits_;
        entries_.splice(entries_.begin(), entries_, existing->second);
        return Result{existing->second->glyph, existing->second->slot};
    }

    if (max_entries_ == 0) {
        // Caching disabled: hand back the bitmap but reserve nothing, so the
        // atlas cannot accumulate slots no entry will ever release.
        return Result{glyph, std::nullopt};
    }

    Entry entry;
    entry.key = key;
    entry.glyph = glyph;
    if (glyph->width > 0 && glyph->height > 0) {
        // One atlas key per cache entry, handed out monotonically: two different
        // bitmaps can never collide on one rectangle, which a derived hash key
        // could not promise.
        entry.atlas_key = next_atlas_key_++;
        if (atlas_.Insert(entry.atlas_key, glyph->width, glyph->height) != AtlasInsert::TooLarge) {
            entry.slot = atlas_.Find(entry.atlas_key);
            entry.in_atlas = entry.slot.has_value();
        }
    }

    const Result result{entry.glyph, entry.slot};
    entries_.push_front(std::move(entry));
    index_.emplace(entries_.front().key, entries_.begin());
    while (entries_.size() > max_entries_) {
        const Entry& victim = entries_.back();
        if (victim.in_atlas) {
            static_cast<void>(atlas_.Erase(victim.atlas_key));
        }
        index_.erase(victim.key);
        entries_.pop_back();
    }
    return result;
}

} // namespace VulkanEngine::Text
