module;

#include <ft2build.h>
#include FT_FREETYPE_H

export module VulkanEngine.Text.GlyphRaster;

import std;
import std.compat;

import FileLoader.Types;

import VulkanEngine.Text.Font;
import VulkanEngine.Text.Atlas;

export namespace VulkanEngine::Text {

// How FreeType is asked to fit an outline to the pixel grid.
//
// Hinting is the reason this rasterizer exists at all: an unhinted outline at
// 11 px is a blurry approximation of the shape, and a UI that draws at small
// sizes needs the stems snapped to whole pixels. FreeType offers two ways to
// get there, and the difference is real:
//
//   * Native runs the font's own hinting programs. It is the best result when
//     the font ships them, and nothing at all when it does not -- many webfonts
//     (the vendored Lato included) carry no TrueType instructions.
//   * Light runs the auto-hinter in its light mode, which works on any outline
//     source and only nudges edges instead of force-snapping every stem, so it
//     keeps the font's proportions at the sizes the UI actually uses.
//
// Light is the default for exactly that reason: it is never absent and never as
// distorted as the normal auto-hinter, while Native stays available for a font
// known to be well hinted.
enum class GlyphHinting {
    Native,
    Light,
    None,
};

// The default the rasterizer applies when a caller does not choose.
inline constexpr GlyphHinting kDefaultGlyphHinting = GlyphHinting::Light;

// Snaps a requested pixel size to the nearest 1/8 px.
//
// A DPI-scaled UI asks for continuously varying sizes -- a 1.25 scale turns
// 14 px into 17.5, an animation can hand over any float at all -- and a cache
// keyed on the raw float would hold a separate bitmap for every value that
// happens to be requested, growing without bound for no visible difference.
// Eighth-pixel steps are finer than the hinting grid (whole pixels) and finer
// than anything a display can show, so quantizing costs nothing visible and
// bounds the number of distinct sizes a cache can see. Zero (and anything
// non-positive) stays zero, which callers use as "do not rasterize".
[[nodiscard]] float QuantizePixelSize(float pixel_size) noexcept;

// One glyph's rendered bitmap.
//
// `coverage` is 8-bit A8 coverage, `width * height` bytes, row-major and
// top-down: row 0 is the visually topmost row. There is no padding between rows
// and no colour -- the value is coverage, so 0 is fully transparent and 255
// fully opaque.
//
// `left` and `top` place the bitmap relative to the pen origin in the usual
// y-down bitmap sense: the pen sits at (0, 0), +x grows right and +y grows
// *down*, so the bitmap's top-left pixel is drawn at (pen_x + left, pen_y + top)
// and `top` is negative for ink that sits above the baseline. For example 'H'
// at 24 px has a positive `left` (it starts inside its advance) and a `top`
// around -17. The y-down sign is deliberate: the same numbers can be added
// straight to a top-left-origin layout without a flip. It is the negation of
// FreeType's `bitmap_top`, and therefore the negation of the y-up
// GlyphMetrics::bearing_y the shaping side reports.
struct RasterGlyph {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t glyph_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::int32_t left = 0;
    std::int32_t top = 0;
    // The glyph's horizontal advance in pixels at the size it was rendered at.
    float advance_x = 0.0f;
    std::vector<std::uint8_t> coverage;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// A fixed pool of FreeType libraries, one per worker, each with its own faces.
//
// FreeType's own documentation is explicit that an FT_Face is not safe to use
// concurrently -- it carries the size, transform, and glyph slot that a load
// mutates -- and that the documented pattern is one FT_Library per thread.
// `thread_local` is the obvious spelling of that and is the wrong one here: the
// global JobScheduler outlives and joins its workers, and a thread_local FT_Library
// destructor would then run during engine teardown while the pool may still be
// running a job, and on a thread the engine does not own. Worse, a worker that
// has gone idle would keep its library and every face it ever parsed alive for
// the life of that thread, with no way to bound or clear them.
//
// An explicit pool makes the lifetime the engine's: slots are created up front,
// a rasterization job acquires one slot and holds it for its whole duration, and
// teardown destroys the faces and libraries in a known order on a known thread.
// The pool is safe to call from any number of threads; when every slot is busy,
// Acquire() waits rather than creating a library on demand.
//
// FT_Faces are cached per slot, keyed by the font's byte buffer identity and its
// face index, so a font is parsed once per worker instead of once per glyph. The
// entry also holds a shared_ptr to the FontFace, which is what keeps the sfnt
// bytes alive for as long as a cached face points into them -- keying on the
// buffer address without owning it would let a destroyed face's address be
// handed to a different font, the same reuse hazard the shaping cache key
// exists to avoid.
class FtLibraryPool {
public:
    // A slot held for the duration of one rasterization job. The slot is
    // released on destruction, so a job that returns early -- or throws -- does
    // not strand it.
    class Lease {
    public:
        Lease() noexcept = default;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        [[nodiscard]] bool IsValid() const noexcept { return pool_ != nullptr; }

        // The FT_Face this slot holds for `font`, created on first use. Returns
        // nullptr when the pool's library failed to initialise or FreeType
        // rejects the bytes. The pointer stays valid for as long as the pool
        // lives; the face is never evicted.
        [[nodiscard]] FT_Face Face(const FontFace& font);

    private:
        friend class FtLibraryPool;
        Lease(FtLibraryPool* pool, std::size_t index) noexcept;

        void Release() noexcept;

        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        FtLibraryPool* pool_ = nullptr;
        std::size_t index_ = 0;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    // One slot per worker thread in the global pool, which is the most that can
    // rasterize at once through ParallelFor().
    FtLibraryPool();
    explicit FtLibraryPool(std::size_t slot_count);
    ~FtLibraryPool();

    FtLibraryPool(const FtLibraryPool&) = delete;
    FtLibraryPool& operator=(const FtLibraryPool&) = delete;
    FtLibraryPool(FtLibraryPool&&) = delete;
    FtLibraryPool& operator=(FtLibraryPool&&) = delete;

    // Waits until a slot is free, then claims it. Callers that hand the lease to
    // a worker must not hold another lease at the same time: the pool has a
    // fixed number of slots and a lease is not reentrant.
    [[nodiscard]] Lease Acquire();

    [[nodiscard]] std::size_t SlotCount() const noexcept { return slots_.size(); }

    // How many FT_Faces are currently cached across all slots. Diagnostics and
    // tests; taking the lock means this is a snapshot, not a reservation.
    [[nodiscard]] std::size_t FaceCount() const;

private:
    struct FaceKey {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        // FontFace::UniqueId(), never reused. The key is deliberately the face's
        // identity and not its byte buffer's address: font hot reload builds a
        // new face, and the allocator may well place its (different) buffer at
        // the address the old one freed -- keying on the address would then hand
        // the new face the previous font's cached FT_Face. The cached entry also
        // owns the face, so the bytes it points into cannot be freed while it is
        // resident.
        std::uint64_t face_id = 0;
        // NOLINTEND(misc-non-private-member-variables-in-classes)

        [[nodiscard]] bool operator==(const FaceKey& other) const noexcept {
            return face_id == other.face_id;
        }
    };

    struct FaceKeyHash {
        [[nodiscard]] std::size_t operator()(const FaceKey& key) const noexcept;
    };

    struct CachedFace {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        FT_Face face = nullptr;
        // Keeps the sfnt buffer alive for as long as `face` points into it.
        std::shared_ptr<const FontFace> owner;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    struct Slot {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        FT_Library library = nullptr;
        std::unordered_map<FaceKey, CachedFace, FaceKeyHash> faces;
        bool in_use = false;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Slot> slots_;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Turns (face, pixel size, glyph id) into a hinted A8 bitmap and packs it into a
// GlyphAtlas.
//
// Rasterization runs on whatever thread calls in, using a pooled FT_Library from
// FtLibraryPool, so a batch of glyphs can be produced by
// JobScheduler::ParallelFor. Results are cached in a bounded, thread-safe LRU
// keyed on the exact identity of the request -- face identity and resource
// version, glyph id, the *quantized* pixel size, and the hinting mode -- so the
// same glyph at the same size is rendered once per process rather than once per
// frame. Quantizing the size in the key is what keeps that set finite; see
// QuantizePixelSize().
//
// The cache hands out shared_ptr<const RasterGlyph>, so a bitmap a caller holds
// stays valid and unchanged after the entry is evicted or the cache cleared.
// Evicting an entry also releases its atlas slot, so the atlas tracks the live
// set instead of filling with rectangles nothing points at.
//
// Atlas access is serialized by the same mutex as the glyph cache: the atlas is
// not internally locked, and every mutation of it happens here, under the lock,
// after rasterization has already finished outside it.
class GlyphRasterizer {
public:
    // `max_cached_glyphs == 0` disables caching: every call rasterizes afresh
    // and no atlas slot is reserved.
    explicit GlyphRasterizer(AtlasConfig atlas_config = {}, std::size_t max_cached_glyphs = 1024);

    GlyphRasterizer(const GlyphRasterizer&) = delete;
    GlyphRasterizer& operator=(const GlyphRasterizer&) = delete;
    GlyphRasterizer(GlyphRasterizer&&) = delete;
    GlyphRasterizer& operator=(GlyphRasterizer&&) = delete;

    // Rasterizes (or looks up) the glyph and returns where its pixels belong.
    // Returns nullopt for a glyph that is missing, a size of zero, a bitmap the
    // atlas cannot hold, or an inkless glyph such as a space, which has nothing
    // to upload in the first place. Get() still returns the cached bitmap in
    // those last two cases.
    [[nodiscard]] std::optional<GlyphSlot> Rasterize(const FontFace& face, std::uint32_t glyph_id,
                                                     float pixel_size,
                                                     GlyphHinting hinting = kDefaultGlyphHinting);

    // The bitmap for (face, glyph, size, hinting), or nullptr when FreeType
    // cannot render it. The result survives eviction and Clear().
    [[nodiscard]] std::shared_ptr<const RasterGlyph> Get(const FontFace& face,
                                                         std::uint32_t glyph_id, float pixel_size,
                                                         GlyphHinting hinting = kDefaultGlyphHinting);

    // The atlas the cached glyphs live in, so a later commit can upload it. The
    // reference is only safe to read while no other thread is rasterizing; an
    // upload pass runs after the workers have drained.
    [[nodiscard]] const GlyphAtlas& Atlas() const noexcept { return atlas_; }

    // Mutable access for the uploader, which has to clear a page's dirty region
    // once its bytes have been copied. The atlas is otherwise private to the
    // rasterizer, and this carries the same contract as Atlas(): the caller must
    // not rasterize concurrently.
    [[nodiscard]] GlyphAtlas& MutableAtlas() noexcept { return atlas_; }

    // The bitmap packed into the rectangle an atlas entry is keyed by, or
    // nullptr when no live entry uses that key. The atlas stores rectangles and
    // deliberately not pixels, so an uploader that has to fill a whole page (a
    // fresh or evicted page is entirely dirty) resolves every live rectangle's
    // pixels through this rather than the atlas growing a byte copy.
    [[nodiscard]] std::shared_ptr<const RasterGlyph> GlyphForAtlasKey(std::uint64_t atlas_key) const;

    [[nodiscard]] std::size_t Size() const;
    // Lifetime hit count. Clear() does not reset it, matching ShapingCache.
    [[nodiscard]] std::uint64_t HitCount() const;
    // Drops every cached glyph and releases the atlas slots they hold.
    void Clear();
    // Drops every cached glyph *and* every atlas page, so the CPU atlas is
    // rebuilt from zero. Clear() keeps the pages and only frees the slots the
    // cached glyphs held; a font reload retires the GPU pages wholesale (a page
    // image cannot be rewritten while a recorded frame may still sample it), so
    // the CPU side has to be dropped to match -- otherwise the next upload would
    // believe a page it must repaint is already clean.
    void Reset();

private:
    struct Key {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        std::uint64_t face_id = 0;
        std::uint32_t resource_version = 0;
        std::uint32_t glyph_id = 0;
        // Already quantized, so two requests for the same eighth-pixel compare
        // exactly equal rather than approximately.
        float pixel_size = 0.0f;
        GlyphHinting hinting = kDefaultGlyphHinting;
        // NOLINTEND(misc-non-private-member-variables-in-classes)

        [[nodiscard]] bool operator==(const Key& other) const noexcept {
            return face_id == other.face_id && resource_version == other.resource_version &&
                   glyph_id == other.glyph_id && pixel_size == other.pixel_size &&
                   hinting == other.hinting;
        }
    };

    struct KeyHash {
        [[nodiscard]] std::size_t operator()(const Key& key) const noexcept;
    };

    struct Entry {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        Key key;
        std::shared_ptr<const RasterGlyph> glyph;
        // nullopt for an inkless glyph or one the atlas could not hold.
        std::optional<GlyphSlot> slot;
        // The atlas key this entry reserved, unique for the entry's life so two
        // different glyphs can never be mistaken for each other's rectangle.
        std::uint64_t atlas_key = 0;
        bool in_atlas = false;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    struct Result {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        std::shared_ptr<const RasterGlyph> glyph;
        std::optional<GlyphSlot> slot;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    [[nodiscard]] Result GetOrRasterize(const FontFace& face, std::uint32_t glyph_id,
                                        float pixel_size, GlyphHinting hinting);

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    FtLibraryPool pool_;
    GlyphAtlas atlas_;
    mutable std::mutex mutex_;
    // Most recently used entry first; the back is the eviction candidate.
    std::list<Entry> entries_;
    std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> index_;
    std::size_t max_entries_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t next_atlas_key_ = 1;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

} // namespace VulkanEngine::Text
