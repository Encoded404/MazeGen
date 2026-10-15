#include <gtest/gtest.h>

import std;
import std.compat;

import VulkanShared.Jobs;

import FileLoader.Types;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.GlyphRaster;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::AtlasRect;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphHinting;
using VulkanEngine::Text::GlyphMetrics;
using VulkanEngine::Text::GlyphRasterizer;
using VulkanEngine::Text::GlyphSlot;
using VulkanEngine::Text::QuantizePixelSize;
using VulkanEngine::Text::RasterGlyph;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

[[nodiscard]] bool Overlaps(const AtlasRect& a, const AtlasRect& b) {
    return a.x < static_cast<std::uint64_t>(b.x) + b.width &&
           b.x < static_cast<std::uint64_t>(a.x) + a.width &&
           a.y < static_cast<std::uint64_t>(b.y) + b.height &&
           b.y < static_cast<std::uint64_t>(a.y) + a.height;
}

[[nodiscard]] bool Inside(const AtlasRect& rect, std::uint32_t page_width,
                          std::uint32_t page_height) {
    return static_cast<std::uint64_t>(rect.x) + rect.width <= page_width &&
           static_cast<std::uint64_t>(rect.y) + rect.height <= page_height;
}

// FNV-1a over the A8 bytes. The value is pinned per FreeType build, so it is
// only ever used as a determinism check, never as a definition of correctness.
[[nodiscard]] constexpr std::uint64_t HashCoverage(const RasterGlyph& glyph) {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const std::uint8_t byte : glyph.coverage) {
        hash ^= byte;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// Owns the resource manager and the loaded resource, because a ResourceHandle
// is a lookup into its manager and not an owner.
class GlyphRasterTest : public ::testing::Test {
protected:
    void SetUp() override {
        handle_ = manager_.LoadFromFile<FontResource>(
            TestFontPath(), ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        FontResource* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());

        face_ = FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);
    }

    [[nodiscard]] std::uint32_t Glyph(char character) const {
        return face_->GlyphForCodepoint(static_cast<std::uint32_t>(character));
    }

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    std::shared_ptr<FontFace> face_;
};

TEST_F(GlyphRasterTest, RasterizesAHintedCapital) {
    const std::uint32_t capital = Glyph('H');
    ASSERT_NE(capital, FontFace::MissingGlyph());

    GlyphRasterizer rasterizer;
    constexpr float kSize = 24.0f;
    const auto glyph = rasterizer.Get(*face_, capital, kSize);
    ASSERT_NE(glyph, nullptr);

    EXPECT_EQ(glyph->glyph_id, capital);
    EXPECT_GT(glyph->width, 0u);
    EXPECT_GT(glyph->height, 0u);
    EXPECT_GT(glyph->advance_x, 0.0f);
    EXPECT_EQ(glyph->coverage.size(),
              static_cast<std::size_t>(glyph->width) * glyph->height);

    // Both kinds of byte must be present: a coverage map that is entirely opaque
    // or entirely clear is not a rendered letter.
    EXPECT_TRUE(std::ranges::any_of(glyph->coverage, [](std::uint8_t v) { return v == 0; }));
    EXPECT_TRUE(std::ranges::any_of(glyph->coverage, [](std::uint8_t v) { return v != 0; }));

    // The hinted ink box is FreeType's answer to the same question GlyphAtSize()
    // answers from the unscaled outline, so they agree to within the grid
    // rounding hinting is allowed to apply.
    const GlyphMetrics metrics = face_->GlyphAtSize(capital, kSize);
    EXPECT_NEAR(static_cast<float>(glyph->width), metrics.width, 2.0f);
    EXPECT_NEAR(static_cast<float>(glyph->height), metrics.height, 2.0f);
    EXPECT_NEAR(static_cast<float>(glyph->left), metrics.bearing_x, 2.0f);
    // RasterGlyph::top is y-down, GlyphMetrics::bearing_y is y-up.
    EXPECT_NEAR(static_cast<float>(glyph->top), -metrics.bearing_y, 2.0f);

    // The bitmap reaches a real atlas slot, padded by the atlas's gutter.
    const auto slot = rasterizer.Rasterize(*face_, capital, kSize);
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(slot->rect.width, glyph->width + 2U);
    EXPECT_EQ(slot->rect.height, glyph->height + 2U);
    const auto found = rasterizer.Atlas().Find(1);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->page, slot->page);
}

// FreeType-version-pinned: this hash is over the A8 coverage bytes of 'H' at
// 32 px with the default hinting, and it changes when FreeType's rasterizer or
// hinting output changes -- which is exactly what a FreeType bump can do. It
// must be updated in the same commit as such a bump, the way the GPU golden
// hashes are. Current pin: FreeType 2.13.3.
constexpr std::uint64_t kHintedCapitalHash32 = 0x4CC168E6D13A7883ULL;

TEST_F(GlyphRasterTest, PinsTheHintedBitmapBytes) {
    const std::uint32_t capital = Glyph('H');
    ASSERT_NE(capital, FontFace::MissingGlyph());

    GlyphRasterizer rasterizer;
    const auto glyph = rasterizer.Get(*face_, capital, 32.0f);
    ASSERT_NE(glyph, nullptr);
    EXPECT_EQ(HashCoverage(*glyph), kHintedCapitalHash32);

    // The pin is a property of this exact request; a different size is a
    // different bitmap.
    const auto smaller = rasterizer.Get(*face_, capital, 16.0f);
    ASSERT_NE(smaller, nullptr);
    EXPECT_NE(HashCoverage(*smaller), kHintedCapitalHash32);
}

// Hinting is the reason this rasterizer exists. If the hinted and unhinted
// outlines produced the same pixels, the UI would be paying for FreeType for
// nothing.
TEST_F(GlyphRasterTest, HintingChangesTheBitmap) {
    const std::uint32_t capital = Glyph('H');
    ASSERT_NE(capital, FontFace::MissingGlyph());

    GlyphRasterizer rasterizer;
    constexpr float kSmallSize = 11.0f;
    const auto none = rasterizer.Get(*face_, capital, kSmallSize, GlyphHinting::None);
    const auto light = rasterizer.Get(*face_, capital, kSmallSize, GlyphHinting::Light);
    const auto native = rasterizer.Get(*face_, capital, kSmallSize, GlyphHinting::Native);
    ASSERT_NE(none, nullptr);
    ASSERT_NE(light, nullptr);
    ASSERT_NE(native, nullptr);

    EXPECT_NE(none->coverage, light->coverage)
        << "the light autohinter must move the outline off the raw one";
    // The vendored Lato carries no native TrueType instructions, so Native and
    // Light both land on an autohinter but not the same one; assert the two are
    // distinct rather than assuming the font has bytecode.
    EXPECT_NE(light->coverage, native->coverage);

    // The default is part of the interface, not an accident of the switch: the
    // omitted argument must be exactly the Light request.
    const auto defaulted = rasterizer.Get(*face_, capital, kSmallSize);
    ASSERT_NE(defaulted, nullptr);
    EXPECT_EQ(defaulted.get(), light.get());
}

// The step is a tuning detail: what callers rely on is that snapping is
// idempotent, monotonic, and lands on a uniform grid no more than half a step
// from the requested size. The step is measured from the quantizer's own outputs -- the smallest gap between
// distinct quantized sizes over a probe ladder fine enough to visit every grid
// point -- so the properties below hold for 1/8 px today and for 1/16 px later.
TEST_F(GlyphRasterTest, QuantizePixelSizeSnapsToTheMeasuredGrid) {
    // Zero, negative or NaN sizes mean "no size" and never become a grid point.
    EXPECT_EQ(QuantizePixelSize(0.0f), 0.0f);
    EXPECT_EQ(QuantizePixelSize(-4.0f), 0.0f);

    // A spread of requested sizes: a 1/64-px ladder fine enough to reveal any
    // plausible grid, plus fractional, sub-step, exact, and huge stragglers.
    std::vector<float> probes;
    probes.reserve(1035);
    for (int i = 1; i <= 1024; ++i) {
        probes.push_back(static_cast<float>(i) / 64.0f);
    }
    for (const float extra : {0.01f, 3.999f, 4.0f, 7.5f, 11.33f, 16.06f, 16.07f,
                              18.75f, 24.001f, 64.0f, 1.0e6f}) {
        probes.push_back(extra);
    }
    std::ranges::sort(probes);

    std::vector<float> quantized;
    quantized.reserve(probes.size());
    for (const float probe : probes) {
        quantized.push_back(QuantizePixelSize(probe));
    }

    // The measured grid step: the smallest positive gap between distinct
    // quantized outputs is one step of whatever grid the quantizer snaps to.
    float step = 0.0f;
    for (std::size_t i = 1; i < quantized.size(); ++i) {
        const float gap = quantized[i] - quantized[i - 1];
        if (gap > 0.0f && (step == 0.0f || gap < step)) {
            step = gap;
        }
    }
    ASSERT_GT(step, 0.0f) << "the quantizer collapsed every size onto one value";

    for (std::size_t i = 0; i < probes.size(); ++i) {
        const float size = quantized[i];
        // Idempotent: snapping an already-snapped size changes nothing.
        EXPECT_FLOAT_EQ(QuantizePixelSize(size), size) << "size " << probes[i];
        // Monotonic: sorted inputs are never reordered by snapping.
        if (i > 0) {
            EXPECT_GE(size, quantized[i - 1]) << "size " << probes[i];
        }
        // On the grid: every output is an exact multiple of the measured step.
        EXPECT_EQ(std::fmod(size, step), 0.0f) << "size " << probes[i];
        // Nearest grid point: never a full step from the request, with a little
        // slack for the float rounding of the multiply-and-round itself. A
        // sub-step size may land on the same grid point as zero, which is fine.
        EXPECT_LE(std::abs(size - probes[i]), step * 0.5f + step / 128.0f)
            << "size " << probes[i];
    }
}

// A space has a real advance and a real glyph but no ink, and a bogus id has no
// glyph at all. Neither is an error: both report "no bitmap" without touching a
// null buffer or dereferencing a face FreeType refused to load.
TEST_F(GlyphRasterTest, HandlesInklessAndMissingGlyphs) {
    const std::uint32_t space = Glyph(' ');
    ASSERT_NE(space, FontFace::MissingGlyph());

    GlyphRasterizer rasterizer;
    const auto blank = rasterizer.Get(*face_, space, 16.0f);
    ASSERT_NE(blank, nullptr);
    EXPECT_EQ(blank->width, 0u);
    EXPECT_EQ(blank->height, 0u);
    EXPECT_TRUE(blank->coverage.empty());
    EXPECT_GT(blank->advance_x, 0.0f);
    // Nothing to upload, so nothing is reserved in the atlas.
    EXPECT_FALSE(rasterizer.Rasterize(*face_, space, 16.0f).has_value());

    constexpr std::uint32_t kBogusGlyphId = 0xFFFFFFU;
    EXPECT_EQ(rasterizer.Get(*face_, kBogusGlyphId, 16.0f), nullptr);
    EXPECT_FALSE(rasterizer.Rasterize(*face_, kBogusGlyphId, 16.0f).has_value());

    // .notdef is a real glyph in Lato; whichever way it renders, it must not
    // crash and it must satisfy the same size invariant when it has ink.
    const auto notdef = rasterizer.Get(*face_, FontFace::MissingGlyph(), 16.0f);
    if (notdef != nullptr) {
        EXPECT_EQ(notdef->coverage.size(),
                  static_cast<std::size_t>(notdef->width) * notdef->height);
    }

    // A zero size is rejected before FreeType sees it.
    EXPECT_EQ(rasterizer.Get(*face_, space, 0.0f), nullptr);
    EXPECT_FALSE(rasterizer.Rasterize(*face_, space, -3.0f).has_value());
}

TEST_F(GlyphRasterTest, CachesRasterizedGlyphsByTheirExactKey) {
    const std::uint32_t capital = Glyph('H');
    ASSERT_NE(capital, FontFace::MissingGlyph());

    GlyphRasterizer rasterizer{AtlasConfig{}, 8};
    EXPECT_EQ(rasterizer.Size(), 0u);
    EXPECT_EQ(rasterizer.HitCount(), 0u);

    const auto first = rasterizer.Get(*face_, capital, 16.0f);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(rasterizer.Size(), 1u);
    EXPECT_EQ(rasterizer.HitCount(), 0u);

    const auto second = rasterizer.Get(*face_, capital, 16.0f);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second.get(), first.get()) << "a hit must reuse the cached bitmap";
    EXPECT_EQ(rasterizer.HitCount(), 1u);

    // Two floats that quantize to the same eighth are the same request.
    const auto nearby = rasterizer.Get(*face_, capital, 16.02f);
    ASSERT_NE(nearby, nullptr);
    EXPECT_EQ(nearby.get(), first.get());
    EXPECT_EQ(rasterizer.HitCount(), 2u);
    EXPECT_EQ(rasterizer.Size(), 1u);

    // A different quantized size is a different key.
    const auto larger = rasterizer.Get(*face_, capital, 16.5f);
    ASSERT_NE(larger, nullptr);
    EXPECT_NE(larger.get(), first.get());
    EXPECT_EQ(rasterizer.Size(), 2u);

    // So is a different hinting mode on the same glyph and size.
    const auto unhinted = rasterizer.Get(*face_, capital, 16.0f, GlyphHinting::None);
    ASSERT_NE(unhinted, nullptr);
    EXPECT_NE(unhinted.get(), first.get());
    EXPECT_EQ(rasterizer.Size(), 3u);

    // And so is another face, even of the same font: identity, not just text.
    auto other_face = FontFace::Create(*handle_.Get());
    ASSERT_NE(other_face, nullptr);
    const std::uint64_t hits_before = rasterizer.HitCount();
    const auto other = rasterizer.Get(*other_face, capital, 16.0f);
    ASSERT_NE(other, nullptr);
    EXPECT_NE(other.get(), first.get());
    EXPECT_EQ(rasterizer.HitCount(), hits_before);

    // A result handed out stays valid after Clear() drops the cache's own
    // reference, and clearing releases the atlas slots with it.
    const std::vector<std::uint8_t> keep = first->coverage;
    rasterizer.Clear();
    EXPECT_EQ(rasterizer.Size(), 0u);
    EXPECT_EQ(rasterizer.Atlas().GlyphCount(), 0u);
    EXPECT_EQ(first->coverage, keep);
}

TEST_F(GlyphRasterTest, StaysBoundedUnderManyDistinctKeys) {
    constexpr std::size_t kMaxEntries = 4;
    GlyphRasterizer rasterizer{AtlasConfig{}, kMaxEntries};

    for (std::uint32_t id = 1; id <= 64; ++id) {
        const auto glyph = rasterizer.Get(*face_, id, 14.0f);
        ASSERT_NE(glyph, nullptr) << "glyph " << id << " failed to rasterize";
        EXPECT_LE(rasterizer.Size(), kMaxEntries) << "the cache outgrew its bound";
        // Evicted entries release their atlas slots, so the atlas tracks the
        // live set instead of the number of glyphs ever seen.
        EXPECT_LE(rasterizer.Atlas().GlyphCount(), kMaxEntries);
    }
    EXPECT_EQ(rasterizer.Size(), kMaxEntries);
    EXPECT_EQ(rasterizer.HitCount(), 0u) << "every key here is distinct";
}

// The test that proves the per-slot library pool is correct: the same batch
// rendered across the pool's workers must be byte-identical to rendering it
// serially on one thread, and the atlas every worker wrote into must still be a
// set of disjoint, in-bounds rectangles.
TEST_F(GlyphRasterTest, RasterizesIdenticallyAcrossThreads) {
    constexpr std::string_view kCharacters = "AVHOxgq.,0123";
    const std::array<float, 3> sizes{11.0f, 16.0f, 24.0f};

    std::vector<std::uint32_t> glyphs;
    glyphs.reserve(kCharacters.size());
    for (const char character : kCharacters) {
        const std::uint32_t glyph = Glyph(character);
        ASSERT_NE(glyph, FontFace::MissingGlyph());
        glyphs.push_back(glyph);
    }

    struct Request {
        std::uint32_t glyph_id;
        float pixel_size;
    };
    std::vector<Request> requests;
    for (const float size : sizes) {
        for (const std::uint32_t glyph : glyphs) {
            requests.push_back(Request{glyph, size});
        }
    }

    GlyphRasterizer serial{AtlasConfig{}, 256};
    std::vector<std::shared_ptr<const RasterGlyph>> expected;
    expected.reserve(requests.size());
    for (const Request& request : requests) {
        expected.push_back(serial.Get(*face_, request.glyph_id, request.pixel_size));
    }

    GlyphRasterizer pooled{AtlasConfig{}, 256};
    std::vector<std::shared_ptr<const RasterGlyph>> actual(requests.size());
    auto& pool = VulkanShared::JobScheduler::Global();
    pool.ParallelFor(requests.size(), [&](std::size_t index) {
        const Request& request = requests[index];
        actual[index] = pooled.Get(*face_, request.glyph_id, request.pixel_size);
    });
    pool.WaitForIdle();

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        ASSERT_NE(actual[i], nullptr) << "request " << i << " failed under the pool";
        ASSERT_NE(expected[i], nullptr);
        EXPECT_EQ(actual[i]->width, expected[i]->width);
        EXPECT_EQ(actual[i]->height, expected[i]->height);
        EXPECT_EQ(actual[i]->left, expected[i]->left);
        EXPECT_EQ(actual[i]->top, expected[i]->top);
        EXPECT_FLOAT_EQ(actual[i]->advance_x, expected[i]->advance_x);
        EXPECT_EQ(actual[i]->coverage, expected[i]->coverage)
            << "request " << i << " differs between the pool and one thread";
    }

    // The atlas was written from many threads; it must still hold a disjoint set
    // of rectangles, each fully inside its page.
    const auto entries = pooled.Atlas().Entries();
    for (const auto& [key, slot] : entries) {
        EXPECT_TRUE(Inside(slot.rect, 1024, 1024)) << "atlas key " << key << " escaped its page";
    }
    for (std::size_t i = 0; i < entries.size(); ++i) {
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            if (entries[i].second.page != entries[j].second.page) {
                continue;
            }
            EXPECT_FALSE(Overlaps(entries[i].second.rect, entries[j].second.rect))
                << "atlas keys " << entries[i].first << " and " << entries[j].first
                << " overlap";
        }
    }
}

// FreeType's rule is one library per concurrently-rasterizing thread, and a job
// must not have its slot taken from it. This exercises the pool directly: every
// worker leases a slot, renders, and releases it, over more rounds than there
// are slots.
TEST_F(GlyphRasterTest, PoolServesConcurrentLeases) {
    const std::uint32_t capital = Glyph('H');
    ASSERT_NE(capital, FontFace::MissingGlyph());

    VulkanEngine::Text::FtLibraryPool pool;
    EXPECT_GE(pool.SlotCount(), 1u);

    constexpr std::size_t kRounds = 128;
    std::atomic<std::size_t> failures{0};
    auto& threads = VulkanShared::JobScheduler::Global();
    threads.ParallelFor(kRounds, [&](std::size_t) {
        auto lease = pool.Acquire();
        if (!lease.IsValid()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // The slot is held for the whole job, across more than one face lookup:
        // the second lookup must be the same cached FT_Face, not a second parse.
        const auto first = lease.Face(*face_);
        const auto second = lease.Face(*face_);
        if (first == nullptr || first != second) {
            failures.fetch_add(1, std::memory_order_relaxed);
        }
    });
    threads.WaitForIdle();

    EXPECT_EQ(failures.load(), 0u);
    // At most one cached face per slot: the two lookups inside one lease must
    // not have parsed the font twice. Which slots the pool happened to schedule
    // is up to the worker pool, so this is a ceiling, not an equality.
    EXPECT_GT(pool.FaceCount(), 0u) << "no slot cached the font at all";
    EXPECT_LE(pool.FaceCount(), pool.SlotCount())
        << "a slot parsed the same font more than once";
}

} // namespace
