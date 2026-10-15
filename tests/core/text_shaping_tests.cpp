#include <gtest/gtest.h>

import std;
import std.compat;

import VulkanShared.Jobs;

import FileLoader.Types;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.Shaping;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapingCache;
using VulkanEngine::Text::ShapeText;
using VulkanEngine::Text::TextDirection;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

[[nodiscard]] std::vector<std::uint32_t> GlyphIds(const ShapedRun& run) {
    std::vector<std::uint32_t> ids;
    ids.reserve(run.glyphs.size());
    for (const auto& glyph : run.glyphs) {
        ids.push_back(glyph.glyph_id);
    }
    return ids;
}

// Owns the resource manager and the loaded resource, because a ResourceHandle
// is a lookup into its manager and not an owner.
class TextShapingTest : public ::testing::Test {
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

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    std::shared_ptr<FontFace> face_;
};

TEST_F(TextShapingTest, ShapesDesignUnitsFromTheFace) {
    const auto glyph_a = face_->GlyphForCodepoint(static_cast<std::uint32_t>('A'));
    ASSERT_NE(glyph_a, FontFace::MissingGlyph());

    const ShapedRun run = ShapeText(*face_, "A");
    ASSERT_EQ(run.glyphs.size(), 1u);
    EXPECT_EQ(run.glyphs[0].glyph_id, glyph_a);
    EXPECT_EQ(run.text, "A");

    // Lato's units-per-em is 2000 and the face's scale is pinned to it, so
    // shaping at that "pixel size" is the identity: the shaped advance must
    // equal the face's design-unit advance exactly. That is what makes one run
    // reusable at every size.
    const auto& metrics = face_->Metrics();
    const float design_units = face_->GlyphAtSize(glyph_a, static_cast<float>(metrics.units_per_em))
                                   .advance_x;
    EXPECT_FLOAT_EQ(run.glyphs[0].advance_x, design_units);
    EXPECT_FLOAT_EQ(run.total_advance_x, design_units);
    EXPECT_GT(run.glyphs[0].advance_x, 0.0f);
}

// Lato's GPOS data carries a real "AV" pair kern: the pair advances 2600 design
// units against 1354 + 1354 = 2708 for the isolated glyphs, so 108 units are
// removed. The two runs are asserted against each other rather than against
// those constants, but the relation is unconditional: kerning makes the pair
// narrower, and disabling it restores the sum exactly.
TEST_F(TextShapingTest, AppliesKerningBetweenAdjacentGlyphs) {
    ShapeOptions kerned{};
    ShapeOptions unkerned{};
    unkerned.kerning = false;

    const ShapedRun pair = ShapeText(*face_, "AV", kerned);
    const ShapedRun pair_without_kerning = ShapeText(*face_, "AV", unkerned);
    ASSERT_EQ(pair.glyphs.size(), 2u);
    ASSERT_EQ(pair_without_kerning.glyphs.size(), 2u);

    const float separate = ShapeText(*face_, "A").total_advance_x +
                           ShapeText(*face_, "V").total_advance_x;

    EXPECT_LT(pair.total_advance_x, separate) << "kerning must tighten the pair";
    EXPECT_FLOAT_EQ(pair_without_kerning.total_advance_x, separate)
        << "without kerning the pair is exactly the sum of its glyphs";
    // The glyphs themselves are untouched by kerning; only the advance differs.
    EXPECT_EQ(GlyphIds(pair), GlyphIds(pair_without_kerning));
}

// Lato's GSUB really does ligate "fi": with `liga` enabled the pair shapes to
// one glyph (glyph 67), and disabling `liga`/`clig` yields the two component
// glyphs the single-character runs produce (61 and 98). Both branches are
// asserted unconditionally because the font is vendored and fixed.
TEST_F(TextShapingTest, LigatesFiWhenLigaturesAreEnabled) {
    ShapeOptions ligated{};
    ShapeOptions unligated{};
    unligated.ligatures = false;

    const ShapedRun one = ShapeText(*face_, "fi", ligated);
    const ShapedRun two = ShapeText(*face_, "fi", unligated);

    ASSERT_EQ(one.glyphs.size(), 1u) << "Lato ligates fi into a single glyph";
    ASSERT_EQ(two.glyphs.size(), 2u) << "disabling liga/clig keeps f and i separate";

    const ShapedRun f = ShapeText(*face_, "f");
    const ShapedRun i = ShapeText(*face_, "i");
    ASSERT_EQ(f.glyphs.size(), 1u);
    ASSERT_EQ(i.glyphs.size(), 1u);
    EXPECT_EQ(two.glyphs[0].glyph_id, f.glyphs[0].glyph_id);
    EXPECT_EQ(two.glyphs[1].glyph_id, i.glyphs[0].glyph_id);
    EXPECT_NE(one.glyphs[0].glyph_id, f.glyphs[0].glyph_id);
    EXPECT_NE(one.glyphs[0].glyph_id, i.glyphs[0].glyph_id);

    // A ligature is narrower than its components here (1145 against 1181 design
    // units), which is the visible reason to keep it on.
    EXPECT_LT(one.total_advance_x, two.total_advance_x);
}

// Clusters are the byte offsets layout uses to map UAX#14 break positions onto
// glyph ranges, so they must index the shaped bytes and never move backwards.
// "The quick brown fox" has no ligature, so it is one glyph per character.
TEST_F(TextShapingTest, ReportsClustersAsByteOffsets) {
    constexpr std::string_view kText = "The quick brown fox";
    const ShapedRun run = ShapeText(*face_, kText);

    ASSERT_EQ(run.glyphs.size(), kText.size());
    EXPECT_EQ(run.text, kText);

    std::uint32_t previous = 0;
    for (std::size_t i = 0; i < run.glyphs.size(); ++i) {
        const std::uint32_t cluster = run.glyphs[i].cluster;
        EXPECT_LT(cluster, kText.size());
        if (i > 0) {
            EXPECT_GE(cluster, previous) << "clusters must be non-decreasing";
        }
        previous = cluster;
    }
    EXPECT_EQ(run.glyphs.front().cluster, 0u);
    EXPECT_EQ(run.glyphs.back().cluster, static_cast<std::uint32_t>(kText.size() - 1));

    // A cluster names the character the glyph came from.
    const auto glyph_at = [&](std::size_t byte) {
        for (const auto& glyph : run.glyphs) {
            if (glyph.cluster == byte) {
                return glyph.glyph_id;
            }
        }
        return FontFace::MissingGlyph() + 1U;
    };
    EXPECT_EQ(glyph_at(0), face_->GlyphForCodepoint(static_cast<std::uint32_t>('T')));
    EXPECT_EQ(glyph_at(1), face_->GlyphForCodepoint(static_cast<std::uint32_t>('h')));
    // Byte 16 is the 'f' of "fox": the cluster of a word-initial glyph is still
    // its own offset, not the offset of the run.
    EXPECT_EQ(glyph_at(16), face_->GlyphForCodepoint(static_cast<std::uint32_t>('f')));
}

TEST_F(TextShapingTest, ShapesEmptyAndWhitespaceOnlyInput) {
    const ShapedRun empty = ShapeText(*face_, "");
    EXPECT_TRUE(empty.Empty());
    EXPECT_FLOAT_EQ(empty.total_advance_x, 0.0f);
    EXPECT_FLOAT_EQ(empty.total_advance_y, 0.0f);
    EXPECT_TRUE(empty.text.empty());

    const ShapedRun spaces = ShapeText(*face_, "   ");
    ASSERT_EQ(spaces.glyphs.size(), 3u);
    for (const auto& glyph : spaces.glyphs) {
        EXPECT_GT(glyph.advance_x, 0.0f) << "a space still advances";
    }
    EXPECT_FLOAT_EQ(spaces.total_advance_x, spaces.glyphs[0].advance_x * 3.0f);
}

// A codepoint the face does not cover must stay in the run as glyph 0. Lato
// has no emoji, so U+1F600 is .notdef: three glyphs for "a<emoji>b", with the
// missing one second, and the trailing glyph's cluster still pointing at byte
// 5 because the emoji is four bytes.
TEST_F(TextShapingTest, KeepsUncoveredCodepointsAsNotdef) {
    const ShapedRun run = ShapeText(*face_, "a\xF0\x9F\x98\x80" "b");
    ASSERT_EQ(run.glyphs.size(), 3u) << "a missing glyph must not be dropped";
    EXPECT_EQ(run.glyphs[0].glyph_id, face_->GlyphForCodepoint(static_cast<std::uint32_t>('a')));
    EXPECT_EQ(run.glyphs[1].glyph_id, FontFace::MissingGlyph());
    EXPECT_EQ(run.glyphs[2].glyph_id, face_->GlyphForCodepoint(static_cast<std::uint32_t>('b')));
    EXPECT_EQ(run.glyphs[1].cluster, 1u);
    EXPECT_EQ(run.glyphs[2].cluster, 5u);
    EXPECT_FLOAT_EQ(run.total_advance_x, run.glyphs[0].advance_x + run.glyphs[1].advance_x +
                                             run.glyphs[2].advance_x);
}

// Direction is applied to the buffer, so a right-to-left run comes back in
// visual order with the clusters reversed; the glyphs are the same set.
TEST_F(TextShapingTest, ReversesGlyphsForRightToLeftText) {
    ShapeOptions rtl{};
    rtl.direction = TextDirection::RightToLeft;

    const ShapedRun ltr_run = ShapeText(*face_, "abc");
    const ShapedRun rtl_run = ShapeText(*face_, "abc", rtl);

    ASSERT_EQ(rtl_run.glyphs.size(), 3u);
    ASSERT_EQ(rtl_run.glyphs.size(), ltr_run.glyphs.size());
    for (std::size_t i = 0; i < rtl_run.glyphs.size(); ++i) {
        EXPECT_EQ(rtl_run.glyphs[i].glyph_id, ltr_run.glyphs[rtl_run.glyphs.size() - 1 - i].glyph_id);
        EXPECT_EQ(rtl_run.glyphs[i].cluster, static_cast<std::uint32_t>(2 - i));
    }
}

class ShapingCacheTest : public TextShapingTest {};

TEST_F(ShapingCacheTest, ReturnsTheSameRunForTheSameKey) {
    ShapingCache cache;

    const auto first = cache.Shape(*face_, "cache me");
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(cache.Size(), 1u);
    EXPECT_EQ(cache.HitCount(), 0u);

    const auto second = cache.Shape(*face_, "cache me");
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second.get(), first.get()) << "a hit must reuse the cached run";
    EXPECT_EQ(cache.HitCount(), 1u);
    EXPECT_EQ(cache.Size(), 1u);
    EXPECT_EQ(GlyphIds(*second), GlyphIds(*first));
}

TEST_F(ShapingCacheTest, NeverReturnsAnotherStringsRun) {
    ShapingCache cache;

    // Different text, and the same text under different options, are different
    // keys. A cache keyed on a text hash alone could confuse the first pair.
    const auto av = cache.Shape(*face_, "AV");
    const auto va = cache.Shape(*face_, "VA");
    ASSERT_NE(av, nullptr);
    ASSERT_NE(va, nullptr);
    EXPECT_NE(GlyphIds(*av), GlyphIds(*va));

    ShapeOptions unkerned{};
    unkerned.kerning = false;
    const auto av_unkerned = cache.Shape(*face_, "AV", unkerned);
    ASSERT_NE(av_unkerned, nullptr);
    EXPECT_NE(av_unkerned.get(), av.get()) << "options are part of the key";
    EXPECT_NE(av_unkerned->total_advance_x, av->total_advance_x);

    EXPECT_EQ(cache.Size(), 3u);
    EXPECT_EQ(cache.HitCount(), 0u);
}

TEST_F(ShapingCacheTest, StaysBoundedUnderManyDistinctStrings) {
    constexpr std::size_t kMaxEntries = 4;
    constexpr std::size_t kDistinct = 64;
    ShapingCache cache{kMaxEntries};

    for (std::size_t i = 0; i < kDistinct; ++i) {
        const std::string text = "entry " + std::to_string(i);
        const auto run = cache.Shape(*face_, text);
        ASSERT_NE(run, nullptr);
        EXPECT_EQ(run->text, text);
        EXPECT_LE(cache.Size(), kMaxEntries) << "the cache must not grow without limit";
    }
    EXPECT_EQ(cache.Size(), kMaxEntries);
    EXPECT_EQ(cache.HitCount(), 0u) << "every string here is distinct";
}

// LRU, not arbitrary eviction: re-requesting an entry must save it from the
// next insertion, and the entry not touched since must be the one that goes.
TEST_F(ShapingCacheTest, EvictsTheLeastRecentlyUsedEntry) {
    ShapingCache cache{3};

    const auto a = cache.Shape(*face_, "A");
    const auto b = cache.Shape(*face_, "B");
    const auto c = cache.Shape(*face_, "C");
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);

    // Touching "A" makes "B" the least recently used.
    const auto a_again = cache.Shape(*face_, "A");
    EXPECT_EQ(a_again.get(), a.get());
    EXPECT_EQ(cache.HitCount(), 1u);

    const auto d = cache.Shape(*face_, "D");
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(cache.Size(), 3u);

    // "B" was evicted, so it shapes afresh; "A" is still the cached run.
    const auto b_again = cache.Shape(*face_, "B");
    ASSERT_NE(b_again, nullptr);
    EXPECT_NE(b_again.get(), b.get());
    EXPECT_EQ(cache.Shape(*face_, "A").get(), a.get());
}

TEST_F(ShapingCacheTest, KeepsAHandedOutRunAliveAcrossClear) {
    ShapingCache cache;
    const auto before = cache.Shape(*face_, "survives");
    ASSERT_NE(before, nullptr);
    const std::vector<std::uint32_t> glyphs = GlyphIds(*before);
    const float advance = before->total_advance_x;
    ASSERT_FALSE(glyphs.empty());

    cache.Clear();
    EXPECT_EQ(cache.Size(), 0u);

    // The caller's shared_ptr is what keeps the run alive; eviction and Clear
    // only drop the cache's own reference.
    EXPECT_EQ(GlyphIds(*before), glyphs);
    EXPECT_FLOAT_EQ(before->total_advance_x, advance);
    EXPECT_EQ(before->text, "survives");
}

TEST_F(ShapingCacheTest, ZeroCapacityDisablesCaching) {
    ShapingCache cache{0};

    const auto first = cache.Shape(*face_, "uncached");
    const auto second = cache.Shape(*face_, "uncached");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(cache.Size(), 0u);
    EXPECT_EQ(cache.HitCount(), 0u);
    EXPECT_EQ(GlyphIds(*first), GlyphIds(*second));
}

// The cache and the face are both shared state: shaping the same strings from
// every worker must produce exactly the single-threaded result. The cache is
// deliberately small so the workers also race on eviction.
TEST_F(ShapingCacheTest, IsSafeToShapeFromManyThreads) {
    constexpr std::size_t kRounds = 64;
    const std::vector<std::string> texts{"AV",   "fi",     "The quick brown fox", "a\xF0\x9F\x98\x80" "b",
                                         "wave", "kern me", "   ",              "\xC3\xA9t\xC3\xA9"};

    std::vector<ShapedRun> baseline;
    baseline.reserve(texts.size());
    for (const std::string& text : texts) {
        baseline.push_back(ShapeText(*face_, text));
    }

    ShapingCache cache{4};
    std::atomic<std::size_t> glyph_mismatches{0};
    std::atomic<std::size_t> advance_mismatches{0};
    std::atomic<std::size_t> cluster_mismatches{0};

    auto& pool = VulkanShared::JobScheduler::Global();
    pool.ParallelFor(kRounds, [&](std::size_t round) {
        const std::string& text = texts[round % texts.size()];
        const auto run = cache.Shape(*face_, text);
        const ShapedRun& expected = baseline[round % texts.size()];
        if (run->glyphs.size() != expected.glyphs.size()) {
            glyph_mismatches.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        for (std::size_t i = 0; i < expected.glyphs.size(); ++i) {
            if (run->glyphs[i].glyph_id != expected.glyphs[i].glyph_id) {
                glyph_mismatches.fetch_add(1, std::memory_order_relaxed);
            }
            if (run->glyphs[i].cluster != expected.glyphs[i].cluster) {
                cluster_mismatches.fetch_add(1, std::memory_order_relaxed);
            }
            if (run->glyphs[i].advance_x != expected.glyphs[i].advance_x) {
                advance_mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    pool.WaitForIdle();

    EXPECT_EQ(glyph_mismatches.load(), 0u) << "glyphs differed under concurrency";
    EXPECT_EQ(cluster_mismatches.load(), 0u) << "clusters differed under concurrency";
    EXPECT_EQ(advance_mismatches.load(), 0u) << "advances differed under concurrency";
    EXPECT_LE(cache.Size(), 4u) << "eviction raced the workers without losing the bound";
}

// The cache keys runs on the face, and it must not key them on the face's
// address: an allocator can hand a destroyed face's address to a different font,
// and two fonts both at resource version 1 would then share every entry. Face
// ids are never reused, which is what makes that impossible. Address reuse
// itself cannot be forced portably -- which is precisely why the key must not
// depend on it -- so this pins the invariant the cache relies on.
TEST(FontFaceIdentityTest, GivesEveryFaceAnIdentityThatIsNeverReused) {
    ResourceManager manager;
    auto handle = manager.LoadFromFile<FontResource>(
        TestFontPath(), ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    FontResource* resource = handle.Get();
    ASSERT_NE(resource, nullptr);

    std::vector<std::uint64_t> ids;
    std::vector<std::shared_ptr<FontFace>> faces;
    for (int i = 0; i < 8; ++i) {
        auto face = FontFace::Create(*resource);
        ASSERT_NE(face, nullptr);
        ids.push_back(face->UniqueId());
        faces.push_back(std::move(face));
    }

    const std::set<std::uint64_t> distinct(ids.begin(), ids.end());
    EXPECT_EQ(distinct.size(), ids.size()) << "a face id was handed out twice";

    // Stable while the face lives, and not recycled to a later face.
    EXPECT_EQ(faces.front()->UniqueId(), ids.front());
    faces.clear();

    auto later = FontFace::Create(*resource);
    ASSERT_NE(later, nullptr);
    EXPECT_EQ(distinct.count(later->UniqueId()), 0u)
        << "a new face reused a destroyed face's id";
}

// Two live faces of the same font are still two entries: the cache is keyed on
// the face, not on the text alone.
TEST_F(ShapingCacheTest, KeepsTwoFacesOfTheSameFontApart) {
    auto other = FontFace::Create(*handle_.Get());
    ASSERT_NE(other, nullptr);
    ASSERT_NE(other->UniqueId(), face_->UniqueId());

    ShapingCache cache;
    const auto first = cache.Shape(*face_, "AV");
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(cache.Size(), 1u);
    ASSERT_EQ(cache.HitCount(), 0u);

    const auto second = cache.Shape(*other, "AV");
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(cache.HitCount(), 0u) << "another face must not hit this face's entry";
    EXPECT_EQ(cache.Size(), 2u);

    // The first face's entry is still its own.
    const auto again = cache.Shape(*face_, "AV");
    EXPECT_EQ(again.get(), first.get());
    EXPECT_EQ(cache.HitCount(), 1u);
}

} // namespace
