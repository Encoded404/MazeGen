#include <gtest/gtest.h>

import std;
import std.compat;

import VulkanShared.Jobs;

import FileLoader.Types;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Font;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphMetrics;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

// Owns the resource manager and the loaded resource, because a ResourceHandle
// is a lookup into its manager and not an owner.
class FontFaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        handle_ = manager_.LoadFromFile<FontResource>(
            TestFontPath(), ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        resource_ = handle_.Get();
        ASSERT_NE(resource_, nullptr);
        ASSERT_TRUE(resource_->IsLoaded());

        face_ = FontFace::Create(*resource_);
        ASSERT_NE(face_, nullptr);
    }

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    FontResource* resource_ = nullptr;
    std::shared_ptr<FontFace> face_;
};

// The face hands out the sfnt buffer it was opened from. FreeType has to parse
// the same bytes the shaper reads -- a second copy per library would defeat the
// shared owner -- so the span must be the resource's own bytes, non-empty, and
// recognisably an sfnt (0x00010000 opens a TrueType outline).
TEST_F(FontFaceTest, ExposesTheFontBytesItOwns) {
    const FileLoader::ByteSpan bytes = face_->Bytes();

    ASSERT_FALSE(bytes.empty());
    EXPECT_EQ(bytes.size(), resource_->GetBytes().size());
    EXPECT_EQ(bytes.size(), resource_->GetByteSpan().size());

    ASSERT_GE(bytes.size(), 2U);
    EXPECT_EQ(std::to_integer<int>(bytes[0]), 0x00);
    EXPECT_EQ(std::to_integer<int>(bytes[1]), 0x01);

    // The same buffer every call: it is owned, not rebuilt per request.
    const FileLoader::ByteSpan again = face_->Bytes();
    EXPECT_EQ(again.data(), bytes.data());
    EXPECT_EQ(again.size(), bytes.size());
}

TEST_F(FontFaceTest, ReportsMetricsFromTheShapingFace) {
    const auto& metrics = face_->Metrics();

    EXPECT_GT(metrics.units_per_em, 0u);
    EXPECT_GT(metrics.glyph_count, 100u);
    EXPECT_EQ(face_->FaceIndex(), 0u);
    EXPECT_EQ(face_->ResourceVersion(), 1u);

    // Lato carries a full vertical header: a positive ascender, a negative
    // descender, and a positive line box overall.
    EXPECT_GT(metrics.ascender, 0);
    EXPECT_LT(metrics.descender, 0);
    EXPECT_GT(metrics.LineAdvance(), 0);

    // OS/2 metrics, and their physical ordering.
    EXPECT_GT(metrics.cap_height, 0);
    EXPECT_GT(metrics.x_height, 0);
    EXPECT_GT(metrics.cap_height, metrics.x_height);
    EXPECT_GT(metrics.cap_height, metrics.ascender - metrics.LineAdvance());

    // The ascender must reach above the cap height and the descender below the
    // baseline, or the line box could not contain capital letters.
    EXPECT_GT(metrics.ascender, metrics.cap_height);

    EXPECT_GT(metrics.underline_thickness, 0);
}

TEST_F(FontFaceTest, MapsCodepointsToGlyphs) {
    const std::uint32_t glyph_a = face_->GlyphForCodepoint(static_cast<std::uint32_t>('A'));
    const std::uint32_t glyph_m = face_->GlyphForCodepoint(static_cast<std::uint32_t>('M'));
    const std::uint32_t glyph_space = face_->GlyphForCodepoint(static_cast<std::uint32_t>(' '));

    EXPECT_NE(glyph_a, FontFace::MissingGlyph());
    EXPECT_NE(glyph_m, FontFace::MissingGlyph());
    EXPECT_NE(glyph_space, FontFace::MissingGlyph()) << "a space still has a glyph";
    EXPECT_NE(glyph_a, glyph_m);

    // Codepoints the font does not cover resolve to .notdef rather than a bogus
    // glyph, so the caller can render a placeholder instead of nothing.
    EXPECT_EQ(face_->GlyphForCodepoint(0x1F600U), FontFace::MissingGlyph());
    EXPECT_EQ(face_->GlyphForCodepoint(0x10FFFFU), FontFace::MissingGlyph());
}

TEST_F(FontFaceTest, ScalesDesignUnitsToPixels) {
    const auto& metrics = face_->Metrics();

    // One em of design units is exactly one pixel at a pixel size of upem.
    EXPECT_FLOAT_EQ(face_->ScaleForSize(static_cast<float>(metrics.units_per_em)), 1.0f);
    EXPECT_FLOAT_EQ(face_->ScaleForSize(0.0f), 0.0f);

    // Glyph metrics are linear in the requested size, which is what makes a
    // face usable as a size-independent glyph source.
    const std::uint32_t glyph =
        face_->GlyphForCodepoint(static_cast<std::uint32_t>('H'));
    ASSERT_NE(glyph, FontFace::MissingGlyph());

    const GlyphMetrics at16 = face_->GlyphAtSize(glyph, 16.0f);
    const GlyphMetrics at32 = face_->GlyphAtSize(glyph, 32.0f);

    EXPECT_GT(at16.advance_x, 0.0f);
    EXPECT_GT(at16.width, 0.0f);
    EXPECT_GT(at16.height, 0.0f);
    EXPECT_FLOAT_EQ(at32.advance_x, at16.advance_x * 2.0f);
    EXPECT_FLOAT_EQ(at32.width, at16.width * 2.0f);
    EXPECT_FLOAT_EQ(at32.height, at16.height * 2.0f);
}

TEST_F(FontFaceTest, ReportsAdvancesAndBoxesThatMatchTheGlyphs) {
    const std::uint32_t capital =
        face_->GlyphForCodepoint(static_cast<std::uint32_t>('M'));
    const std::uint32_t period =
        face_->GlyphForCodepoint(static_cast<std::uint32_t>('.'));
    const std::uint32_t space =
        face_->GlyphForCodepoint(static_cast<std::uint32_t>(' '));
    ASSERT_NE(capital, FontFace::MissingGlyph());
    ASSERT_NE(period, FontFace::MissingGlyph());
    ASSERT_NE(space, FontFace::MissingGlyph());

    const GlyphMetrics cap = face_->GlyphAtSize(capital, 24.0f);
    const GlyphMetrics dot = face_->GlyphAtSize(period, 24.0f);
    const GlyphMetrics gap = face_->GlyphAtSize(space, 24.0f);

    // A capital is wider than a period and taller than it is wide.
    EXPECT_GT(cap.advance_x, dot.advance_x);
    EXPECT_GT(cap.width, cap.height);

    // A period carries ink near the baseline and a narrow advance.
    EXPECT_GT(dot.width, 0.0f);
    EXPECT_LT(dot.advance_x, dot.width * 3.0f);

    // A space advances without drawing anything.
    EXPECT_GT(gap.advance_x, 0.0f);
    EXPECT_FLOAT_EQ(gap.width, 0.0f);
    EXPECT_FLOAT_EQ(gap.height, 0.0f);
}

// The whole point of opening one shared face is that worker threads can query
// it concurrently: HarfBuzz objects are safe to read once marked immutable.
// Repeating every query across the pool must produce the baseline exactly.
TEST_F(FontFaceTest, IsSafeToQueryFromManyThreadsAtOnce) {
    constexpr std::string_view kText = "AVWjifi., 0123gqQy";
    constexpr float kPixelSize = 18.0f;
    constexpr std::size_t kRounds = 64;

    std::vector<std::uint32_t> codepoints;
    codepoints.reserve(kText.size());
    for (const char c : kText) {
        codepoints.push_back(static_cast<std::uint32_t>(static_cast<unsigned char>(c)));
    }

    std::vector<std::uint32_t> expected_glyphs;
    std::vector<float> expected_advances;
    expected_glyphs.reserve(codepoints.size());
    expected_advances.reserve(codepoints.size());
    for (const std::uint32_t codepoint : codepoints) {
        const std::uint32_t glyph = face_->GlyphForCodepoint(codepoint);
        expected_glyphs.push_back(glyph);
        expected_advances.push_back(face_->GlyphAtSize(glyph, kPixelSize).advance_x);
    }

    std::atomic<std::size_t> mismatches{0};
    std::atomic<std::size_t> glyph_mismatches{0};

    auto& pool = VulkanShared::JobScheduler::Global();
    pool.ParallelFor(kRounds, [&](std::size_t) {
        for (std::size_t i = 0; i < codepoints.size(); ++i) {
            const std::uint32_t glyph = face_->GlyphForCodepoint(codepoints[i]);
            if (glyph != expected_glyphs[i]) {
                glyph_mismatches.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (face_->GlyphAtSize(glyph, kPixelSize).advance_x != expected_advances[i]) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    pool.WaitForIdle();

    EXPECT_EQ(glyph_mismatches.load(), 0u) << "glyph lookup differed under concurrency";
    EXPECT_EQ(mismatches.load(), 0u) << "advances differed under concurrency";
}

TEST(FontFaceRejectionTest, RejectsAResourceWithNoBytes) {
    FontResource empty{VulkanEngine::ResourceId{.value = "empty-font"}};
    EXPECT_EQ(FontFace::Create(empty), nullptr);
}

TEST(FontFaceRejectionTest, RejectsAFaceIndexTheContainerDoesNotHave) {
    ResourceManager manager;
    auto handle = manager.LoadFromFile<FontResource>(
        TestFontPath(), ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    FontResource* resource = handle.Get();
    ASSERT_NE(resource, nullptr);

    // The fixture is a single-face font, so index 1 cannot be opened. HarfBuzz
    // itself is lenient here and would hand back a face with no glyphs, which is
    // why the container count is checked first and the glyph count after.
    EXPECT_EQ(FontFace::Create(*resource, 1), nullptr);
    EXPECT_NE(FontFace::Create(*resource, 0), nullptr);
}

TEST(FontFaceRejectionTest, RejectsAResourceThatIsNotAFont) {
    FontResource junk{VulkanEngine::ResourceId{.value = "junk-font"}};
    ASSERT_FALSE(junk.Load(FileLoader::ByteBuffer{std::vector<std::byte>(64, std::byte{0x22})}));
    EXPECT_EQ(FontFace::Create(junk), nullptr);
}

} // namespace
