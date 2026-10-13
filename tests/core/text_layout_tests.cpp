#include <gtest/gtest.h>

import std;
import std.compat;

import FileLoader.Types;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.Shaping;
import VulkanEngine.Text.Layout;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::LayoutOptions;
using VulkanEngine::Text::LayoutText;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeText;
using VulkanEngine::Text::TextAlign;
using VulkanEngine::Text::TextLayout;
using VulkanEngine::Text::ToPhysicalPixels;

constexpr float kPixelSize = 16.0f;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

[[nodiscard]] LayoutOptions OptionsFor(float max_width, TextAlign align = TextAlign::Left,
                                       float line_height_scale = 1.0f) {
    LayoutOptions options{};
    options.pixel_size = kPixelSize;
    options.max_width = max_width;
    options.align = align;
    options.line_height_scale = line_height_scale;
    return options;
}

// The engine's own answer for how wide a piece of text is: shaped and laid out
// with no wrap width, the one line it reports carries the measured width. All
// assertions read widths from layout results like this instead of
// re-implementing the engine's advance-summing arithmetic here.
[[nodiscard]] float MeasuredWidthOf(const FontFace& face, std::string_view text) {
    const TextLayout layout =
        LayoutText(face, ShapeText(face, std::string{text}), OptionsFor(0.0f));
    EXPECT_EQ(layout.lines.size(), 1u) << "unwrapped layout of '" << text << "'";
    return layout.lines.empty() ? 0.0f : layout.lines[0].width;
}

class TextLayoutTest : public ::testing::Test {
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

// A wrap width between the widest word and the narrowest adjacent pair forces
// one word per line: every pair of words is wider than the box, so the greedy
// wrapper has no choice. "alpha bravo charlie delta echo" therefore lays out as
// five lines, one per word, and each line keeps the space that ended it.
TEST_F(TextLayoutTest, WrapsAtBreakOpportunities) {
    const std::string text = "alpha bravo charlie delta echo";
    const std::vector<std::string> words{"alpha", "bravo", "charlie", "delta", "echo"};
    const ShapedRun run = ShapeText(*face_, text);
    ASSERT_EQ(run.glyphs.size(), text.size());

    // Word and pair widths come from the engine itself: lay each out alone and
    // read the line width it reports, instead of summing advances here.
    std::vector<float> word_widths;
    word_widths.reserve(words.size());
    float widest = 0.0f;
    for (const std::string& word : words) {
        const float width = MeasuredWidthOf(*face_, word);
        word_widths.push_back(width);
        widest = std::max(widest, width);
    }
    float narrowest_pair = MeasuredWidthOf(*face_, words[0] + ' ' + words[1]);
    for (std::size_t i = 2; i < words.size(); ++i) {
        narrowest_pair = std::min(
            narrowest_pair, MeasuredWidthOf(*face_, words[i - 1] + ' ' + words[i]));
    }
    ASSERT_GT(narrowest_pair, widest) << "a pair must outgrow a single word";
    const float max_width = (widest + narrowest_pair) * 0.5f;

    const TextLayout layout = LayoutText(*face_, run, OptionsFor(max_width));
    ASSERT_EQ(layout.lines.size(), words.size());
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::size_t first = text.find(words[i]);
        // Each line starts where its word starts and keeps the word, plus the
        // space that ended it (except the last line, which has no space).
        EXPECT_EQ(layout.lines[i].first_glyph, first);
        const std::size_t expected_count =
            words[i].size() + (i + 1 < words.size() ? 1u : 0u);
        EXPECT_EQ(layout.lines[i].glyph_count, expected_count);
        // The line measures exactly the word laid out alone -- the unmeasured
        // trailing space does not count -- and fits the wrap width.
        EXPECT_FLOAT_EQ(layout.lines[i].width, word_widths[i]);
        EXPECT_LE(layout.lines[i].width, max_width);
        EXPECT_FLOAT_EQ(layout.lines[i].offset_x, 0.0f);
    }

    // The lines partition the run: nothing is dropped and nothing is counted
    // twice.
    std::size_t covered = 0;
    for (const auto& line : layout.lines) {
        covered += line.glyph_count;
    }
    EXPECT_EQ(covered, run.glyphs.size());
    EXPECT_FLOAT_EQ(layout.width, widest);
}

TEST_F(TextLayoutTest, DoesNotWrapWithoutAMaxWidth) {
    constexpr std::string_view kText = "The quick brown fox";
    const ShapedRun run = ShapeText(*face_, kText);

    const TextLayout layout = LayoutText(*face_, run, OptionsFor(0.0f));
    ASSERT_EQ(layout.lines.size(), 1u);
    EXPECT_EQ(layout.lines[0].first_glyph, 0u);
    EXPECT_EQ(layout.lines[0].glyph_count, run.glyphs.size());
    // The whole run is one line, and the block width is that line's own width;
    // every number here is read back from the layout itself.
    EXPECT_GT(layout.lines[0].width, 0.0f);
    EXPECT_FLOAT_EQ(layout.width, layout.lines[0].width);
    EXPECT_FLOAT_EQ(layout.lines[0].offset_x, 0.0f);

    const float line_height =
        static_cast<float>(face_->Metrics().LineAdvance()) * face_->ScaleForSize(kPixelSize);
    EXPECT_FLOAT_EQ(layout.height, line_height);
}

// There is no hyphenation: a word with no legal break inside it is never split,
// even when it is far wider than the box. It moves onto a line of its own and
// overflows it.
TEST_F(TextLayoutTest, OverlongWordOverflowsItsOwnLine) {
    const std::string text = "a antidisestablishmentarianism";
    const ShapedRun run = ShapeText(*face_, text);
    const std::size_t word_start = text.find("antidisestablishmentarianism");
    // The box sits just above "a" as the engine itself measures it: wide
    // enough for the first word, far too narrow for the second.
    const float a_width = MeasuredWidthOf(*face_, "a");
    const float word_width =
        MeasuredWidthOf(*face_, "antidisestablishmentarianism");
    ASSERT_GT(word_width, a_width + 5.0f) << "the second word must not fit the box";
    const float max_width = a_width + 5.0f;

    const TextLayout layout = LayoutText(*face_, run, OptionsFor(max_width));
    ASSERT_EQ(layout.lines.size(), 2u);
    EXPECT_EQ(layout.lines[0].first_glyph, 0u);
    EXPECT_EQ(layout.lines[0].glyph_count, word_start);
    EXPECT_EQ(layout.lines[1].first_glyph, word_start);
    EXPECT_EQ(layout.lines[1].glyph_count, run.glyphs.size() - word_start);
    // The unbreakable word is not split: it measures exactly what it measures
    // alone and overflows its line, whatever the box asked for.
    EXPECT_GT(layout.lines[1].width, max_width);
    EXPECT_FLOAT_EQ(layout.lines[1].width, word_width);
    EXPECT_FLOAT_EQ(layout.lines[0].width, a_width);
    EXPECT_LE(layout.lines[0].width, max_width);
}

// Alignment places each line inside the box, and the box is the wrap width when
// there is one and the widest line otherwise. Every width here is read back
// from a layout the engine produced; the test computes no width itself.
TEST_F(TextLayoutTest, AlignsLinesInTheirBox) {
    constexpr std::string_view kText = "Hello\nHi";
    const ShapedRun run = ShapeText(*face_, kText);

    // No wrap width: the box is the widest line, which the engine reports as
    // the layout's own width.
    const TextLayout left = LayoutText(*face_, run, OptionsFor(0.0f, TextAlign::Left));
    const TextLayout center = LayoutText(*face_, run, OptionsFor(0.0f, TextAlign::Center));
    const TextLayout right = LayoutText(*face_, run, OptionsFor(0.0f, TextAlign::Right));
    ASSERT_EQ(left.lines.size(), 2u);
    ASSERT_EQ(center.lines.size(), 2u);
    ASSERT_EQ(right.lines.size(), 2u);
    ASSERT_GT(left.lines[0].width, left.lines[1].width);
    EXPECT_FLOAT_EQ(left.width, left.lines[0].width) << "the box is the widest line";

    EXPECT_FLOAT_EQ(left.lines[0].offset_x, 0.0f);
    EXPECT_FLOAT_EQ(left.lines[1].offset_x, 0.0f);
    EXPECT_FLOAT_EQ(center.lines[0].offset_x, 0.0f);
    EXPECT_FLOAT_EQ(center.lines[1].offset_x, (left.width - center.lines[1].width) * 0.5f);
    EXPECT_FLOAT_EQ(right.lines[0].offset_x, 0.0f);
    EXPECT_FLOAT_EQ(right.lines[1].offset_x, left.width - right.lines[1].width);

    // With a wrap width wider than any line, the box is that width instead.
    const float box_width = left.lines[0].width + 50.0f;
    const TextLayout boxed_center =
        LayoutText(*face_, run, OptionsFor(box_width, TextAlign::Center));
    const TextLayout boxed_right = LayoutText(*face_, run, OptionsFor(box_width, TextAlign::Right));
    ASSERT_EQ(boxed_center.lines.size(), 2u);
    ASSERT_EQ(boxed_right.lines.size(), 2u);
    // Wider boxes measure the same lines.
    EXPECT_FLOAT_EQ(boxed_center.lines[0].width, left.lines[0].width);
    EXPECT_FLOAT_EQ(boxed_center.lines[1].width, left.lines[1].width);
    EXPECT_FLOAT_EQ(boxed_center.lines[0].offset_x, (box_width - left.lines[0].width) * 0.5f);
    EXPECT_FLOAT_EQ(boxed_center.lines[1].offset_x,
                    (box_width - boxed_center.lines[1].width) * 0.5f);
    EXPECT_FLOAT_EQ(boxed_right.lines[0].offset_x, box_width - boxed_right.lines[0].width);
    EXPECT_FLOAT_EQ(boxed_right.lines[1].offset_x, box_width - boxed_right.lines[1].width);

    // An overflowing line is wider than the wrap width, so the box is that
    // line's width; right-aligning it must not push it out of its own box.
    const std::string overflowing = "antidisestablishmentarianism";
    const ShapedRun long_run = ShapeText(*face_, overflowing);
    const TextLayout overflowing_layout =
        LayoutText(*face_, long_run, OptionsFor(10.0f, TextAlign::Right));
    ASSERT_EQ(overflowing_layout.lines.size(), 1u);
    EXPECT_GT(overflowing_layout.width, 10.0f);
    EXPECT_FLOAT_EQ(overflowing_layout.lines[0].offset_x, 0.0f);
}

// Explicit newlines break without a wrap width, an empty line between two of
// them survives, and a run that ends with one does not grow a phantom line
// after it.
TEST_F(TextLayoutTest, BreaksOnExplicitNewlines) {
    constexpr std::string_view kText = "abc\ndef";
    const ShapedRun run = ShapeText(*face_, kText);
    const TextLayout layout = LayoutText(*face_, run, OptionsFor(0.0f));

    ASSERT_EQ(layout.lines.size(), 2u);
    EXPECT_EQ(layout.lines[0].first_glyph, 0u);
    EXPECT_EQ(layout.lines[0].glyph_count, 4u) << "the newline glyph ends the first line";
    ASSERT_LT(run.glyphs[3].cluster, run.text.size());
    EXPECT_EQ(run.text[run.glyphs[3].cluster], '\n') << "glyph 3 is the newline";
    // The newline is shaped as Lato's .notdef, so it must not be measured; the
    // line is exactly as wide as the letters alone measure on their own.
    EXPECT_FLOAT_EQ(layout.lines[0].width, MeasuredWidthOf(*face_, "abc"));
    EXPECT_EQ(layout.lines[1].first_glyph, 4u);
    EXPECT_EQ(layout.lines[1].glyph_count, 3u);
    EXPECT_FLOAT_EQ(layout.lines[1].width, MeasuredWidthOf(*face_, "def"));

    const TextLayout empty_between = LayoutText(*face_, ShapeText(*face_, "a\n\nb"),
                                                OptionsFor(0.0f));
    ASSERT_EQ(empty_between.lines.size(), 3u);
    EXPECT_FLOAT_EQ(empty_between.lines[1].width, 0.0f) << "the middle line is empty";

    const TextLayout leading = LayoutText(*face_, ShapeText(*face_, "\nab"), OptionsFor(0.0f));
    ASSERT_EQ(leading.lines.size(), 2u);
    EXPECT_FLOAT_EQ(leading.lines[0].width, 0.0f);

    // "ab\n" is one line, not two: the newline terminates the line it ends.
    const TextLayout trailing = LayoutText(*face_, ShapeText(*face_, "ab\n"), OptionsFor(0.0f));
    ASSERT_EQ(trailing.lines.size(), 1u);
    EXPECT_EQ(trailing.lines[0].first_glyph, 0u);
    EXPECT_EQ(trailing.lines[0].glyph_count, 3u);
    EXPECT_FLOAT_EQ(trailing.lines[0].width, MeasuredWidthOf(*face_, "ab"));
}

TEST_F(TextLayoutTest, EmptyInputYieldsNoLines) {
    const ShapedRun empty = ShapeText(*face_, "");
    const TextLayout layout = LayoutText(*face_, empty, OptionsFor(100.0f));

    EXPECT_TRUE(layout.lines.empty());
    EXPECT_FLOAT_EQ(layout.width, 0.0f);
    EXPECT_FLOAT_EQ(layout.height, 0.0f);
}

// Line height is the face's own line advance scaled to pixels and multiplied by
// the caller's scale, and the block's height counts the lines, not the gaps.
TEST_F(TextLayoutTest, MeasuresLineHeightFromTheFaceMetrics) {
    constexpr float kLineHeightScale = 1.5f;
    const ShapedRun run = ShapeText(*face_, "a\nb\nc");

    const TextLayout layout = LayoutText(*face_, run, OptionsFor(0.0f, TextAlign::Left, kLineHeightScale));
    ASSERT_EQ(layout.lines.size(), 3u);

    const float expected_line_height = static_cast<float>(face_->Metrics().LineAdvance()) *
                                       face_->ScaleForSize(kPixelSize) * kLineHeightScale;
    EXPECT_GT(expected_line_height, 0.0f);
    for (std::size_t i = 0; i < layout.lines.size(); ++i) {
        EXPECT_FLOAT_EQ(layout.lines[i].advance_y, static_cast<float>(i) * expected_line_height);
    }
    EXPECT_FLOAT_EQ(layout.height, 3.0f * expected_line_height);
    // One glyph per line, and the block width is the widest of the lines the
    // layout itself reports.
    EXPECT_FLOAT_EQ(layout.width, std::max({layout.lines[0].width,
                                            layout.lines[1].width,
                                            layout.lines[2].width}));
}

// The boundary is strict: a line that fills the box exactly is not wrapped.
// "ab c" has a single break opportunity, after the space, so this isolates the
// comparison. Every width comes from the engine: the exact-fit box is the
// unwrapped layout's own width, and the wrapping box is narrower than it by a
// real glyph advance read from the shaped run.
TEST_F(TextLayoutTest, ALineThatExactlyFitsDoesNotWrap) {
    constexpr std::string_view kText = "ab c";
    const ShapedRun run = ShapeText(*face_, kText);
    const TextLayout unwrapped = LayoutText(*face_, run, OptionsFor(0.0f));
    ASSERT_EQ(unwrapped.lines.size(), 1u);
    const float full_width = unwrapped.lines[0].width;
    ASSERT_GT(full_width, 0.0f);

    const TextLayout exact = LayoutText(*face_, run, OptionsFor(full_width));
    ASSERT_EQ(exact.lines.size(), 1u) << "a line exactly as wide as the box still fits";
    EXPECT_FLOAT_EQ(exact.lines[0].width, full_width);
    EXPECT_FLOAT_EQ(exact.width, full_width);

    // Cluster 2 is the space. Asking for a box one real glyph advance narrower
    // makes the text wider than the box by exactly that advance, so it wraps.
    ASSERT_LT(run.glyphs[2].cluster, run.text.size());
    ASSERT_EQ(run.text[run.glyphs[2].cluster], ' ');
    const float space_advance =
        run.glyphs[2].advance_x * face_->ScaleForSize(kPixelSize);
    ASSERT_GT(space_advance, 0.0f);
    const TextLayout tight = LayoutText(*face_, run, OptionsFor(full_width - space_advance));
    ASSERT_EQ(tight.lines.size(), 2u);
    EXPECT_EQ(tight.lines[0].glyph_count, 3u) << "the space stays with the line it ends";
    EXPECT_EQ(tight.lines[1].first_glyph, 3u);
    // Both halves fit the narrower box: the unmeasured trailing space is what
    // got squeezed out, so neither line is wider than what was asked for.
    EXPECT_GT(tight.lines[0].width, 0.0f);
    EXPECT_LE(tight.lines[0].width, full_width - space_advance);
    EXPECT_GT(tight.lines[1].width, 0.0f);
    EXPECT_LE(tight.lines[1].width, full_width - space_advance);
    EXPECT_FLOAT_EQ(tight.width, std::max(tight.lines[0].width, tight.lines[1].width));
}

// The whitespace that triggers a break stays in the line's glyph range but is
// not measured: it cannot push that line past max_width and it cannot become an
// extra line of its own.
TEST_F(TextLayoutTest, TrailingWhitespaceAtABreakIsNotMeasured) {
    constexpr std::string_view kText = "hello world ";
    const ShapedRun run = ShapeText(*face_, kText);
    // The box sits one pixel below "hello world" as the engine measures it, so
    // the pair cannot share a line -- and stays above "hello" alone.
    const float hello_width = MeasuredWidthOf(*face_, "hello");
    const float world_width = MeasuredWidthOf(*face_, "world");
    ASSERT_GT(world_width, 1.0f) << "the box must remain wider than 'hello' alone";
    const float max_width = hello_width + world_width - 1.0f;

    const TextLayout layout = LayoutText(*face_, run, OptionsFor(max_width));
    ASSERT_EQ(layout.lines.size(), 2u) << "the trailing space must not add a third line";

    // The space that caused the break is inside the first line's glyph range..
    EXPECT_EQ(layout.lines[0].first_glyph, 0u);
    EXPECT_EQ(layout.lines[0].glyph_count, 6u) << "the break stays with the line it ends";
    ASSERT_LT(run.glyphs[5].cluster, run.text.size());
    EXPECT_EQ(run.text[run.glyphs[5].cluster], ' ');
    // ..but it is not measured: the line is exactly as wide as "hello" alone
    // and within the box.
    EXPECT_FLOAT_EQ(layout.lines[0].width, hello_width);
    EXPECT_LE(layout.lines[0].width, max_width);

    // The final space is still in the second line's glyph range, unmeasured.
    EXPECT_EQ(layout.lines[1].glyph_count, 6u) << "the final space is still in the line";
    ASSERT_LT(run.glyphs[11].cluster, run.text.size());
    EXPECT_EQ(run.text[run.glyphs[11].cluster], ' ');
    EXPECT_FLOAT_EQ(layout.lines[1].width, world_width);
    EXPECT_LE(layout.lines[1].width, max_width);

    EXPECT_FLOAT_EQ(layout.width, std::max(hello_width, world_width));
}

// ── Logical points -> physical pixels ───────────────────────────────────────
//
// The scale conversion is the seam that lets a caller state a UI size once and
// get the same apparent size on a 1.0 output and on a 1.5 one. It is asserted
// separately from any rasterization because the failure it prevents -- text that
// silently shrinks on a scaled display -- is invisible on an unscaled one.

TEST(ToPhysicalPixelsTest, ScaleOneIsTheIdentity) {
    const auto request = ToPhysicalPixels(18.0f, 24.0f, 110.0f, 460.0f, 1.0f);
    EXPECT_FLOAT_EQ(request.pixel_size, 18.0f);
    EXPECT_FLOAT_EQ(request.x, 24.0f);
    EXPECT_FLOAT_EQ(request.y, 110.0f);
    EXPECT_FLOAT_EQ(request.max_width, 460.0f);
}

TEST(ToPhysicalPixelsTest, FractionalScaleScalesSizeOriginAndWrapWidth) {
    // 1.5 is the actual scale of the fractional-scaling panel this was written
    // against, so it is the case that has to be right.
    const auto request = ToPhysicalPixels(18.0f, 24.0f, 110.0f, 460.0f, 1.5f);
    EXPECT_FLOAT_EQ(request.pixel_size, 27.0f);
    EXPECT_FLOAT_EQ(request.x, 36.0f);
    EXPECT_FLOAT_EQ(request.y, 165.0f);
    EXPECT_FLOAT_EQ(request.max_width, 690.0f);
}

TEST(ToPhysicalPixelsTest, IntegerScaleIsExact) {
    const auto request = ToPhysicalPixels(14.0f, 10.0f, 20.0f, 0.0f, 2.0f);
    EXPECT_FLOAT_EQ(request.pixel_size, 28.0f);
    EXPECT_FLOAT_EQ(request.x, 20.0f);
    EXPECT_FLOAT_EQ(request.y, 40.0f);
}

// "No wrapping" is the absence of a distance, not a distance of zero: scaling it
// would turn every unwrapped request into a wrapped one at width 0.
TEST(ToPhysicalPixelsTest, ZeroWrapWidthStaysUnwrappedAtEveryScale) {
    for (const float scale : {1.0f, 1.25f, 1.5f, 2.0f}) {
        const auto request = ToPhysicalPixels(16.0f, 0.0f, 0.0f, 0.0f, scale);
        EXPECT_FLOAT_EQ(request.max_width, 0.0f) << "scale " << scale;
    }
}

// A platform that cannot answer must not erase the text. A literal zero scale
// would collapse the pixel size and every glyph would vanish.
TEST(ToPhysicalPixelsTest, NonPositiveScaleFallsBackToOne) {
    for (const float scale : {0.0f, -1.0f}) {
        const auto request = ToPhysicalPixels(20.0f, 5.0f, 7.0f, 100.0f, scale);
        EXPECT_FLOAT_EQ(request.pixel_size, 20.0f) << "scale " << scale;
        EXPECT_FLOAT_EQ(request.x, 5.0f) << "scale " << scale;
        EXPECT_FLOAT_EQ(request.y, 7.0f) << "scale " << scale;
        EXPECT_FLOAT_EQ(request.max_width, 100.0f) << "scale " << scale;
    }
}

// The point of the conversion: a request expressed in points occupies the same
// fraction of the window at every scale. If this drifts, UI text changes size
// when the display scale does.
TEST(ToPhysicalPixelsTest, ProportionalSizeIsScaleInvariant) {
    constexpr float kPointSize = 18.0f;
    constexpr float kWindowPoints = 720.0f;
    for (const float scale : {1.0f, 1.25f, 1.5f, 2.0f}) {
        const auto request = ToPhysicalPixels(kPointSize, 0.0f, 0.0f, 0.0f, scale);
        const float fraction = request.pixel_size / (kWindowPoints * scale);
        EXPECT_NEAR(fraction, kPointSize / kWindowPoints, 1e-6f) << "scale " << scale;
    }
}

} // namespace
