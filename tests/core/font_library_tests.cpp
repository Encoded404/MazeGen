#include <gtest/gtest.h>

#include <hb-gpu.h>
#include <msdfgen.h>

import std;

import VulkanEngine.Text.FontLibrary;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Text stack gate.
//
// These tests pin the three text libraries the engine links, at the points
// where a packaging change would otherwise pass unnoticed:
//
//   * FreeType/HarfBuzz versions, which are not observable from a shipped
//     binary and whose header/library agreement is a plausible packaging bug.
//   * hb-gpu availability. The GPU glyph backend (HarfBuzz's implementation of
//     the Slug algorithm) lives in a SEPARATE library, libharfbuzz-gpu, and
//     libharfbuzz.a defines none of the hb_gpu_* symbols. Upstream's vcpkg
//     CMake config does not export that library, so the project overlay adds
//     the target. This test fails to build or link if that ever regresses,
//     rather than silently dropping the backend.
//   * msdfgen, which has no version API at all and so is proven by generating
//     a distance field and checking it describes the shape that was fed in.
// ─────────────────────────────────────────────────────────────────────────────

TEST(FontLibraryTest, ReportsLinkedFreeTypeAndHarfBuzz) {
    const VulkanEngine::Text::LibraryVersions versions =
        VulkanEngine::Text::GetLibraryVersions();

    EXPECT_NE(versions.freetype_runtime_major, 0u)
        << "FreeType could not be initialized, so no rasterizer is available";
    EXPECT_TRUE(versions.FreeTypeHeaderMatchesRuntime())
        << "headers are " << versions.freetype_header_major << "."
        << versions.freetype_header_minor << "." << versions.freetype_header_patch
        << " but the linked library is " << versions.freetype_runtime_major << "."
        << versions.freetype_runtime_minor << "." << versions.freetype_runtime_patch;

    // hb-gpu (the GPU glyph encoder) was introduced in HarfBuzz 14.0.0, and the
    // project overlay raises the port above the registry baseline for it.
    EXPECT_GE(versions.harfbuzz_major, 14u);

    // The summary must serve the function GetLibraryVersions() serves -- name a
    // runtime version for every library the API tracks -- but its exact wording
    // is presentation and is not pinned here. The expected content is rendered
    // from the API's own data, so a rephrased diagnostic that still reports the
    // linked versions passes, and one that dropped a library or drifted from
    // GetLibraryVersions() fails. A real header/runtime mismatch cannot be
    // fabricated in-process, so the (optional) mismatch warning has no wording
    // assertion either.
    const std::string summary = VulkanEngine::Text::FormatLibraryVersions();
    EXPECT_FALSE(summary.empty()) << summary;

    const auto contains = [&summary](const std::string& needle) {
        return summary.find(needle) != std::string::npos;
    };
    const auto triple = [](std::uint32_t major, std::uint32_t minor, std::uint32_t patch) {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    };
    EXPECT_TRUE(contains(triple(versions.freetype_runtime_major,
                                versions.freetype_runtime_minor,
                                versions.freetype_runtime_patch)))
        << summary;
    EXPECT_TRUE(contains(triple(versions.harfbuzz_major, versions.harfbuzz_minor,
                                versions.harfbuzz_micro)))
        << summary;
}

// A second read must agree with the first: GetLibraryVersions() owns the
// temporary FreeType library it creates, so it is safe to call repeatedly and
// leaves no process-global state behind.
TEST(FontLibraryTest, VersionReadIsRepeatableAndSideEffectFree) {
    const VulkanEngine::Text::LibraryVersions first =
        VulkanEngine::Text::GetLibraryVersions();
    const VulkanEngine::Text::LibraryVersions second =
        VulkanEngine::Text::GetLibraryVersions();

    EXPECT_EQ(first.freetype_runtime_major, second.freetype_runtime_major);
    EXPECT_EQ(first.freetype_runtime_minor, second.freetype_runtime_minor);
    EXPECT_EQ(first.freetype_runtime_patch, second.freetype_runtime_patch);
    EXPECT_EQ(first.harfbuzz_major, second.harfbuzz_major);
    EXPECT_EQ(first.harfbuzz_minor, second.harfbuzz_minor);
    EXPECT_EQ(first.harfbuzz_micro, second.harfbuzz_micro);
}

// The encoder and the shader sources the Slug backend is built on must exist in
// the linked library. HLSL is the variant the engine feeds to Slang.
TEST(FontLibraryTest, HbGpuEncoderAndHlslShaderSourcesAreAvailable) {
    hb_gpu_draw_t* draw = hb_gpu_draw_create_or_fail();
    ASSERT_NE(draw, nullptr) << "libharfbuzz-gpu encoder is not linked";

    const char* shared_fragment =
        hb_gpu_shader_source(HB_GPU_SHADER_STAGE_FRAGMENT, HB_GPU_SHADER_LANG_HLSL);
    const char* draw_fragment =
        hb_gpu_draw_shader_source(HB_GPU_SHADER_STAGE_FRAGMENT, HB_GPU_SHADER_LANG_HLSL);
    const char* shared_vertex =
        hb_gpu_shader_source(HB_GPU_SHADER_STAGE_VERTEX, HB_GPU_SHADER_LANG_HLSL);

    ASSERT_NE(shared_fragment, nullptr);
    ASSERT_NE(draw_fragment, nullptr);
    ASSERT_NE(shared_vertex, nullptr);

    // The shared helpers carry the atlas accessor and the vertex-stage dilation;
    // the draw stage carries the coverage math. Both are concatenated into the
    // engine's Slang port, so both must be non-empty.
    EXPECT_FALSE(std::string_view(shared_fragment).empty())
        << "no shared fragment helpers for HLSL";
    EXPECT_FALSE(std::string_view(draw_fragment).empty())
        << "no draw fragment helpers for HLSL";
    EXPECT_FALSE(std::string_view(shared_vertex).empty())
        << "no shared vertex helpers for HLSL";

    // The draw-specific vertex source is documented as empty (all vertex work
    // lives in the shared helpers), and the port concatenates it
    // unconditionally, so it must at least be a valid string.
    const char* draw_vertex =
        hb_gpu_draw_shader_source(HB_GPU_SHADER_STAGE_VERTEX, HB_GPU_SHADER_LANG_HLSL);
    ASSERT_NE(draw_vertex, nullptr);

    hb_gpu_draw_destroy(draw);
}

// msdfgen has no version API, so the link is proven by use: build a square
// directly in pixel coordinates (identity projection) and check that the
// generated field actually describes it. The assertions are aggregate and
// symmetric so they hold regardless of msdfgen's y-axis or normalization
// conventions.
TEST(FontLibraryTest, MsdfgenGeneratesASignedFieldForAKnownShape) {
    constexpr int kSize = 32;
    constexpr double kLo = 8.0;
    constexpr double kHi = 24.0;

    msdfgen::Shape shape;
    // The square below is expressed in bitmap coordinates: x to the right, y
    // downwards, which is how the field is indexed at the bottom of this test.
    shape.inverseYAxis = true;
    shape.contours.emplace_back();
    msdfgen::Contour& contour = shape.contours.back();
    contour.addEdge(msdfgen::EdgeHolder(msdfgen::Point2(kLo, kLo), msdfgen::Point2(kHi, kLo)));
    contour.addEdge(msdfgen::EdgeHolder(msdfgen::Point2(kHi, kLo), msdfgen::Point2(kHi, kHi)));
    contour.addEdge(msdfgen::EdgeHolder(msdfgen::Point2(kHi, kHi), msdfgen::Point2(kLo, kHi)));
    contour.addEdge(msdfgen::EdgeHolder(msdfgen::Point2(kLo, kHi), msdfgen::Point2(kLo, kLo)));
    // A single square contour carries no winding information, so it is left
    // unoriented (even-odd fill) and msdfgen orients it for the non-zero
    // winding rule. Without this the fill rule is inverted and the generator
    // reports the exterior as the interior.
    shape.orientContours();
    msdfgen::edgeColoringSimple(shape, 3.0);

    msdfgen::Bitmap<float, 3> field(kSize, kSize);
    msdfgen::generateMSDF(
        field, shape,
        msdfgen::Projection(msdfgen::Vector2(1.0, 1.0), msdfgen::Vector2(0.0, 0.0)),
        msdfgen::Range(2.0));

    // The reconstruction rule for a multi-channel field: the median of the
    // three channels is the signed distance, and 0.5 is the shape boundary.
    const auto median = [](float r, float g, float b) {
        return std::max(std::min(r, g), std::min(std::max(r, g), b));
    };

    std::size_t inside = 0;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const float* texel = field(x, y);
            if (median(texel[0], texel[1], texel[2]) > 0.5f) {
                ++inside;
                centroid_x += static_cast<double>(x);
                centroid_y += static_cast<double>(y);
            }
        }
    }

    const std::size_t total = static_cast<std::size_t>(kSize) * kSize;
    // A 16x16 square in a 32x32 field covers about a quarter of it.
    ASSERT_GT(inside, total / 8) << "the generated field has no interior";
    EXPECT_LT(inside, total / 2) << "the generated field is not bounded";

    centroid_x /= static_cast<double>(inside);
    centroid_y /= static_cast<double>(inside);
    EXPECT_NEAR(centroid_x, 16.0, 2.0) << "interior is not centred horizontally";
    EXPECT_NEAR(centroid_y, 16.0, 2.0) << "interior is not centred vertically";
}

} // namespace
