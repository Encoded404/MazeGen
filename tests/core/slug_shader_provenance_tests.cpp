#include <gtest/gtest.h>

// The hb-gpu shader sources are read through the third-party C API directly,
// exactly like the text-stack gate test reads hb-gpu.h: the engine links the
// library PRIVATE and never re-exports these headers.
#include <hb-gpu.h>

import std;
import std.compat;

namespace {

// FNV-1a over the exact source string HarfBuzz hands back.
[[nodiscard]] std::uint64_t HashSource(const char* source) {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    if (source == nullptr) {
        return hash;
    }
    for (const char* p = source; *p != '\0'; ++p) {
        hash ^= static_cast<std::uint8_t>(*p);
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// HarfBuzz, HB_GPU_SHADER_LANG_HLSL (shared fragment re-pinned at 14.5.1 for
// the leftward-ray sign fix ported into slug_gpu.slang). These are
// version-pinned facts, not definitions of correctness: they are the source
// the committed port was made from, and a HarfBuzz bump that changes them
// means the port must be redone.
constexpr std::uint64_t kSharedFragmentSourceHash = 0x6A77F5F58F201D7ULL;
constexpr std::uint64_t kSharedVertexSourceHash = 0xDF0141F02DE42987ULL;
constexpr std::uint64_t kDrawFragmentSourceHash = 0x5F5C67FCD6627ECBULL;

[[nodiscard]] std::string ReadPort() {
    const std::filesystem::path path =
        std::filesystem::path{VKENGINE_ENGINE_SHADER_SOURCE_DIR} / "slug_gpu.slang";
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

} // namespace

// Pins the upstream shader sources the Slug port was derived from. On a HarfBuzz
// version bump that changes any of them this fails with the instruction to
// re-port; that is the whole point, because the coverage math is the Slug
// algorithm and silently keeping the old copy would render subtly different
// glyphs.
TEST(SlugShaderProvenanceTest, PinsTheUpstreamHarfBuzzShaderSources) {
    const char* shared_fragment =
        hb_gpu_shader_source(HB_GPU_SHADER_STAGE_FRAGMENT, HB_GPU_SHADER_LANG_HLSL);
    const char* shared_vertex =
        hb_gpu_shader_source(HB_GPU_SHADER_STAGE_VERTEX, HB_GPU_SHADER_LANG_HLSL);
    const char* draw_fragment =
        hb_gpu_draw_shader_source(HB_GPU_SHADER_STAGE_FRAGMENT, HB_GPU_SHADER_LANG_HLSL);
    const char* draw_vertex =
        hb_gpu_draw_shader_source(HB_GPU_SHADER_STAGE_VERTEX, HB_GPU_SHADER_LANG_HLSL);

    const std::string re_port =
        "HarfBuzz's hb-gpu shader source changed. Re-port engine/core/shaders/slug_gpu.slang "
        "(and slug_text.slang if the vertex contract changed) from the new source, re-check the "
        "8-byte atlas accessor and the row-vector dilation, then update this pin.";

    ASSERT_NE(shared_fragment, nullptr);
    ASSERT_NE(shared_vertex, nullptr);
    ASSERT_NE(draw_fragment, nullptr);
    EXPECT_EQ(HashSource(shared_fragment), kSharedFragmentSourceHash) << re_port;
    EXPECT_EQ(HashSource(shared_vertex), kSharedVertexSourceHash) << re_port;
    EXPECT_EQ(HashSource(draw_fragment), kDrawFragmentSourceHash) << re_port;
    // The draw-stage vertex source is empty: all vertex work is the shared
    // hb_gpu_dilate above, which is why slug_text.slang supplies the entry point.
    ASSERT_NE(draw_vertex, nullptr);
    EXPECT_EQ(HashSource(draw_vertex), 0xCBF29CE484222325ULL);
}

// The guard cannot be satisfied by the file merely existing: the committed port
// must still contain both adaptations. A future edit that "fixes" the accessor
// back to the upstream int4 declaration or transposes the dilation fails here.
TEST(SlugShaderProvenanceTest, PortStillCarriesTheTwoAdaptations) {
    const std::string port = ReadPort();
    ASSERT_FALSE(port.empty())
        << "could not read " << VKENGINE_ENGINE_SHADER_SOURCE_DIR << "/slug_gpu.slang";

    // Adaptation 1: the atlas is 8-byte elements, and the four 16-bit halves are
    // sign-extended with 32-bit shifts.
    EXPECT_NE(port.find("StructuredBuffer<int2, CDataLayout> hb_gpu_atlas"), std::string::npos)
        << "the atlas accessor no longer declares 8-byte elements";
    EXPECT_NE(port.find("(raw.x << 16) >> 16"), std::string::npos)
        << "the atlas accessor no longer sign-extends the 16-bit halves";

    // Adaptation 2: the dilation is row-vector, matching every engine shader.
    // The negative check names the upstream *code* form, not the comment that
    // documents it.
    EXPECT_NE(port.find("mul (float4 (position, 0.0, 1.0), m)"), std::string::npos)
        << "hb_gpu_dilate is not in the engine's row-vector convention";
    EXPECT_EQ(port.find("clipPos = mul (m,"), std::string::npos)
        << "hb_gpu_dilate still carries the upstream column-vector multiply";
}
