#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;

namespace {

using namespace VulkanEngine::Textures;

// Fully-capable fake: every candidate format supported.
class FullCaps final : public FormatSupportQuery {
public:
    [[nodiscard]] bool sampled(vk::Format) const override { return true; }
    [[nodiscard]] bool transfer_dst(vk::Format) const override { return true; }
    [[nodiscard]] bool linear_filter(vk::Format) const override { return true; }
    [[nodiscard]] bool color_attachment(vk::Format) const override { return true; }
};

// No block compression at all (rule 6 boundary).
class NoBcCaps final : public FormatSupportQuery {
    static bool IsBc(vk::Format f) {
        return f >= vk::Format::eBc1RgbUnormBlock && f <= vk::Format::eBc7SrgbBlock;
    }
public:
    [[nodiscard]] bool sampled(vk::Format f) const override { return !IsBc(f); }
    [[nodiscard]] bool transfer_dst(vk::Format f) const override { return !IsBc(f); }
    [[nodiscard]] bool linear_filter(vk::Format) const override { return true; }
    [[nodiscard]] bool color_attachment(vk::Format) const override { return false; }
};

class FormatResolveTest : public ::testing::Test {
protected:
    FullCaps caps_{};
};

TEST_F(FormatResolveTest, ColorSemanticReinterpretsToSrgb) {
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eR8G8B8A8Unorm, false, 4, true, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eR8G8B8A8Srgb);
    EXPECT_TRUE(r.reinterpreted);
    EXPECT_FALSE(r.transcoded);
}

TEST_F(FormatResolveTest, DataSemanticKeepsUnorm) {
    const auto r = ResolveUploadFormat(TextureSemantic::ORM, TextureNormalEncoding::Standard,
                                       vk::Format::eR8G8B8A8Unorm, false, 4, true, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eR8G8B8A8Unorm);
    EXPECT_FALSE(r.reinterpreted);
}

TEST_F(FormatResolveTest, SrgbSourceOnDataSemanticReinterpretsToUnorm) {
    // Rule 2: an explicit _SRGB source on a data semantic is not linearized.
    const auto r = ResolveUploadFormat(TextureSemantic::Normal, TextureNormalEncoding::Standard,
                                       vk::Format::eR8G8B8A8Srgb, false, 4, true, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(r.reinterpreted);
}

TEST_F(FormatResolveTest, UncompressedSourceMapsByChannelsAndSemantic) {
    const auto color = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                           vk::Format::eUndefined, false, 4, true, false, caps_);
    EXPECT_EQ(color.format, vk::Format::eR8G8B8A8Srgb);

    const auto data = ResolveUploadFormat(TextureSemantic::Mask, TextureNormalEncoding::Standard,
                                          vk::Format::eUndefined, false, 1, false, false, caps_);
    EXPECT_EQ(data.format, vk::Format::eR8Unorm);

    const auto normal2 = ResolveUploadFormat(TextureSemantic::Normal, TextureNormalEncoding::Standard,
                                             vk::Format::eUndefined, false, 2, false, false, caps_);
    EXPECT_EQ(normal2.format, vk::Format::eR8G8Unorm);
}

TEST_F(FormatResolveTest, HdrNeverDownconverts) {
    const auto r = ResolveUploadFormat(TextureSemantic::HDR, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, false, 4, true, true, caps_);
    EXPECT_EQ(r.format, vk::Format::eR16G16B16A16Sfloat);
}

TEST_F(FormatResolveTest, BasisTwoChannelGoesToBc5) {
    const auto r = ResolveUploadFormat(TextureSemantic::Normal, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, true, 2, false, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eBc5UnormBlock);
    EXPECT_TRUE(r.transcoded);
}

TEST_F(FormatResolveTest, BasisRgbColorPrefersBc7Srgb) {
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, true, 3, false, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eBc7SrgbBlock);
}

TEST_F(FormatResolveTest, BasisAlphaAvoidsBc1) {
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, true, 4, true, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eBc7SrgbBlock);
}

TEST_F(FormatResolveTest, BasisMaskGoesToBc4) {
    const auto r = ResolveUploadFormat(TextureSemantic::Mask, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, true, 1, false, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eBc4UnormBlock);
}

TEST_F(FormatResolveTest, BasisTwoChannelFullEncodingNeverBc5) {
    // Rule 8: Full/Bent require >= 3 channels; 2-channel transcode sources are
    // forced to an RGB target instead of silently landing in BC5.
    const auto full = ResolveUploadFormat(TextureSemantic::Normal, TextureNormalEncoding::Full,
                                          vk::Format::eUndefined, true, 2, false, false, caps_);
    EXPECT_EQ(full.format, vk::Format::eBc7UnormBlock);
}

TEST_F(FormatResolveTest, NativeBc5WithFullEncodingFails) {
    // Rule 8: no runtime decode exists, so a native BC5 source cannot honor
    // Full/Bent — fail the load rather than silently keeping BC5.
    const auto r = ResolveUploadFormat(TextureSemantic::Normal, TextureNormalEncoding::Full,
                                       vk::Format::eBc5UnormBlock, false, 2, false, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eUndefined);
}

TEST_F(FormatResolveTest, NativeBc7ReinterpretsForColor) {
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eBc7UnormBlock, false, 4, true, false, caps_);
    EXPECT_EQ(r.format, vk::Format::eBc7SrgbBlock);
    EXPECT_TRUE(r.reinterpreted);
    EXPECT_FALSE(r.transcoded);
}

TEST_F(FormatResolveTest, NoBcBasisFallsBackToRgba8) {
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eUndefined, true, 3, false, false, NoBcCaps{});
    EXPECT_EQ(r.format, vk::Format::eR8G8B8A8Srgb);
    EXPECT_TRUE(r.transcoded);
}

TEST_F(FormatResolveTest, NoBcNativeBcnFails) {
    // Rule 6: native BCn without textureCompressionBC cannot be decoded —
    // fail with a clear substitution-to-fallback path, never expand to RGBA.
    const auto r = ResolveUploadFormat(TextureSemantic::BaseColor, TextureNormalEncoding::Standard,
                                       vk::Format::eBc7UnormBlock, false, 4, true, false, NoBcCaps{});
    EXPECT_EQ(r.format, vk::Format::eUndefined);
}

}  // namespace
