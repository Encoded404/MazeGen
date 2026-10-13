#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import TestSupport.TextureRenderHarness;
import FileLoader.Types;
import ShaderReflection;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ShaderManager;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;
import VulkanEngine.FileLoaders.TextureLoaders;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;

#include "test_ktx_fixtures.hpp"

namespace {

using namespace VulkanEngine::Textures;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::ShaderSystem::ShaderManager;

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end compressed-format sampling.
//
// A libktx Basis fixture is transcoded two ways: to RGBA8 (reference) and to the
// device-supported BCn target the production resolver picks. Both are uploaded
// through GpuTexture::CreateFromTextureData and sampled by the test fragment
// shader, and the two renders are compared per channel. This is the acceptance
// gate the byte round-trip tests cannot give: it proves the decompressed pixels
// the shader actually receives are correct, not merely that the container bytes
// reached memory.
//
// Gpu label: the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

class TextureRenderGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize()) << backend_.GetErrorMessage();
        heap_.Initialize(backend_, ImageHeapConfig{}, "texture-render-tests");
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(), "");
        harness_ids_ = TestSupport::RegisterHarnessShaders(*shaders_, VKENGINE_TEST_SHADER_DIR);

        TestSupport::TextureRenderHarnessConfig config{};
        config.width = kSize;
        config.height = kSize;
        
        config.vertex_shader = harness_ids_.vertex;
        config.fragment_shader = harness_ids_.sample_fragment;
        ASSERT_TRUE(harness_.Initialize(backend_, *shaders_, config));
    }

    void TearDown() override {
        harness_.Shutdown();
        shaders_.reset();
        heap_.Shutdown();
        backend_.Shutdown();
    }

    static constexpr std::uint32_t kSize = 64;

    // Uploads `data` after transcoding Basis to `target` when required.
    [[nodiscard]] GpuTexture Upload(TextureData data, vk::Format target) {
        if (data.needs_transcode) {
            std::string error;
            if (!VulkanEngine::FileLoaders::Textures::TranscodeBasisToTarget(data, target, &error)) {
                ADD_FAILURE() << "transcode failed: " << error;
                return {};
            }
        }
        auto tex = GpuTexture::CreateFromTextureData(backend_, heap_, data, target);
        if (!tex.IsValid()) {
            ADD_FAILURE() << "CreateFromTextureData failed";
        }
        return tex;
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    std::unique_ptr<ShaderManager> shaders_;
    TestSupport::HarnessShaderIds harness_ids_{};
    TestSupport::TextureRenderHarness harness_{};
};

// Mean absolute per-channel error between two RGBA8 renders, over opaque pixels
// only (the reference is opaque, so alpha is compared too but contributes 0).
[[nodiscard]] std::array<double, 4> MeanAbsError(const std::vector<std::uint8_t>& a,
                                                 const std::vector<std::uint8_t>& b) {
    std::array<double, 4> sums{0.0, 0.0, 0.0, 0.0};
    const std::size_t pixels = std::min(a.size(), b.size()) / 4U;
    for (std::size_t p = 0; p < pixels; ++p) {
        for (std::size_t c = 0; c < 4U; ++c) {
            sums[c] += std::abs(static_cast<double>(a[p * 4U + c]) - static_cast<double>(b[p * 4U + c]));
        }
    }
    if (pixels != 0U) {
        for (double& s : sums) {
            s /= static_cast<double>(pixels);
        }
    }
    return sums;
}

// Loads a Basis fixture built from authored pixels. `channels` selects the
// encoded channel model (see MakeBasisUastc) so the resolver's rule-5 target is
// reachable.
[[nodiscard]] bool LoadBasisFixture(TextureData& out, std::uint32_t width, std::uint32_t height,
                                    std::uint32_t mips, std::uint32_t channels = 4) {
    const auto pixels = TestSupport::MakeTestImage(width, height, /*varied_alpha=*/channels == 4);
    const auto ktx_basis = TestSupport::MakeBasisUastc(width, height, mips, pixels, channels);
    if (ktx_basis.empty()) {
        return false;
    }
    std::string error;
    const FileLoader::ByteBuffer basis_buffer(ktx_basis.begin(), ktx_basis.end());
    return VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer("fixture.ktx2", basis_buffer,
                                                                      out, &error);
}

// Loads the authored RGBA8 image the Basis fixture was compressed from, as the
// reference the compressed sample is measured against. It must be the authored
// pixels, not an RGBA32 transcode of the Basis container: a 2-channel `rrrg`
// Basis transcoded to RGBA32 expands R into RGB and G into A, so its green
// channel is not the authored green.
[[nodiscard]] bool LoadAuthoredRgba8Reference(TextureData& out, std::uint32_t width,
                                              std::uint32_t height, std::uint32_t channels = 4) {
    const auto pixels = TestSupport::MakeTestImage(width, height, /*varied_alpha=*/channels == 4);
    const auto ktx_rgba = TestSupport::MakeKtx2Rgba8(width, height, 1, pixels);
    std::string error;
    const FileLoader::ByteBuffer rgba_buffer(ktx_rgba.begin(), ktx_rgba.end());
    return VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer("fixture.ktx2", rgba_buffer,
                                                                      out, &error);
}

// Resolves the Basis fixture through the production resolver so the test uses
// the same BCn target the runtime picks.
[[nodiscard]] vk::Format ResolveTarget(const TestSupport::HeadlessVulkanBackend& backend,
                                       const TextureData& basis, TextureSemantic semantic,
                                       TextureNormalEncoding encoding) {
    class Query final : public FormatSupportQuery {
    public:
        explicit Query(const VulkanBackend::Vulkan::FormatCapabilities& caps) : caps_(caps) {}
        [[nodiscard]] bool sampled(vk::Format f) const override { return caps_.sampled(f); }
        [[nodiscard]] bool transfer_dst(vk::Format f) const override { return caps_.transfer_dst(f); }
        [[nodiscard]] bool linear_filter(vk::Format f) const override { return caps_.linear_filter(f); }
        [[nodiscard]] bool color_attachment(vk::Format f) const override { return caps_.color_attachment(f); }
    private:
        const VulkanBackend::Vulkan::FormatCapabilities& caps_;
    };
    const Query query(backend.GetCapabilities().GetFormatCapabilities());
    const auto resolved = ResolveUploadFormat(semantic, encoding, basis.source_format,
                                              basis.needs_transcode, basis.source_channels,
                                              basis.source_alpha != 0, basis.is_hdr_source, query);
    return resolved.format;
}

TEST_F(TextureRenderGpuTest, Bc7ColorSamplesMatchTheRgba8Reference) {
    const auto target = ResolveTarget(backend_, TextureData{.needs_transcode = true,
                                                            .source_channels = 4, .source_alpha = 1},
                                      TextureSemantic::BaseColor, TextureNormalEncoding::Standard);
    if (target == vk::Format::eUndefined) {
        GTEST_SKIP() << "no BCn target available on this device";
    }
    // The alpha-bearing 4-channel color path must pick BC7 on a PC device.
    ASSERT_EQ(target, vk::Format::eBc7SrgbBlock) << "expected the BC7 sRGB target";

    TextureData basis{};
    ASSERT_TRUE(LoadBasisFixture(basis, kSize, kSize, 1, /*channels=*/4));
    // The reference is the RGBA32 transcode of the SAME Basis container, so both
    // sides share authored content.
    TextureData reference_data = basis;
    std::string transcode_error;
    ASSERT_TRUE(VulkanEngine::FileLoaders::Textures::TranscodeBasisToTarget(
        reference_data, vk::Format::eR8G8B8A8Srgb, &transcode_error)) << transcode_error;

    auto reference = Upload(std::move(reference_data), vk::Format::eR8G8B8A8Srgb);
    ASSERT_TRUE(reference.IsValid());
    auto compressed = Upload(std::move(basis), target);
    ASSERT_TRUE(compressed.IsValid());

    const auto ref_pixels = harness_.RenderAndReadBack(reference);
    const auto bc_pixels = harness_.RenderAndReadBack(compressed);
    ASSERT_EQ(ref_pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    ASSERT_EQ(bc_pixels.size(), ref_pixels.size());

    const auto error = MeanAbsError(ref_pixels, bc_pixels);
    // BC7 at 4x4 blocks over a smooth gradient: the plan's 6/255 mean bound.
    for (std::size_t c = 0; c < 4U; ++c) {
        EXPECT_LT(error[c], 6.0) << "channel " << c << " mean error " << error[c];
    }
}

TEST_F(TextureRenderGpuTest, Bc5NormalSamplesMatchTheRgba8Reference) {
    // A 2-channel Standard-normal Basis source resolves to BC5, against the
    // authored RGBA8 image as the reference.
    TextureData basis{};
    ASSERT_TRUE(LoadBasisFixture(basis, kSize, kSize, 1, /*channels=*/2));
    ASSERT_EQ(basis.source_channels, 2u) << "2-channel Basis must report two channels";
    const auto target = ResolveTarget(backend_, basis, TextureSemantic::Normal,
                                      TextureNormalEncoding::Standard);
    if (target != vk::Format::eBc5UnormBlock) {
        GTEST_SKIP() << "BC5 not the resolved target on this device";
    }

    TextureData reference{};
    ASSERT_TRUE(LoadAuthoredRgba8Reference(reference, kSize, kSize, /*channels=*/2));
    auto ref_tex = Upload(std::move(reference), vk::Format::eR8G8B8A8Unorm);
    ASSERT_TRUE(ref_tex.IsValid());
    auto bc_tex = Upload(std::move(basis), target);
    ASSERT_TRUE(bc_tex.IsValid());

    const auto ref_pixels = harness_.RenderAndReadBack(ref_tex);
    const auto bc_pixels = harness_.RenderAndReadBack(bc_tex);
    const auto error_metrics = MeanAbsError(ref_pixels, bc_pixels);
    // BC5 encodes two channels; the shader outputs (r,g,0,1). Bound: 4/255.
    EXPECT_LT(error_metrics[0], 4.0);
    EXPECT_LT(error_metrics[1], 4.0);
}

TEST_F(TextureRenderGpuTest, Bc4SingleChannelSamplesMatch) {
    // 1-channel data (Mask) resolves to BC4; the shader outputs (r,0,0,1). The
    // reference is the RGBA32 transcode of the same Basis container.
    TextureData basis{};
    ASSERT_TRUE(LoadBasisFixture(basis, kSize, kSize, 1, /*channels=*/1));
    ASSERT_EQ(basis.source_channels, 1u) << "1-channel Basis must report one channel";
    const auto target = ResolveTarget(backend_, basis, TextureSemantic::Mask,
                                      TextureNormalEncoding::Standard);
    if (target != vk::Format::eBc4UnormBlock) {
        GTEST_SKIP() << "BC4 not the resolved target on this device";
    }

    TextureData reference{};
    ASSERT_TRUE(LoadAuthoredRgba8Reference(reference, kSize, kSize, /*channels=*/1));
    auto ref_tex = Upload(std::move(reference), vk::Format::eR8G8B8A8Unorm);
    auto bc_tex = Upload(std::move(basis), target);
    ASSERT_TRUE(ref_tex.IsValid());
    ASSERT_TRUE(bc_tex.IsValid());

    const auto ref_pixels = harness_.RenderAndReadBack(ref_tex);
    const auto bc_pixels = harness_.RenderAndReadBack(bc_tex);
    const auto error_metrics = MeanAbsError(ref_pixels, bc_pixels);
    EXPECT_LT(error_metrics[0], 6.0);
}

TEST_F(TextureRenderGpuTest, Bc7TwoByTwoBlockSamplesWithoutError) {
    // 2x2 is a single partial block; the edge copy must still sample cleanly.
    TextureData basis{};
    ASSERT_TRUE(LoadBasisFixture(basis, 2, 2, 1, /*channels=*/4));
    const auto target = ResolveTarget(backend_, basis, TextureSemantic::BaseColor,
                                      TextureNormalEncoding::Standard);
    if (target == vk::Format::eUndefined) {
        GTEST_SKIP() << "no BCn target available";
    }

    TextureData reference = basis;
    std::string error;
    ASSERT_TRUE(VulkanEngine::FileLoaders::Textures::TranscodeBasisToTarget(
        reference, vk::Format::eR8G8B8A8Srgb, &error)) << error;
    auto ref_tex = Upload(std::move(reference), vk::Format::eR8G8B8A8Srgb);
    auto bc_tex = Upload(std::move(basis), target);
    ASSERT_TRUE(ref_tex.IsValid());
    ASSERT_TRUE(bc_tex.IsValid());

    const auto ref_pixels = harness_.RenderAndReadBack(ref_tex);
    const auto bc_pixels = harness_.RenderAndReadBack(bc_tex);
    const auto error_metrics = MeanAbsError(ref_pixels, bc_pixels);
    // A 2x2 texel block magnified to 64x64 is flat per block; the two
    // compressed endpoints may differ, so bound generously but require no gross
    // error (e.g. swapped channels or an unsampled image).
    for (double e : error_metrics) {
        EXPECT_LT(e, 16.0);
    }
}

TEST_F(TextureRenderGpuTest, Bc7ThreeByTwoMipCopySamplesCleanly) {
    // 3x2 base with a 2-level chain: both levels are one BC7 block. This is the
    // GPU counterpart of the device-free 3x2 size test, proving the block-exact
    // copy of a non-multiple-of-4 extent samples without validation errors.
    TextureData basis{};
    ASSERT_TRUE(LoadBasisFixture(basis, 3, 2, 2, /*channels=*/4));
    ASSERT_EQ(basis.mip_levels, 2u);
    const auto target = ResolveTarget(backend_, basis, TextureSemantic::BaseColor,
                                      TextureNormalEncoding::Standard);
    if (target == vk::Format::eUndefined) {
        GTEST_SKIP() << "no BCn target available";
    }
    auto tex = Upload(std::move(basis), target);
    ASSERT_TRUE(tex.IsValid());

    const auto pixels_out = harness_.RenderAndReadBack(tex);
    ASSERT_EQ(pixels_out.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
}

}  // namespace
