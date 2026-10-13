#include <gtest/gtest.h>

#include <ktx.h>

#include <stb_image_write.h>

import std;

import vulkan;
import FileLoader.Types;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;
import VulkanEngine.FileLoaders.TextureLoaders;

#include "test_ktx_fixtures.hpp"

namespace {

using namespace VulkanEngine::Textures;

// ── KTX2 fixture builders (libktx write API) ─────────────────────────

struct Ktx2Fixture {
    std::vector<std::byte> bytes;
};

// Creates a KTX2 with the given geometry and fills every (level, layer, face)
// image with a deterministic byte pattern derived from its indices.
template <typename SizeFn>
[[nodiscard]] Ktx2Fixture MakeKtx2(ktx_uint32_t vk_format,
                                   std::uint32_t width, std::uint32_t height,
                                   std::uint32_t levels, std::uint32_t layers,
                                   std::uint32_t faces, bool is_array,
                                   SizeFn&& image_size) {
    ktxTextureCreateInfo info{};
    info.vkFormat = vk_format;
    info.baseWidth = width;
    info.baseHeight = height;
    info.baseDepth = 1;
    info.numDimensions = 2;
    info.numLevels = levels;
    info.numLayers = layers;
    info.numFaces = faces;
    info.isArray = is_array ? KTX_TRUE : KTX_FALSE;
    info.generateMipmaps = KTX_FALSE;

    ktxTexture2* texture = nullptr;
    const KTX_error_code create_result = ktxTexture2_Create(&info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
    if (create_result != KTX_SUCCESS || texture == nullptr) {
        throw std::runtime_error(std::string("ktxTexture2_Create failed: ") + ktxErrorString(create_result));
    }
    auto destroy = [&texture]() noexcept { ktxTexture_Destroy(reinterpret_cast<ktxTexture*>(texture)); };

    for (std::uint32_t level = 0; level < levels; ++level) {
        const ktx_size_t size = image_size(level);
        std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (std::uint32_t face = 0; face < faces; ++face) {
                const std::uint8_t seed = static_cast<std::uint8_t>((level * 31U + layer * 7U + face * 3U) % 251U);
                for (ktx_size_t i = 0; i < size; ++i) {
                    data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((seed + static_cast<std::uint8_t>(i)) & 0xFFU);
                }
                const KTX_error_code set_result = ktxTexture_SetImageFromMemory(
                    reinterpret_cast<ktxTexture*>(texture), level, layer, face,
                    data.data(), size);
                if (set_result != KTX_SUCCESS) {
                    destroy();
                    throw std::runtime_error(std::string("ktxTexture_SetImageFromMemory failed: ") + ktxErrorString(set_result));
                }
            }
        }
    }

    ktx_uint8_t* out_bytes = nullptr;
    ktx_size_t out_size = 0;
    const KTX_error_code write_result = ktxTexture_WriteToMemory(reinterpret_cast<ktxTexture*>(texture), &out_bytes, &out_size);
    if (write_result != KTX_SUCCESS || out_bytes == nullptr) {
        destroy();
        throw std::runtime_error(std::string("ktxTexture_WriteToMemory failed: ") + ktxErrorString(write_result));
    }
    Ktx2Fixture fixture{};
    fixture.bytes.assign(reinterpret_cast<std::byte*>(out_bytes), reinterpret_cast<std::byte*>(out_bytes) + out_size);
    ktxTexture_Destroy(reinterpret_cast<ktxTexture*>(texture));
    std::free(out_bytes);
    return fixture;
}

[[nodiscard]] std::uint32_t MipDim(std::uint32_t base, std::uint32_t level) {
    return level == 0 ? base : std::max<std::uint32_t>(1u, base >> level);
}

[[nodiscard]] ktx_size_t Bc7LevelBytes(std::uint32_t width, std::uint32_t height, std::uint32_t level) {
    const std::uint32_t w = MipDim(width, level);
    const std::uint32_t h = MipDim(height, level);
    return static_cast<ktx_size_t>(((w + 3U) / 4U) * ((h + 3U) / 4U)) * 16U;
}

[[nodiscard]] TextureData LoadFixture(const Ktx2Fixture& fixture) {
    const std::filesystem::path path = "fixture.ktx2";
    const FileLoader::ByteBuffer buffer(fixture.bytes.begin(), fixture.bytes.end());
    TextureData data{};
    std::string error;
    if (!VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer(path, buffer, data, &error)) {
        ADD_FAILURE() << "loader failed: " << error;
        return {};
    }
    return data;
}

// ── Tests ────────────────────────────────────────────────────────────

TEST(TextureLoaderSubresourceTest, Bc7MipChainPassthroughKeepsBlocksAndLevels) {
    // 8x8, 2 levels: 4 blocks + 1 block; native BC7 never expanded.
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eBc7UnormBlock), 8, 8, 2, 1, 1, false,
                                  [](std::uint32_t level) { return Bc7LevelBytes(8, 8, level); });
    const auto data = LoadFixture(fixture);

    EXPECT_EQ(data.source_format, vk::Format::eBc7UnormBlock);
    EXPECT_FALSE(data.needs_transcode);
    EXPECT_EQ(data.mip_levels, 2u);
    EXPECT_EQ(data.array_layers, 1u);
    EXPECT_EQ(data.face_count, 1u);
    ASSERT_EQ(data.subresources.size(), 2u);

    EXPECT_EQ(data.subresources[0].mip, 0u);
    EXPECT_EQ(data.subresources[0].size, 4u * 16u);
    EXPECT_EQ(data.subresources[0].width, 8u);
    EXPECT_EQ(data.subresources[0].height, 8u);
    EXPECT_EQ(data.subresources[1].mip, 1u);
    EXPECT_EQ(data.subresources[1].size, 16u);
    EXPECT_EQ(data.subresources[1].width, 4u);
    EXPECT_EQ(data.subresources[1].height, 4u);
    // KTX2 does not mandate sequential level order inside the data block
    // (libktx writes smaller levels first here); both offsets must land
    // inside the blob without overlapping.
    EXPECT_LT(data.subresources[0].offset + data.subresources[0].size, data.blob.size() + 1U);
    EXPECT_LT(data.subresources[1].offset + data.subresources[1].size, data.blob.size() + 1U);
    EXPECT_NE(data.subresources[0].offset, data.subresources[1].offset);
}

TEST(TextureLoaderSubresourceTest, OddSizeBc7TailMipIsOneBlock) {
    // 3x2 base with 2 levels: tail mip 1x1. Every level is one 16-byte block.
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eBc7UnormBlock), 3, 2, 2, 1, 1, false,
                                  [](std::uint32_t level) { return Bc7LevelBytes(3, 2, level); });
    const auto data = LoadFixture(fixture);

    ASSERT_EQ(data.subresources.size(), 2u);
    EXPECT_EQ(data.subresources[0].size, 16u);
    EXPECT_EQ(data.subresources[1].size, 16u);
    EXPECT_EQ(data.subresources[1].width, 1u);
    EXPECT_EQ(data.subresources[1].height, 1u);
}

TEST(TextureLoaderSubresourceTest, TwoByTwoBc7IsASingleBlock) {
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eBc7UnormBlock), 2, 2, 1, 1, 1, false,
                                  [](std::uint32_t) { return static_cast<ktx_size_t>(16); });
    const auto data = LoadFixture(fixture);

    ASSERT_EQ(data.subresources.size(), 1u);
    EXPECT_EQ(data.subresources[0].size, 16u);
    EXPECT_EQ(data.subresources[0].width, 2u);
    EXPECT_EQ(data.subresources[0].height, 2u);
}

TEST(TextureLoaderSubresourceTest, CubemapFoldsSixFaces) {
    // 6 faces at 4x4 BC7: six 16-byte subresources, gpu_layer = face.
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eBc7UnormBlock), 4, 4, 1, 1, 6, false,
                                  [](std::uint32_t) { return static_cast<ktx_size_t>(16); });
    const auto data = LoadFixture(fixture);

    EXPECT_EQ(data.face_count, 6u);
    EXPECT_EQ(data.array_layers, 1u);
    ASSERT_EQ(data.subresources.size(), 6u);
    for (std::uint32_t face = 0; face < 6; ++face) {
        EXPECT_EQ(data.subresources[face].array_layer, face);
        EXPECT_EQ(data.subresources[face].size, 16u);
    }
    EXPECT_TRUE(data.cube_complete);
}

TEST(TextureLoaderSubresourceTest, TwoDimensionalArrayPreservesLayers) {
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eR8G8B8A8Unorm), 4, 4, 1, 3, 1, true,
                                  [](std::uint32_t) { return static_cast<ktx_size_t>(64); });
    const auto data = LoadFixture(fixture);

    EXPECT_EQ(data.array_layers, 3u);
    ASSERT_EQ(data.subresources.size(), 3u);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        EXPECT_EQ(data.subresources[layer].array_layer, layer);
        EXPECT_EQ(data.subresources[layer].size, 64u);
    }
}

TEST(TextureLoaderSubresourceTest, SrgbFormatIsPreservedNotCollapsed) {
    // Regression: an sRGB KTX2 keeps its _SRGB source format.
    const auto fixture = MakeKtx2(static_cast<ktx_uint32_t>(vk::Format::eR8G8B8A8Srgb), 4, 4, 1, 1, 1, false,
                                  [](std::uint32_t) { return static_cast<ktx_size_t>(64); });
    const auto data = LoadFixture(fixture);
    EXPECT_EQ(data.source_format, vk::Format::eR8G8B8A8Srgb);
}

TEST(TextureLoaderSubresourceTest, PngGetsAFullCpuMipChain) {
    // Uncompressed non-KTX sources carry a complete CPU chain.
    const std::array<std::uint8_t, 4 * 4 * 4> pixels = [] {
        std::array<std::uint8_t, 4 * 4 * 4> p{};
        for (std::size_t i = 0; i < p.size(); i += 4) {
            p[i] = 200; p[i + 1] = 100; p[i + 2] = 50; p[i + 3] = 255;
        }
        return p;
    }();
    const std::filesystem::path path = "fixture.png";
    if (stbi_write_png(path.string().c_str(), 4, 4, 4, pixels.data(), 16) != 1) {
        GTEST_FAIL() << "stbi_write_png failed";
    }

    std::ifstream in(path, std::ios::binary);
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const FileLoader::ByteBuffer buffer(reinterpret_cast<const std::byte*>(raw.data()),
                                        reinterpret_cast<const std::byte*>(raw.data()) + raw.size());
    std::filesystem::remove(path);

    TextureData data{};
    std::string error;
    ASSERT_TRUE(VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer(path, buffer, data, &error)) << error;
    EXPECT_EQ(data.width, 4u);
    EXPECT_EQ(data.height, 4u);
    EXPECT_EQ(data.mip_levels, 3u);
    ASSERT_EQ(data.subresources.size(), 3u);
    EXPECT_EQ(data.blob.size(), (16U + 4U + 1U) * 4U);
    EXPECT_EQ(data.subresources[0].size, 64u);
    EXPECT_EQ(data.subresources[1].size, 16u);
    EXPECT_EQ(data.subresources[2].size, 4u);
    EXPECT_EQ(data.subresources[2].width, 1u);
    EXPECT_EQ(data.subresources[2].height, 1u);
}

// ── DFD channel model (regression) ───────────────────────────────────
//
// The loader feeds source_channels to the upload-format resolver, whose rule 5
// picks BC5 for two independent channels and BC4 for one data channel. Those
// rules are only reachable if the DFD is parsed correctly: ktxTexture2::pDfd
// points at the whole descriptor, whose first word is dfdTotalSize, so the basic
// descriptor block starts one word later. Reading pDfd as the BDB returned
// blockSize=0 for every texture, silently reporting zero channels.

[[nodiscard]] TextureData LoadBasisFixture(std::uint32_t channels) {
    // Alpha varies only for the RGBA case: libktx drops a constant alpha channel.
    const auto pixels = TestSupport::MakeTestImage(16, 16, /*varied_alpha=*/channels == 4);
    const auto ktx_basis = TestSupport::MakeBasisUastc(16, 16, 1, pixels, channels);
    if (ktx_basis.empty()) {
        return {};
    }
    const std::filesystem::path path = "fixture_basis.ktx2";
    const FileLoader::ByteBuffer buffer(ktx_basis.begin(), ktx_basis.end());
    TextureData data{};
    std::string error;
    if (!VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer(path, buffer, data, &error)) {
        ADD_FAILURE() << "loader failed: " << error;
        return {};
    }
    return data;
}

TEST(TextureLoaderSubresourceTest, Rgba8KtxReportsFourChannelsWithAlpha) {
    const auto pixels = TestSupport::MakeTestImage(8, 8);
    const auto fixture = TestSupport::MakeKtx2Rgba8(8, 8, 1, pixels);
    TextureData data{};
    std::string error;
    const FileLoader::ByteBuffer buffer(fixture.begin(), fixture.end());
    ASSERT_TRUE(VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer("fixture.ktx2", buffer,
                                                                           data, &error)) << error;
    EXPECT_EQ(data.source_channels, 4u);
    EXPECT_EQ(data.source_alpha, 1u);
}

// A Basis fixture stays encoded until transcode, so its subresource table is
// built by TranscodeBasisToTarget; the DFD-derived channel count is what the
// loader sets at read time and is what the resolver consumes.

TEST(TextureLoaderSubresourceTest, BasisTwoChannelReportsTwoChannels) {
    const auto data = LoadBasisFixture(/*channels=*/2);
    EXPECT_TRUE(data.needs_transcode);
    EXPECT_EQ(data.source_channels, 2u);
    EXPECT_EQ(data.source_alpha, 0u);
}

TEST(TextureLoaderSubresourceTest, BasisOneChannelReportsOneChannel) {
    const auto data = LoadBasisFixture(/*channels=*/1);
    EXPECT_TRUE(data.needs_transcode);
    EXPECT_EQ(data.source_channels, 1u);
    EXPECT_EQ(data.source_alpha, 0u);
}

TEST(TextureLoaderSubresourceTest, BasisRgbaReportsFourChannelsWithAlpha) {
    const auto data = LoadBasisFixture(/*channels=*/4);
    EXPECT_TRUE(data.needs_transcode);
    EXPECT_EQ(data.source_channels, 4u);
    EXPECT_EQ(data.source_alpha, 1u);
}

TEST(TextureLoaderSubresourceTest, BasisOpaqueRgbReportsThreeChannels) {
    const auto data = LoadBasisFixture(/*channels=*/3);
    EXPECT_TRUE(data.needs_transcode);
    EXPECT_EQ(data.source_channels, 3u);
    EXPECT_EQ(data.source_alpha, 0u);
}

// The two-channel count must actually steer the resolver to BC5, which is the
// reason the channel model exists: rule 5 picks BC5 for two independent
// channels. A device-free fake capability table stands in for the snapshot.
TEST(TextureLoaderSubresourceTest, TwoChannelBasisResolvesToBc5) {
    class AllSupported final : public VulkanEngine::Textures::FormatSupportQuery {
    public:
        [[nodiscard]] bool sampled(vk::Format) const override { return true; }
        [[nodiscard]] bool transfer_dst(vk::Format) const override { return true; }
        [[nodiscard]] bool linear_filter(vk::Format) const override { return true; }
        [[nodiscard]] bool color_attachment(vk::Format) const override { return true; }
    };

    const auto data = LoadBasisFixture(/*channels=*/2);
    const AllSupported caps;
    const auto resolved = VulkanEngine::Textures::ResolveUploadFormat(
        TextureSemantic::Normal, TextureNormalEncoding::Standard, data.source_format,
        data.needs_transcode, data.source_channels, data.source_alpha != 0,
        data.is_hdr_source, caps);
    EXPECT_EQ(resolved.format, vk::Format::eBc5UnormBlock);
}

TEST(TextureLoaderSubresourceTest, OneChannelDataBasisResolvesToBc4) {
    class AllSupported final : public VulkanEngine::Textures::FormatSupportQuery {
    public:
        [[nodiscard]] bool sampled(vk::Format) const override { return true; }
        [[nodiscard]] bool transfer_dst(vk::Format) const override { return true; }
        [[nodiscard]] bool linear_filter(vk::Format) const override { return true; }
        [[nodiscard]] bool color_attachment(vk::Format) const override { return true; }
    };

    const auto data = LoadBasisFixture(/*channels=*/1);
    const AllSupported caps;
    const auto resolved = VulkanEngine::Textures::ResolveUploadFormat(
        TextureSemantic::Mask, TextureNormalEncoding::Standard, data.source_format,
        data.needs_transcode, data.source_channels, data.source_alpha != 0,
        data.is_hdr_source, caps);
    EXPECT_EQ(resolved.format, vk::Format::eBc4UnormBlock);
}

}  // namespace
