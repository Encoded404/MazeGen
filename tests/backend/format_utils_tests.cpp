#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanBackend.Vulkan.FormatUtils;
import VulkanBackend.Vulkan.ImageUtils;

namespace {

using VulkanBackend::Vulkan::FormatTraits;
using VulkanBackend::Vulkan::GetFormatTraits;
using VulkanBackend::Vulkan::MipExtent;
using VulkanBackend::Vulkan::MipLevelCount;
using VulkanBackend::Vulkan::SubresourceByteSize;

TEST(FormatUtilsTest, UncompressedBlockGeometry) {
    const FormatTraits rgba8 = GetFormatTraits(vk::Format::eR8G8B8A8Unorm);
    EXPECT_FALSE(rgba8.compressed);
    EXPECT_EQ(rgba8.block_bytes, 4u);
    EXPECT_EQ(rgba8.srgb_variant, vk::Format::eR8G8B8A8Srgb);
    EXPECT_EQ(rgba8.unorm_variant, vk::Format::eR8G8B8A8Unorm);

    const FormatTraits r8 = GetFormatTraits(vk::Format::eR8Unorm);
    EXPECT_EQ(r8.block_bytes, 1u);

    const FormatTraits rgba16f = GetFormatTraits(vk::Format::eR16G16B16A16Sfloat);
    EXPECT_EQ(rgba16f.block_bytes, 8u);
}

TEST(FormatUtilsTest, CompressedBlockGeometry) {
    const FormatTraits bc7 = GetFormatTraits(vk::Format::eBc7UnormBlock);
    EXPECT_TRUE(bc7.compressed);
    EXPECT_EQ(bc7.block.width, 4u);
    EXPECT_EQ(bc7.block.height, 4u);
    EXPECT_EQ(bc7.block_bytes, 16u);
    EXPECT_EQ(bc7.srgb_variant, vk::Format::eBc7SrgbBlock);

    const FormatTraits bc1 = GetFormatTraits(vk::Format::eBc1RgbaUnormBlock);
    EXPECT_EQ(bc1.block_bytes, 8u);

    const FormatTraits bc5 = GetFormatTraits(vk::Format::eBc5UnormBlock);
    EXPECT_EQ(bc5.block_bytes, 16u);
    EXPECT_EQ(bc5.srgb_variant, vk::Format::eUndefined); // data-only target
}

TEST(FormatUtilsTest, MipExtentsNeverCollapseBelowOne) {
    const auto level0 = MipExtent(64, 32, 1, 0);
    EXPECT_EQ(level0.width, 64u);
    const auto level5 = MipExtent(64, 32, 1, 5);
    EXPECT_EQ(level5.width, 2u);
    EXPECT_EQ(level5.height, 1u);
    const auto tail = MipExtent(3, 2, 1, 2);
    EXPECT_EQ(tail.width, 1u);
    EXPECT_EQ(tail.height, 1u);
    EXPECT_EQ(MipLevelCount(64, 32, 1), 7u);
    EXPECT_EQ(MipLevelCount(3, 2, 1), 2u);
    EXPECT_EQ(MipLevelCount(0, 0, 0), 0u);
}

TEST(FormatUtilsTest, SubresourceSizesAreBlockTight) {
    // RGBA8: byte-exact.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{4, 4, 1}, vk::Format::eR8G8B8A8Unorm), 64);
    // BC7 4x4: exactly one block.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{4, 4, 1}, vk::Format::eBc7UnormBlock), 16);
    // BC7 2x2 tail mip: one block.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{2, 2, 1}, vk::Format::eBc7UnormBlock), 16);
    // BC7 1x1 tail mip: one block.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{1, 1, 1}, vk::Format::eBc7UnormBlock), 16);
    // BC7 3x2: one block (ceil(3/4) x ceil(2/4)).
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{3, 2, 1}, vk::Format::eBc7UnormBlock), 16);
    // BC7 5x4: two blocks.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{5, 4, 1}, vk::Format::eBc7UnormBlock), 32);
    // BC1 4x4: one 8-byte block.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{4, 4, 1}, vk::Format::eBc1RgbUnormBlock), 8);
    // Unknown format reports zero.
    EXPECT_EQ(SubresourceByteSize(vk::Extent3D{4, 4, 1}, vk::Format::eUndefined), 0);
}

TEST(FormatUtilsTest, FullBc7ChainSize) {
    // 64x64 BC7 chain: blocks halve per level: 16^2*16B * 6 levels.
    vk::DeviceSize total = 0;
    for (std::uint32_t level = 0; level < 6; ++level) {
        total += SubresourceByteSize(MipExtent(64, 64, 1, level), vk::Format::eBc7UnormBlock);
    }
    const vk::DeviceSize expected =
        (256 + 64 + 16 + 4 + 1 + 1) * 16; // block counts: 16x16..1x1
    EXPECT_EQ(total, expected);
}

// CalculateImageSize must count bytes through format traits. The old
// implementation counted texels (w*h*d summed over the chain, 1 byte each).
TEST(ImageUtilsTest, CalculateImageSizeAccountsForFormatBytes) {
    using VulkanBackend::Vulkan::ImageUtils;
    // RGBA8 base only: 4x4 = 16 texels * 4 B.
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 1, 1, 1, vk::Format::eR8G8B8A8Unorm), 64);
    // RGBA8 full chain 4x4: 16 + 4 + 1 texels * 4 B.
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 1, 3, 1, vk::Format::eR8G8B8A8Srgb), 84);
    // BC7 full chain 4x4: blocks 1 + 1 + 1 (tail mips are whole blocks) * 16 B.
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 1, 3, 1, vk::Format::eBc7UnormBlock), 48);
    // The old code returned 21 (texels) for the BC7 chain above and 16 for
    // this RGBA8 base — both wrong; assert the exact corrected values.
    // Array layers multiply the whole chain.
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 1, 1, 6, vk::Format::eR8G8B8A8Unorm), 384);
    // 3D texture depth halves per level: level 0 = 32 texels, level 1 = 2x2x1
    // = 4 texels; (32 + 4) * 4 B.
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 2, 2, 1, vk::Format::eR8G8B8A8Unorm), 144);
    // Unknown format reports zero (not a candidate-set format).
    EXPECT_EQ(ImageUtils::CalculateImageSize(4, 4, 1, 1, 1, vk::Format::eUndefined), 0);
}

// The copy region carries the real extent and depth.
TEST(ImageUtilsTest, CreateBufferImageCopyCarriesExtentAndDepth) {
    using VulkanBackend::Vulkan::ImageUtils;
    const auto copy = ImageUtils::CreateBufferImageCopy(
        vk::Format::eBc7UnormBlock, 3, 2, 1,
        vk::ImageAspectFlagBits::eColor, 2, 0, 32);
    EXPECT_EQ(copy.bufferOffset, 32);
    EXPECT_EQ(copy.bufferRowLength, 0u);
    EXPECT_EQ(copy.bufferImageHeight, 0u);
    EXPECT_EQ(copy.imageSubresource.mipLevel, 2u);
    EXPECT_EQ(copy.imageSubresource.baseArrayLayer, 0u);
    EXPECT_EQ(copy.imageSubresource.layerCount, 1u);
    EXPECT_EQ(copy.imageOffset.x, 0);
    EXPECT_EQ(copy.imageExtent.width, 3u);
    EXPECT_EQ(copy.imageExtent.height, 2u);
    EXPECT_EQ(copy.imageExtent.depth, 1u);
    // 3D source: depth carries into the extent in texels.
    const auto copy3d = ImageUtils::CreateBufferImageCopy(
        vk::Format::eR8G8B8A8Unorm, 4, 4, 2, vk::ImageAspectFlagBits::eColor);
    EXPECT_EQ(copy3d.imageExtent.depth, 2u);
    EXPECT_EQ(copy3d.imageSubresource.layerCount, 1u);
}

}  // namespace
