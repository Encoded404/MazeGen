#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.TextureLoaders;

namespace {

using VulkanEngine::Textures::TextureData;
using VulkanEngine::Textures::TextureSubresource;

TextureData MakeRgba8(std::uint32_t width, std::uint32_t height) {
    TextureData data{};
    data.width = width;
    data.height = height;
    data.source_format = vk::Format::eR8G8B8A8Unorm;
    data.source_channels = 4;
    data.source_alpha = 1;
    const std::size_t bytes = static_cast<std::size_t>(width) * height * 4U;
    data.subresources.push_back(TextureSubresource{
        .mip = 0, .array_layer = 0, .offset = 0, .size = bytes,
        .width = width, .height = height, .depth = 1});
    data.blob.resize(bytes);
    return data;
}

TEST(CpuMipChainTest, GeneratesFullChainForSourcelessRgba8) {
    auto data = MakeRgba8(8, 8);
    ASSERT_TRUE(VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(data));
    EXPECT_EQ(data.mip_levels, 4u);  // 8,4,2,1
    ASSERT_EQ(data.subresources.size(), 4u);
    EXPECT_EQ(data.subresources[3].width, 1u);
    EXPECT_EQ(data.subresources[3].height, 1u);
}

TEST(CpuMipChainTest, RejectsSourceThatAlreadyHasAChain) {
    auto data = MakeRgba8(8, 8);
    data.mip_levels = 2;
    data.subresources.push_back(TextureSubresource{.mip = 1, .array_layer = 0, .offset = 256,
                                                    .size = 64, .width = 4, .height = 4, .depth = 1});
    data.blob.resize(320);
    EXPECT_FALSE(VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(data));
}

TEST(CpuMipChainTest, RejectsNonRgba8AndHdrAndMultiLayer) {
    auto hdr = MakeRgba8(8, 8);
    hdr.is_hdr_source = true;
    EXPECT_FALSE(VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(hdr));

    auto bc = MakeRgba8(8, 8);
    bc.source_format = vk::Format::eBc7UnormBlock;
    EXPECT_FALSE(VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(bc));

    auto multi = MakeRgba8(8, 8);
    multi.array_layers = 2;
    EXPECT_FALSE(VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(multi));
}

} // namespace
