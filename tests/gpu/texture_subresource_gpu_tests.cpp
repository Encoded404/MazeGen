#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.ImageUtils;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.TextureLoaders;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;

namespace {

using namespace VulkanEngine::Textures;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::HeapImage;
using VulkanEngine::GpuResources::ImageHeapConfig;

// ── Headless stack shared by these tests ─────────────────────────────

struct HeadlessStack {
    TestSupport::HeadlessVulkanBackend backend{};
    GpuImageHeap heap{};

    [[nodiscard]] bool Initialize() {
        if (!backend.Initialize()) {
            return false;
        }
        return heap.Initialize(backend, ImageHeapConfig{}, "texture-subresource-tests");
    }
};

[[nodiscard]] std::vector<std::byte> DeterministicBlob(std::size_t bytes, std::uint8_t seed) {
    std::vector<std::byte> blob(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        blob[i] = static_cast<std::byte>((seed + static_cast<std::uint8_t>(i)) & 0xFFU);
    }
    return blob;
}

// Builds a single-mip TextureData of `layers` array layers, each of
// `layer_bytes` block-tight bytes, laid out sequentially.
[[nodiscard]] TextureData LayeredData(std::uint32_t width, std::uint32_t height,
                                       std::uint32_t layers, std::size_t layer_bytes,
                                       vk::Format format, std::uint8_t seed,
                                       bool cube_complete = true) {
    TextureData data{};
    data.width = width;
    data.height = height;
    data.mip_levels = 1;
    data.array_layers = layers;
    data.face_count = 1;
    data.source_format = format;
    data.cube_complete = cube_complete;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        data.subresources.push_back(TextureSubresource{
            .mip = 0, .array_layer = layer,
            .offset = layer * layer_bytes, .size = layer_bytes,
            .width = width, .height = height, .depth = 1,
        });
    }
    data.blob = DeterministicBlob(layers * layer_bytes, seed);
    return data;
}

// Uploads via the production path, then reads the image back to a host
// buffer and returns the bytes for comparison.
[[nodiscard]] std::vector<std::byte> UploadAndReadBack(
    HeadlessStack& stack, const TextureData& data, vk::Format resolved,
    std::size_t readback_bytes) {
    auto& backend = stack.backend;
    auto texture = VulkanEngine::GpuResources::GpuTexture::CreateFromTextureData(
        backend, stack.heap, data, resolved);
    if (!texture.IsValid()) {
        ADD_FAILURE() << "CreateFromTextureData failed";
        return {};
    }

    vk::BufferCreateInfo buffer_info({}, static_cast<vk::DeviceSize>(readback_bytes),
                                     vk::BufferUsageFlagBits::eTransferDst);
    vk::raii::Buffer read_buffer(backend.GetDevice(), buffer_info);
    const vk::MemoryRequirements requirements = read_buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo alloc_info(requirements.size,
                                      VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                                          backend.GetPhysicalDevice(), requirements.memoryTypeBits,
                                          vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostVisible) |
                                              vk::MemoryPropertyFlagBits::eHostCoherent));
    vk::raii::DeviceMemory read_memory(backend.GetDevice(), alloc_info);
    read_buffer.bindMemory(read_memory, 0);

    auto& cmd = backend.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    VulkanBackend::Vulkan::ImageUtils::CmdTransitionImageLayout(
        cmd, texture.GetImage(), vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageAspectFlagBits::eColor, 0, 1, 0,
        data.array_layers * data.face_count);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(data.subresources.size());
    for (const auto& sub : data.subresources) {
        vk::BufferImageCopy region{};
        region.bufferOffset = static_cast<vk::DeviceSize>(sub.offset);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = vk::ImageSubresourceLayers(
            vk::ImageAspectFlagBits::eColor, sub.mip, sub.array_layer, 1);
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{sub.width, sub.height, std::max(sub.depth, 1U)};
        regions.push_back(region);
    }
    cmd.copyImageToBuffer(texture.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          read_buffer, regions);

    VulkanBackend::Vulkan::ImageUtils::CmdTransitionImageLayout(
        cmd, texture.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor, 0, 1, 0,
        data.array_layers * data.face_count);

    cmd.end();
    vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
    backend.GetGraphicsQueue().submit(submit, {});
    backend.GetGraphicsQueue().waitIdle();

    void* mapped = read_memory.mapMemory(0, requirements.size);
    std::vector<std::byte> out(static_cast<std::size_t>(readback_bytes));
    std::memcpy(out.data(), mapped, out.size());
    read_memory.unmapMemory();

    texture.Retire(0);
    stack.heap.BeginFrame(1); // device is idle; retire drain frees immediately
    return out;
}

void ExpectSubresourceBytesMatch(const TextureData& data, const std::vector<std::byte>& readback) {
    ASSERT_EQ(readback.size(), data.blob.size());
    for (const auto& sub : data.subresources) {
        for (std::size_t i = 0; i < sub.size; ++i) {
            const std::size_t idx = static_cast<std::size_t>(sub.offset) + i;
            ASSERT_EQ(readback[idx], data.blob[idx])
                << "subresource (mip=" << sub.mip << ", layer=" << sub.array_layer << ") byte " << i;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

TEST(TextureSubresourceGpuTest, Bc7MipChainRoundTripsBlockExact) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    // 8x8 BC7 with a 2-level chain: 4 blocks + 1 block, 16 bytes each.
    TextureData data{};
    data.width = 8;
    data.height = 8;
    data.mip_levels = 2;
    data.source_format = vk::Format::eBc7UnormBlock;
    data.subresources.push_back(TextureSubresource{.mip = 0, .array_layer = 0, .offset = 0, .size = 64, .width = 8, .height = 8, .depth = 1});
    data.subresources.push_back(TextureSubresource{.mip = 1, .array_layer = 0, .offset = 64, .size = 16, .width = 4, .height = 4, .depth = 1});
    data.blob = DeterministicBlob(80, 11);

    const auto readback = UploadAndReadBack(stack, data, vk::Format::eBc7UnormBlock, 80);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, TwoByTwoBc7CopiesWithoutError) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    // 2x2 tail-mip edge case: one 16-byte block for a sub-4x4 extent.
    TextureData data{};
    data.width = 2;
    data.height = 2;
    data.source_format = vk::Format::eBc7UnormBlock;
    data.subresources.push_back(TextureSubresource{.mip = 0, .array_layer = 0, .offset = 0, .size = 16, .width = 2, .height = 2, .depth = 1});
    data.blob = DeterministicBlob(16, 3);

    const auto readback = UploadAndReadBack(stack, data, vk::Format::eBc7UnormBlock, 16);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, Bc5NormalRoundTrips) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    TextureData data{};
    data.width = 16;
    data.height = 16;
    data.mip_levels = 1;
    data.source_format = vk::Format::eBc5UnormBlock;
    data.subresources.push_back(TextureSubresource{.mip = 0, .array_layer = 0, .offset = 0, .size = 16 * 16 / 4, .width = 16, .height = 16, .depth = 1});
    data.blob = DeterministicBlob(64, 7);

    const auto readback = UploadAndReadBack(stack, data, vk::Format::eBc5UnormBlock, 64);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, Bc4MaskRoundTrips) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    TextureData data{};
    data.width = 16;
    data.height = 16;
    data.mip_levels = 1;
    data.source_format = vk::Format::eBc4UnormBlock;
    data.subresources.push_back(TextureSubresource{.mip = 0, .array_layer = 0, .offset = 0, .size = 16 * 16 / 8, .width = 16, .height = 16, .depth = 1});
    data.blob = DeterministicBlob(32, 42);

    const auto readback = UploadAndReadBack(stack, data, vk::Format::eBc4UnormBlock, 32);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, CubeFacesLandInTheirOwnLayers) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    // 6 cube faces at 4x4 RGBA8: each face keeps its own bytes.
    auto data = LayeredData(4, 4, 6, 64, vk::Format::eR8G8B8A8Unorm, 5);
    data.face_count = 6; // 6 faces folded into 6 array layers
    const auto readback = UploadAndReadBack(stack, data, vk::Format::eR8G8B8A8Unorm, 6 * 64);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, TwoDimensionalArrayLayersRoundTrip) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    const auto data = LayeredData(4, 4, 3, 64, vk::Format::eR8G8B8A8Unorm, 9);
    const auto readback = UploadAndReadBack(stack, data, vk::Format::eR8G8B8A8Unorm, 3 * 64);
    ExpectSubresourceBytesMatch(data, readback);
}

TEST(TextureSubresourceGpuTest, HeapIndicesRecycleWithGenerationBumps) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    vk::ImageCreateInfo info{};
    info.imageType = vk::ImageType::e2D;
    info.format = vk::Format::eR8G8B8A8Unorm;
    info.extent = vk::Extent3D{8, 8, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = vk::SampleCountFlagBits::e1;
    info.tiling = vk::ImageTiling::eOptimal;
    info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

    auto first = stack.heap.Allocate(info, {});
    ASSERT_TRUE(first.IsValid());
    const std::uint32_t first_index = first.image_index;
    const std::uint32_t first_generation = first.generation;

    auto second = stack.heap.Allocate(info, {});
    ASSERT_TRUE(second.IsValid());
    EXPECT_NE(second.image_index, first_index);

    stack.heap.Free(first);
    auto recycled = stack.heap.Allocate(info, {});
    ASSERT_TRUE(recycled.IsValid());
    EXPECT_EQ(recycled.image_index, first_index);      // index actually reused
    EXPECT_GT(recycled.generation, first_generation);  // generation guards reuse

    // A stale handle resolves to nothing and cannot free the live record.
    EXPECT_EQ(stack.heap.GetImage(first), vk::Image{nullptr});
    EXPECT_EQ(stack.heap.GetImageView(first), vk::ImageView{nullptr});
    HeapImage stale = first;
    stack.heap.Free(stale);
    EXPECT_NE(stack.heap.GetImage(recycled), vk::Image{nullptr});

    stack.heap.Free(second);
    stack.heap.Free(recycled);
}

TEST(TextureSubresourceGpuTest, RetireFreesAtTheRingDrain) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    vk::ImageCreateInfo info{};
    info.imageType = vk::ImageType::e2D;
    info.format = vk::Format::eR8G8B8A8Unorm;
    info.extent = vk::Extent3D{4, 4, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = vk::SampleCountFlagBits::e1;
    info.tiling = vk::ImageTiling::eOptimal;
    info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

    auto image = stack.heap.Allocate(info, {});
    ASSERT_TRUE(image.IsValid());
    const auto live = stack.heap.GetImage(image);
    ASSERT_NE(live, vk::Image{nullptr});

    stack.heap.Retire(image, /*recording_frame=*/10);
    // Still alive before the drain: a frame could still sample it.
    EXPECT_NE(stack.heap.GetImage(image), vk::Image{nullptr});

    // FIF=1: the retire ring for frame 10 drains at the next frame start.
    stack.heap.BeginFrame(11);
    EXPECT_EQ(stack.heap.GetImage(image), vk::Image{nullptr});
}

TEST(TextureSubresourceGpuTest, SrgbViewFormatReachesTheImage) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }
    HeadlessStack stack;
    ASSERT_TRUE(stack.Initialize());

    // The resolved _SRGB format must be accepted end-to-end (image + view).
    const auto data = LayeredData(8, 8, 1, 8 * 8 * 4, vk::Format::eR8G8B8A8Unorm, 1);
    const auto readback = UploadAndReadBack(stack, data, vk::Format::eR8G8B8A8Srgb, 8 * 8 * 4);
    ExpectSubresourceBytesMatch(data, readback);
}

}  // namespace
