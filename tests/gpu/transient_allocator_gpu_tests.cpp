#include <gtest/gtest.h>

import std;

import vulkan;

import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.TransientAllocator;
import test_gpu;

namespace {

// The shared headless real-device backend (tests/support) provides the device
// and physical device the heap and allocator bind through, exactly as the other
// GPU-labelled tests obtain them; no local bootstrap adapter is needed.
class GpuTransientAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        backend_.Shutdown();
    }

    TestSupport::HeadlessVulkanBackend backend_{};
};

vk::ImageCreateInfo ColorImageInfo(std::uint32_t width, std::uint32_t height, bool aliasable) {
    vk::ImageCreateInfo info{};
    info.imageType = vk::ImageType::e2D;
    info.format = vk::Format::eR8G8B8A8Unorm;
    info.extent = vk::Extent3D{width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = vk::SampleCountFlagBits::e1;
    info.tiling = vk::ImageTiling::eOptimal;
    info.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferDst;
    info.sharingMode = vk::SharingMode::eExclusive;
    info.initialLayout = vk::ImageLayout::eUndefined;
    if (aliasable) {
        info.flags |= vk::ImageCreateFlagBits::eAlias;
    }
    return info;
}

// Slot-0 offset of a resource in the plan the allocator binds through. The plan
// surface is the allocator's public API (the engine's render graph reads the
// same placements), and offset equality is exactly what "aliases to the same
// memory" means, including across FIF slots.
[[nodiscard]] std::optional<std::uint64_t> PlacedOffset(
    const VulkanEngine::GpuResources::TransientPlacementPlan& plan, std::uint32_t resource_index,
    std::uint32_t fif_slot) {
    for (const auto& placement : plan.placements) {
        if (placement.resource_index == resource_index && placement.fif_slot == fif_slot) {
            return placement.offset;
        }
    }
    return std::nullopt;
}

TEST_F(GpuTransientAllocatorTest, ImageHeapBindsSuballocatedImages) {
    VulkanEngine::GpuResources::GpuImageHeap heap;
    ASSERT_TRUE(heap.Initialize(backend_, VulkanEngine::GpuResources::ImageHeapConfig{}, "test-image-heap"));
    EXPECT_GE(heap.GetBufferImageGranularity(), 1u);

    const auto a = heap.Allocate(ColorImageInfo(64, 64, /*aliasable=*/true), vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor});
    const auto b = heap.Allocate(ColorImageInfo(64, 64, /*aliasable=*/true), vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor});
    ASSERT_TRUE(a.IsValid());
    ASSERT_TRUE(b.IsValid());
    EXPECT_FALSE(heap.IsDedicated(a.image_index));
    EXPECT_NE(heap.GetImage(a.image_index), nullptr);
    EXPECT_NE(heap.GetImageView(a.image_index), nullptr);
    EXPECT_NE(heap.GetImage(b.image_index), nullptr);

    auto free_a = a;
    auto free_b = b;
    heap.Free(free_a);
    heap.Free(free_b);
    EXPECT_FALSE(free_a.IsValid());
    heap.Shutdown();
}

TEST_F(GpuTransientAllocatorTest, RuntimePlacesAliasedCopiesPerFifSlot) {
    using VulkanEngine::GpuResources::TransientAllocator;
    using VulkanEngine::GpuResources::TransientKind;

    TransientAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(backend_, "test-transients"));

    std::vector<TransientAllocator::Desc> descs(3);

    descs[0].is_image = true;
    descs[0].requirements = {.name = "a", .kind = TransientKind::Image, .heap_key = 0,
                             .first_pass = 0, .last_pass = 1, .aliasable = true};
    descs[0].image = {.format = vk::Format::eR8G8B8A8Unorm, .width = 64, .height = 64,
                      .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                      .samples = vk::SampleCountFlagBits::e1, .aspect = vk::ImageAspectFlagBits::eColor};

    descs[1].is_image = true;
    descs[1].requirements = {.name = "b", .kind = TransientKind::Image, .heap_key = 0,
                             .first_pass = 2, .last_pass = 3, .aliasable = true};
    descs[1].image = descs[0].image;

    descs[2].is_image = false;
    descs[2].requirements = {.name = "scratch", .kind = TransientKind::Buffer, .heap_key = 1,
                             .size = 4096, .alignment = 256, .first_pass = 0, .last_pass = 3,
                             .aliasable = false};
    descs[2].buffer = {.usage = vk::BufferUsageFlagBits::eStorageBuffer,
                       .memory = vk::MemoryPropertyFlagBits::eDeviceLocal};

    allocator.Sync(descs, /*frames_in_flight=*/2, /*current_frame=*/0);

    for (std::uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_NE(allocator.GetImage(0, slot), nullptr);
        EXPECT_NE(allocator.GetImageView(0, slot), nullptr);
        EXPECT_NE(allocator.GetImage(1, slot), nullptr);

        vk::Buffer buffer{};
        vk::DeviceSize offset = 0;
        vk::DeviceSize size = 0;
        ASSERT_TRUE(allocator.GetBuffer(2, slot, buffer, offset, size));
        EXPECT_NE(buffer, nullptr);
        EXPECT_EQ(size, 4096u);
    }

    // a and b have disjoint lifetimes and are aliasable, so memory is reused:
    // the plan the allocator actually binds shares the same offset between them
    // in both FIF copies.
    const VulkanEngine::GpuResources::TransientPlacementPlan& plan = allocator.GetPlan();
    EXPECT_TRUE(plan.valid);
    EXPECT_EQ(PlacedOffset(plan, 0, 0), PlacedOffset(plan, 1, 0));
    EXPECT_TRUE(PlacedOffset(plan, 0, 0).has_value());
    EXPECT_EQ(PlacedOffset(plan, 0, 1), PlacedOffset(plan, 1, 1));
    EXPECT_TRUE(PlacedOffset(plan, 0, 1).has_value());

    // Reconfiguring retires the old generation; CollectGarbage frees it once the
    // retiring frame is complete. The headless backend reports frames
    // incomplete, so the retired generation lingers until Shutdown; the
    // observable outcome is that the new generation serves the image right away.
    descs[0].image.width = 32;
    allocator.Sync(descs, 2, 1);
    allocator.CollectGarbage(2);
    EXPECT_NE(allocator.GetImage(0, 0), nullptr);

    allocator.Shutdown();
}

}  // namespace