#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.ResourceSystem;
import VulkanEngine.BindlessManager;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;

namespace {

using namespace VulkanEngine::BindlessManager;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;

// Headless bindless checks. The bindless manager needs a real device
// (descriptor set + pool + update-after-bind), so these stay on the gpu label.
class BindlessFallbackGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    GpuImageHeap heap{};
    std::unique_ptr<BindlessManager> bindless{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        ASSERT_TRUE(heap.Initialize(backend, VulkanEngine::GpuResources::ImageHeapConfig{}, "bindless-test"));
        bindless = std::make_unique<BindlessManager>();
        BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = 64;
        ASSERT_TRUE(bindless->Initialize(backend, capacity_config));
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        if (bindless) {
            bindless->Shutdown();
            bindless.reset();
        }
        heap.Shutdown();
        backend.Shutdown();
    }

    [[nodiscard]] GpuTexture MakeSolidTexture(std::uint8_t value) {
        VulkanEngine::Textures::TextureData data{};
        data.width = 2;
        data.height = 2;
        data.source_format = vk::Format::eR8G8B8A8Unorm;
        data.source_channels = 4;
        data.source_alpha = 1;
        data.subresources.push_back(VulkanEngine::Textures::TextureSubresource{
            .mip = 0, .array_layer = 0, .offset = 0, .size = 16,
            .width = 2, .height = 2, .depth = 1,
        });
        data.blob.assign(16, static_cast<std::byte>(value));
        return GpuTexture::CreateFromTextureData(backend, heap, data, vk::Format::eR8G8B8A8Unorm);
    }
};

// Slot 0 is the permanent fallback; allocation starts at slot 1.
TEST_F(BindlessFallbackGpuTest, FallbackOwnsSlotZeroAndAllocationStartsAtOne) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});

    EXPECT_NE(bindless->GetTexture(kFallbackSlot), nullptr);
    EXPECT_NE(bindless->GetTexture(kFallbackSlot)->GetImageView(), vk::ImageView{nullptr});
    EXPECT_NE(bindless->GetTexture(kFallbackSlot)->GetSampler(), vk::Sampler{nullptr});

    const auto first = bindless->AllocateTextureSlot(MakeSolidTexture(0x11), VulkanEngine::ResourceId{"first"});
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->slot, kFallbackSlot + 1U);
    const auto second = bindless->AllocateTextureSlot(MakeSolidTexture(0x22), VulkanEngine::ResourceId{"second"});
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->slot, kFallbackSlot + 2U);
    EXPECT_EQ(bindless->GetTextureId(first->slot)->value, "first");
    EXPECT_EQ(bindless->GetTextureId(second->slot)->value, "second");
    // The fallback is untouched by the allocations.
    EXPECT_EQ(bindless->GetTextureId(kFallbackSlot)->value, "checkerboard_default");
}

// Second SetFallback is ignored: slot 0 is written exactly once.
TEST_F(BindlessFallbackGpuTest, SetFallbackIsOneShot) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    bindless->SetFallback(MakeSolidTexture(0xBB), VulkanEngine::ResourceId{"imposter"});
    EXPECT_EQ(bindless->GetTextureId(kFallbackSlot)->value, "checkerboard_default");
    const auto next = bindless->AllocateTextureSlot(MakeSolidTexture(0x11), VulkanEngine::ResourceId{"next"});
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->slot, 1U);
}

// A reserved slot starts on the fallback descriptor; a commit applies at the
// begin-frame drain only when the recording frame was submitted.
TEST_F(BindlessFallbackGpuTest, CommitPublishesAtTheDrainWhenSubmitted) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    const auto handle = bindless->ReserveSlot(VulkanEngine::ResourceId{"async"});
    ASSERT_TRUE(handle.has_value());

    // Before the commit, the slot samples the fallback.
    EXPECT_EQ(bindless->GetTexture(*handle)->GetImageView(),
              bindless->GetTexture(kFallbackSlot)->GetImageView());

    GpuTexture real = MakeSolidTexture(0x33);
    const vk::ImageView real_view = real.GetImageView();
    bindless->CommitSlot(*handle, std::move(real), /*recording_frame=*/4);

    // Drain at frame 4 + FIF: no op applies before then.
    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(4, [](std::uint32_t) { return true; });
    bindless->BeginFrame(4 + fif, [](std::uint32_t) { return true; });
    ASSERT_NE(bindless->GetTexture(*handle), nullptr);
    EXPECT_EQ(bindless->GetTexture(*handle)->GetImageView(), real_view);
}

// A commit whose recording frame was never submitted is dropped and the slot
// stays on the fallback.
TEST_F(BindlessFallbackGpuTest, DroppedPublishLeavesTheFallback) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    const auto handle = bindless->ReserveSlot(VulkanEngine::ResourceId{"async"});
    ASSERT_TRUE(handle.has_value());

    GpuTexture real = MakeSolidTexture(0x33);
    bindless->CommitSlot(*handle, std::move(real), /*recording_frame=*/4);

    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(4 + fif, [](std::uint32_t) { return false; });
    ASSERT_NE(bindless->GetTexture(*handle), nullptr);
    EXPECT_EQ(bindless->GetTexture(*handle)->GetImageView(),
              bindless->GetTexture(kFallbackSlot)->GetImageView());
}

// A commit for a stale generation (slot re-reserved) is dropped.
TEST_F(BindlessFallbackGpuTest, StaleCommitIsDropped) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    const auto first = bindless->ReserveSlot(VulkanEngine::ResourceId{"first"});
    ASSERT_TRUE(first.has_value());
    const std::uint32_t slot = first->slot;

    // Release and let the decommit apply so the slot recycles (device idle:
    // drain at frame 0's slot immediately).
    bindless->ReleaseSlot(*first, /*recording_frame=*/0);
    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(0 + fif, [](std::uint32_t) { return true; });

    // Re-reserve the same index with a new generation.
    const auto second = bindless->ReserveSlot(VulkanEngine::ResourceId{"second"});
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->slot, slot);
    EXPECT_NE(second->generation, first->generation);

    // Commit with the stale handle: must be ignored (slot keeps the fallback).
    bindless->CommitSlot(*first, MakeSolidTexture(0x77), /*recording_frame=*/0);
    bindless->BeginFrame(fif + 1, [](std::uint32_t) { return true; });
    EXPECT_EQ(bindless->GetTexture(*second)->GetImageView(),
              bindless->GetTexture(kFallbackSlot)->GetImageView());
}

// Capacity is enforced: reserving past it fails cleanly instead of writing OOB.
TEST_F(BindlessFallbackGpuTest, CapacityExhaustionFailsCleanly) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    const std::uint32_t capacity = bindless->Capacity();
    ASSERT_GE(capacity, 2U);
    std::vector<TextureHandle> handles;
    for (std::uint32_t i = 1; i < capacity; ++i) {
        auto handle = bindless->ReserveSlot(VulkanEngine::ResourceId{"slot" + std::to_string(i)});
        ASSERT_TRUE(handle.has_value()) << "i=" << i;
        handles.push_back(*handle);
    }
    EXPECT_FALSE(bindless->ReserveSlot(VulkanEngine::ResourceId{"overflow"}).has_value());
}

// Shutdown clears every slot; the heap Shutdown after it verifies teardown order.
TEST_F(BindlessFallbackGpuTest, ShutdownClearsSlotsAndFallback) {
    bindless->SetFallback(MakeSolidTexture(0xAA), VulkanEngine::ResourceId{"checkerboard_default"});
    static_cast<void>(bindless->AllocateTextureSlot(MakeSolidTexture(0x11), VulkanEngine::ResourceId{"first"}));
    bindless->Shutdown();
    EXPECT_EQ(bindless->GetTexture(kFallbackSlot), nullptr);
    EXPECT_EQ(bindless->GetTexture(1U), nullptr);
    bindless.reset();
    heap.Shutdown(); // must not crash with textures already freed
    SUCCEED();
}

} // namespace
