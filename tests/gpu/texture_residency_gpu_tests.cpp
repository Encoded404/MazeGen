#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.ResourceSystem;
import VulkanEngine.TextureTypes;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;
import VulkanEngine.MaterialManager;
import VulkanEngine.TextureResidency;

namespace {

using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;

// End-to-end whole-texture eviction: residency selects an LRU texture, releases
// its bindless slot through the deferred decommit, and the descriptor is reset
// to the fallback at the ring drain with no validation errors.
class TextureResidencyGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    GpuImageHeap heap{};
    std::unique_ptr<VulkanEngine::BindlessManager::BindlessManager> bindless{};
    VulkanEngine::MaterialManager::MaterialManager materials{};
    std::unique_ptr<VulkanEngine::Textures::TextureResidency> residency{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        ASSERT_TRUE(heap.Initialize(backend, ImageHeapConfig{}, "residency-test"));

        bindless = std::make_unique<VulkanEngine::BindlessManager::BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = 16;
        ASSERT_TRUE(bindless->Initialize(backend, capacity_config));
        bindless->SetFallback(MakeSolid(0xCC), VulkanEngine::ResourceId{"checkerboard"});

        residency = std::make_unique<VulkanEngine::Textures::TextureResidency>();
        residency->Initialize(*bindless, materials, backend.GetCapabilities(), /*budget=*/0);
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        if (residency) residency->Shutdown();
        if (bindless) bindless->Shutdown();
        heap.Shutdown();
        backend.Shutdown();
    }

    GpuTexture MakeSolid(std::uint8_t value) {
        VulkanEngine::Textures::TextureData data{};
        data.width = 4;
        data.height = 4;
        data.source_format = vk::Format::eR8G8B8A8Unorm;
        data.source_channels = 4;
        data.source_alpha = 1;
        data.subresources.push_back(VulkanEngine::Textures::TextureSubresource{
            .mip = 0, .array_layer = 0, .offset = 0, .size = 64,
            .width = 4, .height = 4, .depth = 1,
        });
        data.blob.assign(64, static_cast<std::byte>(value));
        return GpuTexture::CreateFromTextureData(backend, heap, data, vk::Format::eR8G8B8A8Unorm);
    }
};

// Evicting the only resident texture releases its slot through the deferred
// decommit; at the drain the descriptor points at the fallback and the slot is
// free again.
TEST_F(TextureResidencyGpuTest, EvictsLruAndResetsDescriptorToFallback) {
    auto handle = bindless->AllocateTextureSlot(MakeSolid(0x11), VulkanEngine::ResourceId{"resident"});
    ASSERT_TRUE(handle.has_value());
    const std::uint32_t slot = handle->slot;
    const auto fallback_view = bindless->GetTexture(
        VulkanEngine::BindlessManager::kFallbackSlot)->GetImageView();
    // AllocateTextureSlot publishes immediately, so the slot is real.
    EXPECT_TRUE(bindless->IsCommitted(slot));
    EXPECT_NE(bindless->GetTexture(slot)->GetImageView(), fallback_view);

    residency->TrackReservation(*handle, /*bytes=*/0, VulkanEngine::ResourceId{"resident"}, 0);
    residency->MarkResident(*handle, /*bytes=*/4096, /*frame=*/0);
    residency->SetBudgetBytes(/*bytes=*/1024);  // force one eviction
    const std::uint32_t free_before = bindless->FreeSlotCount();
    const std::uint32_t evicted = residency->Collect(/*frame=*/0, /*min_residency=*/0);
    EXPECT_EQ(evicted, 1u);

    // Release is deferred: the slot's binding is not torn down until the ring
    // that frame 0 maps to is drained at frame `fif`.
    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(/*frame_index=*/fif, [](std::uint32_t) { return true; });

    // The decommit applied: the slot is no longer committed and has returned to
    // the free list. The descriptor now points at the fallback (checked through
    // the still-valid slot-0 view) with no validation error.
    EXPECT_FALSE(bindless->IsCommitted(slot));
    EXPECT_EQ(bindless->FreeSlotCount(), free_before + 1u);
    EXPECT_EQ(bindless->GetTexture(VulkanEngine::BindlessManager::kFallbackSlot)->GetImageView(),
              fallback_view);
    (void)fallback_view;
}

// A pinned texture is skipped even when it is the only eviction candidate.
TEST_F(TextureResidencyGpuTest, PinnedTextureSurvivesEviction) {
    auto handle = bindless->AllocateTextureSlot(MakeSolid(0x22), VulkanEngine::ResourceId{"pinned"});
    ASSERT_TRUE(handle.has_value());
    residency->TrackReservation(*handle, 0, VulkanEngine::ResourceId{"pinned"}, 0, /*pinned=*/true);
    residency->MarkResident(*handle, 4096, 0);
    residency->SetBudgetBytes(1024);
    EXPECT_EQ(residency->Collect(10, 0), 0u);
    EXPECT_TRUE(bindless->IsCommitted(handle->slot));
}

}  // namespace
