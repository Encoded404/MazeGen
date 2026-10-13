#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.ResourceSystem;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureUploader;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuTexture;

namespace {

using namespace VulkanEngine::Textures;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;

// End-to-end asynchronous upload: Reserve (fallback-bound) -> worker transcode/
// mip generation -> frame-recorded copies -> frame-gated publish one FIF later.
// The fallback is observable before the publish and the real image after it.
class TextureUploaderGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    GpuImageHeap heap{};
    StagingPool pool{};
    std::unique_ptr<VulkanEngine::BindlessManager::BindlessManager> bindless{};
    std::unique_ptr<TextureUploader> uploader{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        ASSERT_TRUE(heap.Initialize(backend, ImageHeapConfig{}, "uploader-test"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 8u << 20;
        staging_config.capacity_bytes = 32u << 20;
        ASSERT_TRUE(pool.Initialize(backend, staging_config));

        bindless = std::make_unique<VulkanEngine::BindlessManager::BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = 64;
        ASSERT_TRUE(bindless->Initialize(backend, capacity_config));
        bindless->SetFallback(MakeSolid(0xCC), VulkanEngine::ResourceId{"checkerboard"});

        uploader = std::make_unique<TextureUploader>();
        ASSERT_TRUE(uploader->Initialize(backend, heap, pool, *bindless, backend.GetCapabilities()));
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        if (uploader) {
            uploader->Shutdown();
            uploader.reset();
        }
        if (bindless) {
            bindless->Shutdown();
            bindless.reset();
        }
        pool.Shutdown();
        heap.Shutdown();
        backend.Shutdown();
    }

    GpuTexture MakeSolid(std::uint8_t value) {
        TextureData data{};
        data.width = 4;
        data.height = 4;
        data.source_format = vk::Format::eR8G8B8A8Unorm;
        data.source_channels = 4;
        data.source_alpha = 1;
        data.subresources.push_back(TextureSubresource{
            .mip = 0, .array_layer = 0, .offset = 0, .size = 64,
            .width = 4, .height = 4, .depth = 1,
        });
        data.blob.assign(64, static_cast<std::byte>(value));
        return GpuTexture::CreateFromTextureData(backend, heap, data, vk::Format::eR8G8B8A8Unorm);
    }

    static std::shared_ptr<const TextureData> MakeColorData() {
        auto data = std::make_shared<TextureData>();
        data->width = 8;
        data->height = 8;
        data->source_format = vk::Format::eR8G8B8A8Unorm;
        data->source_channels = 4;
        data->source_alpha = 1;
        data->semantic = TextureSemantic::BaseColor;
        const std::size_t bytes = 8u * 8u * 4u;
        data->subresources.push_back(TextureSubresource{
            .mip = 0, .array_layer = 0, .offset = 0, .size = bytes,
            .width = 8, .height = 8, .depth = 1,
        });
        data->blob.resize(bytes);
        for (std::size_t i = 0; i < bytes; i += 4u) {
            data->blob[i + 0] = std::byte{0x40};
            data->blob[i + 1] = std::byte{0x80};
            data->blob[i + 2] = std::byte{0xC0};
            data->blob[i + 3] = std::byte{0xFF};
        }
        return data;
    }
};

// Reserve returns a slot pre-bound to the fallback and never waits.
TEST_F(TextureUploaderGpuTest, ReserveIsImmediateAndFallbackBound) {
    auto reservation = uploader->Reserve(TextureSemantic::BaseColor,
                                          TextureNormalEncoding::Standard, {}, VulkanEngine::ResourceId{"t"});
    ASSERT_TRUE(reservation.has_value());
    EXPECT_NE(reservation->handle.slot, VulkanEngine::BindlessManager::kFallbackSlot);
    // The descriptor is already sampleable: same image view as the fallback.
    EXPECT_EQ(bindless->GetTexture(reservation->handle)->GetImageView(),
              bindless->GetTexture(VulkanEngine::BindlessManager::kFallbackSlot)->GetImageView());
}

// A submitted texture is decoded, copied on a recorded frame, and published one
// FIF cycle later; the binding is the real image only after the drain.
TEST_F(TextureUploaderGpuTest, SubmitStagesRecordsAndPublishesAtTheDrain) {
    auto reservation = uploader->Reserve(TextureSemantic::BaseColor,
                                          TextureNormalEncoding::Standard, {}, VulkanEngine::ResourceId{"async"});
    ASSERT_TRUE(reservation.has_value());
    auto future = uploader->Submit(*reservation, MakeColorData());
    const auto fallback_view =
        bindless->GetTexture(VulkanEngine::BindlessManager::kFallbackSlot)->GetImageView();

    auto& cmd = backend.GetCommandBuffer(0);
    std::uint64_t recorded = 0;
    // Wait for the worker to finish and the frame to record it.
    for (int attempt = 0; attempt < 200 && recorded == 0; ++attempt) {
        uploader->BeginFrame(0, [](std::uint32_t) { return true; });
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        recorded = uploader->RecordUploads(cmd, /*frame_index=*/0, /*budget=*/8u << 20);
        cmd.end();
        if (recorded == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    ASSERT_GT(recorded, 0u) << "upload was never staged";

    // The published binding applies one FIF cycle after the recording frame.
    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(0 + fif, [](std::uint32_t) { return true; });

    ASSERT_NE(bindless->GetTexture(reservation->handle), nullptr);
    EXPECT_NE(bindless->GetTexture(reservation->handle)->GetImageView(), fallback_view);
    ASSERT_TRUE(future.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    EXPECT_EQ(future.get().status, UploadStatus::Resident);
}

// Capacity exhaustion fails Reserve cleanly (caller substitutes the fallback).
TEST_F(TextureUploaderGpuTest, CapacityExhaustionFailsCleanly) {
    const std::uint32_t capacity = bindless->Capacity();
    std::vector<TextureReservation> held;
    for (std::uint32_t i = 1; i < capacity; ++i) {
        auto r = uploader->Reserve(TextureSemantic::BaseColor,
                                    TextureNormalEncoding::Standard, {},
                                    VulkanEngine::ResourceId{"s" + std::to_string(i)});
        ASSERT_TRUE(r.has_value());
        held.push_back(*r);
    }
    EXPECT_FALSE(uploader->Reserve(TextureSemantic::BaseColor,
                                   TextureNormalEncoding::Standard,
                                   {}, VulkanEngine::ResourceId{"overflow"}).has_value());
}

} // namespace
