#include <gtest/gtest.h>
#include <fstream>

#include "test_ktx_fixtures.hpp"

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureUploader;
import VulkanEngine.TextureResidency;
import VulkanEngine.TextureReloader;
import VulkanEngine.TextureWatcher;
import VulkanEngine.BindlessManager;
import VulkanEngine.MaterialManager;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuTexture;

namespace {

using namespace VulkanEngine::Textures;

namespace TextureSystem = VulkanEngine::TextureSystem;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;

// Hot reload end to end: a changed source file is re-decoded, uploaded into a
// new bindless slot, and the old slot is released only after the new binding
// publishes. A frame that recorded against the old slot therefore always sees a
// live image; validation stays enabled throughout.
class TextureHotReloadGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    GpuImageHeap heap{};
    StagingPool pool{};
    std::unique_ptr<VulkanEngine::BindlessManager::BindlessManager> bindless{};
    std::unique_ptr<TextureUploader> uploader{};
    std::unique_ptr<TextureResidency> residency{};
    std::unique_ptr<TextureSystem::TextureWatcher> watcher{};
    std::unique_ptr<TextureReloader> reloader{};
    VulkanEngine::MaterialManager::MaterialManager materials{};
    std::filesystem::path temp_dir{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        ASSERT_TRUE(heap.Initialize(backend, ImageHeapConfig{}, "hot-reload-test"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 8u << 20;
        staging_config.capacity_bytes = 32u << 20;
        ASSERT_TRUE(pool.Initialize(backend, staging_config));

        bindless = std::make_unique<VulkanEngine::BindlessManager::BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = 64;
        ASSERT_TRUE(bindless->Initialize(backend, capacity_config));
        bindless->SetFallback(MakeSolidTexture(0xCC), VulkanEngine::ResourceId{"checkerboard"});

        uploader = std::make_unique<TextureUploader>();
        ASSERT_TRUE(uploader->Initialize(backend, heap, pool, *bindless, backend.GetCapabilities()));

        residency = std::make_unique<TextureResidency>();
        residency->Initialize(*bindless, materials, backend.GetCapabilities(), /*budget=*/0);
        uploader->SetOnReserve([this](VulkanEngine::BindlessManager::TextureHandle handle,
                                      const VulkanEngine::ResourceId& id) {
            residency->RegisterResourceHandle(id, handle, 0);
        });
        uploader->SetOnResident([](auto, auto, auto) {});

        watcher = std::make_unique<TextureSystem::TextureWatcher>();
        watcher->SetDebounce(std::chrono::milliseconds(0));
        reloader = std::make_unique<TextureReloader>();
        reloader->Initialize(*watcher, resources, *bindless, *uploader, *residency, materials);

        temp_dir = std::filesystem::temp_directory_path() / "vkengine-hot-reload-test";
        std::filesystem::create_directories(temp_dir);
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        if (reloader) reloader->Shutdown();
        if (watcher) { watcher->Stop(); watcher.reset(); }
        if (residency) residency->Shutdown();
        if (uploader) { uploader->Shutdown(); uploader.reset(); }
        if (bindless) { bindless->Shutdown(); bindless.reset(); }
        pool.Shutdown();
        heap.Shutdown();
        backend.Shutdown();
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
    }

    VulkanEngine::ResourceManager resources{};

    VulkanEngine::GpuResources::GpuTexture MakeSolidTexture(std::uint8_t value) {
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
        return VulkanEngine::GpuResources::GpuTexture::CreateFromTextureData(
            backend, heap, data, vk::Format::eR8G8B8A8Unorm);
    }

    static std::vector<std::uint8_t> SolidPixels(std::uint32_t size, std::array<std::uint8_t, 4> rgba) {
        std::vector<std::uint8_t> pixels(size * size * 4);
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            pixels[i + 0] = rgba[0];
            pixels[i + 1] = rgba[1];
            pixels[i + 2] = rgba[2];
            pixels[i + 3] = rgba[3];
        }
        return pixels;
    }

    // Uploads `resource` through the async path and records it on a frame until
    // the worker stages it. Returns the reservation.
    std::optional<TextureReservation> UploadAndRecord(const VulkanEngine::ResourceId& id,
                                                      const TextureData& data,
                                                      std::uint32_t frame) {
        auto reservation = uploader->Reserve(TextureSemantic::BaseColor,
                                             TextureNormalEncoding::Standard, {}, id);
        if (!reservation.has_value()) {
            return std::nullopt;
        }
        (void)uploader->Submit(*reservation, std::make_shared<TextureData>(data));
        auto& cmd = backend.GetCommandBuffer(0);
        for (int attempt = 0; attempt < 200; ++attempt) {
            uploader->BeginFrame(frame, [](std::uint32_t) { return true; });
            cmd.reset({});
            cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            const std::uint64_t recorded = uploader->RecordUploads(cmd, frame, 8u << 20);
            cmd.end();
            if (recorded > 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return reservation;
    }
};

// Reloading a changed file swaps the material to a new slot, and the old slot
// is only released after the new binding is sampleable. The in-flight frame
// that sampled the old slot is never left with a dead image.
TEST_F(TextureHotReloadGpuTest, SwapPublishesBeforeOldSlotIsReleased) {
    const auto ktx_v1 = TestSupport::MakeKtx2Rgba8(
        8, 8, 1, SolidPixels(8, {0x10, 0x20, 0x30, 0xFF}));
    const auto ktx_v2 = TestSupport::MakeKtx2Rgba8(
        8, 8, 1, SolidPixels(8, {0xAA, 0xBB, 0xCC, 0xFF}));
    const auto path = temp_dir / "reload.ktx2";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(ktx_v1.data()),
                                                static_cast<std::streamsize>(ktx_v1.size()));

    auto handle = resources.LoadFromFile<VulkanEngine::TextureResource>(
        path, VulkanEngine::ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    const auto reservation = UploadAndRecord(handle.GetId(), handle->GetData(), /*frame=*/0);
    ASSERT_TRUE(reservation.has_value());
    const std::uint32_t old_slot = reservation->handle.slot;

    // Publish the first upload.
    const std::uint32_t fif = backend.GetFramesInFlight();
    bindless->BeginFrame(fif, [](std::uint32_t) { return true; });
    ASSERT_TRUE(bindless->IsCommitted(old_slot));

    // Watch it, then change the file and fire the watcher.
    reloader->WatchResource(handle.GetId(), path, TextureSemantic::BaseColor,
                            TextureNormalEncoding::Standard, {});
    std::ofstream(path, std::ios::binary | std::ios::trunc)
        .write(reinterpret_cast<const char*>(ktx_v2.data()),
               static_cast<std::streamsize>(ktx_v2.size()));
    watcher->NotifyChanged(path.string());

    // Reload is decoded and submitted into a new slot; the old slot stays live.
    reloader->Pump(/*frame_index=*/fif + 1);
    ASSERT_EQ(reloader->PendingSwapCount(), 1u);
    EXPECT_TRUE(bindless->IsCommitted(old_slot));

    // Drive frames so the new upload records and publishes.
    auto& cmd = backend.GetCommandBuffer(0);
    bool recorded = false;
    for (int attempt = 0; attempt < 200 && !recorded; ++attempt) {
        uploader->BeginFrame(fif + 1, [](std::uint32_t) { return true; });
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        recorded = uploader->RecordUploads(cmd, fif + 1, 8u << 20) > 0;
        cmd.end();
        if (!recorded) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(recorded);
    const auto new_handle = residency->HandleForResource(handle.GetId());
    ASSERT_TRUE(new_handle.has_value());
    EXPECT_NE(new_handle->slot, old_slot);
    const std::uint32_t new_slot = new_handle->slot;

    // The publish applies one FIF after the recording frame; only then does the
    // reloader release the old slot.
    bindless->BeginFrame(2 * fif + 1, [](std::uint32_t) { return true; });
    reloader->Pump(/*frame_index=*/2 * fif + 1);
    EXPECT_EQ(reloader->PendingSwapCount(), 0u);
    ASSERT_TRUE(bindless->IsCommitted(new_slot));

    // The old slot's release is itself deferred one more cycle, so it is not
    // torn down while a frame could still be sampling it.
    bindless->BeginFrame(3 * fif + 1, [](std::uint32_t) { return true; });
    EXPECT_FALSE(bindless->IsCommitted(old_slot));
}

// A failed decode leaves the previous binding intact: no swap is scheduled and
// the resource payload is unchanged.
TEST_F(TextureHotReloadGpuTest, BadEditLeavesOldBindingIntact) {
    const auto ktx = TestSupport::MakeKtx2Rgba8(8, 8, 1, SolidPixels(8, {0x11, 0x22, 0x33, 0xFF}));
    const auto path = temp_dir / "bad.ktx2";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(ktx.data()),
                                                static_cast<std::streamsize>(ktx.size()));
    auto handle = resources.LoadFromFile<VulkanEngine::TextureResource>(
        path, VulkanEngine::ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    const auto reservation = UploadAndRecord(handle.GetId(), handle->GetData(), 0);
    ASSERT_TRUE(reservation.has_value());
    const std::uint32_t version_before = handle->GetVersion();

    reloader->WatchResource(handle.GetId(), path, TextureSemantic::BaseColor,
                            TextureNormalEncoding::Standard, {});
    // Corrupt the file, then fire.
    std::ofstream(path, std::ios::binary | std::ios::trunc) << "not an image";
    watcher->NotifyChanged(path.string());
    reloader->Pump(7);
    EXPECT_EQ(reloader->PendingSwapCount(), 0u);
    EXPECT_EQ(handle->GetVersion(), version_before);
}

}  // namespace
