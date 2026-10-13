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
// Out-of-line UV1 variant, end to end through the REAL fragment shader.
//
// The test vertex shader's main_uv1 entry point emits UV0 at location 1 and a
// transposed UV1 at location 7. The shipped standard_mesh.spv main_uv1 entry
// point resolves each texture slot's UV from MaterialData.uv_sets (2 bits per
// slot). With an asymmetric gradient texture, selecting UV1 for the albedo slot
// must change the render versus UV0; equal hashes would mean the variant or the
// selection is dead.
//
// The mesh-path vertex shader (main_indir) is exercised by its own addressing
// tests; this test isolates the fragment side's per-slot selection with the
// exact VS->FS interface the shipped shader declares.
//
// Gpu label: the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

class UvVariantGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize()) << backend_.GetErrorMessage();
        heap_.Initialize(backend_, ImageHeapConfig{}, "uv-variant-tests");
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(), "");

        vs_uv0_ = shaders_->RegisterManual(
            std::format("{}/test_fullscreen_vs.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eVertex, "main");
        vs_uv1_ = shaders_->RegisterManual(
            std::format("{}/test_fullscreen_vs_uv1.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eVertex, "main_uv1");
        frag_uv0_ = shaders_->RegisterManual(
            std::format("{}/StandardMeshFrag.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment, "main");
        frag_uv1_ = shaders_->RegisterManual(
            std::format("{}/StandardMeshFragUV1.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment, "main_uv1");
    }

    void TearDown() override {
        harness_uv0_.Shutdown();
        harness_uv1_.Shutdown();
        shaders_.reset();
        heap_.Shutdown();
        backend_.Shutdown();
    }

    static constexpr std::uint32_t kSize = 64;

    // A gradient: red rises along x, blue rises along y. Sampling UV0=(u,v)
    // versus a transposed UV1=(v,u) yields visibly different colors, so the
    // two renders hash differently whenever the right set is sampled.
    [[nodiscard]] GpuTexture MakeGradientTexture() {
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4U);
        for (std::uint32_t y = 0; y < kSize; ++y) {
            for (std::uint32_t x = 0; x < kSize; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * kSize + x) * 4U;
                pixels[i + 0] = static_cast<std::uint8_t>(x * 255U / (kSize - 1));
                pixels[i + 1] = 120;
                pixels[i + 2] = static_cast<std::uint8_t>(y * 255U / (kSize - 1));
                pixels[i + 3] = 255;
            }
        }
        const auto ktx = TestSupport::MakeKtx2Rgba8(kSize, kSize, 1, pixels);
        TextureData data{};
        std::string error;
        const FileLoader::ByteBuffer buffer(ktx.begin(), ktx.end());
        if (!VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer("fixture.ktx2", buffer, data, &error)) {
            ADD_FAILURE() << "loader failed: " << error;
            return {};
        }
        auto tex = GpuTexture::CreateFromTextureData(backend_, heap_, data, vk::Format::eR8G8B8A8Unorm);
        if (!tex.IsValid()) {
            ADD_FAILURE() << "CreateFromTextureData failed";
        }
        return tex;
    }

    [[nodiscard]] std::uint64_t Hash(const std::vector<std::uint8_t>& pixels) const {
        std::uint64_t hash = 14695981039346656037ULL;
        for (std::uint8_t byte : pixels) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    std::unique_ptr<ShaderManager> shaders_;
    VulkanEngine::ShaderSystem::ShaderId vs_uv0_{};
    VulkanEngine::ShaderSystem::ShaderId vs_uv1_{};
    VulkanEngine::ShaderSystem::ShaderId frag_uv0_{};
    VulkanEngine::ShaderSystem::ShaderId frag_uv1_{};
    TestSupport::TextureRenderHarness harness_uv0_{};
    TestSupport::TextureRenderHarness harness_uv1_{};
};

// Flat, fully-lit scene: the albedo texture is the only thing that varies.
[[nodiscard]] TestSupport::HarnessSceneHeader FlatScene() {
    TestSupport::HarnessSceneHeader scene{};
    scene.ambient_color[0] = 1.0f;
    scene.ambient_color[1] = 1.0f;
    scene.ambient_color[2] = 1.0f;
    scene.ambient_color[3] = 1.0f;  // full ambient -> pure albedo * 1
    scene.light_count = 0;
    return scene;
}

TEST_F(UvVariantGpuTest, UvSetSelectionChangesSamplingAndMatchesGolden) {
    auto albedo = MakeGradientTexture();
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessMaterialData material{};
    material.uv_sets = 0u;  // albedo samples UV0

    TestSupport::TextureRenderHarnessConfig uv0_config{};
    uv0_config.width = kSize;
    uv0_config.height = kSize;
    uv0_config.mode = TestSupport::HarnessMode::Standard;
    uv0_config.vertex_shader = vs_uv0_;
    uv0_config.fragment_shader = frag_uv0_;
    ASSERT_TRUE(harness_uv0_.Initialize(backend_, *shaders_, uv0_config));

    const auto uv0_pixels = harness_uv0_.RenderAndReadBack(albedo, {}, {}, material, FlatScene(), {});
    ASSERT_EQ(uv0_pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    const std::uint64_t uv0_hash = Hash(uv0_pixels);

    // The UV1 variant: the vertex shader emits the transposed UV1 and the
    // fragment selects it for the albedo slot. A fresh harness is used because
    // a harness cannot be re-initialized after Shutdown (its backend state is
    // single-shot).
    TestSupport::TextureRenderHarnessConfig uv1_config{};
    uv1_config.width = kSize;
    uv1_config.height = kSize;
    uv1_config.mode = TestSupport::HarnessMode::Standard;
    uv1_config.vertex_shader = vs_uv1_;
    uv1_config.fragment_shader = frag_uv1_;
    // Slang emits every named wrapper as SPIR-V entry point "main" in its own
    // module, so the pipeline uses "main" for both stages.
    ASSERT_TRUE(harness_uv1_.Initialize(backend_, *shaders_, uv1_config));

    material.uv_sets = 0x1u;  // albedo samples UV1
    const auto uv1_pixels = harness_uv1_.RenderAndReadBack(albedo, {}, {}, material, FlatScene(), {});
    ASSERT_EQ(uv1_pixels.size(), uv0_pixels.size());
    const std::uint64_t uv1_hash = Hash(uv1_pixels);

    // Selecting UV1 for the albedo slot must change the sampled image. A dead
    // branch (no UV1 interpolant, or the selection ignored) makes them equal.
    EXPECT_NE(uv0_hash, uv1_hash)
        << "UV0 and UV1 albedo renders were identical; the out-of-line set is not consumed";

    // Golden hashes pin the shipped fragment selection and the transposed UV1.
    constexpr std::uint64_t kGoldenUv0Hash = 0x3bca043044a02f35ULL;
    constexpr std::uint64_t kGoldenUv1Hash = 0xa482ade04728c305ULL;
    EXPECT_EQ(uv0_hash, kGoldenUv0Hash)
        << "UV0 albedo hash changed (observed 0x" << std::hex << uv0_hash << ")";
    EXPECT_EQ(uv1_hash, kGoldenUv1Hash)
        << "UV1 albedo hash changed (observed 0x" << std::hex << uv1_hash << ")";
}

}  // namespace
