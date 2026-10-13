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
import VulkanEngine.TechniqueManager.UnlitTextureTechnique;
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
// Unlit shader contract, end to end through the REAL unlit.spv.
//
// The trimmed UnlitPerMaterialData (24 bytes) is uploaded to set 5 and the
// shipped fragment shader multiplies the sampled albedo by albedo_factor and
// discards below alpha_cutoff. The harness supplies the exact VS->FS interface
// and a constant texture, so the whole frame is one filtered-constant color.
//
// Gpu label: the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

class UnlitRenderGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize()) << backend_.GetErrorMessage();
        heap_.Initialize(backend_, ImageHeapConfig{}, "unlit-render-tests");
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(), "");
        harness_ids_ = TestSupport::RegisterHarnessShaders(*shaders_, VKENGINE_TEST_SHADER_DIR);

        // The real engine unlit fragment shader, registered straight from the
        // build tree (no generated module needed for a manual registration).
        unlit_frag_ = shaders_->RegisterManual(
            std::format("{}/UnlitFrag.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment);

        TestSupport::TextureRenderHarnessConfig config{};
        config.width = kSize;
        config.height = kSize;
        config.mode = TestSupport::HarnessMode::Unlit;
        config.vertex_shader = harness_ids_.vertex;
        config.fragment_shader = unlit_frag_;
        ASSERT_TRUE(harness_.Initialize(backend_, *shaders_, config));
    }

    void TearDown() override {
        harness_.Shutdown();
        shaders_.reset();
        heap_.Shutdown();
        backend_.Shutdown();
    }

    static constexpr std::uint32_t kSize = 64;

    // A constant texel (opaque) so the whole render is a single stable color;
    // the fragment samples the texel center exactly at this resolution, so the
    // result is filtering-independent and identical across devices.
    [[nodiscard]] GpuTexture MakeConstantTexture(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4U);
        for (std::size_t i = 0; i < pixels.size(); i += 4U) {
            pixels[i + 0] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = 255;
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
    TestSupport::HarnessShaderIds harness_ids_{};
    VulkanEngine::ShaderSystem::ShaderId unlit_frag_{};
    TestSupport::TextureRenderHarness harness_{};
};

// Golden hash for the fixed render below: one constant albedo (200,150,100),
// albedo_factor 1, alpha_cutoff 0.5. The whole image is a single filtered
// constant, so the hash is stable across runs and devices. A deliberate change
// to the shader or the material layout updates it in the same commit.
constexpr std::uint64_t kGoldenUnlitHash = 0xa194e8cfba88a325ULL;

TEST_F(UnlitRenderGpuTest, AlbedoFactorMultipliesTheSample) {
    auto albedo = MakeConstantTexture(200, 150, 100);
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessUnlitMaterialData material{};
    material.albedo_factor[0] = 1.0f;
    material.albedo_factor[1] = 1.0f;
    material.albedo_factor[2] = 1.0f;
    material.albedo_factor[3] = 1.0f;
    material.alpha_cutoff = 0.5f;

    const auto full = harness_.RenderUnlitAndReadBack(albedo, material);
    ASSERT_EQ(full.size(), static_cast<std::size_t>(kSize) * kSize * 4U);

    // The sampled albedo must reach the output: the red channel dominates.
    EXPECT_GT(full[0], 150) << "unlit shader did not output the albedo";

    material.albedo_factor[0] = 0.5f;
    material.albedo_factor[1] = 0.5f;
    material.albedo_factor[2] = 0.5f;
    const auto half = harness_.RenderUnlitAndReadBack(albedo, material);
    ASSERT_EQ(half.size(), full.size());

    // A dead albedo_factor multiply would make these identical.
    EXPECT_NE(Hash(full), Hash(half)) << "albedo_factor had no effect";
    EXPECT_LT(half[0], full[0]) << "0.5 factor did not darken the output";
}

TEST_F(UnlitRenderGpuTest, AlphaCutoffDiscardsBelowThreshold) {
    auto albedo = MakeConstantTexture(200, 150, 100);
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessUnlitMaterialData material{};
    material.albedo_factor[0] = 1.0f;
    material.albedo_factor[1] = 1.0f;
    material.albedo_factor[2] = 1.0f;
    material.albedo_factor[3] = 1.0f;
    // The source alpha is 1.0; a cutoff above it discards every fragment, so
    // the clear color (black, alpha 1) survives.
    material.alpha_cutoff = 2.0f;

    const auto discarded = harness_.RenderUnlitAndReadBack(albedo, material);
    ASSERT_EQ(discarded.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    // The clear is opaque black, so RGB is zero wherever the fragment was
    // discarded; only the alpha byte is one.
    bool any_color = false;
    for (std::size_t i = 0; i < discarded.size(); i += 4U) {
        if (discarded[i + 0] != 0 || discarded[i + 1] != 0 || discarded[i + 2] != 0) {
            any_color = true;
            break;
        }
    }
    EXPECT_FALSE(any_color) << "alpha_cutoff did not discard the fragment";
}

TEST_F(UnlitRenderGpuTest, GoldenHashUnchanged) {
    auto albedo = MakeConstantTexture(200, 150, 100);
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessUnlitMaterialData material{};
    material.albedo_factor[0] = 1.0f;
    material.albedo_factor[1] = 1.0f;
    material.albedo_factor[2] = 1.0f;
    material.albedo_factor[3] = 1.0f;
    material.alpha_cutoff = 0.5f;

    const auto pixels = harness_.RenderUnlitAndReadBack(albedo, material);
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    const std::uint64_t hash = Hash(pixels);
    EXPECT_EQ(hash, kGoldenUnlitHash)
        << "unlit render hash changed (observed 0x" << std::hex << hash
        << "); update kGoldenUnlitHash in the same commit if intentional";
}

}  // namespace
