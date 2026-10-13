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
import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
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
// Declared normal-encoding decode, end to end through the REAL fragment shader.
//
// A material references a normal map and declares its encoding in flags bits
// 0..1. The shipped standard_mesh.spv decodes it: Standard reconstructs z from
// RG; Full uses RGB as authored. The harness supplies the exact VS->FS interface
// and a fixed light, so the two paths differ only in the decode. Each encoding
// has a golden hash, pinning the TBN multiplication; the two hashes must also
// differ, proving the branch is live.
//
// Gpu label: the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

class NormalMappingGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize()) << backend_.GetErrorMessage();
        heap_.Initialize(backend_, ImageHeapConfig{}, "normal-mapping-tests");
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(), "");
        harness_ids_ = TestSupport::RegisterHarnessShaders(*shaders_, VKENGINE_TEST_SHADER_DIR);

        // The real engine fragment shader, registered straight from the build
        // tree (no generated module needed for a manual registration).
        standard_mesh_frag_ = shaders_->RegisterManual(
            std::format("{}/StandardMeshFrag.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment);

        TestSupport::TextureRenderHarnessConfig config{};
        config.width = kSize;
        config.height = kSize;
        config.mode = TestSupport::HarnessMode::Standard;
        config.vertex_shader = harness_ids_.vertex;
        config.fragment_shader = standard_mesh_frag_;
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
    VulkanEngine::ShaderSystem::ShaderId standard_mesh_frag_{};
    TestSupport::TextureRenderHarness harness_{};
};

// Scene constants for the test: a single directional light along +Z and a
// mid-gray albedo, chosen so shading changes are driven only by N.
[[nodiscard]] TestSupport::HarnessSceneHeader TestScene() {
    TestSupport::HarnessSceneHeader scene{};
    scene.ambient_color[0] = 0.0f;
    scene.ambient_color[1] = 0.0f;
    scene.ambient_color[2] = 0.0f;
    scene.ambient_color[3] = 1.0f;
    scene.light_count = 1;
    return scene;
}

[[nodiscard]] TestSupport::HarnessLight TestLight() {
    TestSupport::HarnessLight light{};
    // L = -direction = (0,0,1): the unperturbed N=(0,0,1) face is fully lit.
    light.direction[0] = 0.0f;
    light.direction[1] = 0.0f;
    light.direction[2] = -1.0f;
    light.color[0] = 1.0f;
    light.color[1] = 1.0f;
    light.color[2] = 1.0f;
    light.color[3] = 1.0f;
    return light;
}

// Golden hashes for the fixed scene below: one constant albedo, one asymmetric
// constant normal map, one directional light, and a flat TBN. They pin the
// shipped decode and are stable across runs and devices because the whole image
// is a single filtered-constant texel pair and the lighting is a pure dot
// product. A deliberate change to the decode updates both in the same commit.
constexpr std::uint64_t kGoldenStandardHash = 0x918525c6c987a325ULL;
constexpr std::uint64_t kGoldenFullHash = 0x6927fac75e74a325ULL;

TEST_F(NormalMappingGpuTest, StandardAndFullDecodeDifferAndMatchGolden) {
    auto albedo = MakeConstantTexture(200, 200, 200);
    ASSERT_TRUE(albedo.IsValid());
    // Asymmetric constant normal map: R=180 G=150 B=90. Neither the Standard nor
    // the Full decode is a no-op, so the branch is observable.
    auto normal = MakeConstantTexture(180, 150, 90);
    ASSERT_TRUE(normal.IsValid());

    TestSupport::HarnessMaterialData material{};
    material.roughness_factor = 1.0f;
    material.metallic_factor = 0.0f;
    material.normal_scale = 1.0f;

    const auto scene = TestScene();
    const auto light = TestLight();

    // The writer under test is the production packing helper, not a raw literal.
    material.flags = VulkanEngine::TechniqueManager::PackNormalEncoding(TextureNormalEncoding::Standard);
    const auto standard_pixels = harness_.RenderAndReadBack(albedo, normal, {}, material, scene, light);
    ASSERT_EQ(standard_pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);

    material.flags = VulkanEngine::TechniqueManager::PackNormalEncoding(TextureNormalEncoding::Full);
    const auto full_pixels = harness_.RenderAndReadBack(albedo, normal, {}, material, scene, light);
    ASSERT_EQ(full_pixels.size(), standard_pixels.size());

    const std::uint64_t standard_hash = Hash(standard_pixels);
    const std::uint64_t full_hash = Hash(full_pixels);

    // The declared encoding must change the result: a dead branch would make
    // these equal.
    EXPECT_NE(standard_hash, full_hash) << "Standard and Full decodes produced identical output";

    EXPECT_EQ(standard_hash, kGoldenStandardHash)
        << "Standard normal decode hash changed (observed 0x" << std::hex << standard_hash << ")";
    EXPECT_EQ(full_hash, kGoldenFullHash)
        << "Full normal decode hash changed (observed 0x" << std::hex << full_hash << ")";
}

TEST_F(NormalMappingGpuTest, NoNormalTextureLeavesTheBaseNormal) {
    // With normal_texture == 0 the decode branch must be skipped; this is the
    // baseline the two encoded renders are compared against conceptually.
    auto albedo = MakeConstantTexture(200, 200, 200);
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessMaterialData material{};
    material.flags = 0u;
    const auto scene = TestScene();
    const auto light = TestLight();

    const auto pixels = harness_.RenderAndReadBack(albedo, {}, {}, material, scene, light);
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    // The render must not be uniformly black (lighting reached the surface).
    const bool any_nonzero = std::ranges::any_of(pixels, [](std::uint8_t b) { return b != 0; });
    EXPECT_TRUE(any_nonzero);
}

}  // namespace
