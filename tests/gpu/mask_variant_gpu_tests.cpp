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
// The MASK entry points and the lit-shader fills, end to end through the REAL
// fragment shader SPIR-V.
//
// The MASK entry points (main_mask / main_mask_uv1 compiled into
// StandardMeshFragMask.spv) discard below the material's declared alpha cutoff
// and output a constant opaque alpha — opaque or hole. The base entry point
// (StandardMeshFrag.spv) never discards regardless of the material's flags.
//
// The harness pairs them with the test fullscreen vertex shader, which emits
// the exact VS->FS interface the shipped fragment declares: a +Z plane at
// z=0 spanning x,y in [-1,3] (visible [-1,1]) with worldPos = (x, y, 0) and
// ao=1/roughness=1/metallic=0 packed in the ORM attribute. camera_position is
// pushed at (10, 0, 0).
//
// Gpu label: the default preset stays device-free.
// ─────────────────────────────────────────────────────────────────────────────

class MaskVariantGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize()) << backend_.GetErrorMessage();
        heap_.Initialize(backend_, ImageHeapConfig{}, "mask-variant-tests");
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(), "");

        vs_ = shaders_->RegisterManual(
            std::format("{}/test_fullscreen_vs.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eVertex, "main");
        // Every shipped variant of standard_mesh.slang compiles to its own
        // module with the SPIR-V entry point "main" (slang-spirv-compiler
        // renames the wrapper).
        frag_base_ = shaders_->RegisterManual(
            std::format("{}/StandardMeshFrag.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment, "main");
        frag_mask_ = shaders_->RegisterManual(
            std::format("{}/StandardMeshFragMask.spv", VKENGINE_ENGINE_SHADER_DIR), "",
            ShaderStage::eFragment, "main");
    }

    void TearDown() override {
        for (const auto& harness : harnesses_) {
            harness->Shutdown();
        }
        shaders_.reset();
        heap_.Shutdown();
        backend_.Shutdown();
    }

    static constexpr std::uint32_t kSize = 64;

    // A constant texture with per-channel bytes, including alpha. The alpha
    // byte is what the mask entry points discard on, so it is this test's
    // fixture knob.
    [[nodiscard]] GpuTexture MakeConstantTexture(std::uint8_t r, std::uint8_t g,
                                                 std::uint8_t b, std::uint8_t a) {
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4U);
        for (std::uint32_t y = 0; y < kSize; ++y) {
            for (std::uint32_t x = 0; x < kSize; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * kSize + x) * 4U;
                pixels[i + 0] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = b;
                pixels[i + 3] = a;
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

    // A fresh harness per render (a harness cannot be re-initialized after
    // Shutdown; its backend state is single-shot).
    [[nodiscard]] std::vector<std::uint8_t> Render(
        VulkanEngine::ShaderSystem::ShaderId frag, const GpuTexture& albedo,
        const TestSupport::HarnessMaterialData& material,
        const TestSupport::HarnessSceneHeader& scene,
        const TestSupport::HarnessLight& light = {}) {
        harnesses_.push_back(std::make_unique<TestSupport::TextureRenderHarness>());
        TestSupport::TextureRenderHarnessConfig config{};
        config.width = kSize;
        config.height = kSize;
        config.mode = TestSupport::HarnessMode::Standard;
        config.vertex_shader = vs_;
        config.fragment_shader = frag;
        TestSupport::TextureRenderHarness& harness = *harnesses_.back();
        if (!harness.Initialize(backend_, *shaders_, config)) {
            ADD_FAILURE() << "harness init failed";
            return {};
        }
        return harness.RenderAndReadBack(albedo, {}, {}, material, scene, light);
    }

    [[nodiscard]] std::uint64_t Hash(const std::vector<std::uint8_t>& pixels) const {
        std::uint64_t hash = 14695981039346656037ULL;
        for (std::uint8_t byte : pixels) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    [[nodiscard]] static std::uint32_t LuminanceAt(const std::vector<std::uint8_t>& pixels,
                                                   std::uint32_t x, std::uint32_t y) {
        const std::size_t i = (static_cast<std::size_t>(y) * kSize + x) * 4U;
        return static_cast<std::uint32_t>(pixels[i]) + pixels[i + 1] + pixels[i + 2];
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    std::unique_ptr<ShaderManager> shaders_;
    VulkanEngine::ShaderSystem::ShaderId vs_{};
    VulkanEngine::ShaderSystem::ShaderId frag_base_{};
    VulkanEngine::ShaderSystem::ShaderId frag_mask_{};
    std::vector<std::unique_ptr<TestSupport::TextureRenderHarness>> harnesses_;
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

[[nodiscard]] TestSupport::HarnessMaterialData MaskMaterial(std::uint8_t cutoff) {
    TestSupport::HarnessMaterialData material{};
    material.flags = 4u;  // kAlphaMaskFlag: this payload selects the MASK pair
    material.alpha_cutoff = static_cast<float>(cutoff) / 255.0f;
    return material;
}

// MASK contract, discard path: the whole quad's alpha is below the cutoff, so
// every fragment dies and the clear color comes through — with the output
// alpha opaque (blend is off, so the mask entry's constant 1.0 lands in the
// target byte).
TEST_F(MaskVariantGpuTest, MaskCutoffDiscardsEverything) {
    auto albedo = MakeConstantTexture(128, 128, 128, 128);  // alpha ~0.50
    ASSERT_TRUE(albedo.IsValid());
    const auto pixels = Render(frag_mask_, albedo, MaskMaterial(193), FlatScene());
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    for (std::size_t i = 0; i < pixels.size(); i += 4U) {
        ASSERT_EQ(pixels[i + 0], 0) << "pixel " << i / 4;
        ASSERT_EQ(pixels[i + 1], 0) << "pixel " << i / 4;
        ASSERT_EQ(pixels[i + 2], 0) << "pixel " << i / 4;
        ASSERT_EQ(pixels[i + 3], 255) << "pixel " << i / 4;
    }
}

// MASK contract, keep path: alpha above the cutoff stays fully lit, and every
// output alpha byte is still opaque.
TEST_F(MaskVariantGpuTest, MaskCutoffKeepsLitQuad) {
    auto albedo = MakeConstantTexture(128, 128, 128, 128);
    ASSERT_TRUE(albedo.IsValid());
    const auto pixels = Render(frag_mask_, albedo, MaskMaterial(64), FlatScene());
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    for (std::size_t i = 0; i < pixels.size(); i += 4U) {
        ASSERT_GT(pixels[i + 0], 0) << "pixel " << i / 4;
        ASSERT_GT(pixels[i + 1], 0) << "pixel " << i / 4;
        ASSERT_EQ(pixels[i + 3], 255) << "pixel " << i / 4;
    }
    // Deterministic: an identical re-render hashes the same.
    const auto again = Render(frag_mask_, albedo, MaskMaterial(64), FlatScene());
    EXPECT_EQ(Hash(pixels), Hash(again));
}

// Discard is scoped to the MASK entry points: the same material and texture
// through the base entry point keeps every fragment and carries the real
// albedo alpha out (pins the lit shader's alpha-output fill; the base shader
// ignores the payload's mask flag).
TEST_F(MaskVariantGpuTest, MaskDiscardIsEntryPointScoped) {
    auto albedo = MakeConstantTexture(128, 128, 128, 128);
    ASSERT_TRUE(albedo.IsValid());
    const auto pixels = Render(frag_base_, albedo, MaskMaterial(193), FlatScene());
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);
    for (std::size_t i = 0; i < pixels.size(); i += 4U) {
        ASSERT_GT(pixels[i + 0], 0) << "pixel " << i / 4;
        // albedo.a = 128 must survive the write: the shader outputs the
        // sampled alpha, not a hardcoded 1.0.
        EXPECT_NEAR(static_cast<int>(pixels[i + 3]), 128, 1) << "pixel " << i / 4;
    }
}

// Emissive fill: with no emissive texture, the material factor glows on its
// own (glTF factor-only path). The added rgb must channel-wise brighten the
// fully-lit albedo against a zero-factor render.
TEST_F(MaskVariantGpuTest, EmissiveFactorBrightensChannelWise) {
    auto albedo = MakeConstantTexture(100, 110, 120, 255);
    ASSERT_TRUE(albedo.IsValid());

    TestSupport::HarnessMaterialData material{};
    material.emissive_texture = 0u;  // factor alone glows
    const auto base = Render(frag_base_, albedo, material, FlatScene());
    ASSERT_EQ(base.size(), static_cast<std::size_t>(kSize) * kSize * 4U);

    material.emissive_factor[0] = 0.10f;
    material.emissive_factor[1] = 0.20f;
    material.emissive_factor[2] = 0.30f;
    material.emissive_factor[3] = 1.0f;
    const auto lit = Render(frag_base_, albedo, material, FlatScene());
    ASSERT_EQ(lit.size(), base.size());

    EXPECT_NE(Hash(base), Hash(lit))
        << "the emissive factor had no effect on the render";
    for (std::size_t i = 0; i < base.size(); i += 4U) {
        // 0.10/0.20/0.30 of full scale is >= 24, 48, 73 bytes per channel.
        EXPECT_GE(static_cast<int>(lit[i + 0]) - static_cast<int>(base[i + 0]), 24)
            << "pixel " << i / 4;
        EXPECT_GE(static_cast<int>(lit[i + 1]) - static_cast<int>(base[i + 1]), 48)
            << "pixel " << i / 4;
        EXPECT_GE(static_cast<int>(lit[i + 2]) - static_cast<int>(base[i + 2]), 72)
            << "pixel " << i / 4;
        EXPECT_EQ(lit[i + 3], 255);
    }
}

// Point-light fill: brightness is symmetric about the light, falls off with
// distance, and stays below an equally-intense directional light at the same
// pixel (inverse-square attenuation, not just NdotL).
TEST_F(MaskVariantGpuTest, PointLightFallsOffInverseSquare) {
    auto albedo = MakeConstantTexture(255, 255, 255, 255);
    ASSERT_TRUE(albedo.IsValid());

    // Light one unit in front of the plane's center: L toward +Z as seen from
    // the plane, distance 2 to the closest point.
    TestSupport::HarnessLight point{};
    point.position[0] = 0.0f;
    point.position[1] = 0.0f;
    point.position[2] = 2.0f;
    point.position[3] = 1.0f;  // type = point
    point.color[0] = 1.0f;
    point.color[1] = 1.0f;
    point.color[2] = 1.0f;
    point.color[3] = 3.0f;

    TestSupport::HarnessSceneHeader scene{};
    scene.ambient_color[3] = 1.0f;  // rgb zero: no ambient floor
    scene.light_count = 1;

    const auto pixels = Render(frag_base_, albedo, {}, scene, point);
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kSize) * kSize * 4U);

    // Symmetric fragments (the pixel-index mirror around the light's center)
    // are equidistant in 3D, so their luminance must match.
    const auto center_a = LuminanceAt(pixels, 31, 31);
    const auto center_b = LuminanceAt(pixels, 32, 32);
    EXPECT_LE(static_cast<int>(center_a) - static_cast<int>(center_b), 2);
    EXPECT_LE(static_cast<int>(center_b) - static_cast<int>(center_a), 2);

    // Closer to the light is brighter than farther.
    const auto near_lum = LuminanceAt(pixels, 31, 31);
    const auto far_lum = LuminanceAt(pixels, 40, 32);
    EXPECT_GT(near_lum, far_lum);

    // The same pixel under an equally-intense directional light (NdotL = 1,
    // no distance attenuation) is at least as bright; the point light's
    // inverse-square term keeps it strictly below.
    TestSupport::HarnessLight directional{};
    directional.direction[0] = 0.0f;
    directional.direction[1] = 0.0f;
    directional.direction[2] = -1.0f;  // L = +Z onto the +Z plane
    directional.color[0] = 1.0f;
    directional.color[1] = 1.0f;
    directional.color[2] = 1.0f;
    directional.color[3] = 3.0f;
    const auto dir_pixels = Render(frag_base_, albedo, {}, scene, directional);
    ASSERT_EQ(dir_pixels.size(), pixels.size());
    EXPECT_GT(LuminanceAt(dir_pixels, 31, 31), near_lum)
        << "a point light must not reach the directional light's intensity "
           "at the closest point to it";
}

}  // namespace