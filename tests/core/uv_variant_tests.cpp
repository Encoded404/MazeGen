#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
import VulkanEngine.ShaderManager;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.StandardMeshPipeline;

namespace {

using VulkanEngine::TechniqueManager::DefaultMeshPerMaterialData;
using VulkanEngine::TechniqueManager::DefaultMeshTechnique;
using VulkanEngine::TechniqueManager::GetVariantShaderRow;
using VulkanEngine::TechniqueManager::VariantShaderPair;
namespace DrawKeyState = VulkanEngine::TechniqueManager::DrawKeyState;
namespace TechniquePacking = VulkanEngine::TechniqueManager::TechniquePacking;
using VulkanEngine::ShaderSystem::ShaderId;

std::span<const std::byte> AsBytes(const DefaultMeshPerMaterialData& data) {
    return {reinterpret_cast<const std::byte*>(&data), sizeof(data)};
}

// ── Interface-variant derivation from the material payload ──

TEST(UvVariantTest, DefaultMaterialUsesUv0) {
    DefaultMeshTechnique tech;
    DefaultMeshPerMaterialData mat{};
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(mat)), 0u);
}

TEST(UvVariantTest, Uv1SelectedInAnySlotBecomesVariant1) {
    DefaultMeshTechnique tech;
    // Slot 0 (albedo) selects set 1.
    DefaultMeshPerMaterialData mat{};
    mat.uv_sets = 0x1u;
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(mat)), 1u);

    // Slot 2 (ORM) selects set 1.
    mat.uv_sets = 0x1u << 4;
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(mat)), 1u);
}

TEST(UvVariantTest, HighestSelectedSetWins) {
    DefaultMeshTechnique tech;
    DefaultMeshPerMaterialData mat{};
    // albedo -> set 1, normal -> set 1.
    mat.uv_sets = 0x1u | (0x1u << 2);
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(mat)), 1u);
}

TEST(UvVariantTest, ShortPayloadFallsBackToUv0) {
    DefaultMeshTechnique tech;
    const std::array<std::byte, 4> tiny{};
    EXPECT_EQ(tech.InterfaceVariantForMaterial(tiny), 0u);
}

TEST(UvVariantTest, BaseTechniqueHasNoInterfaceVariants) {
    class Plain final : public VulkanEngine::TechniqueManager::BaseTechnique {};
    Plain plain;
    DefaultMeshPerMaterialData mat{};
    mat.uv_sets = 0x2u;
    EXPECT_EQ(plain.InterfaceVariantForMaterial(AsBytes(mat)), 0u);
    EXPECT_EQ(plain.VariantCount(), 1u);
}

// ── Variant row lookup ──

TEST(UvVariantTest, VariantRowsMapSlotsInOrder) {
    const std::array<VariantShaderPair, 2> rows = {{
        {ShaderId{7}, ShaderId{8}},
        {ShaderId{9}, ShaderId{10}},
    }};
    ShaderId v{};
    ShaderId f{};
    ASSERT_TRUE(GetVariantShaderRow(rows, 0, v, f));
    EXPECT_EQ(v, 7u);
    EXPECT_EQ(f, 8u);
    ASSERT_TRUE(GetVariantShaderRow(rows, 1, v, f));
    EXPECT_EQ(v, 9u);
    EXPECT_EQ(f, 10u);
    EXPECT_FALSE(GetVariantShaderRow(rows, 2, v, f));
}

// ── Composite draw-key packing ──

TEST(UvVariantTest, DrawKeyPacksVariantAndRenderState) {
    const std::uint32_t key = DrawKeyState::PackDrawKey(1u, 0x2u);
    EXPECT_EQ(DrawKeyState::VariantOf(key), 1u);
    EXPECT_EQ(DrawKeyState::RenderStateOf(key), 0x2u);
}

TEST(UvVariantTest, BaseDrawKeyIsZero) {
    EXPECT_EQ(DrawKeyState::PackDrawKey(0u, 0u), 0u);
}

TEST(UvVariantTest, RenderStateRoundTripsThroughCompositeKey) {
    // Transparent blend mode = 2 in the render-state byte; variant 1 must not
    // disturb it.
    const std::uint32_t render_state = 2u << DrawKeyState::BLEND_SHIFT;
    const std::uint32_t key = DrawKeyState::PackDrawKey(1u, render_state);
    EXPECT_EQ(DrawKeyState::RenderStateOf(key), render_state);
    EXPECT_TRUE(DrawKeyState::BlendEnabled(DrawKeyState::RenderStateOf(key)));
    EXPECT_EQ(DrawKeyState::VariantOf(key), 1u);
}

TEST(UvVariantTest, MeshWithoutExtraUvsNeedsNoVariant) {
    VulkanEngine::GpuResources::MeshData md{};
    md.vertices.resize(4);
    EXPECT_EQ(md.HighestOutOfLineSet(), 0u);
    EXPECT_FALSE(md.HasOutOfLineUvs());
}

TEST(UvVariantTest, MeshWithUv1ReportsSetOne) {
    VulkanEngine::GpuResources::MeshData md{};
    md.vertices.resize(4);
    md.uv_extra[0].resize(4);
    EXPECT_EQ(md.HighestOutOfLineSet(), 1u);
    EXPECT_TRUE(md.HasOutOfLineUvs());
}

TEST(UvVariantTest, VariantAndRenderFieldsDoNotOverlap) {
    // The interface variant is stored above the render-state bits, so a
    // transparent (blend) material that also uses UV1 keeps both.
    const std::uint32_t render = 2u << DrawKeyState::BLEND_SHIFT;
    const std::uint32_t key = DrawKeyState::PackDrawKey(1u, render);
    EXPECT_EQ(DrawKeyState::VariantOf(key), 1u);
    EXPECT_EQ(DrawKeyState::RenderStateOf(key), render);
    EXPECT_TRUE(DrawKeyState::BlendEnabled(DrawKeyState::RenderStateOf(key)));
}

}  // namespace
