#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
import VulkanEngine.TechniqueManager.UnlitTextureTechnique;
import VulkanEngine.ShaderManager;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;
import VulkanEngine.StandardMeshPipeline;
import VulkanEngine.MaterialManager;
import VulkanEngine.TechniqueManager;

namespace {

using VulkanEngine::TechniqueManager::DefaultMeshPerMaterialData;
using VulkanEngine::TechniqueManager::DefaultMeshTechnique;
using VulkanEngine::TechniqueManager::GetVariantShaderRow;
using VulkanEngine::TechniqueManager::VariantShaderPair;
namespace DrawKeyState = VulkanEngine::TechniqueManager::DrawKeyState;
namespace TechniquePacking = VulkanEngine::TechniqueManager::TechniquePacking;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::FileLoaders::Mesh::ComputeGltfUvSets;
using VulkanEngine::FileLoaders::Mesh::GltfMaterialInfo;
using VulkanEngine::FileLoaders::Mesh::GltfTextureSlot;

std::span<const std::byte> AsBytes(const DefaultMeshPerMaterialData& data) {
    return {reinterpret_cast<const std::byte*>(&data), sizeof(data)};
}

// Builds a parsed glTF material whose present slots carry `tex_coords[slot]`
// (slot order 0=baseColor, 1=normal, 2=metallicRoughness, 3=emissive) and
// encodes it with the engine's importer, so the payload's uv_sets is the
// encoder's output rather than a hand-packed bit pattern.
[[nodiscard]] DefaultMeshPerMaterialData MaterialSelectingUvSets(
    const std::array<std::uint32_t, 4>& tex_coords) {
    GltfMaterialInfo gltf{};
    for (std::size_t slot = 0; slot < tex_coords.size(); ++slot) {
        GltfTextureSlot& tex = slot == 0 ? gltf.base_color_texture
                             : slot == 1 ? gltf.normal_texture
                             : slot == 2 ? gltf.metallic_roughness_texture
                                         : gltf.emissive_texture;
        tex.present = true;
        tex.tex_coord = tex_coords[slot];
    }
    DefaultMeshPerMaterialData mat{};
    mat.uv_sets = ComputeGltfUvSets(gltf);
    return mat;
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
    EXPECT_EQ(tech.InterfaceVariantForMaterial(
                  AsBytes(MaterialSelectingUvSets({1u, 0u, 0u, 0u}))), 1u);

    // Slot 2 (ORM) selects set 1.
    EXPECT_EQ(tech.InterfaceVariantForMaterial(
                  AsBytes(MaterialSelectingUvSets({0u, 0u, 1u, 0u}))), 1u);
}

TEST(UvVariantTest, HighestSelectedSetWins) {
    DefaultMeshTechnique tech;
    DefaultMeshPerMaterialData mat = MaterialSelectingUvSets({1u, 1u, 0u, 0u});
    // albedo -> set 1, normal -> set 1.
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(mat)), 1u);
}

// Selectors above 1 do NOT select the UV1 pipeline: SlotUV reads any selector
// != 1 as inline UV0, so the variant quantizes to match (TEXCOORD_2+ set
// selectors degrade to UV0 sampling until an out-of-line third UV set exists;
// the old highest-wins mapping only ever "worked" by falling back through the
// row-count check to the base UV0 pipeline anyway).
TEST(UvVariantTest, SetSelectorsAboveOneDegradeToUv0) {
    DefaultMeshTechnique tech;
    // A slot selecting set 2 does not select UV1.
    DefaultMeshPerMaterialData emissive_high = MaterialSelectingUvSets({1u, 0u, 0u, 2u});
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(emissive_high)), 1u);
    DefaultMeshPerMaterialData only_high = MaterialSelectingUvSets({0u, 0u, 0u, 2u});
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(only_high)), 0u);
}

// The mask bit composes with the uv bit: flags bit 2 selects the MASK
// entry-point pairs (slots 2/3), independently of the UV sets the payload
// selects.
TEST(UvVariantTest, MaskFlagSelectsTheMaskPairs) {
    DefaultMeshTechnique tech;
    DefaultMeshPerMaterialData base_mask = MaterialSelectingUvSets({0u, 0u, 0u, 0u});
    base_mask.flags = VulkanEngine::TechniqueManager::kAlphaMaskFlag;
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(base_mask)), 2u);

    DefaultMeshPerMaterialData uv1_mask = MaterialSelectingUvSets({0u, 1u, 0u, 0u});
    uv1_mask.flags = VulkanEngine::TechniqueManager::kAlphaMaskFlag;
    EXPECT_EQ(tech.InterfaceVariantForMaterial(AsBytes(uv1_mask)), 3u);
}

// A texCoord of 1 in any slot selects the out-of-line UV1 sets; the old
// highest-wins semantics did the same numerically, but a later mask bit needs
// a guaranteed {0,1} uv field to compose with.
TEST(UvVariantTest, UvQuantizationMirrorsSlotUv) {
    DefaultMeshTechnique tech;
    EXPECT_EQ(tech.InterfaceVariantForMaterial(
                  AsBytes(MaterialSelectingUvSets({2u, 3u, 1u, 2u}))), 1u);
}

TEST(UvVariantTest, ShortPayloadFallsBackToUv0) {
    DefaultMeshTechnique tech;
    const std::array<std::byte, 4> tiny{};
    EXPECT_EQ(tech.InterfaceVariantForMaterial(tiny), 0u);
}

// ── AlphaMaskSupport contract ──

TEST(UvVariantTest, AlphaMaskExpectationFollowsThePayloadFlag) {
    DefaultMeshTechnique tech;
    DefaultMeshPerMaterialData plain{};
    // Without the flag the material's pair never discards.
    EXPECT_EQ(tech.AlphaMaskExpectation(AsBytes(plain)),
              VulkanEngine::TechniqueManager::AlphaMaskSupport::Never);
    plain.flags = VulkanEngine::TechniqueManager::kAlphaMaskFlag;
    // With it, discard is payload-driven — the render-state bit must agree.
    EXPECT_EQ(tech.AlphaMaskExpectation(AsBytes(plain)),
              VulkanEngine::TechniqueManager::AlphaMaskSupport::Conditional);
    // A short payload cannot carry the flag.
    const std::array<std::byte, 4> tiny{};
    const std::span<const std::byte> tiny_bytes(tiny);
    EXPECT_EQ(tech.AlphaMaskExpectation(tiny_bytes),
              VulkanEngine::TechniqueManager::AlphaMaskSupport::Never);
}

TEST(UvVariantTest, UnlitAlwaysHonorsAlphaCutoff) {
    VulkanEngine::TechniqueManager::UnlitTextureTechnique unlit;
    const std::array<std::byte, 24> payload{};
    const std::span<const std::byte> payload_bytes(payload);
    EXPECT_EQ(unlit.AlphaMaskExpectation(payload_bytes),
              VulkanEngine::TechniqueManager::AlphaMaskSupport::Always);
}

// Registration cross-check: a Cutout material whose payload's entry points
// never discard is rejected instead of silently rendering fully opaque; the
// same payload with the mask flag registers, and an Opaque registration of a
// mask payload is accepted (payload-driven discard renders correctly, just
// ordered as opaque).
TEST(UvVariantTest, CutoutWithoutADiscardCapablePayloadIsRejected) {
    VulkanEngine::TechniqueManager::TechniqueManager tech_mgr{};
    tech_mgr.Register(std::make_unique<DefaultMeshTechnique>());
    VulkanEngine::MaterialManager::MaterialManager materials{};
    materials.SetTechniqueManager(&tech_mgr);
    materials.Initialize(nullptr);

    DefaultMeshPerMaterialData plain{};
    auto rejected = materials.Register<DefaultMeshTechnique>(
        VulkanEngine::MaterialManager::MaterialDesc{
            .blend = VulkanEngine::MaterialManager::BlendMode::Cutout},
        plain);
    EXPECT_FALSE(rejected.Valid());

    DefaultMeshPerMaterialData masked{};
    masked.flags = VulkanEngine::TechniqueManager::kAlphaMaskFlag;
    auto valid = materials.Register<DefaultMeshTechnique>(
        VulkanEngine::MaterialManager::MaterialDesc{
            .blend = VulkanEngine::MaterialManager::BlendMode::Cutout},
        masked);
    EXPECT_TRUE(valid.Valid());

    auto warning_only = materials.Register<DefaultMeshTechnique>(
        VulkanEngine::MaterialManager::MaterialDesc{
            .blend = VulkanEngine::MaterialManager::BlendMode::Opaque},
        masked);
    EXPECT_TRUE(warning_only.Valid());

    materials.Shutdown();
    tech_mgr.Shutdown();
}

TEST(UvVariantTest, BaseTechniqueHasNoInterfaceVariants) {
    class Plain final : public VulkanEngine::TechniqueManager::BaseTechnique {};
    Plain plain;
    DefaultMeshPerMaterialData mat = MaterialSelectingUvSets({2u, 0u, 0u, 0u});
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
    // Transparent blend mode, derived from the shared state constants; variant
    // 1 must not disturb it.
    const std::uint32_t render_state =
        DrawKeyState::BLEND_TRANSPARENT << DrawKeyState::BLEND_SHIFT;
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
    const std::uint32_t render =
        DrawKeyState::BLEND_TRANSPARENT << DrawKeyState::BLEND_SHIFT;
    const std::uint32_t key = DrawKeyState::PackDrawKey(1u, render);
    EXPECT_EQ(DrawKeyState::VariantOf(key), 1u);
    EXPECT_EQ(DrawKeyState::RenderStateOf(key), render);
    EXPECT_TRUE(DrawKeyState::BlendEnabled(DrawKeyState::RenderStateOf(key)));
}

}  // namespace
