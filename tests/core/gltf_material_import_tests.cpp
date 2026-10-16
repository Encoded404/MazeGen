#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

import std;

import vulkan;

import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;
import VulkanEngine.MaterialManager;
import VulkanEngine.MaterialManager.MaterialId;
import VulkanEngine.TechniqueManager;
import VulkanEngine.TechniqueManager.DefaultMeshTechnique;

namespace {

using VulkanEngine::FileLoaders::Mesh::ComputeGltfUvSets;
using VulkanEngine::FileLoaders::Mesh::GltfAlphaMode;
using VulkanEngine::FileLoaders::Mesh::GltfFilter;
using VulkanEngine::FileLoaders::Mesh::GltfImportResult;
using VulkanEngine::FileLoaders::Mesh::GltfMaterialHasTransform;
using VulkanEngine::FileLoaders::Mesh::GltfMaterialInfo;
using VulkanEngine::FileLoaders::Mesh::GltfTextureSlot;
using VulkanEngine::FileLoaders::Mesh::GltfWrap;
using VulkanEngine::FileLoaders::Mesh::ImportGltfMaterials;
using VulkanEngine::FileLoaders::Mesh::kNoGltfSampler;
using VulkanEngine::FileLoaders::Mesh::MapGltfMaterialIndex;
using VulkanEngine::FileLoaders::Mesh::ToSamplerDesc;
using VulkanEngine::MaterialManager::MaterialId;
using VulkanEngine::Textures::TextureNormalEncoding;

// 1x1 PNG, so fastgltf can resolve the image data URI without any real decode.
constexpr const char* kPngDataUri =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

[[nodiscard]] std::string MakeGltfJson() {
    return std::string(R"({
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_texture_transform", "KHR_materials_unlit",
                     "KHR_materials_emissive_strength"],
  "images": [{"uri": ")") + kPngDataUri + R"("}],
  "samplers": [
    {"magFilter": 9728, "minFilter": 9987, "wrapS": 33071, "wrapT": 33648}
  ],
  "textures": [
    {"source": 0, "sampler": 0}
  ],
  "materials": [
    {
      "name": "LitMat",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.5, 0.25, 0.125, 1.0],
        "baseColorTexture": {"index": 0, "texCoord": 1},
        "metallicFactor": 0.3,
        "roughnessFactor": 0.7,
        "metallicRoughnessTexture": {"index": 0, "texCoord": 0}
      },
      "normalTexture": {
        "index": 0, "scale": 2.0, "texCoord": 0,
        "extensions": {"KHR_texture_transform": {"offset": [0.1, 0.2], "scale": [2.0, 3.0]}}
      },
      "occlusionTexture": {"index": 0, "strength": 0.4, "texCoord": 2},
      "emissiveTexture": {"index": 0, "texCoord": 1},
      "emissiveFactor": [0.9, 0.8, 0.7],
      "alphaMode": "MASK",
      "alphaCutoff": 0.3,
      "doubleSided": true
    },
    {
      "name": "UnlitMat",
      "extensions": {
        "KHR_materials_unlit": {},
        "KHR_materials_emissive_strength": {"emissiveStrength": 2.0}
      }
    }
  ]
})";
}

class GltfMaterialImportTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto dir = std::filesystem::temp_directory_path() / "vkengine_gltf_import_tests";
        std::filesystem::create_directories(dir);
        path_ = dir / "fixture.gltf";
        std::ofstream file(path_, std::ios::binary | std::ios::trunc);
        const std::string json = MakeGltfJson();
        file.write(json.data(), static_cast<std::streamsize>(json.size()));
        file.close();
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    std::filesystem::path path_{};
};

TEST_F(GltfMaterialImportTest, ParsesMaterialsTexturesAndSamplers) {
    GltfImportResult result;
    ASSERT_NO_THROW(result = ImportGltfMaterials(path_));

    ASSERT_EQ(result.materials.size(), 2u);
    ASSERT_EQ(result.textures.size(), 1u);
    ASSERT_EQ(result.images.size(), 1u);
    ASSERT_EQ(result.samplers.size(), 1u);

    // Texture resolves to image 0 and sampler 0.
    EXPECT_TRUE(result.textures[0].valid);
    EXPECT_EQ(result.textures[0].image_index, 0u);
    EXPECT_EQ(result.textures[0].sampler_index, 0u);
    EXPECT_FALSE(result.textures[0].basisu);

    // A data URI is inlined by LoadExternalImages into the embedded payload.
    EXPECT_FALSE(result.images[0].embedded_bytes.empty());
}

TEST_F(GltfMaterialImportTest, ParsesLitMaterialFields) {
    const GltfImportResult result = ImportGltfMaterials(path_);
    const GltfMaterialInfo& lit = result.materials[0];

    EXPECT_EQ(lit.name, "LitMat");
    EXPECT_FALSE(lit.unlit);
    EXPECT_EQ(lit.alpha_mode, GltfAlphaMode::Mask);
    EXPECT_FLOAT_EQ(lit.alpha_cutoff, 0.3f);
    EXPECT_TRUE(lit.double_sided);

    EXPECT_FLOAT_EQ(lit.base_color_factor[0], 0.5f);
    EXPECT_FLOAT_EQ(lit.base_color_factor[1], 0.25f);
    EXPECT_FLOAT_EQ(lit.base_color_factor[2], 0.125f);
    EXPECT_FLOAT_EQ(lit.base_color_factor[3], 1.0f);
    EXPECT_FLOAT_EQ(lit.metallic_factor, 0.3f);
    EXPECT_FLOAT_EQ(lit.roughness_factor, 0.7f);
    EXPECT_FLOAT_EQ(lit.emissive_factor[0], 0.9f);
    EXPECT_FLOAT_EQ(lit.emissive_factor[1], 0.8f);
    EXPECT_FLOAT_EQ(lit.emissive_factor[2], 0.7f);
    EXPECT_FLOAT_EQ(lit.normal_scale, 2.0f);
    EXPECT_FLOAT_EQ(lit.occlusion_strength, 0.4f);

    ASSERT_TRUE(lit.base_color_texture.present);
    EXPECT_EQ(lit.base_color_texture.tex_coord, 1u);
    ASSERT_TRUE(lit.normal_texture.present);
    EXPECT_EQ(lit.normal_texture.tex_coord, 0u);
    ASSERT_TRUE(lit.metallic_roughness_texture.present);
    ASSERT_TRUE(lit.occlusion_texture.present);
    ASSERT_TRUE(lit.emissive_texture.present);
    EXPECT_EQ(lit.emissive_texture.tex_coord, 1u);

    // A normal map is authored RGB, so the declared encoding is Full.
    EXPECT_EQ(lit.normal_encoding, TextureNormalEncoding::Full);
}

TEST_F(GltfMaterialImportTest, ParsesTextureTransform) {
    const GltfImportResult result = ImportGltfMaterials(path_);
    const GltfMaterialInfo& lit = result.materials[0];

    ASSERT_TRUE(lit.normal_texture.transform.present);
    EXPECT_FLOAT_EQ(lit.normal_texture.transform.offset_u, 0.1f);
    EXPECT_FLOAT_EQ(lit.normal_texture.transform.offset_v, 0.2f);
    EXPECT_FLOAT_EQ(lit.normal_texture.transform.scale_u, 2.0f);
    EXPECT_FLOAT_EQ(lit.normal_texture.transform.scale_v, 3.0f);
    EXPECT_TRUE(GltfMaterialHasTransform(lit));

    // The unlit material carries no transform.
    EXPECT_FALSE(GltfMaterialHasTransform(result.materials[1]));
}

TEST_F(GltfMaterialImportTest, ParsesUnlitAndEmissiveStrength) {
    const GltfImportResult result = ImportGltfMaterials(path_);
    const GltfMaterialInfo& unlit = result.materials[1];

    EXPECT_EQ(unlit.name, "UnlitMat");
    EXPECT_TRUE(unlit.unlit);
    EXPECT_FLOAT_EQ(unlit.emissive_strength, 2.0f);
    EXPECT_EQ(unlit.normal_encoding, TextureNormalEncoding::Standard);
}

TEST_F(GltfMaterialImportTest, UvSetsFollowTheSlotTexCoords) {
    const GltfImportResult result = ImportGltfMaterials(path_);

    // The texCoord the author declares per slot must resolve the technique's
    // interface variant end to end: the fixture's lit material references
    // baseColor/emissive at TEXCOORD_1 (everything else at TEXCOORD_0) and is
    // authored alphaMode MASK, so it selects the mask+uv1 pair (slot 3) — the
    // pipeline whose fragment honors alphaCutoff, not the discard-free uv1
    // shader the pre-mask mapping silently picked.
    const auto variant = [](const GltfMaterialInfo& material) {
        const auto data = VulkanEngine::TechniqueManager::FromGltfMaterial(material);
        VulkanEngine::TechniqueManager::DefaultMeshTechnique tech;
        return tech.InterfaceVariantForMaterial(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(&data),
                                       sizeof(data)));
    };
    EXPECT_EQ(variant(result.materials[0]), 3u);

    // Authoring TEXCOORD_1 on the normal slot alone (opaque material) must
    // still select UV1.
    GltfMaterialInfo normal_only{};
    normal_only.normal_texture.present = true;
    normal_only.normal_texture.tex_coord = 1u;
    EXPECT_EQ(variant(normal_only), 1u);

    // Selectors above 1 do not select UV1: SlotUV reads any selector != 1 as
    // inline UV0, so an occlusion-only ORM slot at TEXCOORD_2 (no shared
    // metallicRoughness texture) stays on the base pipeline until an
    // out-of-line third UV set exists.
    GltfMaterialInfo occlusion_high{};
    occlusion_high.base_color_texture.present = true;
    occlusion_high.occlusion_texture.present = true;
    occlusion_high.occlusion_texture.tex_coord = 2u;
    EXPECT_EQ(variant(occlusion_high), 0u);
}

TEST_F(GltfMaterialImportTest, SamplerMapsToDeviceDescription) {
    const GltfImportResult result = ImportGltfMaterials(path_);
    const auto desc = ToSamplerDesc(result.samplers[0]);

    EXPECT_EQ(desc.mag_filter, vk::Filter::eNearest);
    // minFilter 9987 = LINEAR_MIPMAP_LINEAR -> linear filter, linear mip mode.
    EXPECT_EQ(desc.min_filter, vk::Filter::eLinear);
    EXPECT_EQ(desc.mip_mode, vk::SamplerMipmapMode::eLinear);
    EXPECT_EQ(desc.address_u, vk::SamplerAddressMode::eClampToEdge);
    EXPECT_EQ(desc.address_v, vk::SamplerAddressMode::eMirroredRepeat);
}

TEST_F(GltfMaterialImportTest, AbsentSamplerUsesGltfDefaults) {
    VulkanEngine::FileLoaders::Mesh::GltfSamplerInfo default_sampler{};
    const auto desc = ToSamplerDesc(default_sampler);
    EXPECT_EQ(desc.min_filter, vk::Filter::eLinear);
    EXPECT_EQ(desc.mag_filter, vk::Filter::eLinear);
    EXPECT_EQ(desc.mip_mode, vk::SamplerMipmapMode::eLinear);
    EXPECT_EQ(desc.address_u, vk::SamplerAddressMode::eRepeat);
    EXPECT_EQ(desc.address_v, vk::SamplerAddressMode::eRepeat);
}

TEST_F(GltfMaterialImportTest, TextureWithNoSamplerReportsNoIndex) {
    // The fixture's texture names sampler 0; a texture without one must report
    // kNoGltfSampler so the importer default applies rather than index 0.
    GltfImportResult result = ImportGltfMaterials(path_);
    ASSERT_EQ(result.textures.size(), 1u);
    EXPECT_NE(result.textures[0].sampler_index, kNoGltfSampler);
}

TEST_F(GltfMaterialImportTest, MapGltfMaterialIndexUsesTheBindingTable) {
    const std::vector<MaterialId> bindings{MaterialId{7}, MaterialId{9}};
    EXPECT_EQ(MapGltfMaterialIndex(0, &bindings).value, 7u);
    EXPECT_EQ(MapGltfMaterialIndex(1, &bindings).value, 9u);
    // Out of range and no table both fall back to the default material.
    EXPECT_EQ(MapGltfMaterialIndex(5, &bindings).value, 0u);
    EXPECT_EQ(MapGltfMaterialIndex(0, nullptr).value, 0u);
}

TEST_F(GltfMaterialImportTest, RejectsMissingFile) {
    EXPECT_THROW((void)ImportGltfMaterials(path_.parent_path() / "does_not_exist.gltf"),
                 std::runtime_error);
}

// End-to-end binding: the parsed material drives a real technique's per-material
// payload, its texCoord selection resolves the interface variant, and its
// render state resolves the draw group. This is the join between the importer
// and the material system, done device-free.
TEST_F(GltfMaterialImportTest, ParsedMaterialResolvesToADrawGroup) {
    const GltfImportResult result = ImportGltfMaterials(path_);
    const GltfMaterialInfo& src = result.materials[0];

    VulkanEngine::TechniqueManager::TechniqueManager tech_mgr{};
    tech_mgr.Register(std::make_unique<VulkanEngine::TechniqueManager::DefaultMeshTechnique>());
    VulkanEngine::MaterialManager::MaterialManager materials{};
    materials.SetTechniqueManager(&tech_mgr);
    materials.Initialize(nullptr);

    VulkanEngine::TechniqueManager::DefaultMeshPerMaterialData data =
        VulkanEngine::TechniqueManager::FromGltfMaterial(src);
    data.albedo_texture = 1; // slot 1 stands in for the resolved bindless slot
    data.emissive_texture = 2;

    // The converter carries the declared encoding, the MASK bit, the transform
    // gate, the per-slot UV selection, and the factors.
    EXPECT_EQ(data.flags & VulkanEngine::TechniqueManager::kNormalEncodingMask,
              static_cast<std::uint32_t>(VulkanEngine::Textures::TextureNormalEncoding::Full));
    EXPECT_NE(data.flags & VulkanEngine::TechniqueManager::kAlphaMaskFlag, 0u);
    EXPECT_NE(data.flags & VulkanEngine::TechniqueManager::kHasTransformFlag, 0u);
    EXPECT_EQ(data.uv_sets, ComputeGltfUvSets(src));
    EXPECT_FLOAT_EQ(data.normal_scale, 2.0f);
    EXPECT_FLOAT_EQ(data.roughness_factor, 0.7f);
    EXPECT_FLOAT_EQ(data.metallic_factor, 0.3f);
    EXPECT_FLOAT_EQ(data.alpha_cutoff, 0.3f);
    EXPECT_FLOAT_EQ(data.emissive_factor[0], 0.9f);

    const auto desc = VulkanEngine::MaterialManager::MaterialDesc{
        .blend = src.alpha_mode == GltfAlphaMode::Mask
                     ? VulkanEngine::MaterialManager::BlendMode::Cutout
                     : VulkanEngine::MaterialManager::BlendMode::Opaque,
        .double_sided = src.double_sided,
        .depth_write = true,
    };
    auto handle = materials.Register<VulkanEngine::TechniqueManager::DefaultMeshTechnique>(desc, data);
    ASSERT_TRUE(handle.Valid());

    // The material's selected UV set is 1 (baseColor/emissive) and it is
    // authored alphaMode MASK, so it must resolve to the mask+uv1 interface
    // variant (slot 3) and intern a correspondingly non-base draw group.
    const auto* entry = tech_mgr.GetTechnique(tech_mgr.GetId<VulkanEngine::TechniqueManager::DefaultMeshTechnique>());
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->InterfaceVariantForMaterial(
                  std::span<const std::byte>(reinterpret_cast<const std::byte*>(&data), sizeof(data))),
              3u);

    // A MASK double-sided material interns a non-zero draw group.
    const VulkanEngine::MaterialManager::MaterialId id{handle.Id()};
    EXPECT_NE(materials.GetGroupForMaterial(id), 0u);
    const auto* state = materials.GetRenderState(id);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->alpha_mask);
    EXPECT_TRUE(state->double_sided);

    materials.Shutdown();
    tech_mgr.Shutdown();
}

}  // namespace

