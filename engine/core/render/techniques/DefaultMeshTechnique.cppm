module;

#include <cassert>
#include <cstddef>
#include <cstring>

export module VulkanEngine.TechniqueManager.DefaultMeshTechnique;

import std;

import vulkan;

import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.GpuResources.BlockArray;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;

export namespace VulkanEngine::TechniqueManager {

// Final material contract: C layout, std430-safe, 64 bytes. The tail is
// float4-packed for unambiguous layout. flags bits 0..1 carry the declared
// TextureNormalEncoding (Standard/Full/Bent); bit 2 alpha_mask; bit 3
// has_transform. The encoding is declared content, never derived from the
// resolved upload format. `standard_mesh.slang` mirrors this struct exactly —
// change both together. No unlit bit: unlit is a separate technique.
struct DefaultMeshPerMaterialData {
    std::uint32_t albedo_texture{0};
    std::uint32_t normal_texture{0};
    std::uint32_t orm_texture{0};       // occlusion(R) roughness(G) metallic(B)
    std::uint32_t emissive_texture{0};
    std::uint32_t flags{0};             // bits0..1 normal encoding; bit2 alpha_mask; bit3 has_transform
    std::uint32_t uv_sets{0};           // 2 bits per texture slot (albedo/normal/orm/emissive)
    float     roughness_factor{1.0f};
    float     metallic_factor{0.0f};
    float     ao_factor{1.0f};
    float     normal_scale{1.0f};
    float     occlusion_strength{1.0f};
    float     alpha_cutoff{0.5f};
    float     emissive_factor[4]{0.0f, 0.0f, 0.0f, 1.0f}; // rgb + strength
};
static_assert(sizeof(DefaultMeshPerMaterialData) == 64);
static_assert(std::is_trivially_copyable_v<DefaultMeshPerMaterialData>);

constexpr std::uint32_t kNormalEncodingMask = 0x3u;
constexpr std::uint32_t kAlphaMaskFlag = 1u << 2;
constexpr std::uint32_t kHasTransformFlag = 1u << 3;

// Packs the declared TextureNormalEncoding into the material's flags bits 0..1.
// The encoding is content the author/importer declares; the shader reads it to
// pick the decode path. The enum values are contiguous from zero, so this is a
// plain masked cast. Use it wherever a material is authored, so a writer can
// never set an out-of-range encoding into the reserved bits.
[[nodiscard]] constexpr std::uint32_t PackNormalEncoding(
    const VulkanEngine::Textures::TextureNormalEncoding encoding) {
    return static_cast<std::uint32_t>(encoding) & kNormalEncodingMask;
}

// Builds the lit technique's per-material payload from an imported glTF
// material. Texture slots are the caller's to resolve (bindless slots differ
// per upload), so `albedo_texture`/`normal_texture`/`orm_texture`/
// `emissive_texture` are left 0 and filled by the caller after it uploads the
// images. Everything the material declares — factors, encoding, UV selection,
// the transform gate and the MASK cutoff — is mapped here so the shader sees a
// consistent contract. The render state (blend/cull) is a MaterialDesc the
// caller derives from alpha_mode/double_sided, not part of this payload.
[[nodiscard]] inline DefaultMeshPerMaterialData FromGltfMaterial(
    const VulkanEngine::FileLoaders::Mesh::GltfMaterialInfo& material) {
    using VulkanEngine::FileLoaders::Mesh::GltfAlphaMode;
    using VulkanEngine::FileLoaders::Mesh::GltfMaterialHasTransform;
    using VulkanEngine::FileLoaders::Mesh::ComputeGltfUvSets;

    DefaultMeshPerMaterialData data{};
    data.flags = PackNormalEncoding(material.normal_encoding);
    if (material.alpha_mode == GltfAlphaMode::Mask) {
        data.flags |= kAlphaMaskFlag;
    }
    if (GltfMaterialHasTransform(material)) {
        data.flags |= kHasTransformFlag;
    }
    data.uv_sets = ComputeGltfUvSets(material);
    data.roughness_factor = material.roughness_factor;
    data.metallic_factor = material.metallic_factor;
    data.ao_factor = material.ao_factor;
    data.normal_scale = material.normal_scale;
    data.occlusion_strength = material.occlusion_strength;
    data.alpha_cutoff = material.alpha_cutoff;
    data.emissive_factor[0] = material.emissive_factor[0];
    data.emissive_factor[1] = material.emissive_factor[1];
    data.emissive_factor[2] = material.emissive_factor[2];
    data.emissive_factor[3] = material.emissive_strength;
    return data;
}

class DefaultMeshTechnique final : public BaseTechnique {
public:
    DefaultMeshTechnique() {
        // Set 5 = first technique custom set (engine sets 0-4 are reserved).
        // Binding 0 = first (only) PerMaterial binding for this technique.
        DeclarePerMaterial<DefaultMeshPerMaterialData>(5, 0);
    }

    // ── MaterialHandle<Tech> requires these static helpers ──

    template<typename T>
    static constexpr bool HasBinding() {
        return std::is_same_v<T, DefaultMeshPerMaterialData>;
    }

    template<typename T>
    static constexpr std::size_t GetOffset() {
        static_assert(std::is_same_v<T, DefaultMeshPerMaterialData>,
                      "DefaultMeshTechnique only has DefaultMeshPerMaterialData");
        return 0;  // only one PerMaterial type, at offset 0 in cpu_data
    }

    template<typename T>
    static constexpr std::uint32_t GetBindingIndex() {
        static_assert(std::is_same_v<T, DefaultMeshPerMaterialData>,
                      "DefaultMeshTechnique only has DefaultMeshPerMaterialData");
        return 0;  // first (only) PerMaterial binding
    }

    // Compile the default mesh pipeline with the given SPIR-V and engine layouts.
    // Returns false if pipeline creation failed.
    [[nodiscard]] bool CompileDefaultMesh(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                            ShaderSystem::ShaderManager& shader_mgr,
                            ShaderSystem::PipelineFactory& pipeline_factory,
                            ShaderSystem::ShaderId vert_id,
                            ShaderSystem::ShaderId frag_id,
                            const VulkanEngine::StandardMeshPipeline::PipelineConfig& config,
                            vk::DescriptorSetLayout bindless_layout,
                            vk::DescriptorSetLayout submesh_vertex_layout,
                            vk::DescriptorSetLayout raw_vertex_layout,
                            vk::DescriptorSetLayout indirection_layout,
                            vk::DescriptorSetLayout scene_uniform_layout = nullptr) {
        return Compile(bootstrap, shader_mgr, pipeline_factory, vert_id, frag_id, config,
                       bindless_layout, submesh_vertex_layout, raw_vertex_layout, indirection_layout,
                       scene_uniform_layout);
    }

    // PackMaterialData uses the BaseTechnique default (TechniquePacking::Pack).

    // Interface-variant slot = uv bit + mask bit, the material's required
    // entry-point pair (slots 0-3 row-indexed in GameEngine):
    //   uv bit: the material samples an out-of-line UV set when any texture
    //     slot's uv_sets selector is exactly 1 (mirroring SlotUV's `== 1`
    //     semantics — selectors above 1 read inline UV0 today, so they must
    //     not select the UV1 pipeline).
    //   mask bit: flags bit 2 (kAlphaMaskFlag); a set flag selects the
    //     alpha-masked entry-point pair whose fragment discards at the
    //     material's alpha_cutoff.
    // This is what keeps the material's pipeline consistent with the UV sets
    // its textures read and with the MASK render contract.
    [[nodiscard]] std::uint32_t InterfaceVariantForMaterial(
        std::span<const std::byte> cpu_data) const override {
        if (cpu_data.size() < sizeof(DefaultMeshPerMaterialData)) return 0;
        DefaultMeshPerMaterialData data{};
        std::memcpy(&data, cpu_data.data(), sizeof(data));
        std::uint32_t uv = 0;
        for (std::uint32_t slot = 0; slot < 4; ++slot) {
            const std::uint32_t set = (data.uv_sets >> (slot * 2u)) & 0x3u;
            if (set == 1u) {
                uv = 1u;
                break;
            }
        }
        const std::uint32_t slot =
            uv | ((data.flags & kAlphaMaskFlag) != 0 ? 2u : 0u);
        assert(slot < 4u && "interface variants are sized for uv bit + mask bit");
        return slot;
    }

    [[nodiscard]] VulkanEngine::GpuResources::BlockArray* GetMaterialBlockArray() {
        return GetBlockArrayForType<DefaultMeshPerMaterialData>();
    }

    // The four texture-slot words the residency layer rewrites on eviction:
    // albedo, normal, orm, emissive. Offsets are computed from the struct so a
    // layout change cannot silently desynchronize them.
    [[nodiscard]] std::span<const std::uint32_t> TextureSlotFieldOffsets() const override {
        static const std::array<std::uint32_t, 4> offsets{
            static_cast<std::uint32_t>(offsetof(DefaultMeshPerMaterialData, albedo_texture)),
            static_cast<std::uint32_t>(offsetof(DefaultMeshPerMaterialData, normal_texture)),
            static_cast<std::uint32_t>(offsetof(DefaultMeshPerMaterialData, orm_texture)),
            static_cast<std::uint32_t>(offsetof(DefaultMeshPerMaterialData, emissive_texture)),
        };
        return offsets;
    }
};

} // namespace VulkanEngine::TechniqueManager
