module;

#include <cstddef>

export module VulkanEngine.TechniqueManager.UnlitTextureTechnique;

import std;

import vulkan;

import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
import VulkanEngine.GpuResources.BlockArray;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

export namespace VulkanEngine::TechniqueManager {

// Unlit technique contract: only what unlit consumes. C layout,
// std430-safe, 24 bytes. `unlit.slang` must be changed in lockstep.
struct UnlitPerMaterialData {
    std::uint32_t albedo_texture{0};
    float albedo_factor[4]{1.0f, 1.0f, 1.0f, 1.0f}; // glTF baseColorFactor
    float alpha_cutoff{0.5f};                       // MASK discard
};

// UnlitTextureTechnique — renders meshes without lighting by sampling the
// per-material albedo texture directly. It has its own trimmed material
// contract (UnlitPerMaterialData at set 5, binding 0), separate from the lit
// technique's DefaultMeshPerMaterialData, so the TechniqueManager compiles it
// into a separate pipeline (paired with the unlit fragment shader).
class UnlitTextureTechnique final : public BaseTechnique {
public:
    UnlitTextureTechnique() {
        DeclarePerMaterial<UnlitPerMaterialData>(5, 0);
    }

    // ── MaterialHandle<Tech> requires these static helpers ──

    template<typename T>
    static constexpr bool HasBinding() {
        return std::is_same_v<T, UnlitPerMaterialData>;
    }

    template<typename T>
    static constexpr std::size_t GetOffset() {
        static_assert(std::is_same_v<T, UnlitPerMaterialData>,
                      "UnlitTextureTechnique only has UnlitPerMaterialData");
        return 0;  // only one PerMaterial type, at offset 0 in cpu_data
    }

    template<typename T>
    static constexpr std::uint32_t GetBindingIndex() {
        static_assert(std::is_same_v<T, UnlitPerMaterialData>,
                      "UnlitTextureTechnique only has UnlitPerMaterialData");
        return 0;  // first (only) PerMaterial binding
    }

    // Compile with an arbitrary vertex shader (typically the engine's
    // main_indir vertex shader) and an unlit fragment shader.
    // Returns false if pipeline creation failed.
    [[nodiscard]] bool CompileUnlit(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
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

    // The single entry point discards below alpha_cutoff unconditionally
    // (unlit.slang), so the MASK contract holds under any blend mode — an
    // opaque unlit material simply never samples below the cutoff.
    [[nodiscard]] AlphaMaskSupport AlphaMaskExpectation(
        std::span<const std::byte> /*cpu_data*/) const override {
        return AlphaMaskSupport::Always;
    }

    // PackMaterialData uses the BaseTechnique default (TechniquePacking::Pack).

    [[nodiscard]] VulkanEngine::GpuResources::BlockArray* GetMaterialBlockArray() {
        return GetBlockArrayForType<UnlitPerMaterialData>();
    }

    // The single albedo texture-slot word the residency layer rewrites on
    // eviction; computed from the struct so the layout stays authoritative.
    [[nodiscard]] std::span<const std::uint32_t> TextureSlotFieldOffsets() const override {
        static const std::array<std::uint32_t, 1> offsets{
            static_cast<std::uint32_t>(offsetof(UnlitPerMaterialData, albedo_texture)),
        };
        return offsets;
    }
};

} // namespace VulkanEngine::TechniqueManager
