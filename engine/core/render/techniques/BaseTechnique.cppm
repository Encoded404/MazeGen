module;

#include <cassert>

export module VulkanEngine.TechniqueManager.BaseTechnique;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.TechniqueManager.TechniqueId;
export import VulkanEngine.GpuResources.BlockArray;
export import VulkanEngine.GpuBuffer;
export import VulkanEngine.GpuResources.StagingPool;
export import VulkanEngine.DescriptorDecl;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

export namespace VulkanEngine::TechniqueManager {

struct DrawKeyPipelineState {
    bool blend_enable{false};
    vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eBack};
    bool depth_write{true};
};

// Pure render-state → pipeline-state derivation for a draw group. Shared by
// pipeline-variant creation and unit-testable without a device: the blend/cull/
// depth bits of a key decide the variant's state. `base_cull_mode` is the
// technique's configured cull mode (a double-sided key overrides it to None).
[[nodiscard]] DrawKeyPipelineState DeriveDrawKeyPipelineState(
    std::uint32_t render_state_key, vk::CullModeFlags base_cull_mode);

// A technique that owns named entry-point rows maps a dense interface-variant
// slot to its (vert, frag) shader rows. Pure, so the slot mapping is
// unit-testable without a device. The slot layout is each technique's
// contract; the default mesh technique uses uv bit + alpha-mask bit (4 rows).
struct VariantShaderPair {
    ShaderSystem::ShaderId vert{};
    ShaderSystem::ShaderId frag{};
};

[[nodiscard]] inline bool GetVariantShaderRow(
    std::span<const VariantShaderPair> rows, std::uint32_t variant_slot,
    ShaderSystem::ShaderId& out_vert, ShaderSystem::ShaderId& out_frag) {
    if (variant_slot >= rows.size()) return false;
    out_vert = rows[variant_slot].vert;
    out_frag = rows[variant_slot].frag;
    return true;
}

// The 32-bit field is split into a technique id (low bits) and a material id
// (high bits). Change TECHNIQUE_BITS to redistribute the budget; keep
// kTechniqueBits in expand.slang in sync.
namespace TechniquePacking {
    inline constexpr std::uint32_t TECHNIQUE_BITS  = 14;                 // → 16384 techniques max
    inline constexpr std::uint32_t MATERIAL_BITS   = 32 - TECHNIQUE_BITS; // → 262144 materials max
    inline constexpr std::uint32_t TECHNIQUE_MASK  = (1u << TECHNIQUE_BITS) - 1;
    // Draw-key table capacity. The low TECHNIQUE_BITS of a packed
    // technique_material carry the draw key (a technique id, and from the
    // variant work a dense interned group id), so every representable key is
    // below this. Every per-key GPU table and the group registry are sized to
    // it, and allocation is capped here so a key can never index out of range.
    inline constexpr std::uint32_t MAX_DRAW_GROUPS = 1u << TECHNIQUE_BITS;
}

// ── Draw-key render-state bits ──
// A draw group key folds a material's render state (blend mode, cull, depth
// write) below a blend-mode byte. This layout is shared by MaterialManager
// (which packs it) and BaseTechnique (which turns it into pipeline state), so
// it lives on the technique layer that MaterialManager already depends on.
namespace DrawKeyState {
    inline constexpr std::uint32_t BLEND_SHIFT = 8;
    inline constexpr std::uint32_t BLEND_MASK  = 0x3u;
    inline constexpr std::uint32_t DOUBLE_SIDED        = 1u << 0;
    inline constexpr std::uint32_t DEPTH_WRITE_DISABLED = 1u << 1;
    inline constexpr std::uint32_t ALPHA_MASK          = 1u << 2;
    // BlendMode values (MaterialManager): Opaque=0, Cutout=1, Transparent=2.
    inline constexpr std::uint32_t BLEND_TRANSPARENT = 2u;
    inline constexpr std::uint32_t BLEND_OPAQUE = 0u;

    [[nodiscard]] constexpr std::uint32_t Blend(std::uint32_t key) {
        return (key >> BLEND_SHIFT) & BLEND_MASK;
    }
    [[nodiscard]] constexpr bool BlendEnabled(std::uint32_t key) {
        return Blend(key) == BLEND_TRANSPARENT;
    }
    [[nodiscard]] constexpr bool DoubleSided(std::uint32_t key) {
        return (key & DOUBLE_SIDED) != 0;
    }
    [[nodiscard]] constexpr bool DepthWriteEnabled(std::uint32_t key) {
        return (key & DEPTH_WRITE_DISABLED) == 0;
    }
    [[nodiscard]] constexpr bool AlphaMask(std::uint32_t key) {
        return (key & ALPHA_MASK) != 0;
    }

    // A draw key is a composite of an interface variant (the entry-point/UV
    // set) in the high bits and render state in the low bits. The variant
    // selects the shader entry-point pair; render state selects blend/cull/
    // depth. Kept as one integer so it intern and pipeline-variant lookup use
    // the same key.
    inline constexpr std::uint32_t VARIANT_SHIFT = 16;
    inline constexpr std::uint32_t VARIANT_MASK  = 0xFFFFu;
    inline constexpr std::uint32_t RENDER_MASK   = 0xFFFFu;

    [[nodiscard]] constexpr std::uint32_t PackDrawKey(std::uint32_t variant_slot,
                                                       std::uint32_t render_state) {
        return ((variant_slot & VARIANT_MASK) << VARIANT_SHIFT) |
               (render_state & RENDER_MASK);
    }
    [[nodiscard]] constexpr std::uint32_t VariantOf(std::uint32_t draw_key) {
        return (draw_key >> VARIANT_SHIFT) & VARIANT_MASK;
    }
    [[nodiscard]] constexpr std::uint32_t RenderStateOf(std::uint32_t draw_key) {
        return draw_key & RENDER_MASK;
    }
}

namespace TechniquePacking {

    // Per-material BlockArray geometry. The fragment shaders hardcode this
    // split as materialId / 256 and materialId % 256, so it must remain 256.
    // MATERIAL_BLOCK_COUNT is the number of descriptor-array elements each
    // technique must expose for its per-material data, and must be able to
    // name every material id the packing above can produce.
    inline constexpr std::uint32_t MATERIAL_BLOCK_SIZE  = 256;
    inline constexpr std::uint32_t MATERIAL_BLOCK_COUNT =
        (1u << MATERIAL_BITS) / MATERIAL_BLOCK_SIZE;
    static_assert((1u << MATERIAL_BITS) % MATERIAL_BLOCK_SIZE == 0,
                  "MATERIAL_BITS must span a whole number of material blocks");

    constexpr std::uint32_t Pack(std::uint32_t material_id, std::uint32_t technique_id) {
        return (material_id << TECHNIQUE_BITS) | (technique_id & TECHNIQUE_MASK);
    }
    constexpr std::uint32_t UnpackTechnique(std::uint32_t packed) {
        return packed & TECHNIQUE_MASK;
    }
    constexpr std::uint32_t UnpackMaterial(std::uint32_t packed) {
        return packed >> TECHNIQUE_BITS;
    }
}

// ── PipelineFlags — per-technique pass participation hints ──
// SceneRenderer packs these into the GPU flag table (UpdateTechniqueFlags);
// the depth prepass filter, occluder selection and the occlusion cull shaders
// consume them.
struct PipelineFlags {
    bool participates_in_depth_pass = true;  // writes depth → occludes others
    bool receives_occlusion = true;          // gets culled by HiZ (set false for transparents)
    bool participates_in_collect = true;     // generates indirect draw commands
    // Conservative post-transform bounds (§5.5): techniques whose vertices can
    // be displaced by compute must set false — they are occludee-only and
    // never occluder candidates unless they prove a post-deform envelope.
    bool bounds_conservative = true;
};

// ── BaseTechnique — abstract base for all rendering techniques ──
class BaseTechnique {
public:
    // Shared descriptor declaration model (sets 0-4 reserved by the engine,
    // technique/application bindings start at set 5).
    using BindingKind = VulkanEngine::Render::DescriptorKind;
    using BindingDecl = VulkanEngine::Render::DescriptorDecl;

    virtual ~BaseTechnique() = default;

    [[nodiscard]] TechniqueId GetId() const { return id_; }
    [[nodiscard]] std::span<const BindingDecl> GetBindings() const { return bindings_; }
    [[nodiscard]] std::size_t GetBindingCount() const { return bindings_.size(); }
    [[nodiscard]] const BindingDecl& GetBinding(std::size_t i) const { return bindings_[i]; }

    // True when this technique declares a PerMaterial binding of exactly the
    // given data type. Callers must not reinterpret a material's packed data as
    // a type the technique did not declare (the layout would not match).
    [[nodiscard]] bool HasPerMaterialOfType(std::type_index type) const {
        return std::any_of(bindings_.begin(), bindings_.end(),
            [type](const BindingDecl& decl) {
                return decl.kind == BindingKind::PerMaterial && decl.type_index == type;
            });
    }

    // ── Typed access to shared data (technique-local, not per-material) ──
    template<typename T>
    const T& ReadShared() const {
        for (std::size_t i = 0; i < bindings_.size(); ++i) {
            if (bindings_[i].kind == BindingKind::Shared &&
                bindings_[i].type_index == std::type_index(typeid(T))) {
                assert(i < shared_cpu_data_.size());
                return *reinterpret_cast<const T*>(shared_cpu_data_[i].data());
            }
        }
        assert(!"Shared binding type not found");
        static T empty{};
        return empty;
    }

    // ── Update shared binding data (writes technique-local CPU buffer, stages upload to GPU) ──
    template<typename T>
    void UpdateShared(const T& data, VulkanEngine::GpuResources::StagingPool& staging) {
        for (std::size_t i = 0; i < bindings_.size(); ++i) {
            if (bindings_[i].kind == BindingKind::Shared &&
                bindings_[i].type_index == std::type_index(typeid(T))) {
                // Copy to technique-local CPU buffer
                assert(i < shared_cpu_data_.size());
                auto* dst = shared_cpu_data_[i].data();
                std::memcpy(dst, &data, sizeof(T));

                // Stage upload to GPU buffer
                assert(i < shared_buffers_.size());
                auto slice = staging.Allocate(sizeof(T), 256);
                if (!slice.has_value()) return;
                std::memcpy(slice->mapped_ptr, &data, sizeof(T));
                staging.RecordBufferCopy(*slice, *shared_buffers_[i].GetBuffer(), 0);
                return;
            }
        }
        assert(!"Shared binding type not found — did you declare it?");
    }

    // ── Block array access by type (PerMaterial only) ──
    template<typename T>
    VulkanEngine::GpuResources::BlockArray* GetBlockArrayForType() {
        std::size_t pm_ordinal = 0;
        for (std::size_t i = 0; i < bindings_.size(); ++i) {
            if (bindings_[i].kind != BindingKind::PerMaterial) continue;
            if (bindings_[i].type_index == std::type_index(typeid(T))) {
                assert(pm_ordinal < block_arrays_.size());
                return &block_arrays_[pm_ordinal];
            }
            ++pm_ordinal;
        }
        assert(!"PerMaterial binding type not found — did you declare it?");
        return nullptr;
    }

    // ── Block array access by declaration index into GetBindings() ──
    // Maps the binding's position among all bindings to its PerMaterial ordinal.
    [[nodiscard]] VulkanEngine::GpuResources::BlockArray* GetBlockArrayForBinding(
        std::size_t binding_index) {
        std::size_t pm_ordinal = 0;
        for (std::size_t i = 0; i < bindings_.size(); ++i) {
            if (bindings_[i].kind != BindingKind::PerMaterial) continue;
            if (i == binding_index) {
                return pm_ordinal < block_arrays_.size() ? &block_arrays_[pm_ordinal] : nullptr;
            }
            ++pm_ordinal;
        }
        return nullptr;
    }

    // ── Block array access by PerMaterial ordinal (0 = first PerMaterial binding) ──
    VulkanEngine::GpuResources::BlockArray* GetBlockArray(std::size_t block_array_index) {
        if (block_array_index < block_arrays_.size()) {
            return &block_arrays_[block_array_index];
        }
        return nullptr;
    }

    // ── GPU resource access ──
    // Variant 0 is the base pipeline (no render-state variation). A draw group
    // selects its variant by the render-state bits of its key; a key with no
    // materialized variant falls back to the base pipeline, so a technique that
    // never sees blend/cull content never pays for extra pipelines.
    [[nodiscard]] vk::Pipeline GetPipeline() const {
        return variants_.empty() ? nullptr : variants_[0].slot.Get();
    }
    [[nodiscard]] vk::Pipeline GetPipeline(std::uint32_t draw_key) const {
        for (const auto& v : variants_) {
            if (v.draw_key == draw_key) return v.slot.Get();
        }
        return GetPipeline();
    }
    [[nodiscard]] vk::PipelineLayout GetPipelineLayout() const { return *pipeline_layout_; }

    // True when a variant for this composite draw key exists. The main pass uses
    // this to decide whether to materialize one before binding.
    [[nodiscard]] bool HasVariant(std::uint32_t draw_key) const {
        return std::any_of(variants_.begin(), variants_.end(),
            [draw_key](const TechniqueVariant& v) {
                return v.draw_key == draw_key;
            });
    }

    // Materialize a pipeline variant for `draw_key` (interface variant +
    // render state) if it does not exist, then return whether the technique has
    // a usable pipeline for it. Requires a prior successful Compile. Safe to
    // call while frames are in flight: a brand-new pipeline is created, no
    // in-flight pipeline is destroyed. Returns the base pipeline's usability
    // when variant creation fails, so the caller still has a pipeline to bind.
    [[nodiscard]] bool EnsureVariant(std::uint32_t draw_key,
                                     ShaderSystem::ShaderManager& shaders,
                                     ShaderSystem::PipelineFactory& factory);

    // Dense interface-variant rows: [0] = default (UV0), [k] = out-of-line UV
    // set k. A technique that owns entry-point rows fills this; a technique
    // that does not leaves it empty and has only slot 0.
    void SetVariantShaderRows(std::span<const VariantShaderPair> rows) {
        variant_shader_rows_.assign(rows.begin(), rows.end());
    }
    [[nodiscard]] std::uint32_t VariantCount() const {
        return variant_shader_rows_.empty() ? 1u
                                            : static_cast<std::uint32_t>(variant_shader_rows_.size());
    }
    [[nodiscard]] bool GetVariantShaders(std::uint32_t variant_slot,
                                         ShaderSystem::ShaderId& out_vert,
                                         ShaderSystem::ShaderId& out_frag) const {
        if (variant_shader_rows_.empty()) {
            if (variant_slot != 0 || variants_.empty()) return false;
            out_vert = variants_[0].vert;
            out_frag = variants_[0].frag;
            return true;
        }
        return GetVariantShaderRow(variant_shader_rows_, variant_slot, out_vert, out_frag);
    }

    // Interface variant a material needs, derived from its serialized
    // PerMaterial payload. The base technique has no interface variants and
    // returns 0; a technique with UV entry-point rows reads its own struct's
    // uv_sets and returns the highest referenced set. Pure and device-free.
    [[nodiscard]] virtual std::uint32_t InterfaceVariantForMaterial(
        std::span<const std::byte> cpu_data) const {
        (void)cpu_data;
        return 0;
    }

    // Byte offsets of the 32-bit bindless texture-slot words inside this
    // technique's PerMaterial payload. The residency layer uses them to find and
    // rewrite a material's texture references when a bindless slot is evicted.
    // The base technique has no texture slots.
    [[nodiscard]] virtual std::span<const std::uint32_t> TextureSlotFieldOffsets() const {
        return {};
    }

    // ── Custom descriptor sets (technique-owned BlockArray/Shared bindings at sets 4+) ──
    [[nodiscard]] std::span<const vk::DescriptorSet> GetCustomDescriptorSets() const {
        return { custom_descriptor_set_handles_.data(), custom_descriptor_set_handles_.size() };
    }

    // ── Material packing — each technique defines how to pack material_id into StaticEntry.technique_material ──
    // The caller passes the material's resolved draw key (a technique id today,
    // a dense interned group id once render-state variants exist). The default
    // packs (material_id << TECHNIQUE_BITS) | draw_key. Override for custom
    // per-material data encoding.
    [[nodiscard]] virtual uint32_t PackMaterialData(uint32_t material_id,
                                                    uint32_t draw_key) const;

    // ── Per-material descriptor arrays ──
    // Bind block `block_index` of the per-material descriptor array for the
    // PerMaterial binding at `binding_index` in GetBindings() (same index the
    // BindingDecl/GpuResources::BlockArray helpers use). Idempotent, and safe
    // while frames are in flight: the binding is partially bound and
    // update-after-bind. Returns false when the index does not name a
    // PerMaterial binding or the BlockArray does not yet contain that block.
    [[nodiscard]] bool EnsureMaterialBlockBound(std::size_t binding_index, std::uint32_t block_index);

    // ── Pass participation flags — see PipelineFlags struct above ──
    PipelineFlags pipeline_flags{}; // NOLINT(misc-non-private-member-variables-in-classes)

    // ── Shutdown — cleanup GPU resources ──
    void Shutdown();

public:
    BaseTechnique() = default;

    // ── Engine set usage (sets 0-4 are always bound at layout slots 0-4) ──
    // Technique/application bindings start at set 5 (kFirstAppDescriptorSet).
    // No opt-in is needed for engine sets.

    // ── Declare a PerMaterial binding ──
    // T is the C++ data type; set/binding are user-specified (per-technique scope).
    // Debug assert fires if set+binding already declared within this technique.
    template<typename T>
    void DeclarePerMaterial(const std::uint32_t set, const std::uint32_t binding) {
        ValidateNoBindingCollision(set, binding);
        BindingDecl decl{};
        decl.set = set;
        decl.binding = binding;
        decl.kind = BindingKind::PerMaterial;
        decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
        decl.stage_flags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        decl.count = 1;
        decl.stride = sizeof(T);
        decl.type_index = std::type_index(typeid(T));
        DeclareBindingImpl(std::move(decl));
    }

    // ── Declare a Shared binding ──
    template<typename T>
    void DeclareShared(const std::uint32_t set, const std::uint32_t binding) {
        ValidateNoBindingCollision(set, binding);
        BindingDecl decl{};
        decl.set = set;
        decl.binding = binding;
        decl.kind = BindingKind::Shared;
        decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
        decl.stage_flags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        decl.count = 1;
        decl.stride = 0;
        decl.type_index = std::type_index(typeid(T));
        DeclareBindingImpl(std::move(decl));
    }

    // ── Set technique ID (called by TechniqueManager during registration) ──
    void SetId(const TechniqueId id) { id_ = id; }

    // ── Compilation (separate from constructor) ──
    // Creates pipeline layout with engine sets 0-4 + custom sets 5+.
    // Builds one BlockArray per PerMaterial binding, one GpuBuffer per Shared binding.
    // Returns false if pipeline creation failed; the technique is then unusable
    // (GetPipeline() returns VK_NULL_HANDLE) and resource setup is skipped.
    [[nodiscard]] bool Compile(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                 ShaderSystem::ShaderManager& shader_mgr,
                 ShaderSystem::PipelineFactory& pipeline_factory,
                 ShaderSystem::ShaderId vert_id,
                 ShaderSystem::ShaderId frag_id,
                 const VulkanEngine::StandardMeshPipeline::PipelineConfig& config,
                 vk::DescriptorSetLayout bindless_layout,
                 vk::DescriptorSetLayout submesh_vertex_layout,
                 vk::DescriptorSetLayout raw_vertex_layout,
                 vk::DescriptorSetLayout indirection_layout,
                 vk::DescriptorSetLayout scene_uniform_layout = nullptr);

    // ── Hot-reload: rebuild the pipeline when a shader version changes ──
    void PollAndRebuild(ShaderSystem::ShaderManager& shaders,
                        ShaderSystem::PipelineFactory& factory,
                        std::uint32_t frame_index);

    // ── Draw-mode switch: re-specialize the vertex-stage draw-mode constant
    // and re-create the pipeline. No-op (returns true) when the technique was
    // never compiled. frame_index selects the retire-ring bucket; callers must
    // have device-idled first so the old pipeline is not in flight. ──
    [[nodiscard]] bool RebuildForDrawMode(ShaderSystem::ShaderManager& shaders,
                                          ShaderSystem::PipelineFactory& factory,
                                          std::uint32_t draw_mode,
                                          std::uint32_t frame_index);

private:
    TechniqueId id_{};
    std::vector<BindingDecl> bindings_;
    std::vector<VulkanEngine::GpuResources::BlockArray> block_arrays_;      // one per PerMaterial binding
    std::vector<VulkanEngine::GpuResources::GpuBuffer> shared_buffers_;      // one per Shared binding
    std::vector<std::vector<std::byte>> shared_cpu_data_;                    // one per Shared binding (technique-local)

    // Per-material descriptor arrays: one entry per PerMaterial binding, tracking
    // where its array lives and which blocks have had a descriptor written.
    struct MaterialArrayBinding {
        std::size_t binding_index = 0;      // index into bindings_ / GetBindings()
        std::size_t block_array_index = 0;  // PerMaterial ordinal into block_arrays_
        vk::DescriptorSet set = nullptr;    // custom set that owns this binding
        std::uint32_t binding_number = 0;   // descriptor binding number within that set
        std::vector<std::uint8_t> block_bound;  // 1 once the block's descriptor is written
    };
    std::vector<MaterialArrayBinding> material_array_bindings_;
    const vk::raii::Device* device_ = nullptr;  // backend device, valid for the technique's lifetime

    vk::raii::PipelineLayout pipeline_layout_ = nullptr;

    // ── Pipeline variants ──
    // One entry per materialized render-state key plus variant 0 (the base
    // pipeline, key 0). Layout, BlockArrays, Shared buffers, and descriptor
    // sets are technique-scoped and shared across variants; only the pipeline
    // differs. Each variant owns its own PipelineSlot so hot reload retires
    // only the pipeline it is rebuilding.
    struct TechniqueVariant {
        std::uint32_t draw_key = 0;   // composite: interface variant << 16 | render state
        ShaderSystem::ShaderId vert{};
        ShaderSystem::ShaderId frag{};
        ShaderSystem::GraphicsPipelineDesc desc{};
        ShaderSystem::PipelineSlot slot;
    };
    std::vector<TechniqueVariant> variants_;  // [0] is the base pipeline
    // Interface-variant shader rows (empty when the technique has only the
    // default pipeline). Set before Compile.
    std::vector<VariantShaderPair> variant_shader_rows_;
    bool compiled_ = false;

    // Base pipeline config, kept to build a new variant's desc on demand (the
    // variant desc differs only in blend/cull/depth state and the draw-mode
    // specialization constant).
    VulkanEngine::StandardMeshPipeline::PipelineConfig pipeline_config_{};

    // Builds a variant's pipeline desc from the base desc + this draw key's
    // render state and interface variant. Pure; the desc owns its color-blend
    // attachments. `vert`/`frag` are the variant's shader rows and their entry
    // points are set from the slot.
    ShaderSystem::GraphicsPipelineDesc MakeVariantDesc(
        const ShaderSystem::GraphicsPipelineDesc& base, std::uint32_t draw_key,
        const VulkanEngine::StandardMeshPipeline::PipelineConfig& config) const;

    // Descriptor pool + sets for custom bindings (sets 4+)
    vk::raii::DescriptorPool descriptor_pool_ = nullptr;
    std::vector<vk::raii::DescriptorSet> custom_descriptor_sets_;
    std::vector<vk::DescriptorSet> custom_descriptor_set_handles_;  // non-owning views for accessor
    std::vector<vk::raii::DescriptorSetLayout> custom_set_layouts_;  // was local in Compile() — fixes lifetime bug

    void DeclareBindingImpl(BindingDecl decl);
    void ValidateNoBindingCollision(std::uint32_t set, std::uint32_t binding) const;
};

} // namespace VulkanEngine::TechniqueManager
