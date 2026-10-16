module;

export module VulkanEngine.MaterialManager:MaterialHandle;

import std;
import std.compat;
import VulkanEngine.TechniqueManager.TechniqueId;
import VulkanEngine.MaterialManager.MaterialId;
import VulkanEngine.TechniqueManager.BaseTechnique;

// ── Design ──
// Lambda-based modify<T>([](T& d) { d.field = value; }) eliminates the
// reference-escape footgun that RAII proxy approaches have.  After the
// lambda returns, the dirty flag and per-binding dirty mask are set
// automatically, and MarkDirty() is called on the MaterialManager.
//
// Usage:
//   auto wood = material_mgr.Register<PBRTechnique>(BlendMode::Opaque, MaterialData{...});
//   wood.modify<MaterialData>([](auto& d) { d.roughness = 0.95f; });
//   float r = wood.read<MaterialData>().roughness;
//
// The technique type Tech must provide static constexpr members:
//   template<typename T> static constexpr std::size_t GetOffset()
//   template<typename T> static constexpr bool     HasBinding()
//   template<typename T> static constexpr std::uint32_t GetBindingIndex()

export namespace VulkanEngine::MaterialManager {

// ── Blend mode ──
enum class BlendMode : std::uint8_t {
    Opaque = 0,
    Cutout,
    Transparent
};

// Authoring descriptor for a material's render state. `blend` selects the
// alpha contract (Opaque / MASK discard / alpha blend); `double_sided` and
// `depth_write` are the orthogonal cull/depth bits. All three resolve to a
// draw key, never to the technique's material payload.
struct MaterialDesc {
    BlendMode blend{BlendMode::Opaque};
    bool double_sided{false};
    bool depth_write{true};
};

// ── Per-material render state ──
// Orthogonal to the technique's material contract: stored on the material and
// resolved to a draw key so the technique's pipeline-variant cache can select
// the matching pipeline. Transparent *ordering* is a separate pass; these only
// pick pipeline state.
struct MaterialRenderState {
    bool double_sided{false};   // disable culling
    bool depth_write{true};
    bool alpha_mask{false};     // derived from BlendMode::Cutout for a MASK discard
    bool operator==(const MaterialRenderState&) const = default;
};

// Derives the stored render state from an authoring descriptor. alpha_mask is
// implied by Cutout, never independently settable. Whether the technique's
// entry points can actually honor the discard is a separate, technique-side
// contract — MaterialManager::Register cross-checks it via
// BaseTechnique::AlphaMaskExpectation and rejects the combination.
[[nodiscard]] constexpr MaterialRenderState DeriveRenderState(const MaterialDesc& desc) {
    return MaterialRenderState{
        .double_sided = desc.double_sided,
        .depth_write = desc.depth_write,
        .alpha_mask = (desc.blend == BlendMode::Cutout),
    };
}

// Packs render state into the key folded into a material's draw group. The
// default descriptor (Opaque, front-culled, depth write) maps to 0, so a
// default material interns the technique's seeded base group and existing
// scenes keep group == technique id. `blend` is included so a Transparent
// material selects a different pipeline than an Opaque one even when the
// boolean bits agree.
[[nodiscard]] constexpr std::uint32_t RenderStateKey(const MaterialDesc& desc) {
    using namespace VulkanEngine::TechniqueManager::DrawKeyState;
    const MaterialRenderState s = DeriveRenderState(desc);
    return (static_cast<std::uint32_t>(desc.blend) << BLEND_SHIFT) |
           (s.double_sided ? DOUBLE_SIDED : 0u) |
           (s.depth_write   ? 0u : DEPTH_WRITE_DISABLED) |
           (s.alpha_mask    ? ALPHA_MASK : 0u);
}

// ── Per-material GPU data entry ──
struct MaterialEntry {
    TechniqueManager::TechniqueId technique_id{0};
    BlendMode blend_mode{BlendMode::Opaque};
    MaterialRenderState render_state{};
    // Resolved draw group for this material, interned from
    // (technique_id, variant_slot, render_state). Written into the low bits of
    // StaticEntry.technique_material by the gather pass; the group carries the
    // interface variant and render state for pipeline selection.
    std::uint32_t group_id{0};
    bool dirty = false;
    std::uint32_t dirty_bindings = 0;
    std::vector<std::byte> cpu_data;  // PerMaterial bindings only, flat buffer
};

template<typename Tech>
class MaterialHandle {
public:
    // Invalid handle: all-null/zero; Valid() is false. Register returns one
    // when a registration is rejected (e.g. a Cutout material whose technique
    // cannot honor alpha_mask).
    MaterialHandle() = default;
    // ── Read access: const ref, never dirties, constexpr offset ──
    template<typename T>
    const T& Read() const {
        static_assert(Tech::template HasBinding<T>(),
                      "Technique does not declare a PerMaterial binding for this type");
        constexpr std::size_t off = Tech::template GetOffset<T>();
        return *reinterpret_cast<const T*>(entry_->cpu_data.data() + off);
    }

    // ── Write access: lambda-based, no reference escape possible ──
    // After func() returns: dirty flag set, per-binding dirty mask updated, MarkDirty called.
    template<typename T, typename Func>
    void Modify(Func&& func) {
        static_assert(Tech::template HasBinding<T>(),
                      "Technique does not declare a PerMaterial binding for this type");
        constexpr std::size_t off = Tech::template GetOffset<T>();
        constexpr std::uint32_t binding_idx = Tech::template GetBindingIndex<T>();
        func(*reinterpret_cast<T*>(entry_->cpu_data.data() + off));
        entry_->dirty = true;
        entry_->dirty_bindings |= (1u << binding_idx);
        mark_dirty_(id_);
    }

    // ── Accessors ──
    [[nodiscard]] std::uint32_t Id() const { return id_; }
    [[nodiscard]] std::uint32_t Generation() const { return generation_; }
    [[nodiscard]] bool Valid() const { return entry_ != nullptr; }

    // ── Generation-checked reference, suitable for long-lived per-object storage ──
    [[nodiscard]] MaterialRef Ref() const { return MaterialRef{id_, generation_}; }

    // Copyable — all copies point to the same MaterialEntry
    MaterialHandle(const MaterialHandle&) = default;
    MaterialHandle& operator=(const MaterialHandle&) = default;
    MaterialHandle(MaterialHandle&&) = default;
    MaterialHandle& operator=(MaterialHandle&&) = default;

private:
    friend class MaterialManager;

    MaterialHandle(std::uint32_t id, std::uint32_t generation, MaterialEntry* entry,
                   std::function<void(std::uint32_t)> mark_dirty)
        : id_(id), generation_(generation), entry_(entry), mark_dirty_(std::move(mark_dirty)) {}

    std::uint32_t id_ = 0;
    std::uint32_t generation_ = 0;
    MaterialEntry* entry_ = nullptr;
    std::function<void(std::uint32_t)> mark_dirty_{};
};

} // namespace VulkanEngine::MaterialManager
