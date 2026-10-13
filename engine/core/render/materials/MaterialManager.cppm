module;

#include <cassert>
#include <logging/logging_macros.hpp>

export module VulkanEngine.MaterialManager;

import std;
import std.compat;

import logiface;

import vulkan;

export import VulkanEngine.MaterialManager.MaterialId;
export import VulkanEngine.BindlessManager.TextureSlot;
export import VulkanEngine.TechniqueManager.TechniqueId;
import VulkanEngine.FileLoaders.TextureLoaders;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.TechniqueManager;

// Import the MaterialHandle template from the module partition
export import :MaterialHandle;

export namespace VulkanEngine::MaterialManager {
    using TechniqueManager::TechniqueId;

void ValidateTextureBlendMode(const VulkanEngine::FileLoaders::Textures::AlphaAnalysis& alpha,
                              BlendMode mode,
                              std::string_view texture_name);

class MaterialManager {
public:
    void Initialize(VulkanEngine::GpuResources::StagingPool* staging_pool = nullptr);
    void Shutdown();

    // Typed registration — technique type inferred from template.
    // Only PerMaterial binding data is passed; Shared data lives on the technique.
    // The legacy BlendMode overload forwards to the MaterialDesc form.
    template<typename Tech, typename... Ts>
    MaterialHandle<Tech> Register(BlendMode blend, const Ts&... data) {
        return Register<Tech>(MaterialDesc{.blend = blend}, data...);
    }

    template<typename Tech, typename... Ts>
    MaterialHandle<Tech> Register(const MaterialDesc& desc, const Ts&... data) {
        static_assert((std::is_trivially_copyable_v<Ts> && ...),
                      "All material data types must be trivially copyable (GPU POD)");

        // Validate technique exists and get its ID
        assert(technique_mgr != nullptr && "TechniqueManager not set — call SetTechniqueManager first");
        const TechniqueId tech_id = technique_mgr->template GetId<Tech>();
        auto* tech_ptr = technique_mgr->GetTechnique(tech_id);
        assert(tech_ptr != nullptr && "Technique not registered for this type");

        // ── Allocate material ID ──
        MaterialId id;
        if (!Free_list.empty()) {
            id = Free_list.back();
            Free_list.pop_back();
        } else {
            id = MaterialId{static_cast<std::uint32_t>(Materials.size())};
            Materials.emplace_back();
            generations_.emplace_back(0u);
        }

        // Fresh generation for this slot lifetime. Generation 0 is reserved so
        // that a default-constructed MaterialRef (value == kInvalidMaterialId)
        // can never accidentally match a live slot.
        if (next_generation_ == 0) next_generation_ = 1;
        const std::uint32_t generation = next_generation_++;
        generations_[id.value] = generation;

        // ── Resolve the material's draw key ──
        // Render state is stored on the material (not the technique payload).
        // The interface variant (the out-of-line UV set the material samples)
        // is derived from the serialized material payload, so a material whose
        // slots select UV1 gets the technique's UV1 entry-point pipeline. Both
        // fold into one composite key interned by the technique manager.
        const MaterialRenderState render_state = DeriveRenderState(desc);

        // Record the technique's texture-slot word offsets so a later eviction
        // can find and rewrite this material's bindless references.
        SetTextureSlotFields(tech_id.value, tech_ptr->TextureSlotFieldOffsets());

        // ── Serialize PerMaterial binding data into flat cpu_data buffer ──
        auto entry = std::make_unique<MaterialEntry>();
        entry->technique_id = tech_id;
        entry->blend_mode = desc.blend;
        entry->render_state = render_state;
        entry->cpu_data.clear();
        auto write_one = [&]<typename U>(const U& d) {
            const auto* bytes = reinterpret_cast<const std::byte*>(&d);
            entry->cpu_data.insert(entry->cpu_data.end(), bytes, bytes + sizeof(U));
        };
        (write_one(data), ...);

        const std::uint32_t variant_slot =
            tech_ptr->InterfaceVariantForMaterial(entry->cpu_data);
        // The intern stores the interface variant and the 16-bit render state
        // separately; the main pass reconstructs the composite draw key with
        // PackDrawKey(variant, render state).
        const std::uint16_t group_id =
            technique_mgr->InternDrawGroup(tech_id.value, static_cast<std::uint16_t>(variant_slot),
                                           RenderStateKey(desc));
        entry->group_id = group_id;

        // ── Immediate first upload via staging → device-local ──
        if (staging_pool && !entry->cpu_data.empty()) {
            const std::size_t total_size = entry->cpu_data.size();
            const auto staging_slice =
                staging_pool->Allocate(static_cast<std::uint64_t>(total_size), 256);
            if (staging_slice.has_value()) {
                std::memcpy(staging_slice->mapped_ptr, entry->cpu_data.data(), total_size);

                for (std::size_t bi = 0; bi < tech_ptr->GetBindingCount(); ++bi) {
                    const auto& binding = tech_ptr->GetBinding(bi);
                    if (binding.kind != TechniqueManager::BaseTechnique::BindingKind::PerMaterial) continue;

                    auto* ba = tech_ptr->GetBlockArrayForBinding(bi);
                    if (ba == nullptr) continue;

                    // The shader derives the block index as materialId /
                    // MATERIAL_BLOCK_SIZE, so the BlockArray geometry must match it.
                    if (ba->EntriesPerBlock() != TechniqueManager::TechniquePacking::MATERIAL_BLOCK_SIZE) {
                        LOGIFACE_LOG(error, "MaterialManager::Register: technique " +
                                     std::to_string(tech_id.value) + " binding " + std::to_string(bi) +
                                     " has entries_per_block " + std::to_string(ba->EntriesPerBlock()) +
                                     " but the shaders expect " +
                                     std::to_string(TechniqueManager::TechniquePacking::MATERIAL_BLOCK_SIZE) +
                                     "; material will not be uploaded");
                        continue;
                    }

                    const std::uint32_t block = id.value / ba->EntriesPerBlock();
                    if (block >= TechniqueManager::TechniquePacking::MATERIAL_BLOCK_COUNT) {
                        LOGIFACE_LOG(error, "MaterialManager::Register: material id " +
                                     std::to_string(id.value) + " exceeds the per-material descriptor "
                                     "array capacity (" +
                                     std::to_string(TechniqueManager::TechniquePacking::MATERIAL_BLOCK_COUNT *
                                                     TechniqueManager::TechniquePacking::MATERIAL_BLOCK_SIZE) +
                                     " materials) for technique " + std::to_string(tech_id.value) +
                                     "; material will not be uploaded");
                        continue;
                    }

                    // Failure is handled by the BlockCount() check below.
                    (void)ba->EnsureCapacity(id.value + 1);
                    if (block >= ba->BlockCount()) {
                        LOGIFACE_LOG(error, "MaterialManager::Register: BlockArray for technique " +
                                     std::to_string(tech_id.value) + " binding " + std::to_string(bi) +
                                     " could not grow to hold material " + std::to_string(id.value) +
                                     "; material will not be uploaded");
                        continue;
                    }

                    if (!tech_ptr->EnsureMaterialBlockBound(bi, block)) {
                        LOGIFACE_LOG(error, "MaterialManager::Register: could not bind material block " +
                                     std::to_string(block) + " for technique " +
                                     std::to_string(tech_id.value) + " binding " + std::to_string(bi) +
                                     "; material will not be uploaded");
                        continue;
                    }

                    staging_pool->RecordBufferCopy(*staging_slice,
                                                   ba->GetBlockArray(block),
                                                   ba->EntrySize() * (static_cast<std::uint64_t>(id.value % ba->EntriesPerBlock())));
                }

                staging_pool->FlushImmediate();
            }
        }

        MaterialEntry* entry_ptr = entry.get();
        Materials[id.value] = std::move(entry);

        return MaterialHandle<Tech>(id.value, generation, entry_ptr,
            [this](const std::uint32_t mid) { MarkDirty(MaterialId{mid}); });
    }

    // ── Batched GPU upload — called once per frame ──
    void FlushDirtyMaterials();

    // ── Material lifecycle ──
    void Destroy(MaterialId id);

    // ── Called by MaterialHandle::modify() ──
    void MarkDirty(MaterialId id);

    // ── Validity ──
    // True when the id names a live material entry. Does not detect slot reuse;
    // use the MaterialRef overload for that.
    [[nodiscard]] bool IsUsable(MaterialId id) const {
        return id.value < Materials.size() && Materials[id.value] != nullptr;
    }

    // True when the generation-checked reference still points at the same
    // material slot lifetime it was taken from.
    [[nodiscard]] bool IsUsable(MaterialRef ref) const {
        return ref.IsSet() &&
               ref.value < Materials.size() &&
               Materials[ref.value] != nullptr &&
               ref.value < generations_.size() &&
               generations_[ref.value] == ref.generation;
    }

    // Generation currently assigned to a slot (0 if the slot does not exist).
    [[nodiscard]] std::uint32_t GetGeneration(MaterialId id) const {
        return id.value < generations_.size() ? generations_[id.value] : 0u;
    }

    // ── Read-only access to any material (type-erased path) ──
    template<typename T>
    const T& Get(const MaterialId id) const {
        assert(IsUsable(id) && "MaterialManager::Get called with an unusable material id");
        const auto& entry = Materials[id.value];
        return *reinterpret_cast<const T*>(entry->cpu_data.data());
    }

    // ── Set technique manager for typed registration ──
    void SetTechniqueManager(VulkanEngine::TechniqueManager::TechniqueManager* mgr) { technique_mgr = mgr; }

    // ── Type-erased access to raw material data for PackMaterialData() ──
    [[nodiscard]] const void* GetRawData(MaterialId id) const {
        if (!IsUsable(id)) return nullptr;
        return Materials[id.value]->cpu_data.data();
    }

    // ── Look up the BaseTechnique for a material (via stored technique_id) ──
    // Null-safe: returns nullptr for an out-of-range, destroyed, or
    // technique-less material instead of dereferencing a dead slot.
    [[nodiscard]] TechniqueManager::BaseTechnique* GetTechniqueForMaterial(MaterialId id) const {
        if (!IsUsable(id) || technique_mgr == nullptr) return nullptr;
        return technique_mgr->GetTechnique(Materials[id.value]->technique_id);
    }

    // Resolved draw key for a material. Returns 0 (the base group) for an
    // unusable id, matching the fallback material's group.
    [[nodiscard]] std::uint16_t GetGroupForMaterial(MaterialId id) const {
        if (!IsUsable(id)) return 0;
        return static_cast<std::uint16_t>(Materials[id.value]->group_id);
    }

    // Per-material render state. Null for an unusable id.
    [[nodiscard]] const MaterialRenderState* GetRenderState(MaterialId id) const {
        if (!IsUsable(id)) return nullptr;
        return &Materials[id.value]->render_state;
    }

    // ── Texture-slot reference tracking (bindless residency/eviction) ──
    // Texture slots are stored as plain uint32_t words inside a technique's
    // PerMaterial payload, so the manager cannot know a technique's slot field
    // layout. The technique advertises the slot words and the manager keeps a
    // slot -> materials index so a released bindless slot can be rewritten to
    // the fallback across every referencing material.

    // Registers one 32-bit texture-slot word at `byte_offset` within the
    // technique's PerMaterial payload. Idempotent per technique id.
    void SetTextureSlotFields(std::uint32_t technique_id, std::span<const std::uint32_t> byte_offsets) {
        auto& fields = texture_slot_fields_[technique_id];
        fields.assign(byte_offsets.begin(), byte_offsets.end());
    }

    // Rewrites every material slot word equal to `slot` to `replacement`. Used
    // when a bindless slot is evicted/decommitted: the descriptor has already
    // been reset to the fallback, so the CPU payload must follow. Marks the
    // changed materials dirty so the new slot reaches the GPU.
    void RewriteTextureSlot(std::uint32_t slot, std::uint32_t replacement);

    // Calls `fn` for each material that currently references `slot`.
    template <typename Fn>
    void ForEachMaterialUsingSlot(std::uint32_t slot, Fn&& fn) const {
        const auto it = slot_users_.find(slot);
        if (it == slot_users_.end()) return;
        for (const MaterialId id : it->second) {
            fn(id);
        }
    }

    // Rebuilds `slot_users_` from every live material's registered slot words.
    // Called after registrations/mutations rather than maintained on each write.
    void RebuildTextureSlotIndex();

    [[nodiscard]] std::span<const std::uint32_t> GetTextureSlotFields(
        std::uint32_t technique_id) const {
        const auto it = texture_slot_fields_.find(technique_id);
        return it != texture_slot_fields_.end()
                   ? std::span<const std::uint32_t>(it->second)
                   : std::span<const std::uint32_t>{};
    }

    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;

    // Default constructible — owned by EngineContext
    MaterialManager() = default;
    ~MaterialManager() = default;

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    // New typed storage (pointer stability via unique_ptr)
    std::vector<std::unique_ptr<MaterialEntry>> Materials{};
    // Parallel to Materials: per-slot lifetime counter. Bumped on allocation
    // and on destroy so MaterialRef values taken earlier become stale.
    std::vector<std::uint32_t> generations_{};
    std::vector<MaterialId> Dirty_list{};
    std::vector<MaterialId> Free_list{};
    std::uint32_t next_generation_{1};
    // technique_id -> byte offsets of the 32-bit texture-slot words in that
    // technique's PerMaterial payload.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> texture_slot_fields_{};
    // slot -> live materials that reference it (rebuilt on demand).
    std::unordered_map<std::uint32_t, std::vector<MaterialId>> slot_users_{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    GpuResources::StagingPool* staging_pool = nullptr; // NOLINT(misc-non-private-member-variables-in-classes)
    TechniqueManager::TechniqueManager* technique_mgr = nullptr; // NOLINT(misc-non-private-member-variables-in-classes)
};

} // namespace VulkanEngine::MaterialManager
