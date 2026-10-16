module;

#include <logging/logging_macros.hpp>

module VulkanEngine.MaterialManager;

import std;
import std.compat;

import logiface;

import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.TechniqueManager;

namespace VulkanEngine::MaterialManager {

void MaterialManager::Initialize(GpuResources::StagingPool* staging_pool) {
    Materials.clear();
    generations_.clear();
    Dirty_list.clear();
    Free_list.clear();
    next_generation_ = 1;
    this->staging_pool = staging_pool;
}

void MaterialManager::Shutdown() {
    Materials.clear();
    generations_.clear();
    Dirty_list.clear();
    Free_list.clear();
    next_generation_ = 1;
    staging_pool = nullptr;
}

void MaterialManager::MarkDirty(MaterialId id) {
    if (id.value >= Materials.size()) return;
    auto& entry = Materials[id.value];
    if (entry && !entry->dirty) {
        entry->dirty = true;
        Dirty_list.push_back(id);
    }
}

void MaterialManager::Destroy(MaterialId id) {
    if (id.value >= Materials.size()) return;
    if (!Materials[id.value]) return;  // already destroyed
    Materials[id.value].reset();

    // Invalidate every outstanding MaterialRef for this slot lifetime. The slot
    // may later be reused by a different material; the generation bump ensures
    // a stale reference cannot alias the new occupant.
    if (next_generation_ == 0) next_generation_ = 1;
    generations_[id.value] = next_generation_++;

    Free_list.push_back(id);
}

void MaterialManager::FlushDirtyMaterials() {
    if (Dirty_list.empty()) return;  // ← common case: zero work
    if (!staging_pool) return;

    // Phase 1: allocate staging for all dirty materials
    struct PendingUpload {
        MaterialId id;
        MaterialEntry* entry;
        VulkanEngine::GpuResources::StagingAlloc slice;
    };
    std::vector<PendingUpload> pending;
    pending.reserve(Dirty_list.size());

    for (const MaterialId id : Dirty_list) {
        if (id.value >= Materials.size()) continue;
        auto& entry = Materials[id.value];
        if (!entry || !entry->dirty) continue;

        auto slice = staging_pool->Allocate(
            static_cast<std::uint64_t>(entry->cpu_data.size()), 256);
        if (!slice.has_value()) continue;
        std::memcpy(slice->mapped_ptr, entry->cpu_data.data(), entry->cpu_data.size());
        pending.push_back({id, entry.get(), *slice});
    }

    // Phase 2: record per-binding buffer copies — only for dirty bindings
    // We need the technique manager to look up binding info
    for (auto& p : pending) {
        // Note: In full implementation, we'd look up the technique from p.entry->technique_id
        // and iterate bindings. For now, we just flush the full cpu_data.
        // The technique lookup requires TechniqueManager which is set via SetTechniqueManager.
        if (technique_mgr) {
            auto* tech = technique_mgr->GetTechnique(p.entry->technique_id);
            if (tech) {
                std::uint32_t mask = p.entry->dirty_bindings;
                for (std::size_t bi = 0; bi < tech->GetBindingCount(); ++bi) {
                    const auto& binding = tech->GetBinding(bi);
                    if (binding.kind != TechniqueManager::BaseTechnique::BindingKind::PerMaterial) {
                        continue;
                    }
                    if (mask & 1u) {
                        auto* ba = tech->GetBlockArrayForBinding(bi);
                        if (ba != nullptr &&
                            (p.id.value / ba->EntriesPerBlock()) < ba->BlockCount()) {
                            staging_pool->RecordBufferCopy(p.slice,
                                ba->GetBlockArray(p.id.value / ba->EntriesPerBlock()),
                                ba->EntrySize() * (static_cast<std::uint64_t>(p.id.value % ba->EntriesPerBlock())));
                        }
                    }
                    mask >>= 1;
                }
            }
        }
        p.entry->dirty = false;
        p.entry->dirty_bindings = 0;
    }

    staging_pool->FlushImmediate();
    Dirty_list.clear();
}

void MaterialManager::RewriteTextureSlot(std::uint32_t slot, std::uint32_t replacement) {
    const auto users_it = slot_users_.find(slot);
    if (users_it == slot_users_.end()) {
        return;
    }
    // Copy: modifying a material can (via MarkDirty) touch Dirty_list, not the
    // users list, but iterate a snapshot to stay safe against reentrancy.
    const std::vector<MaterialId> users = users_it->second;
    for (const MaterialId id : users) {
        if (!IsUsable(id)) continue;
        auto& entry = Materials[id.value];
        const auto fields = GetTextureSlotFields(entry->technique_id.value);
        if (fields.empty()) continue;
        bool changed = false;
        for (const std::uint32_t offset : fields) {
            if (offset + sizeof(std::uint32_t) > entry->cpu_data.size()) continue;
            auto* word = reinterpret_cast<std::uint32_t*>(entry->cpu_data.data() + offset);
            if (*word == slot) {
                *word = replacement;
                changed = true;
            }
        }
        if (changed) {
            MarkDirty(id);
        }
    }
}

void MaterialManager::RebuildTextureSlotIndex() {
    slot_users_.clear();
    for (std::size_t i = 0; i < Materials.size(); ++i) {
        const auto& entry = Materials[i];
        if (!entry) continue;
        const auto fields = GetTextureSlotFields(entry->technique_id.value);
        if (fields.empty()) continue;
        for (const std::uint32_t offset : fields) {
            if (offset + sizeof(std::uint32_t) > entry->cpu_data.size()) continue;
            const auto* word = reinterpret_cast<const std::uint32_t*>(entry->cpu_data.data() + offset);
            slot_users_[*word].push_back(MaterialId{static_cast<std::uint32_t>(i)});
        }
    }
}

void ValidateTextureBlendMode(const VulkanEngine::FileLoaders::Textures::AlphaAnalysis& alpha,
                               BlendMode mode,
                               std::string_view texture_name) {
    const std::string name(texture_name);

    if (mode == BlendMode::Opaque) {
        if (alpha.hasFractionalAlpha) {
            LOGIFACE_LOG(warn, "Warning: texture '" + name + "' contains fractional alpha, "
                         "recommended blend mode is Transparent but current mode is Opaque");
        }
        if (alpha.hasZeroAlpha) {
            LOGIFACE_LOG(warn, "Warning: texture '" + name + "' has zero-alpha pixels, "
                         "may render incorrectly with Opaque blend mode");
        }
    } else if (mode == BlendMode::Cutout) {
        if (alpha.opaqueCoverage >= 1.0f) {
            LOGIFACE_LOG(warn, "Warning: texture '" + name + "' is fully opaque, "
                         "Cutout's discard can never trigger on it (use Opaque)");
        }
        if (alpha.hasFractionalAlpha) {
            LOGIFACE_LOG(warn, "Warning: texture '" + name + "' contains fractional alpha, "
                         "recommended blend mode is Transparent but current mode is Cutout");
        }
    } else if (mode == BlendMode::Transparent) {
        if (alpha.opaqueCoverage >= 1.0f) {
            LOGIFACE_LOG(warn, "Warning: texture '" + name + "' is fully opaque, "
                         "Transparent blend mode is unnecessary (use Opaque)");
        }
    }
}

} // namespace VulkanEngine::MaterialManager
