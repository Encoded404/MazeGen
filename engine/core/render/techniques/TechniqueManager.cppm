module;

#include <logging/logging_macros.hpp>

export module VulkanEngine.TechniqueManager;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanShared.CallbackList;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.TechniqueManager.BaseTechnique;
export import VulkanEngine.TechniqueManager.DefaultMeshTechnique;
export import VulkanEngine.TechniqueManager.UnlitTextureTechnique;

#ifndef UINT16_MAX
constexpr std::uint16_t UINT16_MAX =
    std::numeric_limits<std::uint16_t>::max();
#endif

export namespace VulkanEngine::TechniqueManager {

// TechniquePacking lives in the BaseTechnique module (exported below via the
// BaseTechnique re-export) so BaseTechnique::PackMaterialData can share it.

class TechniqueManager {
public:
    TechniqueManager() = default;
    ~TechniqueManager();

    TechniqueManager(const TechniqueManager&) = delete;
    TechniqueManager& operator=(const TechniqueManager&) = delete;

    VulkanShared::CallbackList<void(std::uint16_t id, vk::Pipeline pipeline, vk::PipelineLayout layout)> on_technique_changed; // NOLINT(misc-non-private-member-variables-in-classes)

    // Get technique by ID
    [[nodiscard]] BaseTechnique* GetTechnique(std::uint16_t technique_id);
    [[nodiscard]] BaseTechnique* GetTechnique(TechniqueId id);

    [[nodiscard]] std::uint16_t GetTechniqueCount() const { return static_cast<std::uint16_t>(techniques_.size()); }

    // Register a typed technique. The technique id must fit the draw-key field
    // (TECHNIQUE_BITS); registering past the cap would produce a key the GPU
    // tables cannot hold, so it is rejected here rather than at draw time.
    template<typename Tech>
        requires std::derived_from<Tech, BaseTechnique>
    TechniqueId Register(std::unique_ptr<Tech> technique) {
        if (techniques_.size() >= TechniquePacking::MAX_DRAW_GROUPS) {
            LOGIFACE_LOG(error, "TechniqueManager::Register: technique capacity (" +
                         std::to_string(TechniquePacking::MAX_DRAW_GROUPS) +
                         ") reached; technique not registered");
            return TechniqueId{UINT16_MAX};
        }
        auto id = TechniqueId{static_cast<std::uint16_t>(techniques_.size())};
        technique->SetId(id);
        type_to_id_[std::type_index(typeid(Tech))] = id;
        Technique t;
        t.base_technique = std::move(technique);
        techniques_.push_back(std::move(t));
        // Seed the technique's base draw group (no variant, no render state).
        // Technique ids are assigned sequentially from zero, so this group gets
        // the technique id, and every per-key table stays index-compatible with
        // the existing technique-id usage.
        (void)InternDrawGroup(id.value, 0, 0);
        return id;
    }

    // ── Draw-key interning ──
    // A draw key names a technique plus its resolved pipeline variant and render
    // state. Keys are dense-interned over the triple so the low TECHNIQUE_BITS
    // of a packed technique_material bound *live groups*, not the key's
    // cardinality. The base group (variant 0, no render state) is seeded at
    // registration; interface-variant and render-state groups are interned on
    // demand as materials register.
    // Returns UINT16_MAX when the cap is reached.
    [[nodiscard]] std::uint16_t InternDrawGroup(std::uint16_t technique_id,
                                                std::uint16_t variant_slot,
                                                std::uint32_t render_state_key) {
        const std::uint64_t intern_key =
            (static_cast<std::uint64_t>(technique_id) << 32) |
            (static_cast<std::uint64_t>(variant_slot) << 16) |
            static_cast<std::uint64_t>(render_state_key & 0xFFFFu);
        if (const auto it = intern_to_group_.find(intern_key); it != intern_to_group_.end()) {
            return it->second;
        }
        if (next_draw_group_ >= TechniquePacking::MAX_DRAW_GROUPS) {
            LOGIFACE_LOG(error, "TechniqueManager::InternDrawGroup: draw-group capacity (" +
                         std::to_string(TechniquePacking::MAX_DRAW_GROUPS) +
                         ") reached; key not interned");
            return UINT16_MAX;
        }
        const auto group = static_cast<std::uint16_t>(next_draw_group_++);
        intern_to_group_.emplace(intern_key, group);
        group_technique_.push_back(technique_id);
        group_variant_.push_back(variant_slot);
        group_render_state_.push_back(render_state_key);
        return group;
    }

    [[nodiscard]] std::uint32_t GetDrawGroupCount() const {
        return static_cast<std::uint32_t>(next_draw_group_);
    }

    // Owning technique of a draw group (for per-group flags/regions). Returns
    // UINT16_MAX for an out-of-range group.
    [[nodiscard]] std::uint16_t GetDrawGroupTechnique(std::uint16_t group) const {
        return group < group_technique_.size() ? group_technique_[group] : UINT16_MAX;
    }

    [[nodiscard]] std::uint16_t GetDrawGroupVariant(std::uint16_t group) const {
        return group < group_variant_.size() ? group_variant_[group] : 0u;
    }

    // Composite draw key (interface variant + render state) of a draw group.
    // This is what the main pass passes to BaseTechnique::EnsureVariant. 0 for
    // an unknown group.
    [[nodiscard]] std::uint32_t GetDrawGroupRenderState(std::uint16_t group) const {
        return group < group_render_state_.size() ? group_render_state_[group] : 0u;
    }

    // Get technique ID by type
    template<typename Tech>
    [[nodiscard]] TechniqueId GetId() const {
        if (const auto it = type_to_id_.find(std::type_index(typeid(Tech))); it != type_to_id_.end()) return it->second;
        return TechniqueId{UINT16_MAX};
    }

    void Shutdown();

    // ── Hot-reload: rebuild technique pipelines whose shaders changed ──
    void PollShaders(ShaderSystem::ShaderManager& shaders,
                     ShaderSystem::PipelineFactory& factory,
                     std::uint32_t frame_index);

    // ── Draw-mode switch: re-specialize every technique pipeline for the new
    // draw mode. Callers must device-idle first. Returns false if any technique
    // failed to re-create (the others are still switched). ──
    [[nodiscard]] bool RebuildForDrawMode(ShaderSystem::ShaderManager& shaders,
                                          ShaderSystem::PipelineFactory& factory,
                                          std::uint32_t draw_mode,
                                          std::uint32_t frame_index);

private:
    friend class BaseTechnique;

    struct Technique {
        std::unique_ptr<BaseTechnique> base_technique;
    };

    std::vector<Technique> techniques_{};
    std::unordered_map<std::type_index, TechniqueId> type_to_id_{};

    // Dense intern of (technique_id, variant_slot, render_state_key) -> the
    // draw key used in the low TECHNIQUE_BITS of technique_material. Variants
    // and render state collapse into the technique id until those axes land, so
    // the map currently holds one entry per registered technique.
    std::unordered_map<std::uint64_t, std::uint16_t> intern_to_group_{};
    std::vector<std::uint16_t> group_technique_{};
    std::vector<std::uint16_t> group_variant_{};
    std::vector<std::uint32_t> group_render_state_{};
    std::uint32_t next_draw_group_{0};
};

}

