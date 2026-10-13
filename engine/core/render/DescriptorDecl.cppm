module;

export module VulkanEngine.DescriptorDecl;

import std;
import std.compat;

import vulkan;

export namespace VulkanEngine::Render {

// ── Shared descriptor declaration model ──
//
// One description of a descriptor binding, shared by techniques and passes so
// the engine composes every pipeline layout the same way. Engine-standard sets
// 0-4 are reserved; application/technique bindings start at set 5.

enum class DescriptorKind : std::uint8_t {
    PerMaterial,   // bindless array indexed by material_id
    Shared,        // single buffer for all users of this declaration
    UniformBuffer,
    SampledImage,
    StorageImage,
    CombinedImageSampler,
};

struct DescriptorDecl {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    DescriptorKind kind = DescriptorKind::Shared;
    vk::DescriptorType descriptor_type = vk::DescriptorType::eStorageBuffer;
    vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eVertex |
                                       vk::ShaderStageFlagBits::eFragment;
    std::uint32_t count = 1;
    vk::DescriptorBindingFlags binding_flags{};
    std::uint32_t stride = 0; // byte size per entry (PerMaterial only)
    std::type_index type_index = typeid(void);
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Engine-standard descriptor sets. Sets 0-4 are always present in a composed
// layout and may only be declared by the engine.
inline constexpr std::uint32_t kFirstAppDescriptorSet = 5;
inline constexpr std::uint32_t kEngineDescriptorSetCount = 5;

struct BindingSetGroup {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t set = 0;
    std::vector<DescriptorDecl> bindings{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct PushConstantRangeDecl {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    vk::ShaderStageFlags stages{};
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

enum class LayoutError : std::uint8_t {
    None,
    ReservedSetUsedByApp,
    EngineSetOutOfRange,
    InvalidSetGap,
    DuplicateBinding,
    EmptyBinding,
    EmptyPushConstantRange,
    OverlappingPushConstantRange,
};

struct ComposedLayout {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::vector<BindingSetGroup> sets{}; // ascending, contiguous from 0
    std::vector<PushConstantRangeDecl> push_ranges{}; // ascending by offset
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] std::size_t SetCount() const { return sets.size(); }
};

// Deterministic grouping: ascending set, ascending binding.
[[nodiscard]] inline std::vector<BindingSetGroup> GroupBindingsBySet(
    const std::vector<DescriptorDecl>& decls) {
    std::unordered_map<std::uint32_t, BindingSetGroup> by_set;
    for (const auto& decl : decls) {
        by_set[decl.set].set = decl.set;
        by_set[decl.set].bindings.push_back(decl);
    }

    std::vector<BindingSetGroup> groups;
    groups.reserve(by_set.size());
    for (auto& group : by_set | std::views::values) {
        std::ranges::sort(group.bindings, [](const DescriptorDecl& a, const DescriptorDecl& b) {
            return a.binding < b.binding;
        });
        groups.push_back(std::move(group));
    }
    std::ranges::sort(groups, [](const BindingSetGroup& a, const BindingSetGroup& b) {
        return a.set < b.set;
    });
    return groups;
}

// Composes a full pipeline layout description with reserved engine sets.
//
// The composer is pure: it validates and orders declarations, and returns a
// deterministic description. It does not create Vulkan objects, so the layout
// contract is unit-testable without a device.
class PipelineLayoutComposer {
public:
    // Engine-standard sets 0-4.
    [[nodiscard]] std::expected<void, LayoutError> AddEngineBindings(
        std::vector<DescriptorDecl> decls) {
        for (const auto& decl : decls) {
            if (decl.set >= kEngineDescriptorSetCount) {
                return std::unexpected(LayoutError::EngineSetOutOfRange);
            }
            if (decl.count == 0) {
                return std::unexpected(LayoutError::EmptyBinding);
            }
            if (HasBinding(engine_decls_, decl.set, decl.binding) ||
                HasBinding(app_decls_, decl.set, decl.binding)) {
                return std::unexpected(LayoutError::DuplicateBinding);
            }
            engine_decls_.push_back(decl);
        }
        return {};
    }

    // Application/technique bindings; every set must be >= 5.
    [[nodiscard]] std::expected<void, LayoutError> AddAppBindings(
        std::vector<DescriptorDecl> decls) {
        for (const auto& decl : decls) {
            if (decl.set < kFirstAppDescriptorSet) {
                return std::unexpected(LayoutError::ReservedSetUsedByApp);
            }
            if (decl.count == 0) {
                return std::unexpected(LayoutError::EmptyBinding);
            }
            if (HasBinding(app_decls_, decl.set, decl.binding) ||
                HasBinding(engine_decls_, decl.set, decl.binding)) {
                return std::unexpected(LayoutError::DuplicateBinding);
            }
            app_decls_.push_back(decl);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, LayoutError> AddPushConstantRange(PushConstantRangeDecl range) {
        if (range.stages == vk::ShaderStageFlags{} || range.size == 0) {
            return std::unexpected(LayoutError::EmptyPushConstantRange);
        }
        push_ranges_.push_back(range);
        return {};
    }

    [[nodiscard]] std::expected<ComposedLayout, LayoutError> Compose() const {
        std::vector<DescriptorDecl> all;
        all.reserve(engine_decls_.size() + app_decls_.size());
        all.insert(all.end(), engine_decls_.begin(), engine_decls_.end());
        all.insert(all.end(), app_decls_.begin(), app_decls_.end());

        ComposedLayout layout{};
        layout.sets = GroupBindingsBySet(all);

        // Reject duplicate (set, binding) and validate contiguity from 0.
        for (const auto& group : layout.sets) {
            for (std::size_t i = 1; i < group.bindings.size(); ++i) {
                if (group.bindings[i].binding == group.bindings[i - 1].binding) {
                    return std::unexpected(LayoutError::DuplicateBinding);
                }
            }
        }
        if (!layout.sets.empty()) {
            const std::uint32_t max_set = layout.sets.back().set;
            if (max_set + 1 != layout.sets.size()) {
                return std::unexpected(LayoutError::InvalidSetGap);
            }
        }

        layout.push_ranges = push_ranges_;
        std::ranges::sort(layout.push_ranges, [](const PushConstantRangeDecl& a,
                                                  const PushConstantRangeDecl& b) {
            if (a.offset != b.offset) {
                return a.offset < b.offset;
            }
            return static_cast<std::uint32_t>(a.stages) < static_cast<std::uint32_t>(b.stages);
        });
        for (std::size_t i = 1; i < layout.push_ranges.size(); ++i) {
            const auto& previous = layout.push_ranges[i - 1];
            const auto& current = layout.push_ranges[i];
            if (current.offset < previous.offset + previous.size) {
                return std::unexpected(LayoutError::OverlappingPushConstantRange);
            }
        }

        return layout;
    }

    [[nodiscard]] std::size_t EngineBindingCount() const { return engine_decls_.size(); }
    [[nodiscard]] std::size_t AppBindingCount() const { return app_decls_.size(); }

private:
    [[nodiscard]] static bool HasBinding(const std::vector<DescriptorDecl>& decls,
                                         std::uint32_t set, std::uint32_t binding) {
        return std::ranges::any_of(decls, [&](const DescriptorDecl& decl) {
            return decl.set == set && decl.binding == binding;
        });
    }

    std::vector<DescriptorDecl> engine_decls_{};
    std::vector<DescriptorDecl> app_decls_{};
    std::vector<PushConstantRangeDecl> push_ranges_{};
};

} // namespace VulkanEngine::Render
