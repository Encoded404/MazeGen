module;

export module VulkanEngine.Components.TransformOverride;

import std;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Mesh.MeshTypes;

export namespace VulkanEngine::Components {

using VulkanEngine::ComposeSubmeshTransform;
using VulkanEngine::SubMeshTransform;

// Per-object, per-submesh placement override.
//
// The asset-side SubMeshTransform places a submesh inside its mesh (for glTF, a
// node's baked world transform) and the entity places the mesh in the world.
// This component replaces the first half for the submeshes it names, so parts of
// one mesh can move independently without the object leaving its entity
// transform.
//
// The contract deliberately mirrors MaterialOverride: keyed by *object-local*
// submesh index (0 .. submesh_count-1), never by the renderer's global submesh
// index, because MeshRegistry reassigns global ranges on re-residency. The
// override is sparse — an index that was never set, or was Clear()ed, falls back
// to the asset transform. The owning mesh is fingerprinted via `mesh_id`; while
// unbound the override adopts the owner's mesh on first resolve, and a later
// mismatch is ignored and reported once rather than applied to a different
// submesh layout.
//
// Data component: dense storage.
struct TransformOverride {
    static constexpr std::uint32_t kUnboundMesh =
        std::numeric_limits<std::uint32_t>::max();

    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    // Mesh this override was authored against, or kUnboundMesh to adopt lazily.
    std::uint32_t mesh_id = kUnboundMesh;

    // Per object-local submesh index; a disengaged entry is "no override".
    std::vector<std::optional<SubMeshTransform>> submeshes;

    // Signatures of discrepancies already reported, so each distinct problem is
    // logged exactly once instead of every frame.
    mutable std::unordered_set<std::uint64_t> reported_signatures;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    // Returns true the first time a discrepancy signature is observed.
    [[nodiscard]] bool ShouldReport(const std::uint64_t signature) const {
        return reported_signatures.insert(signature).second;
    }

    // Pin this override to a specific mesh. Any outstanding diagnostic state is
    // reset so a fresh mismatch is reported.
    void Bind(const std::uint32_t bound_mesh_id) {
        mesh_id = bound_mesh_id;
        reported_signatures.clear();
    }

    void Unbind() {
        mesh_id = kUnboundMesh;
        reported_signatures.clear();
    }

    void Set(const std::uint32_t submesh_index, const SubMeshTransform& transform) {
        if (submesh_index >= submeshes.size()) {
            submeshes.resize(static_cast<std::size_t>(submesh_index) + 1);
        }
        submeshes[submesh_index] = transform;
    }

    void Clear(const std::uint32_t submesh_index) {
        if (submesh_index < submeshes.size()) {
            submeshes[submesh_index] = std::nullopt;
        }
    }

    void ClearAll() {
        submeshes.clear();
        reported_signatures.clear();
    }

    [[nodiscard]] bool Has(const std::uint32_t submesh_index) const {
        return submesh_index < submeshes.size() && submeshes[submesh_index].has_value();
    }

    [[nodiscard]] SubMeshTransform Get(const std::uint32_t submesh_index) const {
        return Has(submesh_index) ? *submeshes[submesh_index] : SubMeshTransform{};
    }
};

// What the resolver did, so the caller can report the cases that would
// otherwise be silent. The resolver itself does not log: it stays a pure,
// device-free function that a test can assert on directly.
enum class TransformOverrideStatus : std::uint8_t {
    NotPresent,    // no override component on the entity
    Applied,       // the override supplied this submesh's transform
    AssetDefault,  // override present, this index unset
    MeshMismatch,  // authored for another mesh: the whole override is ignored
    RangeTooLarge, // more entries than the mesh has submeshes: ignored
};

struct EffectiveSubmeshTransform {
    SubMeshTransform transform{};
    TransformOverrideStatus status{TransformOverrideStatus::NotPresent};
};

// Resolves the placement of one object-local submesh and composes it with the
// entity transform. `asset_local` is the mesh asset's own transform for that
// submesh; `mesh_id` and `submesh_count` describe the owning mesh. Never throws;
// any inconsistency falls back to the asset transform.
[[nodiscard]] inline EffectiveSubmeshTransform ResolveEffectiveSubmeshTransform(
    const SubMeshTransform& entity,
    const SubMeshTransform& asset_local,
    TransformOverride* override_component,
    const std::uint32_t submesh_index,
    const std::uint32_t mesh_id,
    const std::uint32_t submesh_count) noexcept {
    if (override_component == nullptr) {
        return {ComposeSubmeshTransform(entity, asset_local), TransformOverrideStatus::NotPresent};
    }

    if (override_component->mesh_id == TransformOverride::kUnboundMesh) {
        override_component->mesh_id = mesh_id;
    }
    if (override_component->mesh_id != mesh_id) {
        return {ComposeSubmeshTransform(entity, asset_local), TransformOverrideStatus::MeshMismatch};
    }
    if (override_component->submeshes.size() > submesh_count) {
        return {ComposeSubmeshTransform(entity, asset_local), TransformOverrideStatus::RangeTooLarge};
    }
    if (override_component->Has(submesh_index)) {
        return {ComposeSubmeshTransform(entity, override_component->Get(submesh_index)),
                TransformOverrideStatus::Applied};
    }
    return {ComposeSubmeshTransform(entity, asset_local), TransformOverrideStatus::AssetDefault};
}

} // namespace VulkanEngine::Components
