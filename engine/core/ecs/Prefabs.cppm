// Prefabs: first-class composed entity bundles (rung 3 of engine modularization).
//
// Entity construction that composes standard component bundles — transform +
// renderable + physics body + character — used to be factory code re-written
// per game (the runner's cube spawner, every example's spawn list). A prefab is
// content composition through the seams that already exist: plain data
// components added to one entity, with the engine-owned resources (physics
// bodies, streamed meshes) attached by the same OnAdd/OnRemove registry hooks
// the modules own. No new component category, no engine edit.
//
// The physics half follows the ratified contract exactly: RigidBody/Character/
// Collider are pure data; attach/detach happens through hooks; mutations from
// callbacks stay deferred-only. Adding the optional bundles in any order yields
// the same entity (collider rebuilds make body add-order irrelevant).

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.ECS.Prefabs;

import std;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Components.Transform;
import VulkanEngine.Components.MeshReference;

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.Components;
import VulkanEngine.Physics.Character;
#endif

export namespace VulkanEngine::Prefabs {

/// "No mesh" marker for Prefab::mesh_id: the same value MeshReference's unset
/// state carries.
inline constexpr std::uint32_t kNoMesh = std::numeric_limits<std::uint32_t>::max();

/// A composed entity bundle. Everything is optional; a Prefab that names
/// nothing but a pose is a plain Transform entity.
struct Prefab {
    std::string_view name = {};   // diagnostics only; static storage duration
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    /// Renderable bundle: a MeshRegistry id (the entity draws that mesh; the
    /// material state lives in the mesh's submeshes, and the gather pass reads
    /// the entity's own Transform).
    std::uint32_t mesh_id = kNoMesh;
#ifdef VKENGINE_PHYSICS
    std::optional<Physics::RigidBody> rigid_body;
    std::optional<Physics::Collider> collider;
    std::optional<Physics::Character> character;
    std::optional<Physics::Velocity> velocity;
#endif
};

/// Spawn one entity from `prefab`. Returns the persistent EntityId. The bundle
/// order (body before or after collider) is irrelevant by contract: the
/// collider hooks rebuild the body either way.
[[nodiscard]] inline Entity& Spawn(ComponentRegistry& registry, const Prefab& prefab) {
    Entity& entity = registry.CreateEntity();

    // Transform first: the physics attach hook places the body at the
    // entity's authored pose.
    Components::Transform& transform = registry.AddComponent<Components::Transform>(entity);
    transform.position = prefab.position;
    transform.rotation = prefab.rotation;
    transform.scale = prefab.scale;

#ifdef VKENGINE_PHYSICS
    if (prefab.velocity.has_value()) {
        registry.AddComponent<Physics::Velocity>(entity, *prefab.velocity);
    }
    // Body then collider: the body's OnAdd hook already gathers a collider's
    // children (either add order yields the same compound body).
    if (prefab.rigid_body.has_value()) {
        registry.AddComponent<Physics::RigidBody>(entity, *prefab.rigid_body);
    }
    if (prefab.collider.has_value()) {
        registry.AddComponent<Physics::Collider>(entity, *prefab.collider);
    }
    if (prefab.character.has_value()) {
        registry.AddComponent<Physics::Character>(entity, *prefab.character);
    }
#endif

    if (prefab.mesh_id != kNoMesh) {
        registry.AddComponent<Components::MeshReference>(entity).loaded_mesh_id = prefab.mesh_id;
    }
    return entity;
}

} // namespace VulkanEngine::Prefabs