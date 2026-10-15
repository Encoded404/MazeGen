// Component <-> body bridge.
//
// The ECS holds a handle; the physics world owns the body. This module is the
// only place that knows both, and it is the seam the frame contract's three
// physics slots run through:
//
//   PhysicsInput     — authored `Velocity`/`Transform` -> body
//   PhysicsSimulate  — one blocking world step (registered by GameEngine)
//   PhysicsWriteBack — body pose/velocity -> `Transform`/`Velocity`
//
// Attachment happens through the registry's OnAdd/OnRemove hooks, which is the
// engine's ratified seam for owning a resource from a data component. A body
// carries the owning `EntityId` (packed) in its user data, so a contact or a
// query can resolve to a live entity and a stale handle resolves to nothing.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.BodyBridge;

import std;

import VulkanEngine.ECS.Entity;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Physics.World;

export import VulkanEngine.Physics.Types;
export import VulkanEngine.Physics.Components;

export namespace VulkanEngine::Physics {

/// Packs an entity handle into a body's 64-bit user data.
///
/// Both halves are kept, so the value is generational: if the entity is destroyed
/// and its slot reused, the packed id of the old entity no longer resolves.
[[nodiscard]] constexpr std::uint64_t PackEntityId(EntityId id) noexcept {
    return (static_cast<std::uint64_t>(id.generation) << 32) | static_cast<std::uint64_t>(id.index);
}

/// Inverse of `PackEntityId`. A zero payload unpacks to an invalid entity.
[[nodiscard]] constexpr EntityId UnpackEntityId(std::uint64_t packed) noexcept {
    return EntityId{static_cast<std::uint32_t>(packed & 0xffffffffULL),
                    static_cast<std::uint32_t>(packed >> 32)};
}

/// Creates the body for `entity` from `body` and records its handle.
/// Idempotent: an already-attached body is left alone, so re-adding the
/// component (or a hook firing twice) cannot leak a body.
void AttachBody(World& world, Entity& entity, RigidBody& body);

/// Destroys the attached body and clears the handle.
void DetachBody(World& world, RigidBody& body);

/// Recreates the attached body with `RigidBody`'s own shape plus `children`,
/// preserving the live pose, velocity and activation state. Called when a
/// `Collider` is added or removed, because Jolt shapes are immutable once a body
/// owns them.
///
/// The children are passed in rather than looked up: an `OnAdd<Collider>` hook
/// runs before the new component's mask bit is set, so a lookup would find
/// nothing. An empty span is how a removed collider asks for the single-shape
/// body back.
void RebuildBody(World& world, Entity& entity, RigidBody& body,
                 std::span<const ShapeDesc> children);

/// `SystemSlot::PhysicsInput`: pushes authored movement into bodies. Kinematic
/// bodies follow their `Transform` (a non-zero authored `Velocity` wins, which is
/// how a conveyor or a constant-velocity platform is expressed); dynamic bodies
/// are driven only through the typed body API, so nothing here fights the solver.
void ApplyInputs(World& world, ComponentRegistry& registry, float delta_time);

/// `SystemSlot::PhysicsWriteBack`: copies body output into `Transform` (the
/// gameplay-truth fixed-step pose) and `Velocity` (the read channel). Runs after
/// the simulation, so anything constrained After this slot reads post-step state.
void WriteBack(World& world, ComponentRegistry& registry, float delta_time);

} // namespace VulkanEngine::Physics
