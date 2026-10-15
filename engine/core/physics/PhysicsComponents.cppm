// Physics data components.
//
// Both are dense, sparse-set data components with no behaviour: physics work is
// driven by the frame contract's systems, not by a component that updates
// itself. `RigidBody` holds an opaque handle, never an owner — the physics world
// owns the body, and releases it through the registry's OnRemove hook.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Components;

import std;

export import VulkanEngine.Physics.Types;

export namespace VulkanEngine::Physics {

/// A rigid body attached to an entity.
///
/// The body is created from this description when the component is added, using
/// the entity's `Transform` for its initial pose; the handle is written back into
/// `body`. An invalid handle means "not attached" — a body that could not be
/// created (bad shape, body cap reached) leaves the component in that state
/// rather than failing the entity.
///
/// It is acted on only through `Physics::World`'s typed API (`AddForce`,
/// `AddImpulse`, `SetLinearVelocity`, …), never by writing fields after
/// attachment: changing `shape`/`motion`/`mass` here has no effect until the
/// component is removed and re-added.
struct RigidBody {
    PhysicsBodyId body{}; // NOLINT(misc-non-private-member-variables-in-classes)
    ShapeDesc shape{}; // NOLINT(misc-non-private-member-variables-in-classes)
    CollisionLayer layer = CollisionLayer::Default; // NOLINT(misc-non-private-member-variables-in-classes)
    BodyMotionType motion = BodyMotionType::Static; // NOLINT(misc-non-private-member-variables-in-classes)
    /// Explicit mass in kg; `0` derives it from the shape's density.
    float mass = 1.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    float friction = 0.5f; // NOLINT(misc-non-private-member-variables-in-classes)
    float restitution = 0.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    float linear_damping = 0.05f; // NOLINT(misc-non-private-member-variables-in-classes)
    float angular_damping = 0.05f; // NOLINT(misc-non-private-member-variables-in-classes)
    bool ccd = false; // NOLINT(misc-non-private-member-variables-in-classes)
    bool sensor = false; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// The projected velocity channel for gameplay.
///
/// The write-back pass writes the body's live velocity here every step. Reading
/// it is how a game asks "how fast is this thing moving". Writing it is meaningful
/// only for a *kinematic* body, where it becomes that body's driven velocity; a
/// dynamic body's velocity is set through `World::SetLinearVelocity` /
/// `AddImpulse`, which wake it correctly. A sleeping dynamic body is woken by
/// those calls too, so there is no separate "kick" path.
struct Velocity {
    glm::vec3 linear{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::vec3 angular{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// Additional shapes attached to an entity's body.
///
/// With `RigidBody`'s own shape these form one compound body, so one entity can be
/// one collider assembled from several pieces (a table as a top plus four legs).
/// Each child is placed in the body frame by its own position/rotation/scale.
///
/// Jolt's sensor flag is per *body*, not per child shape, so a trigger volume is
/// its own entity with `RigidBody::sensor` set rather than a child here.
///
/// Adding or removing a `Collider` rebuilds the body: Jolt shapes are immutable
/// once a body owns them, so the body is recreated at its current pose with the
/// new shape set. `RigidBody`'s handle therefore changes when this component
/// changes, and code must re-read it (never cache the handle across a rebuild).
struct Collider {
    std::vector<ShapeDesc> children{}; // NOLINT(misc-non-private-member-variables-in-classes)
};

} // namespace VulkanEngine::Physics
