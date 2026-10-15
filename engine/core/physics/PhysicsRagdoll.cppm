// Ragdolls, built programmatically, behind a Jolt-free interface.
//
// A `Ragdoll` is a `JPH::Ragdoll`: an ordered list of rigid parts (one capsule
// each) with a joint from the constraint vocabulary connecting every part to its
// parent. It is built from `RagdollDesc` data — no ObjectStream asset is
// involved — and Jolt's `RagdollSettings::Stabilize` mass/inertia pass plus the
// parent/child collision filter are applied during `Initialize`.
//
// ## Anchors and frames
//
// Like `ConstraintDesc`, ragdoll joint anchors (`joint_anchor`) and axes are
// **world space**, evaluated at creation: a joint is placed where two parts
// meet, which is what a programmatic builder has. Part `position`/`rotation` are
// the part body's center-of-mass frame at spawn.
//
// ## Lifetime
//
// The ragdoll owns its Jolt bodies, constraints and skeleton settings, and
// registers them with the world's physics system. It must be shut down (or
// destroyed) before the `World` is. `RemoveFromPhysicsSystem`/
// `AddToPhysicsSystem` suspend and resume the ragdoll without destroying it.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Ragdoll;

import std;

import VulkanEngine.Physics.World;

export import VulkanEngine.Physics.Types;
export import VulkanEngine.Physics.Constraints;

export namespace VulkanEngine::Physics {

/// One rigid part of a ragdoll: a capsule plus where it starts and which part
/// its joint attaches to.
struct RagdollPartDesc {
    /// Capsule dimensions, in the engine's `ShapeDesc` convention: `height` is
    /// the full capsule height, so the cylindrical half is `height/2 - radius`.
    float radius = 0.2f;
    float height = 1.0f;
    /// Center-of-mass pose of the part at spawn.
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    /// Index of the parent part, or `-1` for the root. Parents must appear
    /// before their children, which is the skeleton's own ordering rule.
    std::int32_t parent = -1;
    /// Joint vocabulary entry connecting this part to its parent. Ignored for
    /// the root.
    ConstraintType joint = ConstraintType::Point;
    /// World-space point both bodies' joint frames are placed at (the shared
    /// articulation point). Required for every non-root part.
    glm::vec3 joint_anchor{0.0f, 0.0f, 0.0f};
    /// World-space axes for `Hinge`/`Slider`; `normal_axis` must be
    /// perpendicular to `axis`.
    glm::vec3 joint_axis{0.0f, 1.0f, 0.0f};
    glm::vec3 joint_normal_axis{1.0f, 0.0f, 0.0f};
    bool use_limits = false;
    float limit_min = 0.0f;
    float limit_max = 0.0f;

    BodyMotionType motion = BodyMotionType::Dynamic;
    CollisionLayer layer = CollisionLayer::Default;
    /// Explicit mass in kg; `0` derives it from the capsule's density, which is
    /// then equalized by Jolt's stabilization pass.
    float mass = 0.0f;
    float friction = 0.5f;
    float restitution = 0.0f;
};

/// A ragdoll as data.
struct RagdollDesc {
    std::vector<RagdollPartDesc> parts;
    /// Free payload stamped on every part body.
    std::uint64_t user_data = 0;
    /// Collision-group id shared by the ragdoll's parts. Distinct live ragdolls
    /// should use distinct ids so their parts only filter against their own
    /// parent/child pairs.
    std::uint32_t collision_group = 1;
    /// Disables collisions between each part and its parent, which is what makes
    /// a chain of touching capsules behave.
    bool disable_parent_child_collisions = true;
    /// Let the ragdoll fall asleep once it settles.
    bool allow_sleeping = true;
};

/// A programmatically built ragdoll.
class Ragdoll {
public:
    Ragdoll();
    ~Ragdoll();

    Ragdoll(const Ragdoll&) = delete;
    Ragdoll& operator=(const Ragdoll&) = delete;
    Ragdoll(Ragdoll&&) = delete;
    Ragdoll& operator=(Ragdoll&&) = delete;

    /// Builds the parts, skeleton and joints, stabilizes them, and adds the
    /// ragdoll to `world`'s physics system. Returns false (with a log) for an
    /// uninitialized world, an empty or malformed part list, or a Jolt failure.
    bool Initialize(World& world, const RagdollDesc& desc);

    /// Removes the ragdoll from the physics system and releases its Jolt state.
    /// Idempotent; must run before the `World` is destroyed.
    void Shutdown();

    [[nodiscard]] bool IsInitialized() const noexcept { return impl_ != nullptr; }

    /// Re-adds a removed ragdoll to the physics system (no-op when already
    /// added).
    void AddToPhysicsSystem();
    /// Removes the ragdoll's bodies and constraints from the physics system
    /// without destroying it (no-op when already removed).
    void RemoveFromPhysicsSystem();
    [[nodiscard]] bool IsInPhysicsSystem() const noexcept;

    [[nodiscard]] std::uint32_t PartCount() const noexcept;
    /// Physics body of one part.
    [[nodiscard]] PhysicsBodyId PartBody(std::uint32_t index) const;

    [[nodiscard]] BodyPose GetPartPose(std::uint32_t index) const;
    void SetPartPose(std::uint32_t index, const BodyPose& pose);
    /// Sets every part pose at once; `poses` must have `PartCount()` entries.
    void SetPose(std::span<const BodyPose> poses);

    /// True while any part is awake.
    [[nodiscard]] bool IsActive() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace VulkanEngine::Physics
