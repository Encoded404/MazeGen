// Joints between bodies, as an engine-owned vocabulary.
//
// This interface names no Jolt type: a caller describes a joint with POD data
// and gets an opaque, generation-checked handle back. The Jolt constraint
// objects, the collision filter that implements `enable_collision`, and the
// per-world registry all live in `PhysicsConstraints.cpp`.
//
// ## Anchor space — one choice, documented
//
// `ConstraintDesc` anchors (`point_a`, `point_b`) and axes (`axis`,
// `normal_axis`) are **world space**, evaluated once at creation. This is the
// space a gameplay caller has: a joint is placed where two things meet, not in
// either body's local frame, and the caller does not know a body's center of
// mass (which is what Jolt's `LocalToBodyCOM` would require). Jolt converts the
// world-space frame to body-local COM at `Create`, so the joint then tracks the
// bodies correctly.
//
// ## Ownership
//
// State is owned by the world (see `detail::WorldExtension`): the registry is
// created on first use, attached to the `World`, and torn down from
// `World::Shutdown` *before* the physics system and its bodies. A constraint
// therefore never outlives its bodies. `DestroyConstraintsFor` is the hook the
// body bridge uses to release a body's joints first; a constraint whose bodies
// are already dead is rejected at creation and can never be made.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Constraints;

import std;

import VulkanEngine.Physics.World;

export import VulkanEngine.Physics.Types;

export namespace VulkanEngine::Physics {

/// The joint vocabulary. `Fixed` welds two bodies at their current relative
/// pose; `Point` pins two anchor points together; `Hinge` allows one rotation
/// axis; `Slider` allows one translation axis; `Distance` holds two points at a
/// distance range.
enum class ConstraintType : std::uint8_t {
    Fixed = 0,
    Point = 1,
    Hinge = 2,
    Slider = 3,
    Distance = 4,
};

/// Diagnostic spelling of a constraint type (failure messages and debug UI).
[[nodiscard]] constexpr std::string_view ConstraintTypeName(ConstraintType type) noexcept {
    switch (type) {
        case ConstraintType::Fixed:    return "fixed";
        case ConstraintType::Point:    return "point";
        case ConstraintType::Hinge:    return "hinge";
        case ConstraintType::Slider:   return "slider";
        case ConstraintType::Distance: return "distance";
    }
    return "<unknown>";
}

/// Opaque, generation-checked handle to a constraint, shaped like every other
/// engine handle: a reused slot is a different id, so a stale handle can never
/// name the joint that replaced it.
struct ConstraintId {
    std::uint32_t slot = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t generation = 0; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] constexpr bool IsValid() const noexcept { return generation != 0; }

    friend constexpr bool operator==(const ConstraintId&, const ConstraintId&) = default;
};

/// Motor drive for `Hinge` and `Slider`. `Velocity` drives at
/// `SetTargetVelocity`; `Position` drives toward the limit-clamped target the
/// constraint's own settings imply. Ignored by the other types.
enum class MotorState : std::uint8_t {
    Off = 0,
    Velocity = 1,
    Position = 2,
};

/// A joint as data. See the file header for the world-space anchor contract.
struct ConstraintDesc {
    ConstraintType type = ConstraintType::Fixed;
    PhysicsBodyId body_a{};
    PhysicsBodyId body_b{};
    /// World-space anchor on body A / body B. `Point` and `Distance` hold these
    /// two points together / at a distance; `Hinge` and `Slider` place the joint
    /// frame at them (set both to the same point for a joint at one location).
    /// `Fixed` ignores them: a weld is defined by the bodies' current relative
    /// pose.
    glm::vec3 point_a{0.0f, 0.0f, 0.0f};
    glm::vec3 point_b{0.0f, 0.0f, 0.0f};
    /// World-space axis for `Hinge` (rotation axis) and `Slider` (slide axis).
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    /// World-space secondary axis for `Hinge`/`Slider`, perpendicular to `axis`;
    /// it fixes the joint's zero orientation (hinge) and its frame (slider).
    glm::vec3 normal_axis{1.0f, 0.0f, 0.0f};
    /// When false the type's natural limit applies: `Hinge` `[-pi, pi]`,
    /// `Slider` unbounded, `Distance` the creation-time separation. When true,
    /// `limit_min`/`limit_max` are used: radians for `Hinge`, metres for
    /// `Slider`/`Distance`. Ignored by `Fixed` and `Point`.
    bool use_limits = false;
    float limit_min = 0.0f;
    float limit_max = 0.0f;
    /// When false, the two bodies stop colliding with each other (implemented
    /// with a per-world collision filter, so a body may carry disabled pairs
    /// with several partners). This is a creation-time property: there is no
    /// setter, because Jolt's broadphase resolves the pair on the next step.
    bool enable_collision = true;
};

/// Creates the joint. Both bodies must be live and distinct; a dead body, a
/// self-joint, a shape-less handle or a world that is not initialized returns an
/// invalid id (with a log). The registry is created lazily and owned by `world`.
[[nodiscard]] ConstraintId CreateConstraint(World& world, const ConstraintDesc& desc);

/// Removes the joint from the physics system and frees its slot. Releasing an
/// invalid or stale handle is a no-op.
void DestroyConstraint(World& world, ConstraintId constraint);

[[nodiscard]] bool IsConstraintValid(World& world, ConstraintId constraint);

/// Live joints the world owns.
[[nodiscard]] std::size_t ConstraintCount(const World& world);

/// Releases every joint that names `body`. The body bridge calls this before
/// destroying a body, because Jolt requires constraints to be removed first.
void DestroyConstraintsFor(World& world, PhysicsBodyId body);

/// Limits for a `Hinge` (radians), `Slider` or `Distance` (metres). Returns
/// false for a stale handle or a type without limits.
[[nodiscard]] bool GetConstraintLimits(World& world, ConstraintId constraint, float& out_min,
                                       float& out_max);
bool SetConstraintLimits(World& world, ConstraintId constraint, float limit_min, float limit_max);

/// Motor drive for a `Hinge`/`Slider`. Returns false for a stale handle or any
/// other type.
bool SetMotorState(World& world, ConstraintId constraint, MotorState state);
/// Target speed for a `Velocity` motor: rad/s for a hinge, m/s for a slider.
bool SetTargetVelocity(World& world, ConstraintId constraint, float velocity);

/// The joint's current coordinate: a hinge's angle in radians, a slider's
/// position in metres. Zero for any other type or a stale handle.
[[nodiscard]] float GetConstraintPosition(World& world, ConstraintId constraint);

} // namespace VulkanEngine::Physics
