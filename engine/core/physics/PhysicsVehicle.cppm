// Wheeled vehicles, as an engine-owned vocabulary.
//
// A `WheeledVehicle` is a `JPH::VehicleConstraint` + `WheeledVehicleController`
// behind a pimpl: this interface names no Jolt type, and a caller describes a
// vehicle with POD data (`VehicleDesc`) and drives it with one input call.
//
// ## Stepping
//
// Jolt's `VehicleConstraint` is itself a `PhysicsStepListener`: its `OnStep`
// runs the suspension/tire update *before* the solver, once per collision step.
// `Initialize` therefore registers the constraint with the physics system twice
// — once as a constraint and once as a step listener — and `Shutdown` removes
// both. Nothing else has to be called each frame; the world's `Step()` drives
// the vehicle.
//
// ## Lifetime
//
// The vehicle owns its Jolt constraint and collision tester. A `WheeledVehicle`
// must be shut down (or destroyed) **before** the `World` it was created on and
// before its chassis body is destroyed — a vehicle constraint references the
// chassis and must be removed first, exactly like any other constraint. This
// mirrors the engine's existing rule that a body outlives the handles into it.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Vehicle;

import std;

import VulkanEngine.Physics.World;

export import VulkanEngine.Physics.Types;

export namespace VulkanEngine::Physics {

/// One wheel of a vehicle, in the chassis body's local space. Every vector is a
/// direction in that space (`suspension_direction` points down, `steering_axis`
/// up, `wheel_up`/`wheel_forward` define the wheel's neutral frame).
struct WheelDesc {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 suspension_direction{0.0f, -1.0f, 0.0f};
    glm::vec3 steering_axis{0.0f, 1.0f, 0.0f};
    glm::vec3 wheel_up{0.0f, 1.0f, 0.0f};
    glm::vec3 wheel_forward{0.0f, 0.0f, 1.0f};
    float radius = 0.3f;
    float width = 0.1f;
    float suspension_min_length = 0.3f;
    float suspension_max_length = 0.5f;
    float suspension_frequency = 1.5f;
    float suspension_damping = 0.5f;
    /// Steering range in radians. `0` makes the wheel non-steering (the usual
    /// choice for rear wheels).
    float max_steer_angle = 0.0f;
    float max_brake_torque = 1500.0f;
    float max_handbrake_torque = 0.0f;
    /// Tire grip, in the scalar slip scale older Jolt exposed as
    /// `WheelSettings::mFrictionSlip` (default 8). Jolt 5.6 replaced that
    /// scalar with friction-vs-slip curves; this value rescales the shipped
    /// curves proportionally, so 8 reproduces Jolt's default grip and every
    /// other value moves peak grip linearly (6 is 25% less grip than stock).
    /// Higher grips harder. This is the tire's own friction, applied against
    /// the contact body's friction by the constraint — unrelated to
    /// body-vs-body contact friction.
    float friction_slip = 1.0f;
    float sideways_friction_slip = 0.8f;
    float inertia = 0.9f;
    float angular_damping = 0.2f;
};

/// An anti-roll bar linking two wheels.
struct AntiRollBarDesc {
    std::uint32_t left_wheel = 0;
    std::uint32_t right_wheel = 1;
    float stiffness = 1000.0f;
};

/// One differential: how engine torque is split between a left and a right
/// wheel. `left_wheel`/`right_wheel` are indices into `VehicleDesc::wheels`;
/// `-1` means "no wheel on this side" (a one-wheel differential).
struct DifferentialDesc {
    std::int32_t left_wheel = 0;
    std::int32_t right_wheel = 1;
    float differential_ratio = 3.42f;
    float left_right_split = 0.5f;
    float engine_torque_ratio = 1.0f;
};

/// A wheeled vehicle as data.
struct VehicleDesc {
    PhysicsBodyId chassis{};
    /// Chassis-local up and forward axes; Jolt expects unit vectors.
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    glm::vec3 forward{0.0f, 0.0f, 1.0f};
    /// Maximum pitch/roll before the vehicle is righted; `pi` disables the
    /// limit. A real car wants something well under `pi`.
    float max_pitch_roll_angle = 0.5f;

    std::vector<WheelDesc> wheels;

    float engine_max_torque = 500.0f;
    float engine_min_rpm = 1000.0f;
    float engine_max_rpm = 6000.0f;

    /// Empty means "infer": one differential per pair of wheels (0/1, 2/3, …),
    /// each taking an equal share of the torque.
    std::vector<DifferentialDesc> differentials;
    std::vector<AntiRollBarDesc> anti_roll_bars;

    /// Object layer the wheel collision query may hit — the ground, normally.
    CollisionLayer wheel_collision_layer = CollisionLayer::Default;
};

/// A vehicle controller: Jolt's `VehicleConstraint` and wheeled controller,
/// owned by this object and registered with the world's physics system.
class WheeledVehicle {
public:
    WheeledVehicle();
    ~WheeledVehicle();

    WheeledVehicle(const WheeledVehicle&) = delete;
    WheeledVehicle& operator=(const WheeledVehicle&) = delete;
    WheeledVehicle(WheeledVehicle&&) = delete;
    WheeledVehicle& operator=(WheeledVehicle&&) = delete;

    /// Creates the constraint, builds the collision tester and registers the
    /// vehicle with `world` as both a constraint and a step listener. Returns
    /// false (with a log) for an uninitialized world, a dead chassis, or a
    /// description with no wheels or an unresolvable differential.
    bool Initialize(World& world, const VehicleDesc& desc);

    /// Removes the vehicle from the physics system. Idempotent; must run before
    /// the chassis body or the world is destroyed.
    void Shutdown();

    [[nodiscard]] bool IsInitialized() const noexcept { return impl_ != nullptr; }

    /// `forward` and `right` are in [-1, 1], `brake` and `handbrake` in [0, 1].
    /// Stored on the controller and applied on the next `World::Step()`.
    void SetDriverInput(float forward, float right, float brake, float handbrake = 0.0f);

    /// The chassis' speed along its own forward axis, in m/s. Negative is
    /// reversing.
    [[nodiscard]] float GetSpeed() const;

    [[nodiscard]] std::uint32_t WheelCount() const noexcept;

    /// World pose of a wheel's center, without its spin. The rotation maps the
    /// engine's wheel model space — the spin axle is model +y, wheel-up model
    /// +x, wheel-forward model +z; the render meshes are authored against the
    /// same convention — to world space, steering and chassis orientation
    /// included. Re-apply the accumulated spin from `GetWheelRotationAngle`
    /// about that +y axle (negated, post-multiplied) to render:
    /// `rotation * angleAxis(-GetWheelRotationAngle(i), {0, 1, 0})`.
    [[nodiscard]] BodyPose GetWheelPose(std::uint32_t index) const;
    /// Accumulated spin of the wheel, radians.
    [[nodiscard]] float GetWheelRotationAngle(std::uint32_t index) const;
    /// Current steering angle of the wheel, radians.
    [[nodiscard]] float GetWheelSteerAngle(std::uint32_t index) const;
    /// Contact the wheel currently rests on, or an invalid id when it is in the
    /// air.
    [[nodiscard]] PhysicsBodyId GetWheelContactBody(std::uint32_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace VulkanEngine::Physics
