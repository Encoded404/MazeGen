// Jolt wheeled-vehicle construction and driving.
//
// The interface (`PhysicsVehicle.cppm`) names no Jolt type; this file builds the
// `VehicleConstraint`, its wheels and the `WheeledVehicleController`, and wires
// the constraint into the physics system as both a constraint and a step
// listener.

module;

#include <logging/logging_macros.hpp>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Jolt/Physics/Vehicle/VehicleTransmission.h>

module VulkanEngine.Physics.Vehicle;

import std;

import logiface;

import VulkanEngine.Physics.World;
import VulkanEngine.Physics.Types;

namespace VulkanEngine::Physics {

namespace {

[[nodiscard]] JPH::Vec3 ToVec3(const glm::vec3& value) {
    return JPH::Vec3(value.x, value.y, value.z);
}
[[nodiscard]] glm::vec3 FromRVec3(JPH::RVec3Arg value) {
    return glm::vec3(static_cast<float>(value.GetX()), static_cast<float>(value.GetY()),
                     static_cast<float>(value.GetZ()));
}
[[nodiscard]] glm::quat FromQuat(const JPH::Quat& value) {
    return glm::quat(value.GetW(), value.GetX(), value.GetY(), value.GetZ());
}
[[nodiscard]] JPH::BodyID ToJoltBodyId(PhysicsBodyId body) noexcept {
    return JPH::BodyID(body.value);
}
[[nodiscard]] PhysicsBodyId FromJoltBodyId(const JPH::BodyID& body) noexcept {
    return PhysicsBodyId{body.GetIndexAndSequenceNumber()};
}

/// Jolt asserts on non-unit direction vectors, so a zero or approximate vector
/// is normalized here rather than crashing the constraint constructor.
[[nodiscard]] glm::vec3 NormalizeOr(const glm::vec3& value, const glm::vec3& fallback) {
    const float length = glm::length(value);
    return length > 1.0e-6f ? value / length : fallback;
}

} // namespace

struct WheeledVehicle::Impl {
    World* world = nullptr;
    PhysicsBodyId chassis{};
    JPH::Ref<JPH::VehicleConstraint> constraint;
    JPH::Ref<JPH::VehicleCollisionTester> tester;
    // Borrowed: the constraint owns its controller.
    JPH::WheeledVehicleController* controller = nullptr;

    [[nodiscard]] JPH::PhysicsSystem* System() const {
        return world != nullptr ? static_cast<JPH::PhysicsSystem*>(world->NativePhysicsSystem())
                                : nullptr;
    }
};

WheeledVehicle::WheeledVehicle() = default;

WheeledVehicle::~WheeledVehicle() {
    Shutdown();
}

bool WheeledVehicle::Initialize(World& world, const VehicleDesc& desc) {
    if (impl_ != nullptr) {
        LOGIFACE_LOG(error, "physics: vehicle is already initialized; shut it down first");
        return false;
    }
    if (!world.IsInitialized()) {
        LOGIFACE_LOG(error, "physics: cannot create a vehicle on an uninitialized world");
        return false;
    }
    JPH::PhysicsSystem* system = static_cast<JPH::PhysicsSystem*>(world.NativePhysicsSystem());
    if (system == nullptr || !desc.chassis.IsValid() || !world.IsBodyValid(desc.chassis)) {
        LOGIFACE_LOG(error, "physics: vehicle chassis is not a live body");
        return false;
    }
    if (desc.wheels.empty()) {
        LOGIFACE_LOG(error, "physics: a vehicle needs at least one wheel");
        return false;
    }

    const JPH::BodyLockInterface& lock = system->GetBodyLockInterface();
    JPH::Body* chassis = lock.TryGetBody(ToJoltBodyId(desc.chassis));
    if (chassis == nullptr) {
        LOGIFACE_LOG(error, "physics: vehicle chassis is not a body Jolt knows");
        return false;
    }

    JPH::VehicleConstraintSettings settings;
    settings.mUp = ToVec3(NormalizeOr(desc.up, {0.0f, 1.0f, 0.0f}));
    settings.mForward = ToVec3(NormalizeOr(desc.forward, {0.0f, 0.0f, 1.0f}));
    settings.mMaxPitchRollAngle = desc.max_pitch_roll_angle;

    const auto wheel_count = static_cast<std::uint32_t>(desc.wheels.size());
    for (const WheelDesc& wheel : desc.wheels) {
        auto* jolt_wheel = new JPH::WheelSettingsWV;
        jolt_wheel->mPosition = ToVec3(wheel.position);
        jolt_wheel->mSuspensionDirection =
            ToVec3(NormalizeOr(wheel.suspension_direction, {0.0f, -1.0f, 0.0f}));
        jolt_wheel->mSteeringAxis = ToVec3(NormalizeOr(wheel.steering_axis, {0.0f, 1.0f, 0.0f}));
        jolt_wheel->mWheelUp = ToVec3(NormalizeOr(wheel.wheel_up, {0.0f, 1.0f, 0.0f}));
        jolt_wheel->mWheelForward =
            ToVec3(NormalizeOr(wheel.wheel_forward, {0.0f, 0.0f, 1.0f}));
        jolt_wheel->mRadius = wheel.radius;
        jolt_wheel->mWidth = wheel.width;
        jolt_wheel->mSuspensionMinLength = wheel.suspension_min_length;
        jolt_wheel->mSuspensionMaxLength = wheel.suspension_max_length;
        jolt_wheel->mSuspensionSpring.mFrequency = wheel.suspension_frequency;
        jolt_wheel->mSuspensionSpring.mDamping = wheel.suspension_damping;
        jolt_wheel->mMaxSteerAngle = wheel.max_steer_angle;
        jolt_wheel->mMaxBrakeTorque = wheel.max_brake_torque;
        jolt_wheel->mMaxHandBrakeTorque = wheel.max_handbrake_torque;
        // Jolt 5.6 models grip as friction-vs-slip curves (y: friction 0..~1.2,
        // x: slip ratio / slip angle); the scalar `friction_slip` vocabulary
        // rescales those curves so 8 is the shipped default and other values
        // move peak grip linearly. Every point's y scales, including the
        // shared zero, so the curve's shape survives.
        const float friction_scale = wheel.friction_slip;
        const float sideways_friction_scale = wheel.sideways_friction_slip;
        if (friction_scale != 1.0f) {
            for (auto& point : jolt_wheel->mLongitudinalFriction.mPoints) {
                point.mY *= friction_scale;
            }
            for (auto& point : jolt_wheel->mLateralFriction.mPoints) {
                point.mY *= sideways_friction_scale;
            }
        }
        jolt_wheel->mInertia = wheel.inertia;
        jolt_wheel->mAngularDamping = wheel.angular_damping;
        settings.mWheels.push_back(JPH::Ref<JPH::WheelSettings>(jolt_wheel));
    }

    auto* controller = new JPH::WheeledVehicleControllerSettings;
    controller->mEngine.mMaxTorque = desc.engine_max_torque;
    controller->mEngine.mMinRPM = desc.engine_min_rpm;
    controller->mEngine.mMaxRPM = desc.engine_max_rpm;

    // Differentials: explicit, or inferred as adjacent left/right pairs sharing
    // the engine torque equally (the common four-wheeled layout).
    std::vector<DifferentialDesc> differentials = desc.differentials;
    if (differentials.empty()) {
        for (std::uint32_t i = 0; i + 1 < wheel_count; i += 2) {
            DifferentialDesc inferred{};
            inferred.left_wheel = static_cast<std::int32_t>(i);
            inferred.right_wheel = static_cast<std::int32_t>(i + 1);
            inferred.engine_torque_ratio = 1.0f;
            differentials.push_back(inferred);
        }
        for (DifferentialDesc& differential : differentials) {
            differential.engine_torque_ratio = 1.0f / static_cast<float>(differentials.size());
        }
    }
    if (differentials.empty()) {
        LOGIFACE_LOG(error, "physics: a vehicle needs at least one differential");
        delete controller;
        return false;
    }
    controller->mDifferentials.resize(differentials.size());
    for (std::size_t i = 0; i < differentials.size(); ++i) {
        const DifferentialDesc& source = differentials[i];
        const auto in_range = [wheel_count](std::int32_t index) {
            return index == -1 || (index >= 0 && static_cast<std::uint32_t>(index) < wheel_count);
        };
        if (!in_range(source.left_wheel) || !in_range(source.right_wheel)) {
            LOGIFACE_LOG(error, "physics: differential references a wheel that does not exist");
            delete controller;
            return false;
        }
        JPH::VehicleDifferentialSettings& target = controller->mDifferentials[i];
        target.mLeftWheel = source.left_wheel;
        target.mRightWheel = source.right_wheel;
        target.mDifferentialRatio = source.differential_ratio;
        target.mLeftRightSplit = source.left_right_split;
        target.mEngineTorqueRatio = source.engine_torque_ratio;
    }
    settings.mController = controller;

    settings.mAntiRollBars.resize(desc.anti_roll_bars.size());
    for (std::size_t i = 0; i < desc.anti_roll_bars.size(); ++i) {
        const AntiRollBarDesc& source = desc.anti_roll_bars[i];
        if (source.left_wheel >= wheel_count || source.right_wheel >= wheel_count) {
            LOGIFACE_LOG(error, "physics: anti-roll bar references a wheel that does not exist");
            return false;
        }
        JPH::VehicleAntiRollBar& target = settings.mAntiRollBars[i];
        target.mLeftWheel = static_cast<int>(source.left_wheel);
        target.mRightWheel = static_cast<int>(source.right_wheel);
        target.mStiffness = source.stiffness;
    }

    auto impl = std::make_unique<Impl>();
    impl->world = &world;
    impl->chassis = desc.chassis;
    impl->constraint = JPH::Ref<JPH::VehicleConstraint>(
        new JPH::VehicleConstraint(*chassis, settings));
    // Jolt's wheel tests need a collision tester; the constraint holds only a
    // ref, so the vehicle owns the instance. A ray is the cheapest stable tester
    // and is what a large ground plane needs.
    impl->tester = JPH::Ref<JPH::VehicleCollisionTester>(new JPH::VehicleCollisionTesterRay(
        static_cast<JPH::ObjectLayer>(desc.wheel_collision_layer)));
    impl->constraint->SetVehicleCollisionTester(impl->tester);
    impl->controller = static_cast<JPH::WheeledVehicleController*>(impl->constraint->GetController());

    // Twice, deliberately: a constraint for the solver, and a step listener for
    // the pre-step suspension/tire update.
    system->AddConstraint(impl->constraint.GetPtr());
    system->AddStepListener(impl->constraint.GetPtr());

    impl_ = std::move(impl);
    LOGIFACE_LOG(info, std::format("physics: vehicle created ({} wheels, {} Nm)",
                                   wheel_count, desc.engine_max_torque));
    return true;
}

void WheeledVehicle::Shutdown() {
    if (impl_ == nullptr) return;
    JPH::PhysicsSystem* system = impl_->System();
    if (system != nullptr && impl_->constraint != nullptr) {
        system->RemoveStepListener(impl_->constraint.GetPtr());
        system->RemoveConstraint(impl_->constraint.GetPtr());
    }
    impl_.reset();
}

void WheeledVehicle::SetDriverInput(float forward, float right, float brake, float handbrake) {
    if (impl_ == nullptr || impl_->controller == nullptr) return;
    impl_->controller->SetDriverInput(forward, right, brake, handbrake);
    // A settled vehicle is asleep; input has to wake it, or the torque is
    // applied to a body the solver never integrates. Jolt's own vehicle sample
    // does exactly this on every input change.
    if (impl_->world != nullptr && (forward != 0.0f || right != 0.0f || brake != 0.0f ||
                                    handbrake != 0.0f)) {
        impl_->world->Activate(impl_->chassis);
    }
}

float WheeledVehicle::GetSpeed() const {
    if (impl_ == nullptr || impl_->constraint == nullptr) return 0.0f;
    const JPH::Body* body = impl_->constraint->GetVehicleBody();
    const JPH::Vec3 forward = body->GetRotation() * impl_->constraint->GetLocalForward();
    return body->GetLinearVelocity().Dot(forward);
}

std::uint32_t WheeledVehicle::WheelCount() const noexcept {
    return impl_ != nullptr && impl_->constraint != nullptr
               ? static_cast<std::uint32_t>(impl_->constraint->GetWheels().size())
               : 0u;
}

BodyPose WheeledVehicle::GetWheelPose(std::uint32_t index) const {
    if (impl_ == nullptr || impl_->constraint == nullptr || index >= WheelCount()) return {};
    // The basis hints describe the wheel model space every wheel render mesh is
    // authored against: the spin axle is model +y (the cylinder axis), wheel-up
    // model +x, wheel-forward model +z. The returned transform therefore maps
    // that model space to world space, with chassis orientation, steering and
    // suspension travel — and the accumulated spin — baked in.
    const JPH::RMat44 transform =
        impl_->constraint->GetWheelWorldTransform(index, JPH::Vec3::sAxisY(), JPH::Vec3::sAxisX());
    // Peel the spin back off, in model space: Jolt accumulates it as a negated
    // rotation about the +y axle (the same convention its own debug renderer
    // uses when drawing wheels), so undoing it is a positive rotation about
    // +y. What stays is the spin-free pose the interface promises; callers
    // re-apply the spin from `GetWheelRotationAngle` where they want it.
    const float spin_angle = impl_->constraint->GetWheel(index)->GetRotationAngle();
    BodyPose pose{};
    pose.position = FromRVec3(transform.GetTranslation());
    pose.rotation = FromQuat(transform.GetQuaternion()) *
                    glm::angleAxis(spin_angle, glm::vec3{0.0f, 1.0f, 0.0f});
    return pose;
}

float WheeledVehicle::GetWheelRotationAngle(std::uint32_t index) const {
    if (impl_ == nullptr || impl_->constraint == nullptr || index >= WheelCount()) return 0.0f;
    return impl_->constraint->GetWheel(index)->GetRotationAngle();
}

float WheeledVehicle::GetWheelSteerAngle(std::uint32_t index) const {
    if (impl_ == nullptr || impl_->constraint == nullptr || index >= WheelCount()) return 0.0f;
    return impl_->constraint->GetWheel(index)->GetSteerAngle();
}

PhysicsBodyId WheeledVehicle::GetWheelContactBody(std::uint32_t index) const {
    if (impl_ == nullptr || impl_->constraint == nullptr || index >= WheelCount()) return {};
    const JPH::BodyID contact = impl_->constraint->GetWheel(index)->GetContactBodyID();
    return contact.IsInvalid() ? PhysicsBodyId{} : FromJoltBodyId(contact);
}

} // namespace VulkanEngine::Physics
