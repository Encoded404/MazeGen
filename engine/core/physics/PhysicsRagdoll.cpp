// Jolt ragdoll construction and lifecycle.
//
// The interface (`PhysicsRagdoll.cppm`) names no Jolt type; this file builds the
// `Skeleton`, the `RagdollSettings::Part` list, the parent-child joints and the
// `JPH::Ragdoll` itself, then hands it to the world's physics system.

module;

#include <logging/logging_macros.hpp>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Constraints/Constraint.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Skeleton/Skeleton.h>

module VulkanEngine.Physics.Ragdoll;

import std;

import logiface;

import VulkanEngine.Physics.World;
import VulkanEngine.Physics.Types;

namespace VulkanEngine::Physics {

namespace {

[[nodiscard]] JPH::Vec3 ToVec3(const glm::vec3& value) {
    return JPH::Vec3(value.x, value.y, value.z);
}
[[nodiscard]] JPH::RVec3 ToRVec3(const glm::vec3& value) {
    return JPH::RVec3(value.x, value.y, value.z);
}
[[nodiscard]] JPH::Quat ToQuat(const glm::quat& value) {
    return JPH::Quat(value.x, value.y, value.z, value.w);
}
[[nodiscard]] PhysicsBodyId FromJoltBodyId(const JPH::BodyID& body) noexcept {
    return PhysicsBodyId{body.GetIndexAndSequenceNumber()};
}
[[nodiscard]] JPH::EMotionType ToJoltMotion(BodyMotionType motion) noexcept {
    switch (motion) {
        case BodyMotionType::Static:    return JPH::EMotionType::Static;
        case BodyMotionType::Kinematic: return JPH::EMotionType::Kinematic;
        case BodyMotionType::Dynamic:   return JPH::EMotionType::Dynamic;
    }
    return JPH::EMotionType::Static;
}

/// Jolt asserts on non-unit direction vectors; a zero or approximate vector is
/// normalized rather than crashing the joint constructor.
[[nodiscard]] glm::vec3 NormalizeOr(const glm::vec3& value, const glm::vec3& fallback) {
    const float length = glm::length(value);
    return length > 1.0e-6f ? value / length : fallback;
}

/// Builds the two-body constraint that attaches `part` to its parent. Every
/// anchor/axis is world space, matching `ConstraintDesc`'s contract.
[[nodiscard]] JPH::Ref<JPH::TwoBodyConstraintSettings> MakeJoint(const RagdollPartDesc& part) {
    const glm::vec3 axis = NormalizeOr(part.joint_axis, {0.0f, 1.0f, 0.0f});
    const glm::vec3 normal = NormalizeOr(part.joint_normal_axis, {1.0f, 0.0f, 0.0f});
    const JPH::RVec3 anchor = ToRVec3(part.joint_anchor);

    switch (part.joint) {
        case ConstraintType::Fixed: {
            auto* settings = new JPH::FixedConstraintSettings;
            settings->mSpace = JPH::EConstraintSpace::WorldSpace;
            settings->mAutoDetectPoint = true;
            return settings;
        }
        case ConstraintType::Point: {
            auto* settings = new JPH::PointConstraintSettings;
            settings->mSpace = JPH::EConstraintSpace::WorldSpace;
            settings->mPoint1 = anchor;
            settings->mPoint2 = anchor;
            return settings;
        }
        case ConstraintType::Hinge: {
            auto* settings = new JPH::HingeConstraintSettings;
            settings->mSpace = JPH::EConstraintSpace::WorldSpace;
            settings->mPoint1 = anchor;
            settings->mPoint2 = anchor;
            settings->mHingeAxis1 = ToVec3(axis);
            settings->mHingeAxis2 = ToVec3(axis);
            settings->mNormalAxis1 = ToVec3(normal);
            settings->mNormalAxis2 = ToVec3(normal);
            if (part.use_limits) {
                settings->mLimitsMin = part.limit_min;
                settings->mLimitsMax = part.limit_max;
            }
            return settings;
        }
        case ConstraintType::Slider: {
            auto* settings = new JPH::SliderConstraintSettings;
            settings->mSpace = JPH::EConstraintSpace::WorldSpace;
            settings->mPoint1 = anchor;
            settings->mPoint2 = anchor;
            settings->mSliderAxis1 = ToVec3(axis);
            settings->mSliderAxis2 = ToVec3(axis);
            settings->mNormalAxis1 = ToVec3(normal);
            settings->mNormalAxis2 = ToVec3(normal);
            if (part.use_limits) {
                settings->mLimitsMin = part.limit_min;
                settings->mLimitsMax = part.limit_max;
            }
            return settings;
        }
        case ConstraintType::Distance: {
            auto* settings = new JPH::DistanceConstraintSettings;
            settings->mSpace = JPH::EConstraintSpace::WorldSpace;
            settings->mPoint1 = anchor;
            settings->mPoint2 = anchor;
            if (part.use_limits) {
                settings->mMinDistance = part.limit_min;
                settings->mMaxDistance = part.limit_max;
            }
            return settings;
        }
    }
    return nullptr;
}

} // namespace

struct Ragdoll::Impl {
    World* world = nullptr;
    // The ragdoll holds a raw pointer to its settings, so the Ref has to stay
    // alive for the ragdoll's whole lifetime — and, by declaration order here,
    // be released after it.
    JPH::Ref<JPH::RagdollSettings> settings;
    JPH::Ref<JPH::Ragdoll> ragdoll;
    bool added = false;

    [[nodiscard]] JPH::PhysicsSystem* System() const {
        return world != nullptr ? static_cast<JPH::PhysicsSystem*>(world->NativePhysicsSystem())
                                : nullptr;
    }
};

Ragdoll::Ragdoll() = default;

Ragdoll::~Ragdoll() {
    Shutdown();
}

bool Ragdoll::Initialize(World& world, const RagdollDesc& desc) {
    if (impl_ != nullptr) {
        LOGIFACE_LOG(error, "physics: ragdoll is already initialized; shut it down first");
        return false;
    }
    if (!world.IsInitialized()) {
        LOGIFACE_LOG(error, "physics: cannot create a ragdoll on an uninitialized world");
        return false;
    }
    JPH::PhysicsSystem* system = static_cast<JPH::PhysicsSystem*>(world.NativePhysicsSystem());
    if (system == nullptr || desc.parts.empty()) {
        LOGIFACE_LOG(error, "physics: a ragdoll needs at least one part and a live world");
        return false;
    }

    auto settings = JPH::Ref<JPH::RagdollSettings>(new JPH::RagdollSettings);
    settings->mSkeleton = JPH::Ref<JPH::Skeleton>(new JPH::Skeleton);
    const auto part_count = static_cast<std::int32_t>(desc.parts.size());
    for (std::int32_t i = 0; i < part_count; ++i) {
        const std::int32_t parent = desc.parts[i].parent;
        if (parent < -1 || parent >= i) {
            LOGIFACE_LOG(error, std::format(
                "physics: ragdoll part {} has parent {} (parents must be earlier parts)", i, parent));
            return false;
        }
        settings->mSkeleton->AddJoint(std::format("part{}", i), static_cast<int>(parent));
    }

    settings->mParts.resize(static_cast<std::size_t>(part_count));
    // Shapes are referenced by raw pointer from `Part`; the settings object does
    // not own them, so they stay alive here until the bodies take their own refs
    // in `CreateRagdoll`.
    std::vector<JPH::RefConst<JPH::Shape>> shapes;
    shapes.reserve(desc.parts.size());

    for (std::int32_t i = 0; i < part_count; ++i) {
        const RagdollPartDesc& part = desc.parts[static_cast<std::size_t>(i)];
        if (part.radius <= 0.0f || part.height < 2.0f * part.radius) {
            LOGIFACE_LOG(error, std::format("physics: ragdoll part {} has invalid capsule "
                                            "dimensions (radius {}, height {})",
                                            i, part.radius, part.height));
            return false;
        }
        const float half_cylinder = std::max(0.0f, part.height * 0.5f - part.radius);
        const JPH::ShapeSettings::ShapeResult result =
            JPH::CapsuleShapeSettings(half_cylinder, part.radius).Create();
        if (!result.IsValid()) {
            LOGIFACE_LOG(error, std::format("physics: cannot cook ragdoll capsule: {}",
                                            result.GetError()));
            return false;
        }
        shapes.push_back(JPH::RefConst<JPH::Shape>(result.Get()));

        JPH::RagdollSettings::Part& body = settings->mParts[static_cast<std::size_t>(i)];
        body.SetShape(shapes.back().GetPtr());
        body.mPosition = ToRVec3(part.position);
        body.mRotation = ToQuat(part.rotation);
        body.mMotionType = ToJoltMotion(part.motion);
        body.mObjectLayer = static_cast<JPH::ObjectLayer>(part.layer);
        body.mUserData = desc.user_data;
        body.mFriction = part.friction;
        body.mRestitution = part.restitution;
        body.mAllowSleeping = desc.allow_sleeping;
        if (part.mass > 0.0f) {
            body.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            body.mMassPropertiesOverride.mMass = part.mass;
        }
        if (part.parent >= 0) {
            JPH::Ref<JPH::TwoBodyConstraintSettings> joint = MakeJoint(part);
            if (joint == nullptr) {
                LOGIFACE_LOG(error, "physics: unknown ragdoll joint type");
                return false;
            }
            body.mToParent = joint;
        }
    }

    // Equalizes mass ratios and inertia between parents and children, so a chain
    // of capsules does not blow up in the solver.
    if (!settings->Stabilize()) {
        LOGIFACE_LOG(error, "physics: ragdoll stabilization failed");
        return false;
    }
    if (desc.disable_parent_child_collisions) {
        settings->DisableParentChildCollisions();
    }
    settings->CalculateBodyIndexToConstraintIndex();
    settings->CalculateConstraintIndexToBodyIdxPair();

    JPH::Ragdoll* created =
        settings->CreateRagdoll(desc.collision_group, desc.user_data, system);
    if (created == nullptr) {
        LOGIFACE_LOG(error, std::format("physics: ragdoll creation failed (body cap {})",
                                        system->GetMaxBodies()));
        return false;
    }

    auto impl = std::make_unique<Impl>();
    impl->world = &world;
    impl->settings = std::move(settings);
    impl->ragdoll = JPH::Ref<JPH::Ragdoll>(created);
    impl->ragdoll->AddToPhysicsSystem(JPH::EActivation::Activate);
    impl->added = true;
    impl_ = std::move(impl);

    LOGIFACE_LOG(info, std::format("physics: ragdoll created ({} parts)", part_count));
    return true;
}

void Ragdoll::Shutdown() {
    if (impl_ == nullptr) return;
    if (impl_->added && impl_->ragdoll != nullptr) {
        impl_->ragdoll->RemoveFromPhysicsSystem();
        impl_->added = false;
    }
    impl_.reset();
}

void Ragdoll::AddToPhysicsSystem() {
    if (impl_ == nullptr || impl_->added) return;
    impl_->ragdoll->AddToPhysicsSystem(JPH::EActivation::Activate);
    impl_->added = true;
}

void Ragdoll::RemoveFromPhysicsSystem() {
    if (impl_ == nullptr || !impl_->added) return;
    impl_->ragdoll->RemoveFromPhysicsSystem();
    impl_->added = false;
}

bool Ragdoll::IsInPhysicsSystem() const noexcept {
    return impl_ != nullptr && impl_->added;
}

std::uint32_t Ragdoll::PartCount() const noexcept {
    return impl_ != nullptr ? static_cast<std::uint32_t>(impl_->ragdoll->GetBodyCount()) : 0u;
}

PhysicsBodyId Ragdoll::PartBody(std::uint32_t index) const {
    if (impl_ == nullptr || index >= PartCount()) return {};
    return FromJoltBodyId(impl_->ragdoll->GetBodyIDs()[index]);
}

BodyPose Ragdoll::GetPartPose(std::uint32_t index) const {
    if (impl_ == nullptr || index >= PartCount()) return {};
    return impl_->world->GetPose(PartBody(index));
}

void Ragdoll::SetPartPose(std::uint32_t index, const BodyPose& pose) {
    if (impl_ == nullptr || index >= PartCount()) return;
    impl_->world->SetPose(PartBody(index), pose);
}

void Ragdoll::SetPose(std::span<const BodyPose> poses) {
    if (impl_ == nullptr) return;
    const std::uint32_t count = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(poses.size()), PartCount());
    for (std::uint32_t i = 0; i < count; ++i) SetPartPose(i, poses[i]);
}

bool Ragdoll::IsActive() const {
    return impl_ != nullptr && impl_->ragdoll->IsActive();
}

} // namespace VulkanEngine::Physics
