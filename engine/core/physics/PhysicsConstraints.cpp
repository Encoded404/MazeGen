// Jolt constraint creation and the per-world constraint registry.
//
// The interface (`PhysicsConstraints.cppm`) names no Jolt type; this file is
// where the Jolt constraint objects, the collision filter that implements
// `enable_collision`, and the world's registry live.
//
// Registry ownership: a `ConstraintStore` is created lazily on first use and
// attached to the `World` through `detail::WorldExtension`. `World::Shutdown`
// calls `OnWorldShutdown`, which removes every constraint from the physics
// system while the bodies are still alive, and releases the store only after the
// bodies are gone. Nothing here can outlive the world it belongs to.

module;

#include <logging/logging_macros.hpp>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>
#include <Jolt/Physics/Collision/CollisionGroup.h>
#include <Jolt/Physics/Collision/GroupFilter.h>
#include <Jolt/Physics/Constraints/Constraint.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>

module VulkanEngine.Physics.Constraints;

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
[[nodiscard]] JPH::BodyID ToJoltBodyId(PhysicsBodyId body) noexcept {
    return JPH::BodyID(body.value);
}

// ── Collision filtering for `enable_collision == false` ─────────────────────
//
// Jolt has no "these two bodies do not collide" flag on a constraint; the
// mechanism is `CollisionGroup` + `GroupFilter`. A body carries one group, so a
// table keyed by a per-constraint subgroup cannot express a body in two
// collision-disabled joints (a chain), which is exactly what ragdolls/vehicles
// need. This filter instead encodes the *body id* (index+sequence, so a reused
// slot is a different id) in the group's `GroupID` and answers from a set of
// disabled pairs — general for any number of joints per body, and only bodies
// that actually participate in a disabled pair ever carry the filter.
//
// `CanCollide` runs on solver threads during broadphase pair finding, so the
// set is behind a shared mutex; constraints are created and destroyed from the
// main thread outside `Step`, so the exclusive side is never contended.
class ConstraintCollisionFilter final : public JPH::GroupFilter {
public:
    [[nodiscard]] bool CanCollide(const JPH::CollisionGroup& group1,
                                  const JPH::CollisionGroup& group2) const override {
        const std::uint32_t a = group1.GetGroupID();
        const std::uint32_t b = group2.GetGroupID();
        if (a == JPH::CollisionGroup::cInvalidGroup || b == JPH::CollisionGroup::cInvalidGroup) {
            return true; // a body without our filter always collides
        }
        const std::shared_lock lock(mutex_);
        return !disabled_pairs_.contains(PairKey(a, b));
    }

    void SetPairDisabled(std::uint32_t a, std::uint32_t b, bool disabled) {
        const std::unique_lock lock(mutex_);
        if (disabled) {
            disabled_pairs_.insert(PairKey(a, b));
        } else {
            disabled_pairs_.erase(PairKey(a, b));
        }
    }

private:
    [[nodiscard]] static std::uint64_t PairKey(std::uint32_t a, std::uint32_t b) noexcept {
        const std::uint32_t lo = std::min(a, b);
        const std::uint32_t hi = std::max(a, b);
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    }

    mutable std::shared_mutex mutex_;
    std::unordered_set<std::uint64_t> disabled_pairs_;
};

/// Per-world constraint state. One instance per `World`, owned by the world's
/// extension slot; `ConstraintId` slots are generational like every engine
/// handle.
class ConstraintStore final : public detail::WorldExtension {
public:
    explicit ConstraintStore(World* world) : world_(world) {}

    /// Removes every constraint from the physics system. Called by
    /// `World::Shutdown` while the physics system and bodies are still alive.
    void OnWorldShutdown() override {
        if (JPH::PhysicsSystem* system = System(); system != nullptr) {
            for (const std::unique_ptr<Slot>& slot : slots_) {
                if (slot != nullptr) system->RemoveConstraint(slot->constraint.GetPtr());
            }
        }
        slots_.clear();
        free_slots_.clear();
        // The filter stays alive through the bodies' `CollisionGroup` references.
        collision_filter_ = nullptr;
    }

    /// `World::DestroyBody` calls this before removing the body, so a destroyed
    /// body never leaves a constraint pointing at freed memory.
    void OnBodyDestroyed(PhysicsBodyId body) override { DestroyFor(body); }

    [[nodiscard]] ConstraintId Create(const ConstraintDesc& desc) {
        JPH::PhysicsSystem* system = System();
        if (system == nullptr) {
            LOGIFACE_LOG(error, "physics: cannot create a constraint on an uninitialized world");
            return {};
        }
        if (!desc.body_a.IsValid() || !desc.body_b.IsValid() || desc.body_a == desc.body_b) {
            LOGIFACE_LOG(error, "physics: a constraint needs two distinct, valid bodies");
            return {};
        }
        if (!world_->IsBodyValid(desc.body_a) || !world_->IsBodyValid(desc.body_b)) {
            LOGIFACE_LOG(error, "physics: a constraint references a body that is not live");
            return {};
        }

        const JPH::BodyLockInterface& lock = system->GetBodyLockInterface();
        JPH::Body* body_a = lock.TryGetBody(ToJoltBodyId(desc.body_a));
        JPH::Body* body_b = lock.TryGetBody(ToJoltBodyId(desc.body_b));
        if (body_a == nullptr || body_b == nullptr) {
            LOGIFACE_LOG(error, "physics: a constraint references a body Jolt does not know");
            return {};
        }

        JPH::TwoBodyConstraint* created = BuildSettings(desc, *body_a, *body_b);
        if (created == nullptr) return {};

        system->AddConstraint(created);

        if (!desc.enable_collision) {
            DisableCollision(desc.body_a, desc.body_b);
        }

        auto slot = std::make_unique<Slot>();
        slot->constraint = JPH::Ref<JPH::Constraint>(created);
        slot->type = desc.type;
        slot->body_a = desc.body_a;
        slot->body_b = desc.body_b;
        slot->disables_collision = !desc.enable_collision;
        slot->generation = next_generation_++;

        std::uint32_t index = 0;
        if (!free_slots_.empty()) {
            index = free_slots_.back();
            free_slots_.pop_back();
            slots_[index] = std::move(slot);
        } else {
            index = static_cast<std::uint32_t>(slots_.size());
            slots_.push_back(std::move(slot));
        }
        return ConstraintId{index, slots_[index]->generation};
    }

    void Destroy(ConstraintId id) {
        const std::optional<std::uint32_t> index = Resolve(id);
        if (!index) return;
        Slot& slot = *slots_[*index];
        if (JPH::PhysicsSystem* system = System(); system != nullptr) {
            system->RemoveConstraint(slot.constraint.GetPtr());
        }
        if (slot.disables_collision && collision_filter_ != nullptr) {
            collision_filter_->SetPairDisabled(slot.body_a.value, slot.body_b.value, false);
        }
        slots_[*index].reset();
        free_slots_.push_back(*index);
    }

    [[nodiscard]] bool Valid(ConstraintId id) const { return Resolve(id).has_value(); }

    [[nodiscard]] std::size_t Count() const noexcept {
        std::size_t live = 0;
        for (const std::unique_ptr<Slot>& slot : slots_) {
            if (slot != nullptr) ++live;
        }
        return live;
    }

    void DestroyFor(PhysicsBodyId body) {
        if (!body.IsValid()) return;
        // Collect first: `Destroy` mutates the slot table.
        std::vector<ConstraintId> matches;
        for (std::uint32_t index = 0; index < slots_.size(); ++index) {
            const Slot* slot = slots_[index].get();
            if (slot != nullptr && (slot->body_a == body || slot->body_b == body)) {
                matches.push_back(ConstraintId{index, slot->generation});
            }
        }
        for (const ConstraintId id : matches) Destroy(id);
    }

    [[nodiscard]] bool GetLimits(ConstraintId id, float& out_min, float& out_max) const {
        const Slot* slot = ResolveSlot(id);
        if (slot == nullptr) return false;
        switch (slot->type) {
            case ConstraintType::Hinge: {
                const auto* hinge = static_cast<const JPH::HingeConstraint*>(slot->constraint.GetPtr());
                out_min = hinge->GetLimitsMin();
                out_max = hinge->GetLimitsMax();
                return true;
            }
            case ConstraintType::Slider: {
                const auto* slider = static_cast<const JPH::SliderConstraint*>(slot->constraint.GetPtr());
                out_min = slider->GetLimitsMin();
                out_max = slider->GetLimitsMax();
                return true;
            }
            case ConstraintType::Distance: {
                const auto* distance =
                    static_cast<const JPH::DistanceConstraint*>(slot->constraint.GetPtr());
                out_min = distance->GetMinDistance();
                out_max = distance->GetMaxDistance();
                return true;
            }
            case ConstraintType::Fixed:
            case ConstraintType::Point:
                break;
        }
        return false;
    }

    bool SetLimits(ConstraintId id, float limit_min, float limit_max) {
        Slot* slot = ResolveSlot(id);
        if (slot == nullptr) return false;
        if (limit_min > limit_max) {
            LOGIFACE_LOG(error, std::format("physics: constraint limits must satisfy min <= max "
                                            "({} > {})",
                                            limit_min, limit_max));
            return false;
        }
        switch (slot->type) {
            case ConstraintType::Hinge:
                static_cast<JPH::HingeConstraint*>(slot->constraint.GetPtr())
                    ->SetLimits(limit_min, limit_max);
                return true;
            case ConstraintType::Slider:
                static_cast<JPH::SliderConstraint*>(slot->constraint.GetPtr())
                    ->SetLimits(limit_min, limit_max);
                return true;
            case ConstraintType::Distance:
                static_cast<JPH::DistanceConstraint*>(slot->constraint.GetPtr())
                    ->SetDistance(limit_min, limit_max);
                return true;
            case ConstraintType::Fixed:
            case ConstraintType::Point:
                break;
        }
        return false;
    }

    bool SetMotor(ConstraintId id, MotorState state) {
        Slot* slot = ResolveSlot(id);
        if (slot == nullptr) return false;
        const JPH::EMotorState jolt_state = ToJoltMotorState(state);
        switch (slot->type) {
            case ConstraintType::Hinge:
                static_cast<JPH::HingeConstraint*>(slot->constraint.GetPtr())
                    ->SetMotorState(jolt_state);
                return true;
            case ConstraintType::Slider:
                static_cast<JPH::SliderConstraint*>(slot->constraint.GetPtr())
                    ->SetMotorState(jolt_state);
                return true;
            case ConstraintType::Fixed:
            case ConstraintType::Point:
            case ConstraintType::Distance:
                break;
        }
        return false;
    }

    bool SetTargetVelocity(ConstraintId id, float velocity) {
        Slot* slot = ResolveSlot(id);
        if (slot == nullptr) return false;
        switch (slot->type) {
            case ConstraintType::Hinge:
                static_cast<JPH::HingeConstraint*>(slot->constraint.GetPtr())
                    ->SetTargetAngularVelocity(velocity);
                return true;
            case ConstraintType::Slider:
                static_cast<JPH::SliderConstraint*>(slot->constraint.GetPtr())
                    ->SetTargetVelocity(velocity);
                return true;
            case ConstraintType::Fixed:
            case ConstraintType::Point:
            case ConstraintType::Distance:
                break;
        }
        return false;
    }

    [[nodiscard]] float Position(ConstraintId id) const {
        const Slot* slot = ResolveSlot(id);
        if (slot == nullptr) return 0.0f;
        switch (slot->type) {
            case ConstraintType::Hinge:
                return static_cast<const JPH::HingeConstraint*>(slot->constraint.GetPtr())
                    ->GetCurrentAngle();
            case ConstraintType::Slider:
                return static_cast<const JPH::SliderConstraint*>(slot->constraint.GetPtr())
                    ->GetCurrentPosition();
            case ConstraintType::Fixed:
            case ConstraintType::Point:
            case ConstraintType::Distance:
                break;
        }
        return 0.0f;
    }

private:
    struct Slot {
        JPH::Ref<JPH::Constraint> constraint;
        ConstraintType type = ConstraintType::Fixed;
        PhysicsBodyId body_a{};
        PhysicsBodyId body_b{};
        bool disables_collision = false;
        std::uint32_t generation = 0;
    };

    [[nodiscard]] JPH::PhysicsSystem* System() const {
        return world_ != nullptr ? static_cast<JPH::PhysicsSystem*>(world_->NativePhysicsSystem())
                                 : nullptr;
    }

    [[nodiscard]] std::optional<std::uint32_t> Resolve(ConstraintId id) const {
        if (!id.IsValid() || id.slot >= slots_.size()) return std::nullopt;
        const Slot* slot = slots_[id.slot].get();
        if (slot == nullptr || slot->generation != id.generation) return std::nullopt;
        return id.slot;
    }

    [[nodiscard]] Slot* ResolveSlot(ConstraintId id) {
        const std::optional<std::uint32_t> index = Resolve(id);
        return index ? slots_[*index].get() : nullptr;
    }
    [[nodiscard]] const Slot* ResolveSlot(ConstraintId id) const {
        const std::optional<std::uint32_t> index = Resolve(id);
        return index ? slots_[*index].get() : nullptr;
    }

    [[nodiscard]] static JPH::EMotorState ToJoltMotorState(MotorState state) noexcept {
        switch (state) {
            case MotorState::Off:      return JPH::EMotorState::Off;
            case MotorState::Velocity: return JPH::EMotorState::Velocity;
            case MotorState::Position: return JPH::EMotorState::Position;
        }
        return JPH::EMotorState::Off;
    }

    /// Builds and creates the Jolt constraint. Returns null (with a log) for an
    /// invalid description; Jolt's `Create` always succeeds for valid settings.
    [[nodiscard]] JPH::TwoBodyConstraint* BuildSettings(const ConstraintDesc& desc, JPH::Body& body_a,
                                                        JPH::Body& body_b) {
        switch (desc.type) {
            case ConstraintType::Fixed: {
                // A weld is the bodies' current relative pose, so auto-detect is
                // the whole description; anchors are deliberately ignored.
                JPH::FixedConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mAutoDetectPoint = true;
                return settings.Create(body_a, body_b);
            }
            case ConstraintType::Point: {
                JPH::PointConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = ToRVec3(desc.point_a);
                settings.mPoint2 = ToRVec3(desc.point_b);
                return settings.Create(body_a, body_b);
            }
            case ConstraintType::Hinge: {
                glm::vec3 axis = desc.axis;
                glm::vec3 normal = desc.normal_axis;
                if (!NormalizeFrame(axis, normal, desc.type)) return nullptr;
                JPH::HingeConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = ToRVec3(desc.point_a);
                settings.mPoint2 = ToRVec3(desc.point_b);
                settings.mHingeAxis1 = ToVec3(axis);
                settings.mHingeAxis2 = ToVec3(axis);
                settings.mNormalAxis1 = ToVec3(normal);
                settings.mNormalAxis2 = ToVec3(normal);
                if (desc.use_limits) {
                    if (desc.limit_min > desc.limit_max) {
                        LOGIFACE_LOG(error, "physics: hinge limits must satisfy min <= max");
                        return nullptr;
                    }
                    settings.mLimitsMin = desc.limit_min;
                    settings.mLimitsMax = desc.limit_max;
                }
                return settings.Create(body_a, body_b);
            }
            case ConstraintType::Slider: {
                glm::vec3 axis = desc.axis;
                glm::vec3 normal = desc.normal_axis;
                if (!NormalizeFrame(axis, normal, desc.type)) return nullptr;
                JPH::SliderConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = ToRVec3(desc.point_a);
                settings.mPoint2 = ToRVec3(desc.point_b);
                settings.mSliderAxis1 = ToVec3(axis);
                settings.mSliderAxis2 = ToVec3(axis);
                settings.mNormalAxis1 = ToVec3(normal);
                settings.mNormalAxis2 = ToVec3(normal);
                if (desc.use_limits) {
                    if (desc.limit_min > desc.limit_max) {
                        LOGIFACE_LOG(error, "physics: slider limits must satisfy min <= max");
                        return nullptr;
                    }
                    settings.mLimitsMin = desc.limit_min;
                    settings.mLimitsMax = desc.limit_max;
                }
                return settings.Create(body_a, body_b);
            }
            case ConstraintType::Distance: {
                JPH::DistanceConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = ToRVec3(desc.point_a);
                settings.mPoint2 = ToRVec3(desc.point_b);
                if (desc.use_limits) {
                    if (desc.limit_min > desc.limit_max) {
                        LOGIFACE_LOG(error, "physics: distance limits must satisfy min <= max");
                        return nullptr;
                    }
                    settings.mMinDistance = desc.limit_min;
                    settings.mMaxDistance = desc.limit_max;
                }
                return settings.Create(body_a, body_b);
            }
        }
        LOGIFACE_LOG(error, std::format("physics: unknown constraint type {}",
                                        static_cast<int>(desc.type)));
        return nullptr;
    }

    /// Normalizes the axis frame in place; a degenerate or non-perpendicular
    /// frame is rejected loudly rather than silently producing a skewed joint.
    [[nodiscard]] static bool NormalizeFrame(glm::vec3& axis, glm::vec3& normal, ConstraintType type) {
        if (glm::length(axis) < 1.0e-6f || glm::length(normal) < 1.0e-6f) {
            LOGIFACE_LOG(error, std::format("physics: {} constraint needs a non-zero axis",
                                            ConstraintTypeName(type)));
            return false;
        }
        axis = glm::normalize(axis);
        normal = glm::normalize(normal);
        // Orthogonalize the normal against the axis so the joint frame is
        // well-defined even when the caller passes an approximation.
        normal = glm::normalize(normal - axis * glm::dot(axis, normal));
        return true;
    }

    void DisableCollision(PhysicsBodyId a, PhysicsBodyId b) {
        JPH::PhysicsSystem* system = System();
        if (system == nullptr) return;
        if (collision_filter_ == nullptr) {
            collision_filter_ = JPH::Ref<ConstraintCollisionFilter>(new ConstraintCollisionFilter());
        }
        collision_filter_->SetPairDisabled(a.value, b.value, true);

        JPH::BodyInterface& bodies = system->GetBodyInterface();
        const JPH::CollisionGroup group_a(collision_filter_.GetPtr(), a.value, 0);
        const JPH::CollisionGroup group_b(collision_filter_.GetPtr(), b.value, 0);
        bodies.SetCollisionGroup(ToJoltBodyId(a), group_a);
        bodies.SetCollisionGroup(ToJoltBodyId(b), group_b);
        // The broadphase only re-evaluates a pair when it is invalidated, so a
        // pair that already had contacts would keep them otherwise.
        bodies.InvalidateContactCache(ToJoltBodyId(a));
        bodies.InvalidateContactCache(ToJoltBodyId(b));
    }

    World* world_ = nullptr;
    JPH::Ref<ConstraintCollisionFilter> collision_filter_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t next_generation_ = 1;
};

// ── Per-world registry lookup ───────────────────────────────────────────────
//
// Keyed by world identity and holding a weak reference: the World's extension
// slot is the strong owner, so a destroyed world's entry expires here and is
// pruned on the next lookup. The map lock is never held while touching Jolt.
std::mutex g_registry_mutex;
std::unordered_map<const World*, std::weak_ptr<ConstraintStore>> g_registries;

[[nodiscard]] std::shared_ptr<ConstraintStore> RegistryFor(const World& world, bool create) {
    const std::scoped_lock lock(g_registry_mutex);
    if (const auto it = g_registries.find(&world); it != g_registries.end()) {
        if (std::shared_ptr<ConstraintStore> store = it->second.lock()) return store;
        g_registries.erase(it);
    }
    if (!create || !world.IsInitialized()) return nullptr;

    auto store = std::make_shared<ConstraintStore>(const_cast<World*>(&world));
    const_cast<World&>(world).AttachExtension(store);
    g_registries.emplace(&world, store);
    return store;
}

} // namespace

ConstraintId CreateConstraint(World& world, const ConstraintDesc& desc) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, true);
    if (store == nullptr) {
        LOGIFACE_LOG(error, "physics: cannot create a constraint on an uninitialized world");
        return {};
    }
    return store->Create(desc);
}

void DestroyConstraint(World& world, ConstraintId constraint) {
    if (const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false)) {
        store->Destroy(constraint);
    }
}

bool IsConstraintValid(World& world, ConstraintId constraint) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr && store->Valid(constraint);
}

std::size_t ConstraintCount(const World& world) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr ? store->Count() : 0;
}

void DestroyConstraintsFor(World& world, PhysicsBodyId body) {
    if (const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false)) {
        store->DestroyFor(body);
    }
}

bool GetConstraintLimits(World& world, ConstraintId constraint, float& out_min, float& out_max) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr && store->GetLimits(constraint, out_min, out_max);
}

bool SetConstraintLimits(World& world, ConstraintId constraint, float limit_min, float limit_max) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr && store->SetLimits(constraint, limit_min, limit_max);
}

bool SetMotorState(World& world, ConstraintId constraint, MotorState state) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr && store->SetMotor(constraint, state);
}

bool SetTargetVelocity(World& world, ConstraintId constraint, float velocity) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr && store->SetTargetVelocity(constraint, velocity);
}

float GetConstraintPosition(World& world, ConstraintId constraint) {
    const std::shared_ptr<ConstraintStore> store = RegistryFor(world, false);
    return store != nullptr ? store->Position(constraint) : 0.0f;
}

} // namespace VulkanEngine::Physics
