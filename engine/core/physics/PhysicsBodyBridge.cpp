// Component <-> body bridge implementation.

module;

#include <logging/logging_macros.hpp>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

module VulkanEngine.Physics.BodyBridge;

import std;

import logiface;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Components.Transform;
import VulkanEngine.Physics.World;

namespace VulkanEngine::Physics {

namespace {

[[nodiscard]] BodyDesc ToBodyDesc(const RigidBody& body, const Components::Transform* transform,
                                  EntityId owner) {
    BodyDesc desc{};
    desc.shape = body.shape;
    desc.layer = body.layer;
    desc.motion = body.motion;
    desc.mass = body.mass;
    desc.friction = body.friction;
    desc.restitution = body.restitution;
    desc.linear_damping = body.linear_damping;
    desc.angular_damping = body.angular_damping;
    desc.ccd = body.ccd;
    desc.sensor = body.sensor;
    if (transform != nullptr) {
        desc.position = transform->position;
        desc.rotation = transform->rotation;
    }
    desc.user_data = PackEntityId(owner);
    return desc;
}

} // namespace

void AttachBody(World& world, Entity& entity, RigidBody& body) {
    if (body.body.IsValid()) return; // already attached
    if (!world.IsInitialized()) return;

    // The hook runs with the registry lock held and before the component's mask
    // bit is set, so the added component arrives by reference and only *other*
    // components are looked up through the entity.
    const auto* transform = entity.GetComponent<Components::Transform>();
    const auto* collider = entity.GetComponent<Collider>();
    const std::span<const ShapeDesc> children =
        collider != nullptr ? std::span<const ShapeDesc>(collider->children) : std::span<const ShapeDesc>{};

    body.body = world.CreateBody(ToBodyDesc(body, transform, entity.GetId()), children);
    if (!body.body.IsValid()) {
        LOGIFACE_LOG(warn, std::format("physics: no body created for entity {} (shape {})",
                                       entity.GetId().index,
                                       ShapeTypeName(body.shape.type)));
        return;
    }

    // An authored starting velocity (a launched projectile, say) applies on the
    // first step; it is written without activating so a static/kinematic body is
    // not needlessly woken.
    if (const auto* velocity = entity.GetComponent<Velocity>()) {
        if (velocity->linear != glm::vec3(0.0f) || velocity->angular != glm::vec3(0.0f)) {
            world.SetLinearAndAngularVelocity(body.body, velocity->linear, velocity->angular, false);
        }
    }

    // A body that lives in the ECS is a body the renderer can draw, so it is
    // tracked for interpolation by default; a system that owns a body outside the
    // ECS tracks it itself. Tracking is what makes `Transform` the gameplay truth
    // and the render pose a blend of the last two fixed steps.
    world.TrackRenderPose(body.body);
}

void DetachBody(World& world, RigidBody& body) {
    if (!body.body.IsValid()) return;
    // Untrack before the destroy: the handle stops naming a body the moment
    // `DestroyBody` returns, and a stale interpolation entry for a recycled slot
    // would blend from a pose that was never this body's.
    world.UntrackRenderPose(body.body);
    world.DestroyBody(body.body);
    body.body = {};
}

void RebuildBody(World& world, Entity& entity, RigidBody& body,
                 std::span<const ShapeDesc> children) {
    if (!body.body.IsValid()) return;

    // Carry the live state across the rebuild: the authored Transform may be a
    // step behind (write-back owns it for dynamic bodies), and a rebuilt body
    // must not lose momentum or silently fall asleep.
    const BodyPose pose = world.GetPose(body.body);
    const glm::vec3 linear = world.GetLinearVelocity(body.body);
    const glm::vec3 angular = world.GetAngularVelocity(body.body);
    const bool active = world.IsActive(body.body);
    // Tracking belongs to the *body*, not to the handle: a rebuild that dropped it
    // would silently stop interpolating a moving platform's load.
    const bool tracked = world.IsRenderPoseTracked(body.body);
    world.UntrackRenderPose(body.body);
    world.DestroyBody(body.body);
    body.body = {};

    auto* transform = entity.GetComponent<Components::Transform>();
    if (transform != nullptr) {
        transform->position = pose.position;
        transform->rotation = pose.rotation;
    }
    body.body = world.CreateBody(ToBodyDesc(body, transform, entity.GetId()), children);
    if (!body.body.IsValid()) {
        LOGIFACE_LOG(warn, std::format("physics: body rebuild failed for entity {}",
                                       entity.GetId().index));
        return;
    }
    if (tracked) {
        // Re-seeding from the carried pose is what makes the rebuild invisible:
        // the render pose is the pose the old body had, from the very next frame,
        // rather than a blend from a pose the new handle never owned.
        world.TrackRenderPose(body.body);
    }
    world.SetLinearAndAngularVelocity(body.body, linear, angular, active);
}

void ApplyInputs(World& world, ComponentRegistry& registry, float delta_time) {
    if (!world.IsInitialized()) return;

    registry.ForEach<RigidBody, Components::Transform>(
        [&world, delta_time](Entity& entity, RigidBody& body, Components::Transform& transform) {
            if (!body.body.IsValid()) return;
            // Only kinematics are moved from the ECS: a dynamic body belongs to
            // the solver, and driving it from here would fight it. Dynamic motion
            // is requested through the typed body API instead.
            if (body.motion != BodyMotionType::Kinematic) return;

            if (const auto* velocity = entity.GetComponent<Velocity>();
                velocity != nullptr && (velocity->linear != glm::vec3(0.0f) ||
                                        velocity->angular != glm::vec3(0.0f))) {
                // A non-zero authored velocity is a *driven* velocity: a
                // conveyor or a constant-velocity platform, which pushes
                // whatever it carries.
                world.SetLinearAndAngularVelocity(body.body, velocity->linear, velocity->angular,
                                                  false);
                return;
            }

            // Otherwise the Transform is the target and Jolt derives the velocity
            // that reaches it exactly within the step, so a moving platform
            // carries what rests on it rather than teleporting through it.
            world.MoveKinematic(body.body, BodyPose{transform.position, transform.rotation},
                                delta_time);
        });
}

void WriteBack(World& world, ComponentRegistry& registry, float delta_time) {
    (void)delta_time;
    if (!world.IsInitialized()) return;

    registry.ForEach<RigidBody, Components::Transform>(
        [&world](Entity& entity, RigidBody& body, Components::Transform& transform) {
            if (!body.body.IsValid()) return;

            // `Transform` is the fixed-step gameplay truth for a body-driven
            // entity; render interpolation samples PhysicsWorld instead, so this
            // stays the step pose rather than a smoothed one.
            const BodyPose pose = world.GetPose(body.body);
            transform.position = pose.position;
            transform.rotation = pose.rotation;

            if (auto* velocity = entity.GetComponent<Velocity>()) {
                velocity->linear = world.GetLinearVelocity(body.body);
                velocity->angular = world.GetAngularVelocity(body.body);
            }
        });
}

} // namespace VulkanEngine::Physics
