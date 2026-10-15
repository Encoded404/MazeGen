// Device-free physics tests.
//
// Physics needs no Vulkan device, so these run under the default (non-gpu) test
// presets. Nothing here creates an instance, a device, or a swapchain — the
// world's only dependency is Jolt and the shared compute pool.

#include <gtest/gtest.h>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

import std;

import VulkanShared.Jobs;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ECS.SystemSchedule;
import VulkanEngine.Components.Transform;
import VulkanEngine.Physics.Types;
import VulkanEngine.Physics.Components;
import VulkanEngine.Physics.World;
import VulkanEngine.Physics.BodyBridge;
import VulkanEngine.Physics.Character;
import VulkanEngine.Physics.Constraints;
import VulkanEngine.Physics.Vehicle;
import VulkanEngine.Physics.Ragdoll;

namespace {

using VulkanEngine::ComponentRegistry;
using VulkanEngine::Components::Transform;
using VulkanEngine::Entity;
using VulkanEngine::EntityId;
using VulkanEngine::SystemSchedule;
using VulkanEngine::SystemSlot;
using VulkanEngine::Physics::AttachBody;
using VulkanEngine::Physics::BodyDesc;
using VulkanEngine::Physics::BodyMotionType;
using VulkanEngine::Physics::BodyPose;
using VulkanEngine::Physics::Character;
using VulkanEngine::Physics::CharacterDesc;
using VulkanEngine::Physics::CharacterId;
using VulkanEngine::Physics::Collider;
using VulkanEngine::Physics::CollisionLayer;
using VulkanEngine::Physics::ConstraintCount;
using VulkanEngine::Physics::ConstraintDesc;
using VulkanEngine::Physics::ConstraintId;
using VulkanEngine::Physics::ConstraintType;
using VulkanEngine::Physics::CreateConstraint;
using VulkanEngine::Physics::DestroyConstraint;
using VulkanEngine::Physics::DestroyConstraintsFor;
using VulkanEngine::Physics::GetConstraintLimits;
using VulkanEngine::Physics::GetConstraintPosition;
using VulkanEngine::Physics::IsConstraintValid;
using VulkanEngine::Physics::MotorState;
using VulkanEngine::Physics::SetConstraintLimits;
using VulkanEngine::Physics::SetMotorState;
using VulkanEngine::Physics::SetTargetVelocity;
using VulkanEngine::Physics::VehicleDesc;
using VulkanEngine::Physics::WheelDesc;
using VulkanEngine::Physics::WheeledVehicle;
using VulkanEngine::Physics::Ragdoll;
using VulkanEngine::Physics::RagdollDesc;
using VulkanEngine::Physics::RagdollPartDesc;
using VulkanEngine::Physics::ContactEvent;
using VulkanEngine::Physics::ContactPhase;
using VulkanEngine::Physics::DetachBody;
using VulkanEngine::Physics::AttachCharacter;
using VulkanEngine::Physics::DetachCharacter;
using VulkanEngine::Physics::PackEntityId;
using VulkanEngine::Physics::RebuildBody;
using VulkanEngine::Physics::PhysicsBodyId;
using VulkanEngine::Physics::PhysicsDebugFaceCulling;
using VulkanEngine::Physics::PhysicsDebugGeometry;
using VulkanEngine::Physics::PhysicsDebugSettings;
using VulkanEngine::Physics::PhysicsWorldConfig;
using VulkanEngine::Physics::RayCastRequest;
using VulkanEngine::Physics::RigidBody;
using VulkanEngine::Physics::ShapeCastRequest;
using VulkanEngine::Physics::ShapeDesc;
using VulkanEngine::Physics::ShapeType;
using VulkanEngine::Physics::UnpackEntityId;
using VulkanEngine::Physics::Velocity;
using VulkanEngine::Physics::World;
using VulkanEngine::Physics::ApplyInputs;
using VulkanEngine::Physics::StepCharacters;
using VulkanEngine::Physics::WriteBack;

/// A registry wired to a world exactly the way `GameEngine::Setup` wires it
/// (hooks plus the three frame-contract slots), with a small world so a test
/// does not pay for the 65536-body default arrays.
class PhysicsEcs {
public:
    PhysicsEcs() {
        PhysicsWorldConfig config{};
        config.max_bodies = 1024;
        config.max_body_pairs = 4096;
        config.max_contact_constraints = 1024;
        world_.Initialize(config);

        registry_.OnAdd<RigidBody>([this](Entity& entity, RigidBody& body) {
            AttachBody(world_, entity, body);
        });
        registry_.OnRemove<RigidBody>([this](Entity& /*entity*/, RigidBody& body) {
            DetachBody(world_, body);
        });
        // A collider change means a new shape set; Jolt shapes are immutable once
        // a body owns them, so the body is rebuilt at its live pose (the same
        // wiring GameEngine::Setup installs).
        registry_.OnAdd<Collider>([this](Entity& entity, Collider& collider) {
            if (auto* body = entity.GetComponent<RigidBody>()) {
                RebuildBody(world_, entity, *body, collider.children);
            }
        });
        registry_.OnRemove<Collider>([this](Entity& entity, Collider& /*collider*/) {
            if (auto* body = entity.GetComponent<RigidBody>()) {
                RebuildBody(world_, entity, *body, {});
            }
        });
        registry_.OnAdd<Character>([this](Entity& entity, Character& character) {
            AttachCharacter(world_, entity, character);
        });
        registry_.OnRemove<Character>([this](Entity& /*entity*/, Character& character) {
            DetachCharacter(world_, character);
        });
    }

    [[nodiscard]] EntityId SpawnBody(const RigidBody& body, glm::vec3 position) {
        Entity& entity = registry_.CreateEntity();
        const EntityId id = entity.GetId();
        registry_.AddComponent<Transform>(entity).position = position;
        registry_.AddComponent<RigidBody>(entity, body);
        return id;
    }

    /// One fixed step of the real frame contract: input (bodies, then swept
    /// characters), simulate, write back.
    void Step(int steps, float delta_time = 1.0f / 60.0f) {
        for (int i = 0; i < steps; ++i) {
            ApplyInputs(world_, registry_, delta_time);
            StepCharacters(world_, registry_, delta_time);
            world_.Step();
            WriteBack(world_, registry_, delta_time);
        }
    }

    [[nodiscard]] EntityId SpawnCharacter(const CharacterDesc& desc, glm::vec3 position) {
        Entity& entity = registry_.CreateEntity();
        const EntityId id = entity.GetId();
        registry_.AddComponent<Transform>(entity).position = position;
        Character character{};
        character.desc = desc;
        registry_.AddComponent<Character>(entity, character);
        return id;
    }

    [[nodiscard]] Character* GetCharacter(EntityId id) {
        Entity* entity = registry_.TryGetEntity(id);
        return entity != nullptr ? registry_.GetComponent<Character>(*entity) : nullptr;
    }

    [[nodiscard]] World& WorldRef() { return world_; }
    [[nodiscard]] ComponentRegistry& Registry() { return registry_; }
    /// Adds the projected-velocity channel so write-back has somewhere to land.
    Velocity& AddVelocity(EntityId id, glm::vec3 linear, glm::vec3 angular = glm::vec3(0.0f)) {
        Velocity velocity{};
        velocity.linear = linear;
        velocity.angular = angular;
        return registry_.AddComponent<Velocity>(*registry_.TryGetEntity(id), velocity);
    }
    [[nodiscard]] Transform* GetTransform(EntityId id) {
        Entity* entity = registry_.TryGetEntity(id);
        return entity != nullptr ? registry_.GetComponent<Transform>(*entity) : nullptr;
    }
    [[nodiscard]] RigidBody* GetRigidBody(EntityId id) {
        Entity* entity = registry_.TryGetEntity(id);
        return entity != nullptr ? registry_.GetComponent<RigidBody>(*entity) : nullptr;
    }
    [[nodiscard]] Velocity* GetVelocity(EntityId id) {
        Entity* entity = registry_.TryGetEntity(id);
        return entity != nullptr ? registry_.GetComponent<Velocity>(*entity) : nullptr;
    }

private:
    ComponentRegistry registry_;
    World world_;
};

[[nodiscard]] RigidBody BoxBody(glm::vec3 extent, BodyMotionType motion,
                                CollisionLayer layer = CollisionLayer::Default, float mass = 1.0f) {
    RigidBody body{};
    body.shape.type = ShapeType::Box;
    body.shape.box_extent = extent;
    body.motion = motion;
    body.layer = layer;
    body.mass = mass;
    body.friction = 0.5f;
    body.restitution = 0.0f;
    return body;
}

TEST(PhysicsBridge, EntityIdPackingRoundTripsAndKeepsTheGeneration) {
    const EntityId id{1234, 7};
    const std::uint64_t packed = PackEntityId(id);
    EXPECT_EQ(UnpackEntityId(packed), id);
    // A zero payload is the "no entity" value a body without an owner carries.
    EXPECT_FALSE(UnpackEntityId(0).IsValid());
}

TEST(PhysicsEcs, AddingTheComponentAttachesABodyAndRemovingItDetaches) {
    PhysicsEcs sim;
    const EntityId entity = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 5.0f, 0.0f});

    RigidBody* body = sim.GetRigidBody(entity);
    ASSERT_NE(body, nullptr);
    ASSERT_TRUE(body->body.IsValid()) << "the OnAdd hook did not attach a body";
    EXPECT_EQ(sim.WorldRef().BodyCount(), 1u);
    // The body carries the owning entity, generation included.
    EXPECT_EQ(UnpackEntityId(sim.WorldRef().GetUserData(body->body)), entity);

    const PhysicsBodyId handle = body->body;
    Entity* raw = sim.Registry().TryGetEntity(entity);
    ASSERT_NE(raw, nullptr);
    sim.Registry().RemoveComponent<RigidBody>(*raw);

    EXPECT_EQ(sim.WorldRef().BodyCount(), 0u);
    EXPECT_FALSE(sim.WorldRef().IsBodyValid(handle)) << "the OnRemove hook did not destroy the body";
}

TEST(PhysicsEcs, DestroyingTheEntityDestroysItsBodyAndInvalidatesTheHandle) {
    PhysicsEcs sim;
    const EntityId entity = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 5.0f, 0.0f});
    ASSERT_EQ(sim.WorldRef().BodyCount(), 1u);
    const PhysicsBodyId handle = sim.GetRigidBody(entity)->body;

    Entity* raw = sim.Registry().TryGetEntity(entity);
    ASSERT_NE(raw, nullptr);
    sim.Registry().DestroyEntity(*raw);

    EXPECT_EQ(sim.WorldRef().BodyCount(), 0u);
    EXPECT_FALSE(sim.WorldRef().IsBodyValid(handle));
    // A stale handle resolves to nothing, so a physics event carrying it is
    // dropped rather than resolving to the slot's next occupant.
    EXPECT_EQ(sim.Registry().TryGetEntity(entity), nullptr);
}

TEST(PhysicsEcs, ShapeCacheDeduplicatesGeometryButNotPlacement) {
    PhysicsEcs sim;
    for (int i = 0; i < 5; ++i) {
        (void)sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                            {static_cast<float>(i), 5.0f, 0.0f});
    }
    EXPECT_EQ(sim.WorldRef().BodyCount(), 5u);
    EXPECT_EQ(sim.WorldRef().CachedShapeCount(), 1u)
        << "placement must be a decorator, not part of the cooked geometry";

    (void)sim.SpawnBody(BoxBody({2.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic), {0.0f, 5.0f, 0.0f});
    EXPECT_EQ(sim.WorldRef().CachedShapeCount(), 2u);
}

TEST(PhysicsEcs, AFallingBodyRestsOnTheGroundAtItsAnalyticHeight) {
    PhysicsEcs sim;
    // A wide static slab with its top face at y = 0.
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default,
                                0.0f),
                        {0.0f, -0.5f, 0.0f});

    RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
    crate.mass = 10.0f;
    const EntityId crate_id = sim.SpawnBody(crate, {0.0f, 6.0f, 0.0f});

    sim.Step(300);

    const Transform* transform = sim.GetTransform(crate_id);
    ASSERT_NE(transform, nullptr);
    // Half the box's height: the analytic resting height for a box on a plane.
    EXPECT_NEAR(transform->position.y, 0.5f, 0.05f);
    EXPECT_NEAR(transform->position.x, 0.0f, 0.05f);
    EXPECT_NEAR(transform->position.z, 0.0f, 0.05f);

    const Velocity* velocity = sim.GetVelocity(crate_id);
    if (velocity != nullptr) {
        EXPECT_NEAR(velocity->linear.y, 0.0f, 0.1f);
    }
}

TEST(PhysicsEcs, VelocityIsProjectedFromTheBodyDuringWriteBack) {
    PhysicsEcs sim;
    const EntityId entity = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 50.0f, 0.0f});
    (void)sim.AddVelocity(entity, {0.0f, 0.0f, 0.0f});

    sim.Step(30);

    // Still falling after half a second: the projection must show it.
    EXPECT_LT(sim.GetVelocity(entity)->linear.y, 0.0f);
    EXPECT_LT(sim.GetTransform(entity)->position.y, 50.0f);
}

TEST(PhysicsEcs, AKinematicTransformReachesItsTargetWithinTheStep) {
    PhysicsEcs sim;
    const EntityId entity = sim.SpawnBody(BoxBody({4.0f, 0.5f, 4.0f}, BodyMotionType::Kinematic),
                                          {0.0f, 0.0f, 0.0f});

    Transform* transform = sim.GetTransform(entity);
    ASSERT_NE(transform, nullptr);
    transform->position = {5.0f, 0.0f, 0.0f};

    sim.Step(1);

    // Jolt derives the velocity that lands exactly on the target in one step, so
    // a platform carries what rests on it instead of teleporting through it.
    EXPECT_NEAR(sim.GetTransform(entity)->position.x, 5.0f, 0.01f);

    sim.Step(10);
    EXPECT_NEAR(sim.GetTransform(entity)->position.x, 5.0f, 0.01f);
    ASSERT_TRUE(sim.GetRigidBody(entity)->body.IsValid());
}

TEST(PhysicsEcs, AnAuthoredKinematicVelocityIsDrivenNotTreatedAsATarget) {
    PhysicsEcs sim;
    const EntityId entity = sim.SpawnBody(BoxBody({4.0f, 0.5f, 4.0f}, BodyMotionType::Kinematic),
                                          {0.0f, 0.0f, 0.0f});
    (void)sim.AddVelocity(entity, {2.0f, 0.0f, 0.0f});

    sim.Step(60); // one second at 2 m/s

    EXPECT_NEAR(sim.GetTransform(entity)->position.x, 2.0f, 0.05f);
}

TEST(PhysicsEcs, ImpulsesAndForcesGoThroughTheTypedBodyApi) {
    PhysicsEcs sim;
    RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
    crate.mass = 10.0f;
    const EntityId entity = sim.SpawnBody(crate, {0.0f, 100.0f, 0.0f});
    const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
    ASSERT_TRUE(body.IsValid());

    sim.WorldRef().AddImpulse(body, {20.0f, 0.0f, 0.0f}); // 20 N*s on 10 kg -> 2 m/s
    sim.Step(1);

    EXPECT_NEAR(sim.WorldRef().GetLinearVelocity(body).x, 2.0f, 0.05f);

    // A sleeping body is woken by the same API; this is why there is no separate
    // "kick a sleeping body" path in the component.
    sim.WorldRef().SetLinearVelocity(body, {0.0f, 0.0f, 0.0f});
    sim.Step(600); // let it come to rest and sleep
    sim.WorldRef().AddImpulse(body, {30.0f, 0.0f, 0.0f});
    sim.Step(1);
    EXPECT_NEAR(sim.WorldRef().GetLinearVelocity(body).x, 3.0f, 0.1f);
}

TEST(PhysicsBodyTuning, MassChangesLiveAndScalesInertiaWithIt) {
    PhysicsEcs sim;
    // Two identical crates, one retuned to a heavier mass after creation.
    RigidBody spinner = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
    spinner.mass = 10.0f;
    const EntityId light = sim.SpawnBody(spinner, {0.0f, 100.0f, 0.0f});
    const EntityId heavy = sim.SpawnBody(spinner, {0.0f, 130.0f, 0.0f});
    const PhysicsBodyId light_body = sim.GetRigidBody(light)->body;
    const PhysicsBodyId heavy_body = sim.GetRigidBody(heavy)->body;
    ASSERT_TRUE(light_body.IsValid());
    EXPECT_NEAR(sim.WorldRef().GetBodyMass(light_body), 10.0f, 1.0e-4f);

    sim.WorldRef().SetBodyMass(heavy_body, 40.0f);
    EXPECT_NEAR(sim.WorldRef().GetBodyMass(heavy_body), 40.0f, 1.0e-4f);

    // The same torque on the same shape (and so `ScaleToMass`-scaled inertia):
    // after one step the heavy crate spins a quarter as fast as the light one.
    sim.WorldRef().AddTorque(light_body, {0.0f, 1000.0f, 0.0f});
    sim.WorldRef().AddTorque(heavy_body, {0.0f, 1000.0f, 0.0f});
    sim.Step(1);
    const float light_w = std::abs(sim.WorldRef().GetAngularVelocity(light_body).y);
    const float heavy_w = std::abs(sim.WorldRef().GetAngularVelocity(heavy_body).y);
    ASSERT_GT(light_w, 0.0f);
    EXPECT_NEAR(light_w / heavy_w, 4.0f, 0.5f)
        << "inertia did not grow proportionally with the retuned mass";
}

TEST(PhysicsBodyTuning, AMassChangeIsRejectedOnAStaticBody) {
    PhysicsEcs sim;
    const EntityId ground = sim.SpawnBody(
        BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
        {0.0f, -0.5f, 0.0f});
    const PhysicsBodyId body = sim.GetRigidBody(ground)->body;
    ASSERT_TRUE(body.IsValid());

    // A static body has no motion properties: the getter reports 0 (the mass a
    // dynamic body feels against it) and the setter is ignored with a log.
    EXPECT_EQ(sim.WorldRef().GetBodyMass(body), 0.0f);
    sim.WorldRef().SetBodyMass(body, 100.0f);
    EXPECT_EQ(sim.WorldRef().GetBodyMass(body), 0.0f);
    EXPECT_TRUE(sim.WorldRef().IsBodyValid(body));
}

TEST(PhysicsBodyTuning, ANonPositiveOrDeadMassChangeIsIgnored) {
    PhysicsEcs sim;
    RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
    crate.mass = 10.0f;
    const EntityId entity = sim.SpawnBody(crate, {0.0f, 100.0f, 0.0f});
    const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
    ASSERT_TRUE(body.IsValid());

    sim.WorldRef().SetBodyMass(body, 0.0f);
    sim.WorldRef().SetBodyMass(body, -5.0f);
    EXPECT_NEAR(sim.WorldRef().GetBodyMass(body), 10.0f, 1.0e-4f)
        << "a non-positive mass must be rejected, not zero the body out";

    const PhysicsBodyId dead{};
    sim.WorldRef().SetBodyMass(dead, 5.0f);
    EXPECT_FALSE(sim.WorldRef().IsBodyValid(dead));
    EXPECT_NEAR(sim.WorldRef().GetBodyMass(dead), 0.0f, 1.0e-6f);
}

TEST(PhysicsBodyTuning, FrictionAndRestitutionChangeLive) {
    PhysicsEcs sim;
    RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic); // friction 0.5
    const EntityId entity = sim.SpawnBody(crate, {0.0f, 100.0f, 0.0f});
    const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
    ASSERT_TRUE(body.IsValid());

    EXPECT_NEAR(sim.WorldRef().GetBodyFriction(body), 0.5f, 1.0e-4f);
    EXPECT_NEAR(sim.WorldRef().GetBodyRestitution(body), 0.0f, 1.0e-4f);
    sim.WorldRef().SetBodyFriction(body, 0.9f);
    sim.WorldRef().SetBodyRestitution(body, 0.3f);
    EXPECT_NEAR(sim.WorldRef().GetBodyFriction(body), 0.9f, 1.0e-4f);
    EXPECT_NEAR(sim.WorldRef().GetBodyRestitution(body), 0.3f, 1.0e-4f);
}

TEST(PhysicsBodyTuning, AFrictionlessBodySlidesFartherThanAGrippyOne) {
    // Two identical sliding worlds that differ only in the crate's live
    // friction: the solver must pick the change up on the next step.
    const auto slide_distance = [](float friction) {
        PhysicsEcs sim;
        (void)sim.SpawnBody(
            BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
            {0.0f, -0.5f, 0.0f});
        RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
        const EntityId entity = sim.SpawnBody(crate, {0.0f, 1.05f, 0.0f});
        const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
        sim.WorldRef().SetBodyFriction(body, friction);
        sim.WorldRef().SetLinearVelocity(body, {10.0f, 0.0f, 0.0f});
        // Two full seconds: the grippy crate comes to a stop, the frictionless
        // one keeps coasting, so the separation is the whole effect.
        sim.Step(120);
        return sim.WorldRef().GetPose(body).position.x;
    };
    const float free = slide_distance(0.02f);
    const float grippy = slide_distance(1.0f);
    EXPECT_GT(free, grippy * 2.0f)
        << std::format("frictionless slide {:.2f} m vs grippy slide {:.2f} m", free, grippy);
}

TEST(PhysicsBodyTuning, AHighRestitutionBodyBouncesHigherThanALowOne) {
    // Drop two identical worlds from 5 m; measure how high each rises after its
    // first ground contact.
    const auto bounce_height = [](float restitution) {
        PhysicsEcs sim;
        (void)sim.SpawnBody(
            BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
            {0.0f, -0.5f, 0.0f});
        RigidBody crate = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
        const EntityId entity = sim.SpawnBody(crate, {0.0f, 5.0f, 0.0f});
        const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
        sim.WorldRef().SetBodyRestitution(body, restitution);
        bool touched = false;
        float max_y = -1.0f;
        for (int i = 0; i < 180; ++i) {
            sim.Step(1);
            const float y = sim.WorldRef().GetPose(body).position.y;
            if (y < 1.2f) touched = true;
            if (touched && y > max_y) max_y = y;
        }
        return max_y;
    };
    const float dead = bounce_height(0.0f);
    const float bouncy = bounce_height(0.9f);
    EXPECT_GT(bouncy, dead + 1.0f)
        << std::format("bounced to {:.2f} m vs dead drop settled to {:.2f} m", bouncy, dead);
}

TEST(PhysicsEcs, CompoundColliderMakesOneBodyFromSeveralShapes) {
    PhysicsEcs sim;
    // A table: one top plus four legs, as a single compound body.
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    RigidBody table = BoxBody({2.0f, 0.1f, 1.0f}, BodyMotionType::Dynamic);
    const EntityId id = sim.SpawnBody(table, {0.0f, 5.0f, 0.0f});
    ASSERT_TRUE(sim.GetRigidBody(id)->body.IsValid());

    const PhysicsBodyId before = sim.GetRigidBody(id)->body;
    Collider collider{};
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
            ShapeDesc leg{};
            leg.type = ShapeType::Box;
            leg.box_extent = {0.1f, 0.5f, 0.1f};
            leg.position = {1.6f * static_cast<float>(sx), -0.6f, 0.8f * static_cast<float>(sz)};
            collider.children.push_back(leg);
        }
    }
    sim.Registry().AddComponent<Collider>(*sim.Registry().TryGetEntity(id), collider);

    // One entity is still exactly one body: the ground is the only other one.
    EXPECT_EQ(sim.WorldRef().BodyCount(), 2u);
    // ...but the body was rebuilt, because Jolt shapes are immutable once owned.
    const PhysicsBodyId after = sim.GetRigidBody(id)->body;
    EXPECT_TRUE(after.IsValid());
    EXPECT_NE(after, before);
    EXPECT_FALSE(sim.WorldRef().IsBodyValid(before));

    // The compound really is bigger: with the legs attached it rests on them, so
    // its centre is a top-thickness plus a leg-height above the ground.
    sim.Step(400);
    // Leg centre at -0.6 with half height 0.05... the leg's bottom face is at
    // -0.85 in the body frame, so the body rests with its centre 0.85 above the
    // ground. Without the legs it would lie flat at 0.05.
    EXPECT_NEAR(sim.GetTransform(id)->position.y, 0.85f, 0.15f) << "leg height below the table top";
}

TEST(PhysicsEcs, RemovingTheColliderRestoresTheSingleShapeBody) {
    PhysicsEcs sim;
    const EntityId id = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                      {0.0f, 5.0f, 0.0f});
    Collider collider{};
    ShapeDesc child{};
    child.type = ShapeType::Box;
    child.box_extent = {0.5f, 0.5f, 0.5f};
    child.position = {2.0f, 0.0f, 0.0f};
    collider.children.push_back(child);
    sim.Registry().AddComponent<Collider>(*sim.Registry().TryGetEntity(id), collider);
    const PhysicsBodyId compound = sim.GetRigidBody(id)->body;
    EXPECT_TRUE(compound.IsValid());

    sim.Registry().RemoveComponent<Collider>(*sim.Registry().TryGetEntity(id));
    const PhysicsBodyId restored = sim.GetRigidBody(id)->body;
    EXPECT_TRUE(restored.IsValid());
    EXPECT_NE(restored, compound);
    EXPECT_EQ(sim.WorldRef().BodyCount(), 1u);
}

TEST(PhysicsEcs, BodySequenceNumbersDistinguishAReusedIndex) {
    PhysicsEcs sim;
    const EntityId first = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                         {0.0f, 5.0f, 0.0f});
    const PhysicsBodyId old_handle = sim.GetRigidBody(first)->body;

    sim.Registry().DestroyEntity(*sim.Registry().TryGetEntity(first));

    const EntityId second = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 5.0f, 0.0f});
    const PhysicsBodyId new_handle = sim.GetRigidBody(second)->body;

    EXPECT_NE(new_handle, old_handle);
    EXPECT_FALSE(sim.WorldRef().IsBodyValid(old_handle));
    EXPECT_TRUE(sim.WorldRef().IsBodyValid(new_handle));
    if (new_handle.index() == old_handle.index()) {
        // The slot was reused: only the generation makes the handles distinct,
        // which is exactly what stops a stale handle naming the new body.
        EXPECT_NE(new_handle.sequence(), old_handle.sequence());
    }
}

TEST(PhysicsTypes, ShapeHashIsContentAddressed) {
    ShapeDesc box{};
    box.type = ShapeType::Box;
    box.box_extent = {1.0f, 2.0f, 3.0f};
    ShapeDesc copy = box;
    EXPECT_EQ(box.Hash(), copy.Hash());

    copy.box_extent = {1.0f, 2.0f, 4.0f};
    EXPECT_NE(box.Hash(), copy.Hash());

    // Hull geometry hashes by content, so two descriptors that share a mesh
    // agree even when they do not share the vector.
    const std::vector<glm::vec3> source{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    ShapeDesc hull{};
    hull.type = ShapeType::ConvexHull;
    hull.points = std::make_shared<const std::vector<glm::vec3>>(source);
    const std::size_t first = hull.Hash();
    hull.points = std::make_shared<const std::vector<glm::vec3>>(source);
    EXPECT_EQ(first, hull.Hash());

    const std::vector<glm::vec3> other{{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    hull.points = std::make_shared<const std::vector<glm::vec3>>(other);
    EXPECT_NE(first, hull.Hash());
}

TEST(PhysicsTypes, BodyIdsAreGenerationChecked) {
    const PhysicsBodyId invalid{};
    EXPECT_FALSE(invalid.IsValid());

    // The same encoding Jolt uses: index in the low 23 bits, sequence above.
    // Body index 0 is valid, so validity is not "index != 0".
    const PhysicsBodyId body{(7u << 23) | 3u};
    EXPECT_TRUE(body.IsValid());
    EXPECT_EQ(body.index(), 3u);
    EXPECT_EQ(body.sequence(), 7u);
    EXPECT_EQ(body, (PhysicsBodyId{(7u << 23) | 3u}));
    EXPECT_NE(body, (PhysicsBodyId{(8u << 23) | 3u}));
    EXPECT_NE(body, (PhysicsBodyId{(7u << 23) | 4u}));
}

TEST(PhysicsWorld, InitializeAndShutdownAreIdempotent) {
    World world;
    EXPECT_FALSE(world.IsInitialized());
    EXPECT_EQ(world.StepCount(), 0u);
    EXPECT_EQ(world.BodyCount(), 0u);

    ASSERT_TRUE(world.Initialize());
    EXPECT_TRUE(world.IsInitialized());
    // A second Initialize on a live world is a no-op that reports success.
    EXPECT_TRUE(world.Initialize());

    world.Shutdown();
    EXPECT_FALSE(world.IsInitialized());
    world.Shutdown(); // idempotent
    EXPECT_FALSE(world.IsInitialized());
}

TEST(PhysicsWorld, RejectsANonPositiveFixedStep) {
    World world;
    PhysicsWorldConfig config{};
    config.fixed_delta_time = 0.0f;
    EXPECT_FALSE(world.Initialize(config));
    EXPECT_FALSE(world.IsInitialized());
}

TEST(PhysicsWorld, StepsAnEmptySceneCleanly) {
    World world;
    ASSERT_TRUE(world.Initialize());

    for (int i = 0; i < 240; ++i) world.Step();

    EXPECT_EQ(world.StepCount(), 240u);
    EXPECT_EQ(world.LastUpdateErrors(), 0u);
    EXPECT_EQ(world.BodyCount(), 0u);
}

TEST(PhysicsWorld, StepsWithASinglePoolWorkerAndWithNone) {
    // The bridge must make progress in every pool configuration, including a
    // fully single-threaded one where the waiting thread is the only executor.
    auto& scheduler = VulkanShared::JobScheduler::Global();

    for (const unsigned int workers : {1u, 0u}) {
        scheduler.Configure(VulkanShared::JobBudget{.workers = workers});
        ASSERT_EQ(scheduler.WorkerCount(), workers);

        World world;
        ASSERT_TRUE(world.Initialize());
        EXPECT_EQ(world.JobConcurrency(), std::max(1u, workers));

        for (int i = 0; i < 30; ++i) world.Step();
        EXPECT_EQ(world.StepCount(), 30u);
        EXPECT_EQ(world.LastUpdateErrors(), 0u);
        world.Shutdown();
    }

    scheduler.Configure(VulkanShared::JobBudget{});
}

TEST(PhysicsWorld, JobConcurrencyNeverExceedsJoltsCeiling) {
    // Jolt supports at most 32 concurrent jobs regardless of pool size.
    auto& scheduler = VulkanShared::JobScheduler::Global();
    const unsigned int original = scheduler.WorkerCount();
    scheduler.Configure(VulkanShared::JobBudget{.workers = 64});

    World world;
    ASSERT_TRUE(world.Initialize());
    EXPECT_LE(world.JobConcurrency(), 32u);
    world.Shutdown();

    scheduler.Configure(VulkanShared::JobBudget{.workers = original});
}

TEST(PhysicsWorld, AllocatorOverrideIsInstalledAndCounted) {
    const auto before = VulkanEngine::Physics::detail::JoltMemory();
    {
        World world;
        ASSERT_TRUE(world.Initialize());
        const auto after_init = VulkanEngine::Physics::detail::JoltMemory();
        // Building the world allocates through the engine's hooks, which is the
        // only observable proof the override is in effect.
        EXPECT_GT(after_init.total_allocations, before.total_allocations);
        EXPECT_GT(after_init.total_bytes, before.total_bytes);
        EXPECT_GT(after_init.live_allocations, 0u);
    }
}

TEST(PhysicsWorld, BootstrapIsReferenceCountedAcrossWorlds) {
    EXPECT_FALSE(VulkanEngine::Physics::detail::IsJoltBootstrapped());

    World first;
    ASSERT_TRUE(first.Initialize());
    EXPECT_TRUE(VulkanEngine::Physics::detail::IsJoltBootstrapped());

    World second;
    ASSERT_TRUE(second.Initialize());
    first.Shutdown();
    EXPECT_TRUE(VulkanEngine::Physics::detail::IsJoltBootstrapped());

    second.Shutdown();
    EXPECT_FALSE(VulkanEngine::Physics::detail::IsJoltBootstrapped());
}

TEST(PhysicsWorld, LayerPolicyIsTableDriven) {
    World world;
    ASSERT_TRUE(world.Initialize());

    EXPECT_TRUE(world.LayersCollide(CollisionLayer::Default, CollisionLayer::Default));
    EXPECT_TRUE(world.LayersCollide(CollisionLayer::Default, CollisionLayer::Character));
    EXPECT_TRUE(world.LayersCollide(CollisionLayer::Character, CollisionLayer::Debris));
    // Sensors detect ordinary bodies...
    EXPECT_TRUE(world.LayersCollide(CollisionLayer::Sensor, CollisionLayer::Default));
    EXPECT_TRUE(world.LayersCollide(CollisionLayer::Sensor, CollisionLayer::Projectile));
    // ...but a sensor/sensor pair cannot mean anything.
    EXPECT_FALSE(world.LayersCollide(CollisionLayer::Sensor, CollisionLayer::Sensor));
}

TEST(PhysicsSchedule, TheSimulateSlotOccupiesTheFrameContract) {
    World world;
    ASSERT_TRUE(world.Initialize());

    ComponentRegistry registry;
    SystemSchedule schedule;
    std::vector<std::string> order;
    schedule.Add(
        SystemSlot::PhysicsSimulate,
        [&](ComponentRegistry&, float) {
            world.Step();
            order.push_back("simulate");
        },
        "physics.simulate");
    schedule.AddBefore(
        SystemSlot::PhysicsSimulate,
        [&](ComponentRegistry&, float) { order.push_back("input"); }, "gameplay.input");
    schedule.AddAfter(
        SystemSlot::PhysicsSimulate,
        [&](ComponentRegistry&, float) { order.push_back("writeback"); }, "gameplay.writeback");

    for (int step = 0; step < 3; ++step) schedule.Run(registry, 1.0f / 60.0f);

    EXPECT_EQ(world.StepCount(), 3u);
    const std::vector<std::string> expected{
        "input", "simulate", "writeback",
        "input", "simulate", "writeback",
        "input", "simulate", "writeback",
    };
    EXPECT_EQ(order, expected);
}


// ── Rung 4: events, queries, debug visualization ────────────────────────────

TEST(PhysicsEvents, AContactIsReportedOnceWhenTheBodiesSettle) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId crate = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                        {0.0f, 1.2f, 0.0f});

    std::vector<ContactEvent> events;
    sim.WorldRef().SetContactCallback([&events](const ContactEvent& event) { events.push_back(event); });

    sim.Step(120); // long enough to land and stop moving

    std::size_t added = 0;
    std::size_t removed = 0;
    for (const ContactEvent& event : events) {
        if (event.phase == ContactPhase::Added) ++added;
        if (event.phase == ContactPhase::Removed) ++removed;
    }
    EXPECT_GE(added, 1u) << "the crate never reported touching the ground";
    // Jolt reports a synthetic Removed when a body falls asleep, so Removed is
    // allowed — but never more often than Added, and never before the first Added.
    EXPECT_LE(removed, added);
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.front().phase, ContactPhase::Added);

    // Every event resolves to the live crate entity, with its generation intact.
    const std::uint64_t crate_payload = PackEntityId(crate);
    bool saw_crate = false;
    for (const ContactEvent& event : events) {
        if (event.user_data_a == crate_payload || event.user_data_b == crate_payload) {
            saw_crate = true;
            const EntityId resolved = event.user_data_a == crate_payload ? UnpackEntityId(event.user_data_a)
                                                                         : UnpackEntityId(event.user_data_b);
            EXPECT_NE(sim.Registry().TryGetEntity(resolved), nullptr);
        }
    }
    EXPECT_TRUE(saw_crate);
}

TEST(PhysicsEvents, ReplayOrderIsDeterministicAcrossRuns) {
    // Two identical worlds, two steps each, must replay the same sequence: the
    // job order is not deterministic, the replayed order is.
    auto run = [] {
        PhysicsEcs sim;
        (void)sim.SpawnBody(BoxBody({20.0f, 1.0f, 20.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                            {0.0f, -0.5f, 0.0f});
        for (int i = 0; i < 6; ++i) {
            (void)sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                {static_cast<float>(i) * 2.0f - 5.0f, 0.6f + static_cast<float>(i), 0.0f});
        }
        std::vector<std::uint64_t> order;
        sim.WorldRef().SetContactCallback([&order](const ContactEvent& event) {
            order.push_back((event.user_data_a << 32) ^ event.user_data_b ^
                            (static_cast<std::uint64_t>(event.phase) << 60));
        });
        sim.Step(90);
        return order;
    };

    const std::vector<std::uint64_t> first = run();
    const std::vector<std::uint64_t> second = run();
    EXPECT_FALSE(first.empty());
    EXPECT_EQ(first, second);
}

TEST(PhysicsEvents, AnOverlappingSensorReportsWithoutResolving) {
    PhysicsEcs sim;
    // A sensor volume and a dynamic body that falls through it.
    RigidBody sensor = BoxBody({2.0f, 2.0f, 2.0f}, BodyMotionType::Static, CollisionLayer::Sensor, 0.0f);
    sensor.sensor = true;
    const EntityId sensor_id = sim.SpawnBody(sensor, {0.0f, 5.0f, 0.0f});
    const EntityId faller = sim.SpawnBody(BoxBody({0.5f, 0.5f, 0.5f}, BodyMotionType::Dynamic),
                                         {0.0f, 9.0f, 0.0f});

    std::size_t sensor_contacts = 0;
    sim.WorldRef().SetContactCallback([&](const ContactEvent& event) {
        if (event.sensor) ++sensor_contacts;
    });

    sim.Step(120);

    EXPECT_GT(sensor_contacts, 0u) << "the sensor never reported an overlap";
    // The sensor did not stop the faller: it is a trigger, not a collider.
    EXPECT_LT(sim.GetTransform(faller)->position.y, 5.0f);
    EXPECT_TRUE(sim.GetRigidBody(sensor_id)->body.IsValid());
}

TEST(PhysicsEvents, ActivationIsReplayedWithTheOwningEntity) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId crate = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                         {0.0f, 6.0f, 0.0f});

    std::vector<std::uint64_t> payloads;
    sim.WorldRef().SetActivationCallback([&payloads](const auto& event) {
        payloads.push_back(event.user_data);
    });

    sim.Step(600); // land, settle and fall asleep

    EXPECT_FALSE(payloads.empty()) << "no activation events were replayed";
    const std::uint64_t crate_payload = PackEntityId(crate);
    const bool saw_crate = std::ranges::find(payloads, crate_payload) != payloads.end();
    EXPECT_TRUE(saw_crate);
}

TEST(PhysicsQueries, RayCastFindsTheExpectedEntity) {
    PhysicsEcs sim;
    const EntityId floor = sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                                        {0.0f, -0.5f, 0.0f});
    const EntityId crate = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                        {0.0f, 4.0f, 0.0f});

    RayCastRequest request{};
    request.from = {0.0f, 10.0f, 0.0f};
    request.to = {0.0f, -10.0f, 0.0f};
    const auto hit = sim.WorldRef().CastRay(request);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(UnpackEntityId(sim.WorldRef().GetUserData(hit->body)), crate);
    // The top face of a box centred at y = 4 with half extent 0.5.
    EXPECT_NEAR(hit->position.y, 4.5f, 0.05f);
    EXPECT_NEAR(hit->normal.y, 1.0f, 0.05f);

    // A ray that misses everything resolves to nothing.
    RayCastRequest miss{};
    miss.from = {100.0f, 10.0f, 100.0f};
    miss.to = {100.0f, -10.0f, 100.0f};
    EXPECT_FALSE(sim.WorldRef().CastRay(miss).has_value());

    // The layer mask is honoured: a mask with only Character set cannot hit
    // either body, which are Default and Dynamic.
    RayCastRequest masked = request;
    masked.layer_mask = 1u << static_cast<std::uint32_t>(CollisionLayer::Character);
    EXPECT_FALSE(sim.WorldRef().CastRay(masked).has_value());

    // Motion-type filtering is honoured too.
    RayCastRequest dynamics_only = request;
    dynamics_only.hit_static = false;
    const auto dynamic_hit = sim.WorldRef().CastRay(dynamics_only);
    ASSERT_TRUE(dynamic_hit.has_value());
    EXPECT_EQ(UnpackEntityId(sim.WorldRef().GetUserData(dynamic_hit->body)), crate);
    EXPECT_NE(floor, crate);
}

TEST(PhysicsQueries, ShapeSweepStopsBeforeTheBodyItWouldHit) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});

    ShapeCastRequest request{};
    request.shape.type = ShapeType::Sphere;
    request.shape.radius = 0.5f;
    request.from = {0.0f, 5.0f, 0.0f};
    request.to = {0.0f, 0.0f, 0.0f};

    const auto hit = sim.WorldRef().CastShape(request);
    ASSERT_TRUE(hit.has_value());
    // The sphere's centre stops one radius above the ground plane.
    EXPECT_NEAR(hit->position.y, 0.5f, 0.1f);
    EXPECT_GT(hit->normal.y, 0.5f);
}

TEST(PhysicsDebug, SolidTrianglesCarryJoltsBackFaceCullingContract) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({2.0f, 1.0f, 2.0f}, BodyMotionType::Static, CollisionLayer::Default,
                                0.0f),
                        {0.0f, -0.5f, 0.0f});

    PhysicsDebugSettings settings{};
    settings.draw_shapes = true;
    settings.wireframe = false;
    PhysicsDebugGeometry solid;
    sim.WorldRef().CollectDebugGeometry(settings, solid);
    ASSERT_FALSE(solid.triangles.empty());
    ASSERT_EQ(solid.triangles.size() % 3, 0u);
    // Jolt draws a solid box through `DrawGeometry` with `ECullMode::CullBackFace`
    // and documents `DrawTriangle` as "a single back face culled triangle": the
    // renderer realizes the cull, so every triangle must carry that contract to
    // the overlay, which owns the screen-space winding test.
    for (const auto& vertex : solid.triangles) {
        ASSERT_EQ(vertex.culling, PhysicsDebugFaceCulling::BackFace);
    }
}

TEST(PhysicsDebug, CollectedGeometryIsNonEmptyAndFilterable) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({2.0f, 1.0f, 2.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    (void)sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic), {0.0f, 4.0f, 0.0f});

    PhysicsDebugGeometry geometry;
    PhysicsDebugSettings settings{};
    settings.draw_shapes = true;
    settings.wireframe = true;
    sim.WorldRef().CollectDebugGeometry(settings, geometry);
    EXPECT_FALSE(geometry.lines.empty()) << "wireframe shapes should emit lines";
    EXPECT_EQ(geometry.lines.size() % 2, 0u) << "a line list is pairs of vertices";

    PhysicsDebugGeometry solid;
    settings.wireframe = false;
    sim.WorldRef().CollectDebugGeometry(settings, solid);
    EXPECT_FALSE(solid.triangles.empty()) << "solid shapes should emit triangles";
    EXPECT_EQ(solid.triangles.size() % 3, 0u) << "a triangle list is triples of vertices";

    // A layer mask that excludes everything draws nothing.
    PhysicsDebugGeometry masked;
    settings.layer_mask = 1u << static_cast<std::uint32_t>(CollisionLayer::Character);
    sim.WorldRef().CollectDebugGeometry(settings, masked);
    EXPECT_TRUE(masked.lines.empty());
    EXPECT_TRUE(masked.triangles.empty());

    // Clearing resets both lists, which is what a per-frame overlay relies on.
    masked.Clear();
    EXPECT_TRUE(masked.lines.empty());
}


// ── Rung 5: characters ──────────────────────────────────────────────────────

/// A 1.8 m humanoid capsule: radius 0.4, so the cylindrical half is 0.5 and the
/// resting centre sits 0.9 above the surface.
[[nodiscard]] CharacterDesc HumanoidDesc() {
    CharacterDesc desc{};
    desc.shape.type = ShapeType::Capsule;
    desc.shape.radius = 0.4f;
    desc.shape.height = 1.8f;
    desc.max_slope_angle_degrees = 50.0f;
    return desc;
}

TEST(PhysicsCharacter, FallsAndLandsOnTheGround) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {0.0f, 4.0f, 0.0f});
    ASSERT_TRUE(sim.GetCharacter(id)->character.IsValid());

    sim.Step(180);

    EXPECT_TRUE(sim.GetCharacter(id)->on_ground) << "the character never landed";
    // Capsule total half height = height/2 = 0.9, so it rests with its centre there.
    EXPECT_NEAR(sim.GetTransform(id)->position.y, 0.9f, 0.1f);
    EXPECT_EQ(sim.WorldRef().CharacterCount(), 1u);
}

TEST(PhysicsCharacter, WalksUpAStepStaircase) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    // Three 0.25 m steps marching in +X, each 0.8 m deep. 0.25 is below the
    // character's 0.4 m step height, so it should climb rather than stop.
    for (int i = 0; i < 3; ++i) {
        RigidBody step = BoxBody({0.4f, 0.5f, 5.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f);
        (void)sim.SpawnBody(step, {0.4f + static_cast<float>(i) * 0.8f,
                                   -0.25f + 0.25f * static_cast<float>(i + 1), 0.0f});
    }

    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {-6.0f, 0.9f, 0.0f});
    sim.GetCharacter(id)->desired_velocity = {1.5f, 0.0f, 0.0f};

    // Track the high point rather than the final pose: the staircase has no wall
    // past its top, so a character that climbed it walks off the far end and
    // falls back to the ground. Climbing is the interesting part.
    float highest = 0.0f;
    bool on_ground_at_top = false;
    for (int step = 0; step < 600; ++step) {
        sim.Step(1);
        const float y = sim.GetTransform(id)->position.y;
        if (y > highest) {
            highest = y;
            on_ground_at_top = sim.GetCharacter(id)->on_ground;
        }
    }

    EXPECT_GT(sim.GetTransform(id)->position.x, 2.0f) << "the character did not cross the staircase";
    // The top step's surface is at y = 0.75, so a standing character centres at 1.65.
    EXPECT_GT(highest, 1.4f) << "the character did not climb the steps";
    EXPECT_TRUE(on_ground_at_top) << "the character climbed without ever standing on a step";
}

TEST(PhysicsCharacter, WalksUpASlope) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    // A 25-degree ramp: shallower than the character's 50-degree limit, so it is
    // walkable ground rather than a wall.
    Entity& ramp = sim.Registry().CreateEntity();
    auto& ramp_transform = sim.Registry().AddComponent<Transform>(ramp);
    ramp_transform.position = {6.0f, 0.6f, 0.0f};
    ramp_transform.rotation = glm::angleAxis(glm::radians(25.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    RigidBody ramp_body = BoxBody({6.0f, 1.0f, 4.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f);
    sim.Registry().AddComponent<RigidBody>(ramp, ramp_body);

    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {-1.0f, 0.9f, 0.0f});
    sim.GetCharacter(id)->desired_velocity = {1.2f, 0.0f, 0.0f};

    sim.Step(480);

    // Height is gained by walking up the ramp, and the character stays on it.
    EXPECT_GT(sim.GetTransform(id)->position.y, 1.2f);
    EXPECT_TRUE(sim.GetCharacter(id)->on_ground);
}

TEST(PhysicsCharacter, RidesAMovingPlatform) {
    PhysicsEcs sim;
    RigidBody platform = BoxBody({4.0f, 0.5f, 4.0f}, BodyMotionType::Kinematic);
    const EntityId platform_id = sim.SpawnBody(platform, {0.0f, 0.0f, 0.0f});
    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {0.0f, 1.1f, 0.0f});

    sim.Step(90); // settle onto the platform
    ASSERT_TRUE(sim.GetCharacter(id)->on_ground) << "the character never stood on the platform";
    const float start_x = sim.GetTransform(id)->position.x;

    // Drive the platform 3 m in +X over one second. The kinematic body is moved
    // by its authored Transform, and the character rides the surface velocity.
    for (int i = 0; i < 60; ++i) {
        sim.GetTransform(platform_id)->position.x += 3.0f / 60.0f;
        sim.Step(1);
    }

    EXPECT_GT(sim.GetTransform(id)->position.x - start_x, 2.0f)
        << "the character did not ride the platform";
    EXPECT_TRUE(sim.GetCharacter(id)->on_ground);
}

TEST(PhysicsCharacter, DestroyingTheCharacterReleasesItAndInvalidatesTheHandle) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {0.0f, 2.0f, 0.0f});
    const CharacterId handle = sim.GetCharacter(id)->character;
    ASSERT_TRUE(handle.IsValid());
    ASSERT_EQ(sim.WorldRef().CharacterCount(), 1u);

    sim.Registry().DestroyEntity(*sim.Registry().TryGetEntity(id));

    EXPECT_EQ(sim.WorldRef().CharacterCount(), 0u);
    EXPECT_FALSE(sim.WorldRef().IsCharacterValid(handle));
}

// ── Render interpolation ─────────────────────────────────────────────────────
//
// `FrameStepClock` produces `interpolation_alpha` per rendered frame; physics is
// its first consumer. The contract these tests pin down: a tracked body keeps the
// last two fixed-step poses, `alpha` blends between them, and a body nobody
// tracked returns its live pose (which is also what an untracked body costs: a
// single map lookup, not a per-step entry).

void ExpectPoseNear(const BodyPose& actual, const BodyPose& expected, float tolerance) {
    EXPECT_NEAR(actual.position.x, expected.position.x, tolerance);
    EXPECT_NEAR(actual.position.y, expected.position.y, tolerance);
    EXPECT_NEAR(actual.position.z, expected.position.z, tolerance);
    // Quaternions are a double cover, so compare against whichever sign lands
    // nearest rather than assuming the same hemisphere.
    const float same = std::abs(glm::dot(actual.rotation, expected.rotation));
    EXPECT_NEAR(same, 1.0f, tolerance) << "the interpolated rotation does not match the step pose";
}

[[nodiscard]] BodyDesc BoxDesc(glm::vec3 extent, BodyMotionType motion, float mass = 1.0f) {
    BodyDesc desc{};
    desc.shape.type = ShapeType::Box;
    desc.shape.box_extent = extent;
    desc.motion = motion;
    desc.mass = mass;
    desc.friction = 0.5f;
    return desc;
}

/// A body created straight on the facade carries its pose in the descriptor: the
/// ECS bridge is what reads a `Transform`.
[[nodiscard]] BodyDesc PlacedBoxDesc(glm::vec3 extent, glm::vec3 position,
                                     BodyMotionType motion = BodyMotionType::Dynamic,
                                     float mass = 1.0f) {
    BodyDesc desc = BoxDesc(extent, motion, mass);
    desc.position = position;
    return desc;
}

TEST(PhysicsInterpolation, ATrackedBodyBlendsTheLastTwoFixedSteps) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId entity = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 8.0f, 0.0f});
    const PhysicsBodyId body = sim.GetRigidBody(entity)->body;
    World& world = sim.WorldRef();

    // Attaching through the bridge is what tracks a body: a body the ECS owns is
    // a body the renderer can draw.
    ASSERT_TRUE(world.IsRenderPoseTracked(body)) << "AttachBody did not track the body for rendering";

    sim.Step(10); // let it gather speed, but stay in the air

    const BodyPose before = world.GetPose(body);
    sim.Step(1);
    const BodyPose after = world.GetPose(body);
    ASSERT_LT(after.position.y, before.position.y) << "the crate is not falling";

    // alpha 0 is the pose the previous step ended on, alpha 1 is where this step
    // left the body — exactly, not approximately: the endpoints are not blended.
    ExpectPoseNear(world.GetRenderPose(body, 0.0f), before, 1e-5f);
    ExpectPoseNear(world.GetRenderPose(body, 1.0f), after, 1e-5f);
    // alpha is clamped, so a clock that overshoots cannot extrapolate past the
    // step's own pose.
    ExpectPoseNear(world.GetRenderPose(body, -3.0f), before, 1e-5f);
    ExpectPoseNear(world.GetRenderPose(body, 3.0f), after, 1e-5f);

    // Halfway between the two step poses, and strictly inside them.
    const BodyPose half = world.GetRenderPose(body, 0.5f);
    EXPECT_NEAR(half.position.y, 0.5f * (before.position.y + after.position.y), 1e-4f);
    EXPECT_GT(half.position.y, after.position.y);
    EXPECT_LT(half.position.y, before.position.y);
    EXPECT_NEAR(half.position.x, before.position.x, 1e-5f);
    EXPECT_NEAR(half.position.z, before.position.z, 1e-5f);

    // The blend is a *render* pose: the gameplay truth in `Transform` stays the
    // fixed-step pose the last step wrote.
    const Transform* transform = sim.GetTransform(entity);
    ASSERT_NE(transform, nullptr);
    EXPECT_NEAR(transform->position.y, after.position.y, 1e-5f);
}

TEST(PhysicsInterpolation, AnUntrackedBodyReturnsItsLivePose) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 256;
    ASSERT_TRUE(world.Initialize(config));

    (void)world.CreateBody(BoxDesc({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, 0.0f));
    // A body created straight on the facade, with nobody tracking it.
    const PhysicsBodyId loose =
        world.CreateBody(PlacedBoxDesc({1.0f, 1.0f, 1.0f}, {0.0f, 7.0f, 0.0f}));
    ASSERT_TRUE(loose.IsValid());
    EXPECT_FALSE(world.IsRenderPoseTracked(loose));

    world.Step();
    const BodyPose live = world.GetPose(loose);
    for (const float alpha : {0.0f, 0.25f, 0.5f, 1.0f}) {
        ExpectPoseNear(world.GetRenderPose(loose, alpha), live, 1e-6f);
    }
}

TEST(PhysicsInterpolation, UntrackingReturnsTheLivePoseAgain) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 256;
    ASSERT_TRUE(world.Initialize(config));

    (void)world.CreateBody(BoxDesc({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, 0.0f));
    const PhysicsBodyId body =
        world.CreateBody(PlacedBoxDesc({1.0f, 1.0f, 1.0f}, {0.0f, 9.0f, 0.0f}));
    ASSERT_TRUE(body.IsValid());

    world.TrackRenderPose(body);
    EXPECT_TRUE(world.IsRenderPoseTracked(body));
    world.Step();
    world.Step();

    // Tracked and falling: alpha 0 is the pose the previous step ended on, which
    // sits *above* the live pose this step produced. That one-step lag is the
    // whole point of interpolating.
    const BodyPose live = world.GetPose(body);
    EXPECT_GT(world.GetRenderPose(body, 0.0f).position.y, live.position.y);

    world.UntrackRenderPose(body);
    EXPECT_FALSE(world.IsRenderPoseTracked(body));
    // Untracked, every alpha is the live pose — including the ones that were
    // interpolated a moment ago.
    ExpectPoseNear(world.GetRenderPose(body, 0.0f), live, 1e-6f);
    ExpectPoseNear(world.GetRenderPose(body, 0.5f), live, 1e-6f);
    ExpectPoseNear(world.GetRenderPose(body, 1.0f), live, 1e-6f);

    // Untracking twice, and untracking something that was never tracked, are both
    // no-ops rather than errors.
    world.UntrackRenderPose(body);
    world.UntrackRenderPose(PhysicsBodyId{0xffffu});
    world.UntrackRenderPose(PhysicsBodyId{});
}

TEST(PhysicsInterpolation, DestroyingATrackedBodyDropsItsInterpolationState) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 256;
    ASSERT_TRUE(world.Initialize(config));

    const PhysicsBodyId body =
        world.CreateBody(PlacedBoxDesc({1.0f, 1.0f, 1.0f}, {0.0f, 3.0f, 0.0f}));
    world.TrackRenderPose(body);
    EXPECT_TRUE(world.IsRenderPoseTracked(body));

    world.DestroyBody(body);

    EXPECT_FALSE(world.IsRenderPoseTracked(body));
    EXPECT_FALSE(world.IsBodyValid(body));
}

TEST(PhysicsInterpolation, ARebuiltBodyKeepsInterpolatingFromItsCarriedPose) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId entity = sim.SpawnBody(BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
                                          {0.0f, 8.0f, 0.0f});
    World& world = sim.WorldRef();
    sim.Step(10);

    const PhysicsBodyId old_body = sim.GetRigidBody(entity)->body;
    const BodyPose carried = world.GetPose(old_body);
    ASSERT_TRUE(world.IsRenderPoseTracked(old_body));

    // A collider change rebuilds the body at its live pose (the hooks in
    // `GameEngine::Setup` do exactly this).
    Entity* raw = sim.Registry().TryGetEntity(entity);
    ASSERT_NE(raw, nullptr);
    Collider collider{};
    sim.Registry().AddComponent<Collider>(*raw, collider);

    const PhysicsBodyId new_body = sim.GetRigidBody(entity)->body;
    ASSERT_TRUE(new_body.IsValid());
    ASSERT_NE(new_body, old_body) << "the rebuild did not replace the handle";
    // The tracking moved to the new handle: the old one and its state are gone.
    EXPECT_TRUE(world.IsRenderPoseTracked(new_body));
    EXPECT_FALSE(world.IsRenderPoseTracked(old_body));
    // And it is continuous: the render pose is the pose the body was rebuilt at,
    // not a blend from a pose the new handle never owned.
    ExpectPoseNear(world.GetRenderPose(new_body, 0.0f), carried, 1e-5f);
    ExpectPoseNear(world.GetRenderPose(new_body, 1.0f), carried, 1e-5f);

    // The next step picks the interpolation back up. Its own rotation point is the
    // carried pose the rebuild seeded, so alpha 0 still reads the rebuild's pose
    // and alpha 1 is where the step moved the body to.
    sim.Step(1);
    ExpectPoseNear(world.GetRenderPose(new_body, 0.0f), carried, 1e-5f);
    EXPECT_LT(world.GetRenderPose(new_body, 1.0f).position.y, carried.position.y);
}

TEST(PhysicsInterpolation, ACharacterBlendsTheLastTwoSweeps) {
    PhysicsEcs sim;
    (void)sim.SpawnBody(BoxBody({40.0f, 1.0f, 40.0f}, BodyMotionType::Static, CollisionLayer::Default, 0.0f),
                        {0.0f, -0.5f, 0.0f});
    const EntityId id = sim.SpawnCharacter(HumanoidDesc(), {0.0f, 8.0f, 0.0f});
    World& world = sim.WorldRef();
    const CharacterId character = sim.GetCharacter(id)->character;
    ASSERT_TRUE(world.IsCharacterRenderPoseTracked(character)) << "AttachCharacter did not track the character";

    sim.Step(20); // still falling

    const BodyPose before = world.GetCharacterPose(character);
    sim.Step(1);
    const BodyPose after = world.GetCharacterPose(character);
    ASSERT_LT(after.position.y, before.position.y) << "the character is not falling";

    // The character is swept in the `PhysicsInput` slot and captured in `Step`,
    // so one rotation point governs both kinds of actor.
    ExpectPoseNear(world.GetCharacterRenderPose(character, 0.0f), before, 1e-5f);
    ExpectPoseNear(world.GetCharacterRenderPose(character, 1.0f), after, 1e-5f);
    const BodyPose half = world.GetCharacterRenderPose(character, 0.5f);
    EXPECT_GT(half.position.y, after.position.y);
    EXPECT_LT(half.position.y, before.position.y);

    // Untracked characters read their live pose, exactly like untracked bodies.
    EXPECT_FALSE(world.IsCharacterRenderPoseTracked(CharacterId{}));
    world.UntrackCharacterRenderPose(character);
    EXPECT_FALSE(world.IsCharacterRenderPoseTracked(character));
    ExpectPoseNear(world.GetCharacterRenderPose(character, 0.0f), after, 1e-6f);
}

// ── Cooked-shape disk cache ──────────────────────────────────────────────────
//
// A cooked shape is a cache, never primary data: the blob is keyed by (Jolt
// version, shape hash) and every read path failure is a miss that re-cooks. These
// tests run the *real* disk protocol against a temp directory, so a schema change
// that quietly breaks restoration shows up here rather than as a slow load.

/// A clean cache directory, named after the test. Each gtest case is its own ctest
/// entry, so the run stamp keeps two concurrent runs (or two agents in one build
/// directory) from sharing a directory.
[[nodiscard]] std::filesystem::path FreshShapeCacheDir(std::string_view name) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "vkeng_shape_cache_test" /
        std::format("{}_{}_{}", name, stamp, sequence.fetch_add(1));
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    return directory;
}

[[nodiscard]] std::size_t CountCacheFiles(const std::filesystem::path& directory) {
    std::size_t count = 0;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (entry.is_regular_file(error)) ++count;
    }
    return count;
}

[[nodiscard]] PhysicsWorldConfig ShapeCacheConfig(const std::filesystem::path& directory) {
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 256;
    config.shape_cache_directory = directory;
    return config;
}

/// The box the cache tests use, so two worlds of a pair hash identically. A body
/// created straight on the facade carries its pose in the descriptor (the ECS
/// bridge is what reads a `Transform`).
[[nodiscard]] BodyDesc CachedBoxDesc(glm::vec3 position = {0.0f, 5.0f, 0.0f}) {
    return PlacedBoxDesc({1.0f, 1.0f, 1.0f}, position);
}

TEST(PhysicsShapeCache, ACookedShapeIsRestoredFromDiskInsteadOfBeingRecooked) {
    const std::filesystem::path directory = FreshShapeCacheDir("ACookedShapeIsRestoredFromDiskInsteadOfBeingRecooked");

    PhysicsBodyId first_body{};
    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        first_body = world.CreateBody(CachedBoxDesc());
        ASSERT_TRUE(first_body.IsValid());
        // The first world has nothing to restore, so its one geometry is cooked
        // and written.
        EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 1u);
    }
    EXPECT_GE(CountCacheFiles(directory), 1u) << "nothing was written to the cache directory";

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        const PhysicsBodyId body = world.CreateBody(CachedBoxDesc());
        ASSERT_TRUE(body.IsValid());
        EXPECT_EQ(world.ShapeCacheDiskHits(), 1u) << "the second world did not restore the blob";
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
        // A restored shape has to be a *usable* shape, not merely a non-null
        // pointer: the same extent, in the same place.
        EXPECT_TRUE(world.IsBodyValid(body));
        RayCastRequest request{};
        request.from = {0.0f, 10.0f, 0.0f};
        request.to = {0.0f, -10.0f, 0.0f};
        const auto hit = world.CastRay(request);
        ASSERT_TRUE(hit.has_value()) << "a ray straight down misses the restored box";
        EXPECT_EQ(hit->body, body);
        // The top face of a box centred at y = 5 with half extent 0.5.
        EXPECT_NEAR(hit->position.y, 5.5f, 0.05f);
        EXPECT_NEAR(hit->normal.y, 1.0f, 0.05f);

        // Within one world the in-memory cache still serves repeats, so a scene
        // of a thousand crates cooks one shape and touches the disk once.
        const PhysicsBodyId second = world.CreateBody(CachedBoxDesc({2.0f, 5.0f, 0.0f}));
        ASSERT_TRUE(second.IsValid());
        EXPECT_EQ(world.ShapeCacheDiskHits(), 1u);
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
        EXPECT_EQ(world.CachedShapeCount(), 1u);
    }
}

TEST(PhysicsShapeCache, WithTheCacheDisabledEveryShapeIsCookedAndNothingIsWritten) {
    const std::filesystem::path directory = FreshShapeCacheDir("WithTheCacheDisabledEveryShapeIsCookedAndNothingIsWritten");

    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 256;
    ASSERT_TRUE(world.Initialize(config)); // shape_cache_directory left empty
    ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());

    // The counters only move when the cache is on: "disabled" is not a miss.
    EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
    EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
    std::error_code error;
    EXPECT_FALSE(std::filesystem::exists(directory, error)) << "a disabled cache created a directory";
}

TEST(PhysicsShapeCache, ACorruptBlobIsAMissThatRecooksAndHealsTheCache) {
    const std::filesystem::path directory = FreshShapeCacheDir("ACorruptBlobIsAMissThatRecooksAndHealsTheCache");

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());
    }

    // Corrupt the *payload* while leaving the header intact: this is the case the
    // header alone cannot catch, and the one that would hand Jolt a garbage shape
    // id. Flip the last byte, which is always payload.
    std::size_t corrupted = 0;
    {
        std::error_code error;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(directory, error)) {
            if (!entry.is_regular_file(error)) continue;
            std::vector<char> bytes;
            {
                std::ifstream in(entry.path(), std::ios::binary);
                bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            }
            ASSERT_FALSE(bytes.empty()) << "the cache blob is unexpectedly empty";
            bytes.back() = static_cast<char>(bytes.back() ^ 0x5a);
            std::ofstream out(entry.path(), std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            ++corrupted;
        }
    }
    ASSERT_EQ(corrupted, 1u);

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        const PhysicsBodyId body = world.CreateBody(CachedBoxDesc());
        // The world did not fail, and the shape was re-cooked rather than
        // restored from the damaged bytes.
        EXPECT_TRUE(body.IsValid());
        EXPECT_TRUE(world.IsBodyValid(body));
        EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 1u);

        RayCastRequest request{};
        request.from = {0.0f, 10.0f, 0.0f};
        request.to = {0.0f, -10.0f, 0.0f};
        const auto hit = world.CastRay(request);
        ASSERT_TRUE(hit.has_value()) << "the re-cooked box is not there";
        EXPECT_EQ(hit->body, body);
    }

    // The re-cook wrote the blob back, so the damage healed itself.
    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());
        EXPECT_EQ(world.ShapeCacheDiskHits(), 1u) << "the cache did not heal after a corrupt blob";
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
    }
}

TEST(PhysicsShapeCache, GarbageInTheCacheDirectoryIsAMissAndNeverAFailure) {
    const std::filesystem::path directory = FreshShapeCacheDir("GarbageInTheCacheDirectoryIsAMissAndNeverAFailure");

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());
    }
    ASSERT_GE(CountCacheFiles(directory), 1u);

    // Replace every blob with something that is not a shape blob at all, and add
    // an extra file that merely *looks* like one. Neither may be trusted on the
    // strength of its name.
    std::error_code error;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        std::ofstream out(entry.path(), std::ios::binary | std::ios::trunc);
        out << "not a jolt shape, not even close";
    }
    {
        std::ofstream bogus(directory / "shape_00000000_0000000000000000.jshape",
                            std::ios::binary | std::ios::trunc);
        bogus << "definitely not a shape";
    }

    World world;
    ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
    const PhysicsBodyId body = world.CreateBody(CachedBoxDesc());
    ASSERT_TRUE(body.IsValid()) << "a garbage cache directory failed the world";
    EXPECT_TRUE(world.IsBodyValid(body));
    EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
    EXPECT_EQ(world.ShapeCacheDiskMisses(), 1u);

    RayCastRequest request{};
    request.from = {0.0f, 10.0f, 0.0f};
    request.to = {0.0f, -10.0f, 0.0f};
    const auto hit = world.CastRay(request);
    ASSERT_TRUE(hit.has_value()) << "the re-cooked box is not there";
    EXPECT_NEAR(hit->position.y, 5.5f, 0.05f);
}

TEST(PhysicsShapeCache, AnUnusableCacheDirectoryDisablesTheCacheInsteadOfFailingTheWorld) {
    const std::filesystem::path path =
        FreshShapeCacheDir("AnUnusableCacheDirectoryDisablesTheCacheInsteadOfFailingTheWorld");
    // A regular file where the directory should be: the cache cannot be created,
    // and a read-only data directory is the real-world version of this.
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "not a directory";
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(path, error));

    World world;
    ASSERT_TRUE(world.Initialize(ShapeCacheConfig(path))) << "an unusable cache directory failed the world";
    const PhysicsBodyId body = world.CreateBody(CachedBoxDesc());
    EXPECT_TRUE(world.IsBodyValid(body));
    // The cache is simply off: neither counter moves, and the "directory" is
    // still the file the host put there.
    EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
    EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
    EXPECT_TRUE(std::filesystem::is_regular_file(path, error));
}

TEST(PhysicsShapeCache, ADifferentShapeIsADifferentBlob) {
    const std::filesystem::path directory = FreshShapeCacheDir("ADifferentShapeIsADifferentBlob");

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());
        ASSERT_TRUE(world.CreateBody(PlacedBoxDesc({2.0f, 1.0f, 3.0f}, {4.0f, 5.0f, 0.0f})).IsValid());
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 2u);
        EXPECT_EQ(world.ShapeCacheDiskHits(), 0u);
    }
    EXPECT_EQ(CountCacheFiles(directory), 2u) << "two geometries should be two blobs";

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        ASSERT_TRUE(world.CreateBody(CachedBoxDesc()).IsValid());
        ASSERT_TRUE(world.CreateBody(PlacedBoxDesc({2.0f, 1.0f, 3.0f}, {4.0f, 5.0f, 0.0f})).IsValid());
        // Keyed by geometry hash: each shape finds *its* blob, and neither is
        // served the other's.
        EXPECT_EQ(world.ShapeCacheDiskHits(), 2u);
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
    }
}

TEST(PhysicsShapeCache, ShapesWithPlacementAndScaleAreCookedThenCached) {
    const std::filesystem::path directory = FreshShapeCacheDir("ShapesWithPlacementAndScaleAreCookedThenCached");

    // Placement and scale are decorators over the cooked geometry, so two bodies
    // that differ only in where a shape sits share one blob — the cache keys on
    // geometry, exactly like the in-memory cache.
    BodyDesc placed = CachedBoxDesc();
    placed.shape.position = {0.0f, 0.5f, 0.0f};
    placed.shape.scale = {2.0f, 1.0f, 1.0f};
    placed.position = {0.0f, 5.0f, 0.0f};

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        const PhysicsBodyId body = world.CreateBody(placed);
        ASSERT_TRUE(body.IsValid());
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 1u);
        EXPECT_EQ(CountCacheFiles(directory), 1u);
    }

    {
        World world;
        ASSERT_TRUE(world.Initialize(ShapeCacheConfig(directory)));
        const PhysicsBodyId body = world.CreateBody(placed);
        ASSERT_TRUE(body.IsValid());
        EXPECT_EQ(world.ShapeCacheDiskHits(), 1u);
        EXPECT_EQ(world.ShapeCacheDiskMisses(), 0u);
        // The decorators are rebuilt from the descriptor on top of the restored
        // geometry, so the body is the scaled, offset box it asked for: a scaled
        // 2x1x1 box is 4 wide, and a ray down the middle of it hits y = 5.5.
        RayCastRequest request{};
        request.from = {0.0f, 10.0f, 0.0f};
        request.to = {0.0f, -10.0f, 0.0f};
        const auto hit = world.CastRay(request);
        ASSERT_TRUE(hit.has_value());
        EXPECT_NEAR(hit->position.y, 6.0f, 0.05f) << "the restored box lost its placement";
    }
}

// ── Save / replay / rollback ────────────────────────────────────────────────
//
// These tests exercise the rung-7 determinism checkpoint directly on a `World`,
// without the ECS: the bridge's iteration order is what makes replay
// deterministic in the engine, but the *contract* being tested here is that the
// world's solver state is reproducible and restorable.

[[nodiscard]] PhysicsWorldConfig DeterminismConfig() {
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 1024;
    return config;
}

/// The scripted scene, as data in creation order. `CreateBody` consumes it in
/// order and `CreateBodyWithID` replays it positionally — the same discipline
/// the ECS bridge gives the real engine.
[[nodiscard]] std::vector<BodyDesc> DeterminismSceneDescs() {
    std::vector<BodyDesc> descs;

    BodyDesc ground{};
    ground.shape.type = ShapeType::Box;
    ground.shape.box_extent = {24.0f, 1.0f, 24.0f};
    ground.motion = BodyMotionType::Static;
    ground.mass = 0.0f;
    ground.position = {0.0f, -0.5f, 0.0f};
    descs.push_back(ground);

    // A slightly leaning stack, so contacts and warm-start impulses are a real
    // part of the state — that is what a snapshot must carry to be replayable.
    for (int i = 0; i < 6; ++i) {
        BodyDesc box{};
        box.shape.type = ShapeType::Box;
        box.shape.box_extent = {1.0f, 1.0f, 1.0f};
        box.motion = BodyMotionType::Dynamic;
        box.mass = 1.0f + 0.5f * static_cast<float>(i);
        box.position = {0.05f * static_cast<float>(i), 1.0f + 1.05f * static_cast<float>(i),
                        -0.04f * static_cast<float>(i)};
        box.rotation = glm::quat(glm::radians(glm::vec3(1.5f * static_cast<float>(i),
                                                        2.0f * static_cast<float>(i),
                                                        0.5f * static_cast<float>(i))));
        descs.push_back(box);
    }

    // A body crossing the scene, so the script has motion beyond a settled stack.
    BodyDesc thrown{};
    thrown.shape.type = ShapeType::Sphere;
    thrown.shape.radius = 0.4f;
    thrown.motion = BodyMotionType::Dynamic;
    thrown.mass = 2.0f;
    thrown.position = {-4.0f, 4.0f, 0.0f};
    thrown.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    descs.push_back(thrown);

    return descs;
}

void BuildDeterminismScene(World& world) {
    for (const BodyDesc& desc : DeterminismSceneDescs()) {
        const PhysicsBodyId body = world.CreateBody(desc);
        ASSERT_TRUE(body.IsValid());
    }
}

TEST(PhysicsDeterminism, TwoIdenticalWorldsProduceEqualSnapshots) {
    World first;
    World second;
    ASSERT_TRUE(first.Initialize(DeterminismConfig()));
    ASSERT_TRUE(second.Initialize(DeterminismConfig()));
    BuildDeterminismScene(first);
    BuildDeterminismScene(second);

    // Same body set, same order: ids are part of the state, so this is the
    // precondition for the hashes to be comparable at all.
    ASSERT_EQ(first.BodyIds(), second.BodyIds());

    for (int i = 0; i < 150; ++i) {
        first.Step();
        second.Step();
    }

    const auto first_snapshot = first.SaveSnapshot();
    const auto second_snapshot = second.SaveSnapshot();
    ASSERT_FALSE(first_snapshot.empty());
    EXPECT_EQ(World::SnapshotHash(first_snapshot), World::SnapshotHash(second_snapshot));
}

TEST(PhysicsDeterminism, SnapshotHashIsIndependentOfJobScheduling) {
    // The rung's real claim: Jolt's solver is deterministic given identical
    // inputs, so the number of workers changes nothing observable. The pool is
    // reconfigured around each run and restored afterwards.
    auto& scheduler = VulkanShared::JobScheduler::Global();
    const unsigned int many = std::max(2u, std::thread::hardware_concurrency());
    const std::array<unsigned int, 3> budgets{0u, 1u, many};
    std::array<std::uint64_t, budgets.size()> hashes{};

    for (std::size_t i = 0; i < budgets.size(); ++i) {
        scheduler.Configure(VulkanShared::JobBudget{.workers = budgets[i]});
        World world;
        ASSERT_TRUE(world.Initialize(DeterminismConfig()));
        BuildDeterminismScene(world);
        for (int step = 0; step < 200; ++step) world.Step();
        hashes[i] = World::SnapshotHash(world.SaveSnapshot());
        world.Shutdown();
    }

    scheduler.Configure(VulkanShared::JobBudget{});

    EXPECT_EQ(hashes[0], hashes[1]) << "0 vs 1 worker diverged";
    EXPECT_EQ(hashes[0], hashes[2]) << "0 vs " << many << " workers diverged";
}

TEST(PhysicsSnapshot, RollbackRestoresTheStateAndReproducesTheContinuation) {
    World world;
    ASSERT_TRUE(world.Initialize(DeterminismConfig()));
    BuildDeterminismScene(world);

    for (int i = 0; i < 60; ++i) world.Step();
    const std::vector<std::byte> anchor = world.SaveSnapshot();
    const std::uint64_t anchor_hash = World::SnapshotHash(anchor);
    ASSERT_FALSE(anchor.empty());

    // Step past the anchor and capture where the untouched timeline goes.
    for (int i = 0; i < 90; ++i) world.Step();
    const std::uint64_t continuation_hash = World::SnapshotHash(world.SaveSnapshot());
    ASSERT_NE(anchor_hash, continuation_hash) << "the script did not actually move the state";

    // Roll back: the restored world must hash to exactly the anchor.
    ASSERT_TRUE(world.RestoreSnapshot(anchor));
    EXPECT_EQ(World::SnapshotHash(world.SaveSnapshot()), anchor_hash);

    // Replaying the same M steps from the restored state must reproduce the
    // original continuation bit for bit. This is the rollback contract: contacts
    // and constraint warm-start state are part of the snapshot, not just poses.
    for (int i = 0; i < 90; ++i) world.Step();
    EXPECT_EQ(World::SnapshotHash(world.SaveSnapshot()), continuation_hash);
}

TEST(PhysicsSnapshot, RebuildingWithFixedBodyIdsReproducesTheSnapshot) {
    const std::vector<BodyDesc> descs = DeterminismSceneDescs();

    // The original run: ids come from Jolt's allocation order.
    World original;
    ASSERT_TRUE(original.Initialize(DeterminismConfig()));
    for (const BodyDesc& desc : descs) ASSERT_TRUE(original.CreateBody(desc).IsValid());
    const std::vector<PhysicsBodyId> ids = original.BodyIds();
    ASSERT_EQ(ids.size(), descs.size());
    for (int i = 0; i < 120; ++i) original.Step();
    const std::uint64_t original_hash = World::SnapshotHash(original.SaveSnapshot());

    // The replay: the same script with the recorded ids, which is what a
    // rollback/replay loader does with `CreateBodyWithID`.
    World replay;
    ASSERT_TRUE(replay.Initialize(DeterminismConfig()));
    for (std::size_t i = 0; i < descs.size(); ++i) {
        ASSERT_TRUE(replay.CreateBodyWithID(descs[i], ids[i]).IsValid());
    }
    EXPECT_EQ(replay.BodyIds(), ids);
    for (int i = 0; i < 120; ++i) replay.Step();

    EXPECT_EQ(World::SnapshotHash(replay.SaveSnapshot()), original_hash);
}

TEST(PhysicsSnapshot, CreateBodyWithIDRejectsAnIdAlreadyInUse) {
    World world;
    ASSERT_TRUE(world.Initialize(DeterminismConfig()));
    const std::vector<BodyDesc> descs = DeterminismSceneDescs();
    const PhysicsBodyId first = world.CreateBody(descs.front());
    ASSERT_TRUE(first.IsValid());
    ASSERT_EQ(world.BodyCount(), 1u);

    // Handing out the same id twice must fail loudly, not silently renumber: a
    // second body under one id would break every snapshot that names it.
    const PhysicsBodyId duplicate = world.CreateBodyWithID(descs[1], first);
    EXPECT_FALSE(duplicate.IsValid());
    EXPECT_EQ(world.BodyCount(), 1u);
}

TEST(PhysicsSnapshot, ARejectedSnapshotLeavesTheWorldUntouched) {
    World world;
    ASSERT_TRUE(world.Initialize(DeterminismConfig()));
    BuildDeterminismScene(world);
    for (int i = 0; i < 30; ++i) world.Step();
    const std::uint64_t before = World::SnapshotHash(world.SaveSnapshot());

    // Not our frame at all.
    EXPECT_FALSE(world.RestoreSnapshot(std::span<const std::byte>{}));
    const std::vector<std::byte> foreign(256, std::byte{0xAB});
    EXPECT_FALSE(world.RestoreSnapshot(foreign));

    // A real snapshot, truncated and then bit-flipped in the payload.
    const std::vector<std::byte> good = world.SaveSnapshot();
    ASSERT_GT(good.size(), 16u);
    EXPECT_FALSE(world.RestoreSnapshot(std::span<const std::byte>(good).first(good.size() - 1)));
    std::vector<std::byte> corrupt = good;
    corrupt.back() ^= std::byte{0xFF};
    EXPECT_FALSE(world.RestoreSnapshot(corrupt));

    // Every rejection left the world exactly where it was.
    EXPECT_EQ(World::SnapshotHash(world.SaveSnapshot()), before);
}

TEST(PhysicsSnapshot, ASnapshotFromADifferentBodySetIsRejected) {
    World source;
    World target;
    ASSERT_TRUE(source.Initialize(DeterminismConfig()));
    ASSERT_TRUE(target.Initialize(DeterminismConfig()));
    BuildDeterminismScene(source);
    BuildDeterminismScene(target);
    // The target holds one more body than the source: Jolt's restore matches
    // bodies by id and would misalign part-way through, so this has to be
    // rejected before any byte reaches it.
    ASSERT_TRUE(target.CreateBody(DeterminismSceneDescs().front()).IsValid());

    for (int i = 0; i < 30; ++i) {
        source.Step();
        target.Step();
    }

    const std::uint64_t target_before = World::SnapshotHash(target.SaveSnapshot());
    EXPECT_FALSE(target.RestoreSnapshot(source.SaveSnapshot()));
    EXPECT_EQ(World::SnapshotHash(target.SaveSnapshot()), target_before)
        << "a rejected snapshot changed the target world";
}

// ── Constraints ─────────────────────────────────────────────────────────────

[[nodiscard]] PhysicsWorldConfig ConstraintsConfig() {
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 1024;
    config.max_contact_constraints = 1024;
    return config;
}

[[nodiscard]] PhysicsBodyId MakeWorldBox(World& world, glm::vec3 extent, BodyMotionType motion,
                                         glm::vec3 position, float mass = 1.0f) {
    BodyDesc desc{};
    desc.shape.type = ShapeType::Box;
    desc.shape.box_extent = extent;
    desc.motion = motion;
    desc.mass = motion == BodyMotionType::Static ? 0.0f : mass;
    desc.position = position;
    return world.CreateBody(desc);
}

TEST(PhysicsConstraints, AHingePendulumSwingsAndItsAngleChanges) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    // A rod hanging from the anchor, offset so gravity gives it torque.
    const PhysicsBodyId rod =
        MakeWorldBox(world, {0.2f, 2.0f, 0.2f}, BodyMotionType::Dynamic, {1.5f, -1.5f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Hinge;
    desc.body_a = anchor;
    desc.body_b = rod;
    desc.point_a = {0.0f, 0.0f, 0.0f};
    desc.point_b = {0.0f, 0.0f, 0.0f}; // joint at the anchor, in world space
    desc.axis = {0.0f, 0.0f, 1.0f};    // rotate in the XY plane
    desc.normal_axis = {1.0f, 0.0f, 0.0f};

    const ConstraintId hinge = CreateConstraint(world, desc);
    ASSERT_TRUE(hinge.IsValid());
    EXPECT_EQ(ConstraintCount(world), 1u);
    // A joint created with world-space frames starts at angle zero.
    EXPECT_NEAR(GetConstraintPosition(world, hinge), 0.0f, 1.0e-3f);

    for (int i = 0; i < 150; ++i) world.Step();

    EXPECT_GT(std::abs(GetConstraintPosition(world, hinge)), 0.1f)
        << "the pendulum did not swing about its hinge";
}

TEST(PhysicsConstraints, HingeLimitsClampTheAngle) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyId rod =
        MakeWorldBox(world, {0.2f, 2.0f, 0.2f}, BodyMotionType::Dynamic, {1.5f, -1.5f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Hinge;
    desc.body_a = anchor;
    desc.body_b = rod;
    desc.point_a = {0.0f, 0.0f, 0.0f};
    desc.point_b = {0.0f, 0.0f, 0.0f};
    desc.axis = {0.0f, 0.0f, 1.0f};
    desc.normal_axis = {1.0f, 0.0f, 0.0f};
    desc.use_limits = true;
    desc.limit_min = -0.2f;
    desc.limit_max = 0.2f;

    const ConstraintId hinge = CreateConstraint(world, desc);
    ASSERT_TRUE(hinge.IsValid());

    float lo = 0.0f;
    float hi = 0.0f;
    ASSERT_TRUE(GetConstraintLimits(world, hinge, lo, hi));
    EXPECT_NEAR(lo, -0.2f, 1.0e-5f);
    EXPECT_NEAR(hi, 0.2f, 1.0e-5f);

    for (int i = 0; i < 180; ++i) world.Step();

    const float angle = GetConstraintPosition(world, hinge);
    // Gravity pulls the rod toward its hanging pose, which is well past the
    // limit; the clamp is what has to stop it.
    EXPECT_GT(std::abs(angle), 0.1f) << "the pendulum never reached its limit";
    EXPECT_GE(angle, -0.3f) << "the hinge exceeded its lower limit";
    EXPECT_LE(angle, 0.3f) << "the hinge exceeded its upper limit";
}

TEST(PhysicsConstraints, AFixedConstraintWeldsTwoBodiesTogether) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId first =
        MakeWorldBox(world, {0.5f, 0.5f, 0.5f}, BodyMotionType::Dynamic, {0.0f, 5.0f, 0.0f});
    const PhysicsBodyId second =
        MakeWorldBox(world, {0.5f, 0.5f, 0.5f}, BodyMotionType::Dynamic, {2.0f, 5.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Fixed; // a weld is the bodies' current relative pose
    desc.body_a = first;
    desc.body_b = second;
    const ConstraintId weld = CreateConstraint(world, desc);
    ASSERT_TRUE(weld.IsValid());

    // Let the pair leave its spawn pose, then check the frame between them.
    for (int i = 0; i < 5; ++i) world.Step();
    const glm::vec3 reference = world.GetPose(second).position - world.GetPose(first).position;

    for (int i = 0; i < 180; ++i) world.Step();
    const glm::vec3 live = world.GetPose(second).position - world.GetPose(first).position;

    EXPECT_NEAR(live.x, reference.x, 0.05f);
    EXPECT_NEAR(live.y, reference.y, 0.05f);
    EXPECT_NEAR(live.z, reference.z, 0.05f);
    // The weld also survives the fall: both bodies are well below their spawn.
    EXPECT_LT(world.GetPose(first).position.y, 4.9f);
    EXPECT_NEAR(world.GetPose(first).position.y, world.GetPose(second).position.y, 0.05f);
}

TEST(PhysicsConstraints, APointConstraintHoldsABodyAtItsJointDistance) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.1f, 0.1f, 0.1f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    // The body starts 2 m from the pivot; the point joint is at the pivot.
    const PhysicsBodyId body =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {1.4f, -1.4f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Point;
    desc.body_a = anchor;
    desc.body_b = body;
    desc.point_a = {0.0f, 0.0f, 0.0f};
    desc.point_b = {0.0f, 0.0f, 0.0f};
    const ConstraintId point = CreateConstraint(world, desc);
    ASSERT_TRUE(point.IsValid());

    for (int i = 0; i < 180; ++i) world.Step();

    const float radius = glm::length(world.GetPose(body).position - world.GetPose(anchor).position);
    EXPECT_NEAR(radius, 1.9799f, 0.1f) << "the point joint did not hold the body at its radius";
}

TEST(PhysicsConstraints, ADistanceConstraintKeepsASeparation) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.1f, 0.1f, 0.1f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    // Start well outside the range; the constraint must pull the body in to it.
    const PhysicsBodyId body =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {5.0f, 0.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Distance;
    desc.body_a = anchor;
    desc.body_b = body;
    // Both anchors sit at the bodies' centers: an anchor far from a body's
    // center of mass is a long lever, and the solver reacts to it with a huge
    // torque. The joint then holds the two centers at the requested separation.
    desc.point_a = {0.0f, 0.0f, 0.0f};
    desc.point_b = {5.0f, 0.0f, 0.0f};
    desc.use_limits = true;
    desc.limit_min = 3.0f;
    desc.limit_max = 3.0f; // a rigid rod
    const ConstraintId distance = CreateConstraint(world, desc);
    ASSERT_TRUE(distance.IsValid());

    for (int i = 0; i < 240; ++i) world.Step();

    const float separation =
        glm::length(world.GetPose(body).position - world.GetPose(anchor).position);
    EXPECT_NEAR(separation, 3.0f, 0.1f) << "the distance joint did not hold its separation";
}

TEST(PhysicsConstraints, DisablingCollisionLetsConstrainedBodiesOverlap) {
    // Two welded bodies spawned exactly coincident. With collision enabled the
    // overlap generates contacts (and the solver fights the weld); with
    // collision disabled the pair never collides and simply stays overlapped.
    // The observable is the contact stream, not a separation: it is the direct
    // statement of "these two bodies do not collide", independent of how the
    // solver resolves two constraints that disagree.
    const auto run = [](bool enable_collision) {
        World world;
        EXPECT_TRUE(world.Initialize(ConstraintsConfig()));
        const PhysicsBodyId first =
            MakeWorldBox(world, {0.5f, 0.5f, 0.5f}, BodyMotionType::Dynamic, {0.0f, 10.0f, 0.0f});
        const PhysicsBodyId second =
            MakeWorldBox(world, {0.5f, 0.5f, 0.5f}, BodyMotionType::Dynamic, {0.0f, 10.0f, 0.0f});

        int contacts = 0;
        world.SetContactCallback([&contacts](const ContactEvent&) { ++contacts; });

        ConstraintDesc desc{};
        desc.type = ConstraintType::Fixed;
        desc.body_a = first;
        desc.body_b = second;
        desc.enable_collision = enable_collision;
        EXPECT_TRUE(CreateConstraint(world, desc).IsValid());

        for (int i = 0; i < 60; ++i) world.Step();

        const float separation =
            glm::length(world.GetPose(second).position - world.GetPose(first).position);
        return std::pair{contacts, separation};
    };

    const auto [disabled_contacts, disabled_separation] = run(false);
    const auto [enabled_contacts, enabled_separation] = run(true);

    // Disabled: the bodies stay overlapped and never report a contact.
    EXPECT_EQ(disabled_contacts, 0) << "the disabled pair still collided";
    EXPECT_LT(disabled_separation, 0.05f) << "the constrained bodies did not overlap";
    // Enabled: the very same overlap is a real collision.
    EXPECT_GT(enabled_contacts, 0) << "collision was not respected between the bodies";
    EXPECT_GT(enabled_separation, disabled_separation)
        << "collision made no difference to the constrained pair";
}

TEST(PhysicsConstraints, StaleHandlesAreInvalidAndDestroyingThemIsSafe) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyId body =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -2.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Point;
    desc.body_a = anchor;
    desc.body_b = body;
    const ConstraintId first = CreateConstraint(world, desc);
    ASSERT_TRUE(first.IsValid());
    ASSERT_TRUE(IsConstraintValid(world, first));

    DestroyConstraint(world, first);
    EXPECT_FALSE(IsConstraintValid(world, first));
    EXPECT_EQ(ConstraintCount(world), 0u);
    // Releasing a stale handle twice is a no-op, not a crash.
    DestroyConstraint(world, first);
    EXPECT_FALSE(IsConstraintValid(world, ConstraintId{}));

    // A rejected description leaves the world with no constraint.
    ConstraintDesc dead = desc;
    dead.body_b = PhysicsBodyId{}; // invalid
    EXPECT_FALSE(CreateConstraint(world, dead).IsValid());
    ConstraintDesc self_joint = desc;
    self_joint.body_a = body;
    self_joint.body_b = body;
    EXPECT_FALSE(CreateConstraint(world, self_joint).IsValid());
    EXPECT_EQ(ConstraintCount(world), 0u);
}

TEST(PhysicsConstraints, DestroyingABodyReleasesItsConstraintsAutomatically) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyId middle =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -2.0f, 0.0f});
    const PhysicsBodyId end =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -4.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Point;
    desc.body_a = anchor;
    desc.body_b = middle;
    ASSERT_TRUE(CreateConstraint(world, desc).IsValid());
    desc.body_a = middle;
    desc.body_b = end;
    ASSERT_TRUE(CreateConstraint(world, desc).IsValid());
    ASSERT_EQ(ConstraintCount(world), 2u);

    // No explicit release: `World::DestroyBody` reaches the registry through
    // the world's extension, so a body can never be destroyed out from under a
    // live constraint.
    world.DestroyBody(middle);
    EXPECT_FALSE(world.IsBodyValid(middle));
    EXPECT_EQ(ConstraintCount(world), 0u);
    EXPECT_TRUE(world.IsBodyValid(anchor));
    EXPECT_TRUE(world.IsBodyValid(end));
}

TEST(PhysicsConstraints, DestroyConstraintsForReleasesABodysJoints) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyId middle =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -2.0f, 0.0f});
    const PhysicsBodyId end =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -4.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Point;
    desc.body_a = anchor;
    desc.body_b = middle;
    ASSERT_TRUE(CreateConstraint(world, desc).IsValid());
    desc.body_a = middle;
    desc.body_b = end;
    const ConstraintId second = CreateConstraint(world, desc);
    ASSERT_TRUE(second.IsValid());
    ASSERT_EQ(ConstraintCount(world), 2u);

    // Destroying a body must not leave a dangling joint: this is the bridge's
    // release hook, and it takes out both of the middle body's joints.
    DestroyConstraintsFor(world, middle);
    EXPECT_EQ(ConstraintCount(world), 0u);
    EXPECT_FALSE(IsConstraintValid(world, second));

    // World teardown with live constraints is safe (the registry is torn down
    // before the bodies); building and destroying one more makes that explicit.
    desc.body_a = anchor;
    desc.body_b = end;
    ASSERT_TRUE(CreateConstraint(world, desc).IsValid());
}

TEST(PhysicsConstraints, MotorsDriveAHingeAndTargetVelocityIsReadable) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    const PhysicsBodyId anchor =
        MakeWorldBox(world, {0.2f, 0.2f, 0.2f}, BodyMotionType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyId rod =
        MakeWorldBox(world, {0.2f, 2.0f, 0.2f}, BodyMotionType::Dynamic, {0.0f, -2.0f, 0.0f});

    ConstraintDesc desc{};
    desc.type = ConstraintType::Hinge;
    desc.body_a = anchor;
    desc.body_b = rod;
    desc.point_a = {0.0f, 0.0f, 0.0f};
    desc.point_b = {0.0f, 0.0f, 0.0f};
    desc.axis = {0.0f, 0.0f, 1.0f};
    desc.normal_axis = {1.0f, 0.0f, 0.0f};
    const ConstraintId hinge = CreateConstraint(world, desc);
    ASSERT_TRUE(hinge.IsValid());

    ASSERT_TRUE(SetMotorState(world, hinge, MotorState::Velocity));
    ASSERT_TRUE(SetTargetVelocity(world, hinge, 2.0f));
    for (int i = 0; i < 60; ++i) world.Step();

    // The motor spun the rod about its hinge, against gravity.
    EXPECT_GT(std::abs(GetConstraintPosition(world, hinge)), 0.1f);
    ASSERT_TRUE(SetMotorState(world, hinge, MotorState::Off));
    // Limits/motors are hinge/slider-only.
    EXPECT_FALSE(SetConstraintLimits(world, ConstraintId{}, -1.0f, 1.0f));
}

// ── Vehicles ────────────────────────────────────────────────────────────────

TEST(PhysicsVehicle, AThrottledVehicleDrivesForwardAndStaysUpright) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 4096;
    config.max_contact_constraints = 2048;
    ASSERT_TRUE(world.Initialize(config));

    // A large static ground plane with its top face at y = 0.
    BodyDesc ground{};
    ground.shape.type = ShapeType::Box;
    ground.shape.box_extent = {400.0f, 1.0f, 400.0f};
    ground.motion = BodyMotionType::Static;
    ground.mass = 0.0f;
    ground.position = {0.0f, -0.5f, 0.0f};
    ASSERT_TRUE(world.CreateBody(ground).IsValid());

    BodyDesc chassis{};
    chassis.shape.type = ShapeType::Box;
    chassis.shape.box_extent = {2.0f, 0.5f, 4.0f};
    chassis.motion = BodyMotionType::Dynamic;
    chassis.mass = 1500.0f;
    chassis.friction = 0.5f;
    chassis.linear_damping = 0.0f;
    chassis.angular_damping = 0.0f;
    chassis.position = {0.0f, 1.2f, 0.0f};
    const PhysicsBodyId chassis_body = world.CreateBody(chassis);
    ASSERT_TRUE(chassis_body.IsValid());

    VehicleDesc desc{};
    desc.chassis = chassis_body;
    desc.up = {0.0f, 1.0f, 0.0f};
    desc.forward = {0.0f, 0.0f, 1.0f};
    desc.max_pitch_roll_angle = 0.6f;
    desc.engine_max_torque = 800.0f;

    // Rear-left, rear-right, front-left, front-right — the order the
    // differentials below refer to.
    struct WheelSpec {
        float x;
        float z;
        bool steers;
        bool handbrake;
    };
    constexpr WheelSpec kSpecs[4] = {
        {-1.0f, -1.4f, false, true},
        {1.0f, -1.4f, false, true},
        {-1.0f, 1.4f, true, false},
        {1.0f, 1.4f, true, false},
    };
    for (const WheelSpec& spec : kSpecs) {
        WheelDesc wheel{};
        wheel.position = {spec.x, -0.5f, spec.z};
        wheel.radius = 0.4f;
        wheel.width = 0.2f;
        wheel.suspension_min_length = 0.2f;
        wheel.suspension_max_length = 0.5f;
        wheel.suspension_frequency = 1.5f;
        wheel.suspension_damping = 0.5f;
        wheel.max_steer_angle = spec.steers ? 0.5f : 0.0f;
        wheel.max_brake_torque = 1500.0f;
        wheel.max_handbrake_torque = spec.handbrake ? 4000.0f : 0.0f;
        desc.wheels.push_back(wheel);
    }
    desc.differentials.resize(2);
    desc.differentials[0] = {.left_wheel = 0, .right_wheel = 1};
    desc.differentials[1] = {.left_wheel = 2, .right_wheel = 3};
    desc.differentials[0].engine_torque_ratio = 0.5f;
    desc.differentials[1].engine_torque_ratio = 0.5f;
    desc.anti_roll_bars.push_back({.left_wheel = 0, .right_wheel = 1, .stiffness = 2000.0f});
    desc.anti_roll_bars.push_back({.left_wheel = 2, .right_wheel = 3, .stiffness = 2000.0f});

    WheeledVehicle vehicle;
    ASSERT_TRUE(vehicle.Initialize(world, desc));
    EXPECT_EQ(vehicle.WheelCount(), 4u);

    // Let the suspension settle onto the ground.
    for (int i = 0; i < 60; ++i) world.Step();
    const float start_z = world.GetPose(chassis_body).position.z;
    EXPECT_TRUE(vehicle.GetWheelContactBody(0).IsValid()) << "the rear wheels are not on the ground";

    // Full throttle for two seconds.
    vehicle.SetDriverInput(1.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 120; ++i) world.Step();

    const BodyPose end = world.GetPose(chassis_body);
    const float distance = end.position.z - start_z;
    EXPECT_GT(distance, 4.0f) << "the vehicle did not drive forward";
    EXPECT_LT(distance, 80.0f) << "the vehicle drove an implausible distance";
    EXPECT_GT(vehicle.GetSpeed(), 1.0f) << "the vehicle has no forward speed";

    const glm::vec3 up = end.rotation * glm::vec3(0.0f, 1.0f, 0.0f);
    EXPECT_GT(up.y, 0.85f) << "the vehicle tipped over";
    // The wheel spin and steer read back through the wrapper.
    EXPECT_TRUE(vehicle.GetWheelContactBody(2).IsValid());
    EXPECT_NE(vehicle.GetWheelRotationAngle(0), 0.0f);

    vehicle.Shutdown();
    EXPECT_FALSE(vehicle.IsInitialized());
}

TEST(PhysicsVehicle, GetWheelPoseReportsSpinFreeWorldOrientation) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 4096;
    config.max_contact_constraints = 2048;
    ASSERT_TRUE(world.Initialize(config));

    // Same scene as the driving test: flat ground, a level chassis, four
    // wheels with the front pair steering.
    BodyDesc ground{};
    ground.shape.type = ShapeType::Box;
    ground.shape.box_extent = {400.0f, 1.0f, 400.0f};
    ground.motion = BodyMotionType::Static;
    ground.mass = 0.0f;
    ground.position = {0.0f, -0.5f, 0.0f};
    ASSERT_TRUE(world.CreateBody(ground).IsValid());

    BodyDesc chassis{};
    chassis.shape.type = ShapeType::Box;
    chassis.shape.box_extent = {2.0f, 0.5f, 4.0f};
    chassis.motion = BodyMotionType::Dynamic;
    chassis.mass = 1500.0f;
    chassis.friction = 0.5f;
    chassis.linear_damping = 0.0f;
    chassis.angular_damping = 0.0f;
    chassis.position = {0.0f, 1.2f, 0.0f};
    const PhysicsBodyId chassis_body = world.CreateBody(chassis);
    ASSERT_TRUE(chassis_body.IsValid());

    VehicleDesc desc{};
    desc.chassis = chassis_body;
    desc.up = {0.0f, 1.0f, 0.0f};
    desc.forward = {0.0f, 0.0f, 1.0f};
    desc.engine_max_torque = 800.0f;
    for (const auto& [x, z, steers] : std::to_array({
             std::tuple{-1.0f, -1.4f, false},
             std::tuple{1.0f, -1.4f, false},
             std::tuple{-1.0f, 1.4f, true},
             std::tuple{1.0f, 1.4f, true},
         })) {
        WheelDesc wheel{};
        wheel.position = {x, -0.5f, z};
        wheel.radius = 0.4f;
        wheel.width = 0.2f;
        wheel.suspension_min_length = 0.2f;
        wheel.suspension_max_length = 0.5f;
        wheel.suspension_frequency = 1.5f;
        wheel.suspension_damping = 0.5f;
        wheel.max_steer_angle = steers ? 0.5f : 0.0f;
        desc.wheels.push_back(wheel);
    }
    desc.differentials.resize(2);
    desc.differentials[0] = {.left_wheel = 0, .right_wheel = 1};
    desc.differentials[1] = {.left_wheel = 2, .right_wheel = 3};

    WheeledVehicle vehicle;
    ASSERT_TRUE(vehicle.Initialize(world, desc));

    // Let the suspension settle onto the ground.
    for (int i = 0; i < 60; ++i) world.Step();
    const BodyPose settled = world.GetPose(chassis_body);
    ASSERT_GT((settled.rotation * glm::vec3(0.0f, 1.0f, 0.0f)).y, 0.99f)
        << "the chassis is not level";

    // The pose rotation maps the wheel model space (spin axle +y, wheel-up
    // +x, wheel-forward +z) to world: the axle runs along the chassis' right
    // side (forward x up = -x for this layout), and the wheel center sits one
    // radius above the ground.
    const BodyPose at_rest = vehicle.GetWheelPose(0);
    const glm::vec3 axle = at_rest.rotation * glm::vec3(0.0f, 1.0f, 0.0f);
    EXPECT_NEAR(axle.x, -1.0f, 0.05f) << "the axle does not run along the chassis' x axis";
    EXPECT_NEAR(axle.y, 0.0f, 0.05f);
    EXPECT_NEAR(axle.z, 0.0f, 0.05f);
    const glm::vec3 up = at_rest.rotation * glm::vec3(1.0f, 0.0f, 0.0f);
    EXPECT_GT(up.y, 0.95f) << "model +x does not map to the wheel-up direction";
    const glm::vec3 fwd = at_rest.rotation * glm::vec3(0.0f, 0.0f, 1.0f);
    EXPECT_GT(fwd.z, 0.95f) << "model +z does not map to the wheel-forward direction";
    EXPECT_NEAR(at_rest.position.y, 0.4f, 0.05f)
        << "the wheel center is not riding at its radius above the ground";

    // Drive forward from rest, taking the measurement while the accumulated
    // spin is still under half a revolution so the roll direction is
    // unambiguous.
    float spin = 0.0f;
    for (int i = 0; i < 120; ++i) {
        vehicle.SetDriverInput(1.0f, 0.0f, 0.0f, 0.0f);
        world.Step();
        spin = vehicle.GetWheelRotationAngle(0);
        if (std::abs(spin) >= 0.3f) break;
    }
    EXPECT_GE(std::abs(spin), 0.3f) << "the wheel did not spin while driving";
    EXPECT_LT(std::abs(spin), 2.5f) << "the spin wrapped past half a turn";

    // The pose rotation excludes the spin: while the wheel rotates underneath
    // it, the pose stays pinned to the settled orientation (up to the small
    // pitch and yaw the accelerating chassis picks up).
    const BodyPose rolling = vehicle.GetWheelPose(0);
    const float drift = 2.0f * std::acos(std::min(1.0f,
        std::abs(glm::dot(rolling.rotation, at_rest.rotation))));
    EXPECT_LT(drift, 0.3f) << "the pose rotation moved with the wheel spin";

    // Re-applying the spin per the documented composition reproduces the full
    // wheel transform: the pattern's up direction tilts toward travel, i.e.
    // the wheel rolls forward rather than backward.
    const glm::quat full =
        rolling.rotation * glm::angleAxis(-spin, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 pattern = full * glm::vec3(1.0f, 0.0f, 0.0f);
    EXPECT_GT(glm::dot(pattern, settled.rotation * glm::vec3(0.0f, 0.0f, 1.0f)), 0.1f)
        << "the spin composition does not roll the wheel forward";

    // Steering input shows up in the pose rotation: the model forward maps to
    // the steered forward the steer angle defines about the +y steering axis.
    vehicle.SetDriverInput(0.0f, 1.0f, 0.0f, 0.0f);
    for (int i = 0; i < 10; ++i) world.Step();
    const float steer = vehicle.GetWheelSteerAngle(2);
    EXPECT_GE(std::abs(steer), 0.05f) << "the front wheels did not steer";
    const BodyPose steered = vehicle.GetWheelPose(2);
    const glm::vec3 steered_fwd = steered.rotation * glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::vec3 expected_fwd =
        world.GetPose(chassis_body).rotation *
        (glm::angleAxis(steer, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::vec3(0.0f, 0.0f, 1.0f));
    EXPECT_GT(glm::dot(steered_fwd, expected_fwd), 0.99f)
        << "the pose rotation does not carry the steering";

    vehicle.Shutdown();
}

TEST(PhysicsVehicle, AnUninitializedVehicleRejectsADeadChassis) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    WheeledVehicle vehicle;
    VehicleDesc desc{};
    desc.chassis = PhysicsBodyId{}; // invalid
    desc.wheels.push_back(WheelDesc{});
    EXPECT_FALSE(vehicle.Initialize(world, desc));

    // No wheels is also rejected.
    const PhysicsBodyId chassis =
        MakeWorldBox(world, {1.0f, 0.5f, 2.0f}, BodyMotionType::Dynamic, {0.0f, 5.0f, 0.0f});
    desc.chassis = chassis;
    desc.wheels.clear();
    EXPECT_FALSE(vehicle.Initialize(world, desc));
}

// ── Ragdolls ────────────────────────────────────────────────────────────────

/// Three capsules stacked vertically, each point-jointed to the one above.
[[nodiscard]] RagdollDesc ThreePartRagdoll() {
    RagdollDesc desc{};
    desc.collision_group = 7;
    for (int i = 0; i < 3; ++i) {
        RagdollPartDesc part{};
        part.radius = 0.2f;
        part.height = 1.0f;
        part.position = {0.0f, 3.0f - static_cast<float>(i), 0.0f};
        part.parent = i - 1;
        part.joint = ConstraintType::Point;
        // The shared articulation point between part i and part i-1.
        part.joint_anchor = {0.0f, 3.5f - static_cast<float>(i), 0.0f};
        desc.parts.push_back(part);
    }
    return desc;
}

TEST(PhysicsRagdoll, AThreePartRagdollFallsSettlesAndComesToRest) {
    World world;
    PhysicsWorldConfig config{};
    config.max_bodies = 256;
    config.max_body_pairs = 4096;
    config.max_contact_constraints = 2048;
    ASSERT_TRUE(world.Initialize(config));

    BodyDesc ground{};
    ground.shape.type = ShapeType::Box;
    ground.shape.box_extent = {60.0f, 1.0f, 60.0f};
    ground.motion = BodyMotionType::Static;
    ground.mass = 0.0f;
    ground.position = {0.0f, -0.5f, 0.0f};
    ASSERT_TRUE(world.CreateBody(ground).IsValid());

    Ragdoll ragdoll;
    ASSERT_TRUE(ragdoll.Initialize(world, ThreePartRagdoll()));
    ASSERT_EQ(ragdoll.PartCount(), 3u);
    EXPECT_TRUE(ragdoll.IsInPhysicsSystem());
    for (std::uint32_t i = 0; i < ragdoll.PartCount(); ++i) {
        EXPECT_TRUE(ragdoll.PartBody(i).IsValid());
    }

    // Nudge the root so the chain topples instead of balancing perfectly.
    world.SetAngularVelocity(ragdoll.PartBody(0), {0.0f, 0.0f, 1.5f});

    for (int i = 0; i < 600; ++i) world.Step();

    constexpr std::array<float, 3> kSpawnHeights{3.0f, 2.0f, 1.0f};
    for (std::uint32_t i = 0; i < 3; ++i) {
        const BodyPose pose = ragdoll.GetPartPose(i);
        EXPECT_LT(pose.position.y, kSpawnHeights[i]) << "part " << i << " did not fall";
        EXPECT_GT(pose.position.y, -0.5f) << "part " << i << " fell through the ground";
        EXPECT_LT(glm::length(world.GetLinearVelocity(ragdoll.PartBody(i))), 0.5f)
            << "part " << i << " never came to rest";
        EXPECT_LT(glm::length(world.GetAngularVelocity(ragdoll.PartBody(i))), 1.0f)
            << "part " << i << " is still spinning";
    }

    // Whole-ragdoll pose write and read-back.
    std::array<BodyPose, 3> lifted{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        lifted[i].position = {4.0f, 6.0f - static_cast<float>(i), 0.0f};
    }
    ragdoll.SetPose(lifted);
    for (std::uint32_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(ragdoll.GetPartPose(i).position.y, lifted[i].position.y, 0.05f);
    }
}

TEST(PhysicsRagdoll, RemovingAndReaddingItTakesTheBodiesOutAndBack) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    Ragdoll ragdoll;
    ASSERT_TRUE(ragdoll.Initialize(world, ThreePartRagdoll()));

    const PhysicsBodyId root = ragdoll.PartBody(0);
    ASSERT_TRUE(root.IsValid());
    EXPECT_TRUE(world.IsBodyValid(root));

    ragdoll.RemoveFromPhysicsSystem();
    EXPECT_FALSE(ragdoll.IsInPhysicsSystem());
    EXPECT_FALSE(world.IsBodyValid(root)) << "the parts stayed in the physics system";

    ragdoll.AddToPhysicsSystem();
    EXPECT_TRUE(ragdoll.IsInPhysicsSystem());
    EXPECT_TRUE(world.IsBodyValid(root));

    ragdoll.Shutdown();
    EXPECT_FALSE(ragdoll.IsInitialized());
    EXPECT_FALSE(world.IsBodyValid(root));
}

TEST(PhysicsRagdoll, AMalformedPartListIsRejected) {
    World world;
    ASSERT_TRUE(world.Initialize(ConstraintsConfig()));
    Ragdoll ragdoll;

    RagdollDesc empty{};
    EXPECT_FALSE(ragdoll.Initialize(world, empty));

    // A parent that is not an earlier part cannot form a skeleton.
    RagdollDesc forward_reference{};
    RagdollPartDesc first{};
    first.parent = 1;
    forward_reference.parts.push_back(first);
    RagdollPartDesc second{};
    second.parent = 0;
    forward_reference.parts.push_back(second);
    EXPECT_FALSE(ragdoll.Initialize(world, forward_reference));

    // A capsule whose radius exceeds its height is not cookable.
    RagdollDesc bad_capsule{};
    RagdollPartDesc fat{};
    fat.radius = 1.0f;
    fat.height = 0.5f;
    bad_capsule.parts.push_back(fat);
    EXPECT_FALSE(ragdoll.Initialize(world, bad_capsule));
    EXPECT_FALSE(ragdoll.IsInitialized());
}

} // namespace
