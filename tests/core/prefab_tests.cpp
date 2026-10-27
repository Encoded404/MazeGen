// Device-free component-context and prefab tests (rung 3 of engine
// modularization).
//
// The component context is a granted service view; prefabs compose standard
// component bundles through the same registry hooks the engine modules own.
// The Jolt-backed prefab tests are compiled only when VKENGINE_PHYSICS is
// enabled; Jolt needs no Vulkan device, so nothing here creates a Vulkan object.

#include <gtest/gtest.h>

#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

import std;

import VulkanEngine.Input;
import VulkanEngine.ECS.ComponentContext;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ECS.Prefabs;
import VulkanEngine.Components.Transform;
import VulkanEngine.Components.MeshReference;
#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.Types;
import VulkanEngine.Physics.Components;
import VulkanEngine.Physics.World;
import VulkanEngine.Physics.Character;
import VulkanEngine.Physics.BodyBridge;
#endif

namespace {

using namespace VulkanEngine;
using Prefabs::Prefab;
using Prefabs::kNoMesh;
using Components::Transform;
using Components::MeshReference;
#ifdef VKENGINE_PHYSICS
using Physics::RigidBody;
using Physics::Collider;
using Physics::Character;
using Physics::CharacterDesc;
using Physics::CollisionLayer;
using Physics::BodyMotionType;
using Physics::ShapeType;
using Physics::ShapeDesc;
using Physics::World;
#endif

#ifdef VKENGINE_PHYSICS
[[nodiscard]] RigidBody BoxBody(glm::vec3 extent, BodyMotionType motion, float mass = 1.0f) {
    RigidBody body{};
    body.shape.type = ShapeType::Box;
    body.shape.box_extent = extent;
    body.motion = motion;
    body.layer = CollisionLayer::Default;
    body.mass = mass;
    body.friction = 0.5f;
    return body;
}
#endif

} // namespace

TEST(ComponentContextTest, GrantsAreIndependentViewsNotTheEngine) {
    // A default context grants nothing: components read null and decide.
    ComponentContext empty{};
    EXPECT_EQ(empty.Input(), nullptr);
#ifdef VKENGINE_PHYSICS
    EXPECT_EQ(empty.PhysicsWorld(), nullptr);
#endif

    VulkanEngine::Input::InputSystem input{};
    ComponentContext granted{};
    granted.input = &input;
    EXPECT_EQ(granted.Input(), &input);
}

TEST(ComponentContextTest, UnownedContextDefaultsAreHarmless) {
    ComponentRegistry registry;
    class Probe : public Component {
    public:
        void Update(float /*delta_time*/, const ComponentContext& context) override {
            seen_ = &context;
        }
        [[nodiscard]] const ComponentContext* Seen() const noexcept { return seen_; }

    private:
        const ComponentContext* seen_ = nullptr;
    };
    Probe& probe = registry.AddComponent<Probe>(registry.CreateEntity());

    // The default-arg path: a registry update with no explicit context.
    registry.UpdateAllComponentsAsync(0.016f);
    EXPECT_EQ(probe.Seen()->Input(), nullptr);
}

TEST(PrefabTest, APoseOnlyPrefabIsAPlainTransformEntity) {
    ComponentRegistry registry;
    const EntityId id =
        Prefabs::Spawn(registry, Prefab{
                                     .name = "marker",
                                     .position = {1.0f, 2.0f, 3.0f},
                                     .scale = {2.0f, 2.0f, 2.0f},
                                 })
            .GetId();

    Entity* entity = registry.TryGetEntity(id);
    ASSERT_NE(entity, nullptr);
    const auto* transform = entity->GetComponent<Transform>();
    ASSERT_NE(transform, nullptr);
    EXPECT_EQ(transform->position, (glm::vec3{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(transform->scale, (glm::vec3{2.0f, 2.0f, 2.0f}));
    EXPECT_FALSE(entity->HasComponent<MeshReference>());
}

#ifdef VKENGINE_PHYSICS
TEST(PrefabTest, AFullBundleComposesOnOneEntity) {
    ComponentRegistry registry;
    World world;
    ASSERT_TRUE(world.Initialize());

    // The engine's own attachments (physics hooks) are what a prefab relies on,
    // wiring identical to the engine's physics module.
    registry.OnAdd<RigidBody>(
        [&](Entity& entity, RigidBody& body) { VulkanEngine::Physics::AttachBody(world, entity, body); });
    registry.OnRemove<RigidBody>(
        [&](Entity& /*entity*/, RigidBody& body) { VulkanEngine::Physics::DetachBody(world, body); });
    registry.OnAdd<Collider>([&](Entity& entity, Collider& collider) {
        if (auto* body = entity.GetComponent<RigidBody>()) {
            VulkanEngine::Physics::RebuildBody(world, entity, *body, collider.children);
        }
    });

    CharacterDesc character_desc{};

    Prefab prefab{
        .name = "crate",
        .position = {0.0f, 4.0f, 0.0f},
        .mesh_id = 7,
        .rigid_body = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic),
        .character = Character{.desc = character_desc},
    };
    // (A body and a character on one entity only for the composition check; the
    // composed example uses one or the other.)

    Entity& entity = Prefabs::Spawn(registry, prefab);
    const EntityId id = entity.GetId();

    EXPECT_TRUE(entity.HasComponent<Transform>());
    EXPECT_TRUE(entity.HasComponent<MeshReference>());
    EXPECT_EQ(entity.GetComponent<MeshReference>()->loaded_mesh_id, 7u);
    EXPECT_TRUE(entity.HasComponent<RigidBody>());
    EXPECT_EQ(world.BodyCount(), 1u)
        << "the body was attached through the engine's OnAdd hook, not by the factory";

    registry.DestroyEntity(*registry.TryGetEntity(id));
    EXPECT_EQ(world.BodyCount(), 0u);
    world.Shutdown();
}

TEST(PrefabTest, ColliderBundlesComposeIntoCompoundBodies) {
    ComponentRegistry registry;
    World world;
    ASSERT_TRUE(world.Initialize());

    registry.OnAdd<RigidBody>(
        [&](Entity& entity, RigidBody& body) { VulkanEngine::Physics::AttachBody(world, entity, body); });
    registry.OnRemove<RigidBody>(
        [&](Entity& /*entity*/, RigidBody& body) { VulkanEngine::Physics::DetachBody(world, body); });
    registry.OnAdd<Collider>([&](Entity& entity, Collider& collider) {
        if (auto* body = entity.GetComponent<RigidBody>()) {
            VulkanEngine::Physics::RebuildBody(world, entity, *body, collider.children);
        }
    });

    Prefab prefab{};
    prefab.position = {0.0f, 8.0f, 0.0f};
    prefab.rigid_body = BoxBody({1.0f, 1.0f, 1.0f}, BodyMotionType::Dynamic);
    ShapeDesc leg{};
    leg.type = ShapeType::Box;
    leg.box_extent = {0.5f, 0.5f, 0.5f};
    leg.position = {2.0f, 0.0f, 0.0f};
    Collider collider{};
    collider.children.push_back(leg);
    prefab.collider = collider;

    Entity& entity = Prefabs::Spawn(registry, prefab);
    ASSERT_TRUE(world.BodyCount() == 1);

    prefab.mesh_id = kNoMesh;
    // Removing the collider through the registry restores the single-shape
    // body — the bundle is not special-cased anywhere.
    registry.RemoveComponent<Collider>(entity);
    EXPECT_TRUE(entity.GetComponent<RigidBody>()->body.IsValid());
    world.Shutdown();
}
#endif
