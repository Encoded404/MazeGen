// Component context: the narrow service view granted to object components
// (rung 3 of engine modularization).
//
// Object components need engine services (input, physics, config), but
// construction-site injection forces games into per-type wiring — every factory
// repeats the engine context it collected. The fix keeps the ECS contract
// intact: the engine builds ONE view of the services an object component is
// allowed to see and hands it to Component::Update. The component never
// reaches the world or the engine; it reaches what was granted.
//
// The struct is a plain data view over granted service pointers, so
// `UpdateAllComponentsAsync` can copy it into each parallel job. Services not
// granted (null when the subsystem is not compiled in or not initialized) read
// as null — a component decides what to do without them, exactly like the
// engine's no-singleton rule.

module;

#include <cassert>

export module VulkanEngine.ECS.ComponentContext;

import std;

import VulkanEngine.Input;

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.World;
#endif

export namespace VulkanEngine {

struct ComponentContext {
    // Input bindings and edge/level queries. Granted by the engine (the app's
    // InputSystem), never owned here.
    VulkanEngine::Input::InputSystem* input = nullptr; // NOLINT(misc-non-private-member-variables-in-classes)

#ifdef VKENGINE_PHYSICS
    // The physics world facade, for steering/diagnosing components that need
    // it (queries, velocity control). Granted when the physics module is
    // running; null otherwise.
    VulkanEngine::Physics::World* physics_world = nullptr; // NOLINT(misc-non-private-member-variables-in-classes)
#endif

    [[nodiscard]] VulkanEngine::Input::InputSystem* Input() const noexcept {
        return input;
    }
#ifdef VKENGINE_PHYSICS
    [[nodiscard]] VulkanEngine::Physics::World* PhysicsWorld() const noexcept {
        return physics_world;
    }
#endif
};

} // namespace VulkanEngine