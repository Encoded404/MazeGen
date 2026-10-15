// The gameplay-character component and its ECS bridge.
//
// A `CharacterVirtual` is not a rigid body: it is a swept shape that the engine
// moves itself, outside the solver step. That is what buys stair walking, slope
// sticking and moving-platform rides without contorting the solver, and it is
// why characters are stepped in `SystemSlot::PhysicsInput` (before the solver
// step) rather than being simulated.
//
// Like `RigidBody`, the component holds an opaque handle and the world owns the
// state; the registry's OnAdd/OnRemove hooks are the ownership seam, and
// `Transform` is the character's output pose for gameplay and rendering.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Character;

import std;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Physics.World;

export import VulkanEngine.Physics.Types;

export namespace VulkanEngine::Physics {

/// An input-driven gameplay character.
///
/// Data component: the handle names world-owned state, the fields are the
/// authored input and the read-back. Gameplay writes `desired_velocity` (world
/// space, m/s) — horizontal intent plus an optional upward jump impulse — and
/// reads `on_ground` and the entity's `Transform` for the resulting pose.
///
/// Vertical motion is engine-owned unless gameplay asks for a jump: when
/// `desired_velocity.y` is zero the character keeps its own vertical speed and
/// accumulates gravity, which is what makes it fall and land correctly.
struct Character {
    CharacterId character{}; // NOLINT(misc-non-private-member-variables-in-classes)
    CharacterDesc desc{}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::vec3 desired_velocity{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    bool on_ground = false; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// Creates the world-owned character for `entity` at its `Transform` pose.
void AttachCharacter(World& world, Entity& entity, Character& character);

/// Destroys the world-owned character.
void DetachCharacter(World& world, Character& character);

/// `SystemSlot::PhysicsInput`: advances every character one sweep, then writes its
/// pose into the entity's `Transform`. Runs before the solver step, so a character
/// pushed by a kinematic platform sees that platform's new position.
void StepCharacters(World& world, ComponentRegistry& registry, float delta_time);

} // namespace VulkanEngine::Physics
