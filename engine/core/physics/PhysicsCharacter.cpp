// Gameplay-character bridge implementation.

module;

#include <logging/logging_macros.hpp>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

module VulkanEngine.Physics.Character;

import std;

import logiface;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Components.Transform;
import VulkanEngine.Physics.World;
import VulkanEngine.Physics.BodyBridge;

namespace VulkanEngine::Physics {

void AttachCharacter(World& world, Entity& entity, Character& character) {
    if (character.character.IsValid()) return;
    if (!world.IsInitialized()) return;

    BodyPose pose{};
    if (const auto* transform = entity.GetComponent<Components::Transform>()) {
        pose.position = transform->position;
        pose.rotation = transform->rotation;
    }

    character.character =
        world.CreateCharacter(character.desc, pose, PackEntityId(entity.GetId()));
    if (!character.character.IsValid()) {
        LOGIFACE_LOG(warn, std::format("physics: no character created for entity {} (shape {})",
                                       entity.GetId().index,
                                       ShapeTypeName(character.desc.shape.type)));
        return;
    }

    // A character in the ECS is one the renderer draws, so it is interpolated
    // like a body: the sweep runs in the `PhysicsInput` slot and `Step` captures
    // the result, so both kinds of actor blend from one rotation point.
    world.TrackCharacterRenderPose(character.character);
}

void DetachCharacter(World& world, Character& character) {
    if (!character.character.IsValid()) return;
    world.UntrackCharacterRenderPose(character.character);
    world.DestroyCharacter(character.character);
    character.character = {};
}

void StepCharacters(World& world, ComponentRegistry& registry, float delta_time) {
    if (!world.IsInitialized()) return;

    const glm::vec3 gravity = world.Config().gravity;

    registry.ForEach<Character, Components::Transform>(
        [&world, gravity, delta_time](Entity& /*entity*/, Character& character,
                                      Components::Transform& transform) {
            if (!character.character.IsValid()) return;

            // The character's pose is authoritative; the Transform mirrors it so
            // gameplay and rendering read one value.
            BodyPose pose = world.GetCharacterPose(character.character);
            transform.position = pose.position;
            transform.rotation = pose.rotation;

            const bool on_ground = world.IsCharacterOnGround(character.character);
            glm::vec3 velocity = character.desired_velocity;
            if (character.desired_velocity.y == 0.0f) {
                // No jump requested: keep the character's own vertical speed and
                // integrate gravity, so it falls and lands instead of hovering.
                velocity.y = on_ground ? 0.0f
                                       : world.GetCharacterVelocity(character.character).y +
                                             gravity.y * delta_time;
            }
            if (on_ground) {
                // Add the surface velocity: a moving platform carries the
                // character rather than sliding out from under it.
                velocity += world.GetCharacterGroundVelocity(character.character);
            }
            world.SetCharacterVelocity(character.character, velocity);

            world.StepCharacter(character.character, delta_time);

            pose = world.GetCharacterPose(character.character);
            transform.position = pose.position;
            transform.rotation = pose.rotation;
            character.on_ground = world.IsCharacterOnGround(character.character);
        });
}

} // namespace VulkanEngine::Physics
