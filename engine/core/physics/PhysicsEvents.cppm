// Contact and activation events, in engine form.
//
// Jolt calls its listeners from several solver threads at once, in whatever order
// the work happened to complete, and only while all bodies are locked. The facade
// therefore does the only two things that are safe there: it copies the values it
// needs (body ids, user data, contact manifold) into a queue, and nothing else.
// Once the step has finished the queue is drained, sorted deterministically, and
// replayed into the game-facing callbacks on the stepping thread.
//
// That split is what makes the observable order a contract instead of a race:
// `ContactEvent`s replay ordered by the owning entity pair, then phase, then
// sub-shape pair, and a game callback may therefore only *queue* structural
// changes (DestroyEntity is deferred by the ECS; AddComponent from a physics
// callback is not allowed at all).

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Events;

import std;

export import VulkanEngine.Physics.Types;

export namespace VulkanEngine::Physics {

/// What happened to a contact this step.
enum class ContactPhase : std::uint8_t {
    Added = 0,     // a new contact pair
    Persisted = 1, // the same pair is still touching
    Removed = 2,   // the pair stopped touching
};

[[nodiscard]] constexpr std::string_view ContactPhaseName(ContactPhase phase) noexcept {
    switch (phase) {
        case ContactPhase::Added:     return "added";
        case ContactPhase::Persisted: return "persisted";
        case ContactPhase::Removed:   return "removed";
    }
    return "<unknown>";
}

/// One contact, resolved to the engine's handles.
///
/// `user_data_a`/`user_data_b` are the bodies' 64-bit payloads — the entity
/// handles the body bridge packed — so a handler can resolve the participants
/// without a second lookup. `body_a`/`body_b` are in Jolt's canonical order
/// (body id ascending), which is stable for a given pair.
struct ContactEvent {
    PhysicsBodyId body_a{};
    PhysicsBodyId body_b{};
    std::uint64_t user_data_a = 0;
    std::uint64_t user_data_b = 0;
    ContactPhase phase = ContactPhase::Added;
    /// First contact point of the manifold, in world space (zero for Removed,
    /// where the bodies can no longer be read).
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    /// Manifold normal: the direction to move body B out of body A.
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    float penetration = 0.0f;
    std::uint32_t sub_shape_a = 0;
    std::uint32_t sub_shape_b = 0;
    /// True when the contact is a trigger overlap rather than a collision.
    bool sensor = false;
};

/// A body woke up or went to sleep. Replayed in the same sorted pass.
struct ActivationEvent {
    PhysicsBodyId body{};
    std::uint64_t user_data = 0;
    bool activated = false;
};

/// Replay callbacks. Called once per fixed step on the stepping thread, after the
/// simulation, in the deterministic order described above.
using ContactCallback = std::function<void(const ContactEvent&)>;
using ActivationCallback = std::function<void(const ActivationEvent&)>;

} // namespace VulkanEngine::Physics
