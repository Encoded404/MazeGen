module;

#include <glm/glm.hpp> //NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> //NOLINT(misc-include-cleaner)

export module Examples.BasicScene.Components.TransformControlComponent;

import std;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Components.Transform;

export namespace Examples::BasicScene::Components {

enum class RotationMode : std::uint8_t {
    Euler,
    Quaternion
};

// Object component: a debug-panel-controlled transform driver. Stored as a
// stable object, so examples may hold a pointer to it across frames.
class TransformControlComponent : public VulkanEngine::Component {
public:
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    RotationMode rotation_mode = RotationMode::Euler;
    glm::vec3 rotation_euler{0.0f, 0.0f, 0.0f};
    glm::quat rotation_quat{1.0f, 0.0f, 0.0f, 0.0f};
    int texture_slot = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    void Update(float /*delta_time*/, const VulkanEngine::ComponentContext& /*context*/) override {
        auto* transform = GetOwner() != nullptr ? GetOwner()->GetComponent<VulkanEngine::Components::Transform>() : nullptr;
        if (transform) {
            transform->position = position;
            if (rotation_mode == RotationMode::Euler) {
                transform->rotation = glm::quat{glm::radians(rotation_euler)};
            } else {
                transform->rotation = rotation_quat;
            }
        }
    }
};

}
