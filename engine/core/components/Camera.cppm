module;

#include <glm/glm.hpp> //NOLINT(misc-include-cleaner)
#include <glm/gtc/matrix_transform.hpp> //NOLINT(misc-include-cleaner)

export module VulkanEngine.Components.Camera;

export import VulkanEngine.ECS.ComponentRegistry;

import std;

export namespace VulkanEngine::Components {

enum class ProjectionMode : std::uint8_t {
    Perspective,
    Orthographic
};

// How GetViewMatrix() derives the camera orientation:
//   LookAtTarget - aim from `position` at `target` (default, historical behaviour).
//   Direction    - aim from `position` along `forward`, ignoring `target` entirely.
enum class CameraOrientation : std::uint8_t {
    LookAtTarget,
    Direction
};

// Data component: camera parameters plus the matrix helpers that read them.
// Dense storage, so do not cache a Camera* -- store the owning EntityId.
struct Camera {
    glm::vec3 position{0.0f, 0.0f, 3.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::vec3 target{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::vec3 up{0.0f, 1.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::vec3 forward{0.0f, 0.0f, -1.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    float fov_degrees = 60.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    float orthographic_size = 10.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    float near_plane = 0.1f; // NOLINT(misc-non-private-member-variables-in-classes)
    float far_plane = 10000.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    ProjectionMode projection_mode = ProjectionMode::Perspective; // NOLINT(misc-non-private-member-variables-in-classes)
    CameraOrientation orientation = CameraOrientation::LookAtTarget; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] glm::mat4 GetViewMatrix() const {
        if (orientation == CameraOrientation::Direction) {
            // A zero-length forward would make lookAt emit NaNs; fall back to the
            // target-based path rather than producing a degenerate matrix. lookAt
            // normalises the direction itself, so no pre-normalise is needed here.
            if (glm::dot(forward, forward) > 0.0f) {
                return glm::lookAt(position, position + forward, up);
            }
        }
        return glm::lookAt(position, target, up);
    }

    [[nodiscard]] glm::mat4 GetProjectionMatrix(float aspect) const {
        if (projection_mode == ProjectionMode::Orthographic) {
            const float half_h = orthographic_size * 0.5f;
            const float half_w = half_h * aspect;
            return glm::ortho(-half_w, half_w, -half_h, half_h, near_plane, far_plane);
        }
        const float fov_rad = fov_degrees * (std::numbers::pi_v<float> / 180.0f);
        return glm::perspective(fov_rad, aspect, near_plane, far_plane);
    }
};

} // namespace VulkanEngine::Components
