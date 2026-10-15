// Physics debug overlay.
//
// Collects the world's debug geometry once per frame and submits it to the
// current ImGui frame: a "Physics" toggles window plus the projected geometry,
// drawn through ImGui's background draw list. The engine has no debug line
// pass, so the draw list is the whole renderer here -- and consequently this
// module creates no render pass, shader or pipeline.
//
// No Jolt type appears in this interface. The world's facade hands over plain
// vertices with packed colors (see PhysicsDebugGeometry); this module only
// knows how to project, clip, wind-cull and draw them.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.DebugOverlay;

import std;

import VulkanEngine.Components.Camera;

export import VulkanEngine.Physics.World;

export namespace VulkanEngine::Physics {

/// The overlay's projection, as a pure function so device-free tests can pin
/// the clipping and winding conventions the draw loop relies on.
///
/// The triangle is clipped against the camera plane (Sutherland-Hodgman on clip
/// `w`) before projection, which is what keeps large surfaces from vanishing
/// when the camera enters their footprint: Jolt emits a box's top face as a
/// single triangle per half-quad, and a whole-primitive rejection would drop
/// it whenever one corner falls behind the camera. `signed_area` is the double
/// area in ImGui screen space after the overlay's y flip — the flip inverts
/// handedness, so a triangle wound per Jolt's (right-handed, CCW from outside)
/// contract is front-facing exactly when `signed_area` is negative. An
/// orthographic projection never trips the clip (`w` is constant there and
/// behind-camera geometry projects correctly anyway).
struct DebugProjectedPolygon {
    std::size_t count = 0; ///< projected vertex count; 0 means nothing is visible
    float signed_area = 0.0f; ///< negative = front-facing per Jolt's winding
    std::array<glm::vec2, 4> points{}; ///< screen-space corners, `count` of them used
};

/// Clips `triangle` to the camera-plane-forward part and projects it.
[[nodiscard]] DebugProjectedPolygon ProjectDebugTriangle(const glm::mat4& view_projection,
                                                         float display_width, float display_height,
                                                         const glm::vec3& a, const glm::vec3& b,
                                                         const glm::vec3& c);

/// Clips the segment to its camera-plane-forward part and projects it. Returns
/// false when the whole segment is behind the camera.
[[nodiscard]] bool ProjectDebugSegment(const glm::mat4& view_projection, float display_width,
                                       float display_height, const glm::vec3& from,
                                       const glm::vec3& to, glm::vec2& out_from,
                                       glm::vec2& out_to);

/// Owns the debug-overlay state (toggles + the frame's collected geometry) and
/// draws both the toggles window and the projected geometry.
class PhysicsDebugOverlay {
public:
    /// Collects geometry from `world` and submits it to the current ImGui frame.
    /// `camera` supplies view/projection; `aspect_ratio` is width/height.
    ///
    /// Must be called inside an ImGui frame (after `ImGuiSystem::NewFrame`),
    /// which is what the engine's `draw_callbacks` seam provides. Safe with an
    /// empty world or empty geometry. Closing the window hides the toggles but
    /// not the geometry: the window is the controls, `WindowOpen` is not a
    /// per-frame enable.
    void Draw(World& world, const Components::Camera& camera, float aspect_ratio);

    /// The toggles the next `Draw` collects with. UI code may write them; the
    /// overlay itself also writes the layer-mask bits.
    [[nodiscard]] PhysicsDebugSettings& Settings() noexcept;
    [[nodiscard]] const PhysicsDebugSettings& Settings() const noexcept;

    /// Geometry collected on the last Draw (diagnostics and tests).
    [[nodiscard]] const PhysicsDebugGeometry& Geometry() const noexcept;

    /// True when the overlay window is open (it starts open).
    [[nodiscard]] bool WindowOpen() const noexcept;
    void SetWindowOpen(bool open) noexcept;

private:
    PhysicsDebugSettings settings_{};
    PhysicsDebugGeometry geometry_{};
    bool window_open_ = true;
};

} // namespace VulkanEngine::Physics
