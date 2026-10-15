// Physics debug overlay implementation.
//
// The projection mirrors the scene's own convention: the engine draws the world
// through a negative-height viewport (clip-space y is up), while ImGui's draw
// list is y-down, so the y axis is flipped back here. Drawing happens in the
// background draw list so the overlay sits behind any UI the app opens.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

#include <imgui.h>

module VulkanEngine.Physics.DebugOverlay;

import std;

import VulkanEngine.Components.Camera;
import VulkanEngine.Physics.World;

namespace VulkanEngine::Physics {

namespace {

/// Jolt packs colors as 0xAABBGGRR -- r in the low byte, then g, b, a -- which
/// is exactly IM_COL32's layout. The decode/re-encode is deliberate: it states
/// the contract instead of depending on the two packings agreeing by accident.
[[nodiscard]] ImU32 ToImGuiColor(std::uint32_t packed) noexcept {
    const std::uint32_t a = (packed >> 24) & 0xffu;
    const std::uint32_t b = (packed >> 16) & 0xffu;
    const std::uint32_t g = (packed >> 8) & 0xffu;
    const std::uint32_t r = packed & 0xffu;
    return IM_COL32(r, g, b, a);
}

/// Clip-space `w` below this is treated as "at or behind the camera". The clip
/// happens against this constant plane rather than dropping the whole primitive
/// per vertex, which is what used to make the vehicle example's floor slab
/// vanish: Jolt emits its top face as one triangle per half-quad spanning the
/// whole 60 m slab, so both top-face triangles always had a corner behind the
/// camera and were dropped entirely.
///
/// `w` is clip `w` from the combined view-projection: for the perspective
/// projection this is view-space depth, so the clip plane sits (almost) in the
/// camera plane. The epsilon keeps the perspective divide well-conditioned; an
/// orthographic projection has `w == 1` everywhere, so nothing is ever clipped.
constexpr float kMinClipW = 1.0e-3f;

/// World -> ImGui screen space. Returns false when the vertex is behind the
/// camera or its clip `w` is degenerate, which is what keeps a division by
/// (near) zero out of the projection.
[[nodiscard]] bool ProjectToScreen(const glm::mat4& view_projection,
                                   const glm::vec3& world_position,
                                   float width, float height, ImVec2& out) noexcept {
    const glm::vec4 clip = view_projection * glm::vec4(world_position, 1.0f);
    if (clip.w <= 1.0e-6f) return false;
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    // The scene's viewport has a negative height; ImGui's draw list is y-down.
    out.x = (ndc.x * 0.5f + 0.5f) * width;
    out.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height;
    return true;
}

/// Clip-space `w` of a world position; linear in the position, so edge
/// intersections computed on it match a world-space lerp.
[[nodiscard]] float ClipW(const glm::mat4& view_projection,
                          const glm::vec3& world_position) noexcept {
    return (view_projection * glm::vec4(world_position, 1.0f)).w;
}

/// One checkbox label per collision layer. A literal, not
/// `CollisionLayerName`'s view: ImGui wants a NUL-terminated `const char*`.
[[nodiscard]] const char* LayerLabel(CollisionLayer layer) noexcept {
    switch (layer) {
        case CollisionLayer::Default:    return "Default";
        case CollisionLayer::Sensor:     return "Sensor";
        case CollisionLayer::Debris:     return "Debris";
        case CollisionLayer::Character:  return "Character";
        case CollisionLayer::Projectile: return "Projectile";
        case CollisionLayer::Count:      break;
    }
    return "Unknown";
}

constexpr std::uint32_t kLayerCount = static_cast<std::uint32_t>(CollisionLayer::Count);
/// Every layer's bit, used to materialize `layer_mask == 0` ("all") once a
/// checkbox is touched, so unchecking one layer means "all but this one".
constexpr std::uint32_t kAllLayerBits = (1u << kLayerCount) - 1u;

/// Guards against an aspect of 0 (a minimized or not-yet-sized window) without
/// silently changing a legitimate value.
[[nodiscard]] float SafeAspect(float aspect_ratio) noexcept {
    return aspect_ratio > 1.0e-6f ? aspect_ratio : 1.0f;
}

} // namespace

bool ProjectDebugSegment(const glm::mat4& view_projection, const float display_width,
                         const float display_height, const glm::vec3& from, const glm::vec3& to,
                         glm::vec2& out_from, glm::vec2& out_to) {
    const float w_from = ClipW(view_projection, from);
    const float w_to = ClipW(view_projection, to);
    if (w_from < kMinClipW && w_to < kMinClipW) return false;

    // Liang-Barsky-style endpoint clip against the `w == kMinClipW` plane.
    glm::vec3 a = from;
    glm::vec3 b = to;
    if (w_from < kMinClipW) {
        a = glm::mix(from, to, (kMinClipW - w_from) / (w_to - w_from));
    } else if (w_to < kMinClipW) {
        b = glm::mix(to, from, (kMinClipW - w_to) / (w_from - w_to));
    }

    ImVec2 from_screen{};
    ImVec2 to_screen{};
    if (!ProjectToScreen(view_projection, a, display_width, display_height, from_screen) ||
        !ProjectToScreen(view_projection, b, display_width, display_height, to_screen)) {
        return false;
    }
    out_from = glm::vec2(from_screen.x, from_screen.y);
    out_to = glm::vec2(to_screen.x, to_screen.y);
    return true;
}

DebugProjectedPolygon ProjectDebugTriangle(const glm::mat4& view_projection,
                                           const float display_width, const float display_height,
                                           const glm::vec3& a, const glm::vec3& b,
                                           const glm::vec3& c) {
    DebugProjectedPolygon result;
    const glm::vec3 input[3] = {a, b, c};
    constexpr std::size_t input_count = 3;

    // Sutherland-Hodgman against the single camera plane: keep the inside
    // vertex of each edge, then the edge/plane intersection for edges that
    // cross. A triangle yields at most a quad.
    const float w[input_count] = {ClipW(view_projection, a), ClipW(view_projection, b),
                                  ClipW(view_projection, c)};
    glm::vec3 clipped[4];
    std::size_t clipped_count = 0;
    for (std::size_t i = 0; i < input_count; ++i) {
        const std::size_t next = (i + 1) % input_count;
        const bool current_in = w[i] >= kMinClipW;
        const bool next_in = w[next] >= kMinClipW;
        if (current_in) clipped[clipped_count++] = input[i];
        if (current_in != next_in) {
            const float t = (kMinClipW - w[i]) / (w[next] - w[i]);
            clipped[clipped_count++] = glm::mix(input[i], input[next], t);
        }
    }
    if (clipped_count < 3) return result;

    ImVec2 screen[4];
    for (std::size_t i = 0; i < clipped_count; ++i) {
        if (!ProjectToScreen(view_projection, clipped[i], display_width, display_height,
                             screen[i])) {
            return DebugProjectedPolygon{}; // unreachable post-clip; keeps the contract
        }
        result.points[i] = glm::vec2(screen[i].x, screen[i].y);
    }

    // Shoelace double area over the (convex, same-orientation) clipped polygon.
    float signed_area = 0.0f;
    for (std::size_t i = 0; i < clipped_count; ++i) {
        const glm::vec2& current = result.points[i];
        const glm::vec2& next = result.points[(i + 1) % clipped_count];
        signed_area += current.x * next.y - next.x * current.y;
    }
    result.count = clipped_count;
    result.signed_area = signed_area;
    return result;
}

void PhysicsDebugOverlay::Draw(World& world, const Components::Camera& camera,
                               float aspect_ratio) {
    geometry_.Clear();
    world.CollectDebugGeometry(settings_, geometry_);

    // The ImGui viewport is the space the draw list is in, and what the backend
    // render targets, so screen positions come from it rather than from the
    // caller's aspect (which is only for the projection).
    const ImVec2 display = ImGui::GetIO().DisplaySize;

    if (window_open_) {
        ImGui::SetNextWindowSize(ImVec2(320.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Physics", &window_open_);

        ImGui::Checkbox("Draw shapes", &settings_.draw_shapes);
        ImGui::Checkbox("Wireframe", &settings_.wireframe);
        ImGui::Checkbox("Bounding boxes", &settings_.draw_bounding_boxes);
        ImGui::Checkbox("Centers of mass", &settings_.draw_centers_of_mass);
        ImGui::Checkbox("Constraints", &settings_.draw_constraints);
        ImGui::Checkbox("Constraint limits", &settings_.draw_constraint_limits);

        ImGui::SeparatorText("Collision layers");
        for (std::uint32_t i = 0; i < kLayerCount; ++i) {
            const std::uint32_t bit = 1u << i;
            bool checked = settings_.layer_mask == 0u || (settings_.layer_mask & bit) != 0u;
            if (ImGui::Checkbox(LayerLabel(static_cast<CollisionLayer>(i)), &checked)) {
                if (settings_.layer_mask == 0u) settings_.layer_mask = kAllLayerBits;
                if (checked) {
                    settings_.layer_mask |= bit;
                } else {
                    settings_.layer_mask &= ~bit;
                }
            }
        }

        ImGui::Separator();
        ImGui::Text("lines: %llu  triangles: %llu",
                    static_cast<unsigned long long>(geometry_.lines.size()),
                    static_cast<unsigned long long>(geometry_.triangles.size()));
        ImGui::Text("bodies: %u  fixed steps: %llu", world.BodyCount(),
                    static_cast<unsigned long long>(world.StepCount()));
        ImGui::End();
    }

    if (!(display.x > 0.0f) || !(display.y > 0.0f)) return;
    const glm::mat4 view_projection =
        camera.GetProjectionMatrix(SafeAspect(aspect_ratio)) * camera.GetViewMatrix();

    ImDrawList* const draw_list = ImGui::GetBackgroundDrawList();
    if (draw_list == nullptr) return;

    // A line list is pairs of vertices: Jolt's wireframe shapes and its
    // constraint lines both arrive this way. A segment that crosses the camera
    // plane is clipped to its forward part instead of being dropped whole.
    for (std::size_t i = 0; i + 1 < geometry_.lines.size(); i += 2) {
        glm::vec2 a{};
        glm::vec2 b{};
        if (!ProjectDebugSegment(view_projection, display.x, display.y, geometry_.lines[i].position,
                                 geometry_.lines[i + 1].position, a, b)) {
            continue;
        }
        draw_list->AddLine(ImVec2(a.x, a.y), ImVec2(b.x, b.y),
                           ToImGuiColor(geometry_.lines[i].color), 1.0f);
    }

    // A triangle list is triples: the solid shapes Jolt emits when wireframe is
    // off. Each primitive is clipped to the part in front of the camera and
    // filled as a fan; the cull contract the renderer recorded realizes Jolt's
    // "back face culled triangle" semantics against the winding, which the y
    // flip below inverted: negative signed area is front-facing here. Jolt
    // emits a box's faces as whole-footprint triangles, so a partially
    // behind-camera triangle is clipped, not skipped — that is what keeps the
    // floor the car drives on rendered.
    for (std::size_t i = 0; i + 2 < geometry_.triangles.size(); i += 3) {
        const auto clipped = ProjectDebugTriangle(view_projection, display.x, display.y,
                                                  geometry_.triangles[i].position,
                                                  geometry_.triangles[i + 1].position,
                                                  geometry_.triangles[i + 2].position);
        if (clipped.count < 3) continue;

        switch (geometry_.triangles[i].culling) {
            case PhysicsDebugFaceCulling::BackFace:
                // Positive signed area = wound away from the camera.
                if (clipped.signed_area >= 0.0f) continue;
                break;
            case PhysicsDebugFaceCulling::FrontFace:
                if (clipped.signed_area <= 0.0f) continue;
                break;
            case PhysicsDebugFaceCulling::None:
                break;
        }

        const ImU32 color = ToImGuiColor(geometry_.triangles[i].color);
        for (std::size_t j = 1; j + 1 < clipped.count; ++j) {
            draw_list->AddTriangleFilled(ImVec2(clipped.points[0].x, clipped.points[0].y),
                                         ImVec2(clipped.points[j].x, clipped.points[j].y),
                                         ImVec2(clipped.points[j + 1].x, clipped.points[j + 1].y),
                                         color);
            if (settings_.wireframe) {
                draw_list->AddTriangle(ImVec2(clipped.points[0].x, clipped.points[0].y),
                                       ImVec2(clipped.points[j].x, clipped.points[j].y),
                                       ImVec2(clipped.points[j + 1].x, clipped.points[j + 1].y),
                                       color, 1.0f);
            }
        }
    }
}

PhysicsDebugSettings& PhysicsDebugOverlay::Settings() noexcept { return settings_; }

const PhysicsDebugSettings& PhysicsDebugOverlay::Settings() const noexcept { return settings_; }

const PhysicsDebugGeometry& PhysicsDebugOverlay::Geometry() const noexcept { return geometry_; }

bool PhysicsDebugOverlay::WindowOpen() const noexcept { return window_open_; }

void PhysicsDebugOverlay::SetWindowOpen(bool open) noexcept { window_open_ = open; }

} // namespace VulkanEngine::Physics
