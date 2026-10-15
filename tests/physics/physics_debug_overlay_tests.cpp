// Device-free tests for the physics debug overlay's projection contract.
//
// The overlay draws Jolt's debug shapes through ImGui's painter's-algorithm
// draw list, so it realizes two renderer-side conventions itself: near-plane
// clipping of primitives that cross the camera plane, and winding-based face
// culling per Jolt's "back face culled triangle" contract. These tests pin the
// pure projection helpers the draw loop uses.

#include <gtest/gtest.h>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/matrix_transform.hpp> // NOLINT(misc-include-cleaner)

import std;

import VulkanEngine.Physics.DebugOverlay;

namespace {

using VulkanEngine::Physics::DebugProjectedPolygon;
using VulkanEngine::Physics::ProjectDebugSegment;
using VulkanEngine::Physics::ProjectDebugTriangle;

constexpr float kWidth = 1280.0f;
constexpr float kHeight = 720.0f;

/// The vehicle example's chase-camera pose with the car at the origin.
[[nodiscard]] glm::mat4 ChaseViewProjection() {
    const glm::mat4 view = glm::lookAt(glm::vec3{0.0f, 3.2f, -8.5f},
                                       glm::vec3{0.0f, 0.6f, 0.0f},
                                       glm::vec3{0.0f, 1.0f, 0.0f});
    return glm::perspective(glm::radians(60.0f), kWidth / kHeight, 0.1f, 100.0f) * view;
}

/// Top face of a unit cube at (0, 1, 8), wound per Jolt's contract: the
/// right-hand-rule normal points out of the shape (+y here), and the camera
/// looks down at it, so the face is front-facing.
[[nodiscard]] glm::vec3 FrontFaceVertexA() noexcept { return {-0.5f, 1.0f, 7.5f}; }
[[nodiscard]] glm::vec3 FrontFaceVertexB() noexcept { return {-0.5f, 1.0f, 8.5f}; }
[[nodiscard]] glm::vec3 FrontFaceVertexC() noexcept { return {0.5f, 1.0f, 8.5f}; }

} // namespace

TEST(PhysicsDebugOverlay, FrontFacingTrianglesHaveNegativeScreenSpaceArea) {
    const glm::mat4 view_projection = ChaseViewProjection();
    const glm::vec3 a = FrontFaceVertexA();
    const glm::vec3 b = FrontFaceVertexB();
    const glm::vec3 c = FrontFaceVertexC();
    ASSERT_GT(glm::dot(glm::cross(b - a, c - a), glm::vec3{0.0f, 1.0f, 0.0f}), 0.0f)
        << "test setup: the triangle must be wound per Jolt's contract";

    const DebugProjectedPolygon projected =
        ProjectDebugTriangle(view_projection, kWidth, kHeight, a, b, c);
    ASSERT_EQ(projected.count, 3u);
    // The overlay's y flip inverts handedness: a Jolt front face is negative.
    // A cull implemented with the opposite sign would drop exactly the visible
    // surfaces, which is the trap this pins down.
    EXPECT_LT(projected.signed_area, 0.0f);
}

TEST(PhysicsDebugOverlay, ReversedWindingIsBackFacing) {
    const glm::mat4 view_projection = ChaseViewProjection();
    const DebugProjectedPolygon projected = ProjectDebugTriangle(
        view_projection, kWidth, kHeight, FrontFaceVertexC(), FrontFaceVertexB(),
        FrontFaceVertexA());
    ASSERT_EQ(projected.count, 3u);
    EXPECT_GT(projected.signed_area, 0.0f);
}

TEST(PhysicsDebugOverlay, TriangleCrossingTheCameraPlaneIsClippedNotDropped) {
    const glm::mat4 view_projection = ChaseViewProjection();

    // The vehicle example's floor top face: it spans the whole 60 m slab and
    // the chase camera sits inside the footprint, so half the primitive is
    // behind the camera plane. Dropping whole primitives here is what used to
    // leave the floor rendered as edge slivers instead of the driving surface.
    const glm::vec3 a{-30.0f, 0.0f, -30.0f};
    const glm::vec3 b{-30.0f, 0.0f, 30.0f};
    const glm::vec3 c{30.0f, 0.0f, 30.0f};
    ASSERT_GT(glm::dot(glm::cross(b - a, c - a), glm::vec3{0.0f, 1.0f, 0.0f}), 0.0f)
        << "test setup: wound per Jolt, front-facing toward the chase camera";

    const DebugProjectedPolygon projected =
        ProjectDebugTriangle(view_projection, kWidth, kHeight, a, b, c);
    ASSERT_EQ(projected.count, 4u)
        << "a half-behind-camera triangle must clip to a quad, not vanish";
    EXPECT_LT(projected.signed_area, 0.0f);
    for (std::size_t i = 0; i < projected.count; ++i) {
        EXPECT_TRUE(std::isfinite(projected.points[i].x));
        EXPECT_TRUE(std::isfinite(projected.points[i].y));
    }
}

TEST(PhysicsDebugOverlay, TriangleFullyBehindTheCameraIsDropped) {
    const glm::mat4 view_projection = ChaseViewProjection();
    const DebugProjectedPolygon projected = ProjectDebugTriangle(
        view_projection, kWidth, kHeight, glm::vec3{0.0f, 3.2f, -20.0f},
        glm::vec3{1.0f, 3.2f, -20.0f}, glm::vec3{0.0f, 4.2f, -20.0f});
    EXPECT_EQ(projected.count, 0u);
}

TEST(PhysicsDebugOverlay, SegmentCrossingTheCameraPlaneKeepsItsForwardPart) {
    const glm::mat4 view_projection = ChaseViewProjection();
    glm::vec2 from{};
    glm::vec2 to{};
    EXPECT_TRUE(ProjectDebugSegment(view_projection, kWidth, kHeight,
                                    glm::vec3{0.0f, 3.2f, -30.0f},
                                    glm::vec3{0.0f, 3.2f, 10.0f}, from, to));
    // Orthographic-style whole-primitive bounds only hold for the endpoint that
    // is genuinely in front; the clipped endpoint legitimately extends past the
    // display, so only finiteness is asserted for it.
    EXPECT_TRUE(std::isfinite(from.x) && std::isfinite(from.y));
    EXPECT_TRUE(std::isfinite(to.x) && std::isfinite(to.y));
    EXPECT_GE(to.x, 0.0f);
    EXPECT_LE(to.x, kWidth);
    EXPECT_GE(to.y, 0.0f);
    EXPECT_LE(to.y, kHeight);
}

TEST(PhysicsDebugOverlay, SegmentFullyBehindTheCameraIsDropped) {
    const glm::mat4 view_projection = ChaseViewProjection();
    glm::vec2 from{};
    glm::vec2 to{};
    EXPECT_FALSE(ProjectDebugSegment(view_projection, kWidth, kHeight,
                                     glm::vec3{0.0f, 3.2f, -30.0f},
                                     glm::vec3{1.0f, 3.2f, -30.0f}, from, to));
}