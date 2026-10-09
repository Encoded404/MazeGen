#include <gtest/gtest.h>

import std;

import VulkanEngine.Mesh.MeshTypes;

namespace {

using VulkanEngine::ComposeSubmeshTransform;
using VulkanEngine::MeshVertexVec3;
using VulkanEngine::SubMeshTransform;

constexpr float kEps = 1e-5f;
constexpr float kHalfSqrt2 = 0.70710678f;

SubMeshTransform MakeTrs(MeshVertexVec3 position,
                         float rx, float ry, float rz, float rw,
                         MeshVertexVec3 scale) {
    SubMeshTransform t{};
    t.position = position;
    t.rotation[0] = rx;
    t.rotation[1] = ry;
    t.rotation[2] = rz;
    t.rotation[3] = rw;
    t.scale = scale;
    return t;
}

SubMeshTransform Identity() { return SubMeshTransform{}; }

// q and -q name the same rotation, and the shader consumes the quaternion
// directly, so compare rotations up to sign.
void ExpectSameRotation(const float actual[4], const float expected[4]) {
    const float dot = actual[0] * expected[0] + actual[1] * expected[1] +
                      actual[2] * expected[2] + actual[3] * expected[3];
    const float sign = dot < 0.0f ? -1.0f : 1.0f;
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(actual[i], sign * expected[i], kEps) << "rotation component " << i;
    }
}

void ExpectSameTransform(const SubMeshTransform& actual, const SubMeshTransform& expected) {
    EXPECT_NEAR(actual.position.x, expected.position.x, kEps);
    EXPECT_NEAR(actual.position.y, expected.position.y, kEps);
    EXPECT_NEAR(actual.position.z, expected.position.z, kEps);
    ExpectSameRotation(actual.rotation, expected.rotation);
    EXPECT_NEAR(actual.scale.x, expected.scale.x, kEps);
    EXPECT_NEAR(actual.scale.y, expected.scale.y, kEps);
    EXPECT_NEAR(actual.scale.z, expected.scale.z, kEps);
}

constexpr MeshVertexVec3 kOrigin{0.0f, 0.0f, 0.0f};
constexpr MeshVertexVec3 kUnitScale{1.0f, 1.0f, 1.0f};

// 90 degrees about +Z: maps (1,0,0) to (0,1,0), and composed with itself to
// (-1,0,0). The sign convention is the right-hand rule, matching the render
// path's rotate-by-quaternion.
SubMeshTransform RotZ90() {
    return MakeTrs(kOrigin, 0.0f, 0.0f, kHalfSqrt2, kHalfSqrt2, kUnitScale);
}

TEST(SubMeshTransformCompose, IdentityIsNeutral) {
    const SubMeshTransform t = MakeTrs(MeshVertexVec3{1.0f, 2.0f, 3.0f},
                                       0.0f, 0.0f, kHalfSqrt2, kHalfSqrt2,
                                       MeshVertexVec3{2.0f, 3.0f, 4.0f});

    ExpectSameTransform(ComposeSubmeshTransform(Identity(), Identity()), Identity());
    ExpectSameTransform(ComposeSubmeshTransform(Identity(), t), t);
    ExpectSameTransform(ComposeSubmeshTransform(t, Identity()), t);
}

TEST(SubMeshTransformCompose, DefaultTransformIsIdentity) {
    const SubMeshTransform t{};
    EXPECT_FLOAT_EQ(t.position.x, 0.0f);
    EXPECT_FLOAT_EQ(t.position.y, 0.0f);
    EXPECT_FLOAT_EQ(t.position.z, 0.0f);
    EXPECT_FLOAT_EQ(t.rotation[0], 0.0f);
    EXPECT_FLOAT_EQ(t.rotation[1], 0.0f);
    EXPECT_FLOAT_EQ(t.rotation[2], 0.0f);
    EXPECT_FLOAT_EQ(t.rotation[3], 1.0f);
    EXPECT_FLOAT_EQ(t.scale.x, 1.0f);
    EXPECT_FLOAT_EQ(t.scale.y, 1.0f);
    EXPECT_FLOAT_EQ(t.scale.z, 1.0f);
}

TEST(SubMeshTransformCompose, AddsTranslationsWhenNeitherRotates) {
    const auto parent = MakeTrs(MeshVertexVec3{1.0f, 2.0f, 3.0f}, 0.0f, 0.0f, 0.0f, 1.0f, kUnitScale);
    const auto child = MakeTrs(MeshVertexVec3{10.0f, 20.0f, 30.0f}, 0.0f, 0.0f, 0.0f, 1.0f, kUnitScale);

    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.position.x, 11.0f, kEps);
    EXPECT_NEAR(out.position.y, 22.0f, kEps);
    EXPECT_NEAR(out.position.z, 33.0f, kEps);
}

TEST(SubMeshTransformCompose, ParentRotationRotatesTheChildOffset) {
    const auto parent = RotZ90();
    const auto child = MakeTrs(MeshVertexVec3{1.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 0.0f, 1.0f, kUnitScale);

    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.position.x, 0.0f, kEps);
    EXPECT_NEAR(out.position.y, 1.0f, kEps);
    EXPECT_NEAR(out.position.z, 0.0f, kEps);
    ExpectSameRotation(out.rotation, parent.rotation);
}

TEST(SubMeshTransformCompose, ChildRotationDoesNotMoveTheParentPosition) {
    const auto parent = MakeTrs(MeshVertexVec3{5.0f, 6.0f, 7.0f}, 0.0f, 0.0f, kHalfSqrt2, kHalfSqrt2, kUnitScale);
    const auto child = RotZ90();

    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.position.x, 5.0f, kEps);
    EXPECT_NEAR(out.position.y, 6.0f, kEps);
    EXPECT_NEAR(out.position.z, 7.0f, kEps);
}

TEST(SubMeshTransformCompose, ParentScaleScalesTheChildOffset) {
    const auto parent = MakeTrs(kOrigin, 0.0f, 0.0f, 0.0f, 1.0f, MeshVertexVec3{2.0f, 3.0f, 4.0f});
    const auto child = MakeTrs(MeshVertexVec3{1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, 0.0f, 1.0f, kUnitScale);

    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.position.x, 2.0f, kEps);
    EXPECT_NEAR(out.position.y, 3.0f, kEps);
    EXPECT_NEAR(out.position.z, 4.0f, kEps);
}

TEST(SubMeshTransformCompose, ScalesMultiplyComponentwise) {
    const auto parent = MakeTrs(kOrigin, 0.0f, 0.0f, 0.0f, 1.0f, MeshVertexVec3{2.0f, 3.0f, 4.0f});
    const auto child = MakeTrs(kOrigin, 0.0f, 0.0f, 0.0f, 1.0f, MeshVertexVec3{0.5f, 2.0f, 0.25f});

    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.scale.x, 1.0f, kEps);
    EXPECT_NEAR(out.scale.y, 6.0f, kEps);
    EXPECT_NEAR(out.scale.z, 1.0f, kEps);
}

TEST(SubMeshTransformCompose, ChildRotationComposesWithoutMovingItsOffset) {
    const auto parent = RotZ90();
    const auto child = MakeTrs(MeshVertexVec3{1.0f, 0.0f, 0.0f}, 0.0f, 0.0f, kHalfSqrt2, kHalfSqrt2, kUnitScale);

    // Only the parent's rotation moves the child's origin: (1,0,0) -> (0,1,0).
    // The child's own rotation does not translate it; it composes instead, so
    // two 90-degree rotations about Z become 180 degrees: (1,0,0) -> (-1,0,0)
    // for any geometry the child carries.
    const auto out = ComposeSubmeshTransform(parent, child);
    EXPECT_NEAR(out.position.x, 0.0f, kEps);
    EXPECT_NEAR(out.position.y, 1.0f, kEps);
    EXPECT_NEAR(out.position.z, 0.0f, kEps);

    const float kRotZ180[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    ExpectSameRotation(out.rotation, kRotZ180);
}

TEST(SubMeshTransformCompose, ComposedRotationStaysUnitLength) {
    const auto parent = MakeTrs(kOrigin, 0.1f, 0.2f, 0.3f, 0.9273618f, kUnitScale);
    const auto child = MakeTrs(kOrigin, 0.5f, -0.5f, 0.5f, 0.5f, kUnitScale);

    const auto out = ComposeSubmeshTransform(parent, child);
    const float length = std::sqrt(out.rotation[0] * out.rotation[0] +
                                   out.rotation[1] * out.rotation[1] +
                                   out.rotation[2] * out.rotation[2] +
                                   out.rotation[3] * out.rotation[3]);
    EXPECT_NEAR(length, 1.0f, kEps);
}

} // namespace
