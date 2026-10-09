#include <gtest/gtest.h>

import std;

import VulkanEngine.Mesh.MeshTypes;
import VulkanEngine.Components.TransformOverride;

namespace {

using VulkanEngine::MeshVertexVec3;
using VulkanEngine::SubMeshTransform;
using VulkanEngine::Components::EffectiveSubmeshTransform;
using VulkanEngine::Components::ResolveEffectiveSubmeshTransform;
using VulkanEngine::Components::TransformOverride;
using VulkanEngine::Components::TransformOverrideStatus;

constexpr float kEps = 1e-5f;

SubMeshTransform MakeTrs(const float px, const float py, const float pz,
                         const float sx, const float sy, const float sz) {
    SubMeshTransform t{};
    t.position = MeshVertexVec3{px, py, pz};
    t.scale = MeshVertexVec3{sx, sy, sz};
    return t;
}

void ExpectPosition(const SubMeshTransform& actual, const float x, const float y, const float z) {
    EXPECT_NEAR(actual.position.x, x, kEps);
    EXPECT_NEAR(actual.position.y, y, kEps);
    EXPECT_NEAR(actual.position.z, z, kEps);
}

void ExpectScale(const SubMeshTransform& actual, const float x, const float y, const float z) {
    EXPECT_NEAR(actual.scale.x, x, kEps);
    EXPECT_NEAR(actual.scale.y, y, kEps);
    EXPECT_NEAR(actual.scale.z, z, kEps);
}

constexpr std::uint32_t kMeshId = 42;
constexpr std::uint32_t kSubmeshCount = 3;

// No override at all: the asset transform still places the submesh, composed
// with the entity so the object stays movable.
TEST(TransformOverrideResolve, NoOverrideComposesEntityAndAsset) {
    const auto entity = MakeTrs(10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    const auto asset = MakeTrs(0.0f, 2.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        entity, asset, nullptr, 0, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::NotPresent);
    ExpectPosition(resolved.transform, 10.0f, 2.0f, 0.0f);
}

TEST(TransformOverrideResolve, IdentityAssetLeavesTheEntityPlacement) {
    const auto entity = MakeTrs(1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        entity, SubMeshTransform{}, nullptr, 0, kMeshId, kSubmeshCount);

    ExpectPosition(resolved.transform, 1.0f, 2.0f, 3.0f);
    ExpectScale(resolved.transform, 2.0f, 2.0f, 2.0f);
}

TEST(TransformOverrideResolve, AppliedOverrideReplacesTheAssetAndComposesWithTheEntity) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId);
    override_component.Set(1, MakeTrs(4.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));

    const auto entity = MakeTrs(10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    const auto asset = MakeTrs(0.0f, 7.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        entity, asset, &override_component, 1, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::Applied);
    // The override replaces the asset's placement, then the entity still places
    // the object: the asset's +Y is gone, the entity's +X remains.
    ExpectPosition(resolved.transform, 14.0f, 0.0f, 0.0f);
}

TEST(TransformOverrideResolve, UnsetIndexFallsBackToTheAssetTransform) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId);
    override_component.Set(0, MakeTrs(4.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));

    const auto asset = MakeTrs(0.0f, 7.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        SubMeshTransform{}, asset, &override_component, 2, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::AssetDefault);
    ExpectPosition(resolved.transform, 0.0f, 7.0f, 0.0f);
}

TEST(TransformOverrideResolve, AdoptsTheMeshOnFirstResolve) {
    TransformOverride override_component{};
    ASSERT_EQ(override_component.mesh_id, TransformOverride::kUnboundMesh);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        SubMeshTransform{}, SubMeshTransform{}, &override_component, 0, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::AssetDefault);
    EXPECT_EQ(override_component.mesh_id, kMeshId);
}

TEST(TransformOverrideResolve, MeshMismatchIgnoresTheWholeOverride) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId + 1);
    override_component.Set(0, MakeTrs(4.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));

    const auto asset = MakeTrs(0.0f, 7.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        SubMeshTransform{}, asset, &override_component, 0, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::MeshMismatch);
    ExpectPosition(resolved.transform, 0.0f, 7.0f, 0.0f);
}

TEST(TransformOverrideResolve, RangeTooLargeIgnoresTheWholeOverride) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId);
    // Four entries for a three-submesh mesh: applied to the wrong layout, so the
    // whole override is dropped rather than trusted.
    override_component.Set(3, MakeTrs(4.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));

    const auto asset = MakeTrs(0.0f, 7.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        SubMeshTransform{}, asset, &override_component, 3, kMeshId, kSubmeshCount);

    EXPECT_EQ(resolved.status, TransformOverrideStatus::RangeTooLarge);
    ExpectPosition(resolved.transform, 0.0f, 7.0f, 0.0f);
}

TEST(TransformOverrideResolve, ClearedIndexFallsBackToTheAssetTransform) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId);
    override_component.Set(0, MakeTrs(4.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));
    ASSERT_TRUE(override_component.Has(0));

    override_component.Clear(0);
    EXPECT_FALSE(override_component.Has(0));

    const auto asset = MakeTrs(0.0f, 7.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    const EffectiveSubmeshTransform resolved = ResolveEffectiveSubmeshTransform(
        SubMeshTransform{}, asset, &override_component, 0, kMeshId, kSubmeshCount);
    EXPECT_EQ(resolved.status, TransformOverrideStatus::AssetDefault);
    ExpectPosition(resolved.transform, 0.0f, 7.0f, 0.0f);
}

TEST(TransformOverrideComponent, ReportsEachSignatureOnce) {
    TransformOverride override_component{};
    EXPECT_TRUE(override_component.ShouldReport(7));
    EXPECT_FALSE(override_component.ShouldReport(7));
    EXPECT_TRUE(override_component.ShouldReport(8));
}

TEST(TransformOverrideComponent, ClearAllResetsEntriesAndDiagnostics) {
    TransformOverride override_component{};
    override_component.Bind(kMeshId);
    override_component.Set(0, MakeTrs(1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f));
    static_cast<void>(override_component.ShouldReport(1));

    override_component.ClearAll();

    EXPECT_TRUE(override_component.submeshes.empty());
    EXPECT_FALSE(override_component.Has(0));
    // Diagnostics reset, so a fresh problem is reported again.
    EXPECT_TRUE(override_component.ShouldReport(1));
}

} // namespace
