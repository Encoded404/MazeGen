#include <gtest/gtest.h>

import std;

import VulkanEngine.SceneRenderer;
import VulkanEngine.TechniqueManager.BaseTechnique;

namespace {

using VulkanEngine::SceneRenderer::SceneRenderer;
namespace TechniquePacking = VulkanEngine::TechniqueManager::TechniquePacking;

// The draw-key tables must span the full technique bit width. A packed key is
// the low TECHNIQUE_BITS of a technique_material, so every representable key is
// below the table capacity; sizing to 256 would silently drop or OOB-index a
// key at or above 256.

TEST(DrawKeyBoundTest, DrawGroupCapacitySpansTheTechniqueBits) {
    EXPECT_EQ(SceneRenderer::MAX_DRAW_GROUPS, 1u << TechniquePacking::TECHNIQUE_BITS);
}

TEST(DrawKeyBoundTest, EveryPackedKeyFitsTheTable) {
    constexpr std::uint32_t capacity = SceneRenderer::MAX_DRAW_GROUPS;
    // The largest key the packing can produce, at the maximum material id.
    const std::uint32_t max_material = (1u << TechniquePacking::MATERIAL_BITS) - 1u;
    const std::uint32_t packed = TechniquePacking::Pack(max_material, capacity - 1u);
    EXPECT_EQ(TechniquePacking::UnpackTechnique(packed), capacity - 1u);
    EXPECT_LT(TechniquePacking::UnpackTechnique(packed), capacity);

    // The mask can never emit a key at or above the capacity for any technique
    // id, including ones that overflow the field.
    EXPECT_LT(TechniquePacking::UnpackTechnique(
                  TechniquePacking::Pack(0u, capacity + 7u)),
              capacity);

    // Regression for the old hardcoded 256: a key of 256 must round-trip and
    // fit the widened table.
    const std::uint32_t key256 = TechniquePacking::UnpackTechnique(
        TechniquePacking::Pack(3u, 256u));
    EXPECT_EQ(key256, 256u);
    EXPECT_LT(key256, capacity);
}

}  // namespace
