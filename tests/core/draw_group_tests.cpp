#include <gtest/gtest.h>

import std;

import VulkanEngine.MaterialManager;
import VulkanEngine.TechniqueManager;

namespace {

using VulkanEngine::MaterialManager::BlendMode;
using VulkanEngine::MaterialManager::DeriveRenderState;
using VulkanEngine::MaterialManager::MaterialDesc;
using VulkanEngine::MaterialManager::RenderStateKey;
using VulkanEngine::TechniqueManager::BaseTechnique;
using VulkanEngine::TechniqueManager::TechniqueManager;
namespace TechniquePacking = VulkanEngine::TechniqueManager::TechniquePacking;

// A minimal concrete technique so the manager can be exercised without a
// device. Registration never touches the GPU; it only assigns ids and interns
// the base draw group.
class TestTechnique final : public BaseTechnique {};

// ── Render state ──

TEST(DrawGroupTest, CutoutImpliesAlphaMask) {
    EXPECT_TRUE(DeriveRenderState(MaterialDesc{.blend = BlendMode::Cutout}).alpha_mask);
    EXPECT_FALSE(DeriveRenderState(MaterialDesc{.blend = BlendMode::Opaque}).alpha_mask);
    EXPECT_FALSE(DeriveRenderState(MaterialDesc{.blend = BlendMode::Transparent}).alpha_mask);
}

TEST(DrawGroupTest, RenderStateKeyDistinguishesBlendModes) {
    const MaterialDesc opaque{.blend = BlendMode::Opaque};
    const MaterialDesc transparent{.blend = BlendMode::Transparent};
    EXPECT_NE(RenderStateKey(opaque), RenderStateKey(transparent));
}

TEST(DrawGroupTest, RenderStateKeyDistinguishesDoubleSidedAndDepthWrite) {
    const MaterialDesc base{};
    MaterialDesc double_sided = base;
    double_sided.double_sided = true;
    MaterialDesc no_depth = base;
    no_depth.depth_write = false;
    EXPECT_NE(RenderStateKey(base), RenderStateKey(double_sided));
    EXPECT_NE(RenderStateKey(base), RenderStateKey(no_depth));
    EXPECT_NE(RenderStateKey(double_sided), RenderStateKey(no_depth));
}

TEST(DrawGroupTest, DefaultRenderStateKeyIsZero) {
    // The default descriptor must intern the seeded base group so existing
    // scenes keep group == technique id.
    EXPECT_EQ(RenderStateKey(MaterialDesc{}), 0u);
}

TEST(DrawGroupTest, RenderStateKeyIsStable) {
    const MaterialDesc desc{.blend = BlendMode::Cutout, .double_sided = true, .depth_write = false};
    EXPECT_EQ(RenderStateKey(desc), RenderStateKey(desc));
}

// ── Draw-group interning ──

TEST(DrawGroupTest, BaseGroupMatchesTechniqueId) {
    TechniqueManager mgr;
    const auto id = mgr.Register(std::make_unique<TestTechnique>());
    EXPECT_EQ(mgr.GetDrawGroupCount(), 1u);
    // The base group (variant 0, no render state) carries the technique id, so
    // the flag table and region table stay index-compatible with technique ids.
    EXPECT_EQ(mgr.GetDrawGroupTechnique(id.value), id.value);
    EXPECT_EQ(mgr.GetDrawGroupVariant(id.value), 0u);
}

TEST(DrawGroupTest, InterningIsDenseAndIdempotent) {
    TechniqueManager mgr;
    const auto a = mgr.Register(std::make_unique<TestTechnique>());
    const auto b = mgr.Register(std::make_unique<TestTechnique>());
    // Distinct techniques yield pairwise-distinct ids inside the dense range;
    // which id each technique lands on is not part of the contract.
    EXPECT_NE(a.value, b.value);
    EXPECT_LT(a.value, TechniquePacking::MAX_DRAW_GROUPS);
    EXPECT_LT(b.value, TechniquePacking::MAX_DRAW_GROUPS);
    EXPECT_EQ(mgr.GetDrawGroupCount(), 2u);

    // Interning a new triple claims some fresh dense group: the concrete id is
    // unspecified, only that it is in range, unclaimed, and owned by `a`.
    const std::uint16_t g = mgr.InternDrawGroup(a.value, 0, 0x40u);
    EXPECT_LT(g, TechniquePacking::MAX_DRAW_GROUPS);
    EXPECT_NE(g, a.value);
    EXPECT_NE(g, b.value);
    EXPECT_EQ(mgr.GetDrawGroupCount(), 3u);
    EXPECT_EQ(mgr.GetDrawGroupTechnique(g), a.value);

    // The same triple returns the same group.
    EXPECT_EQ(mgr.InternDrawGroup(a.value, 0, 0x40u), g);
    EXPECT_EQ(mgr.GetDrawGroupCount(), 3u);

    // A different technique's identical state is a different group.
    const std::uint16_t gb = mgr.InternDrawGroup(b.value, 0, 0x40u);
    EXPECT_NE(gb, g);
    EXPECT_LT(gb, TechniquePacking::MAX_DRAW_GROUPS);
    EXPECT_EQ(mgr.GetDrawGroupTechnique(gb), b.value);
}

TEST(DrawGroupTest, EveryInterredGroupFitsTheKeyField) {
    TechniqueManager mgr;
    const auto id = mgr.Register(std::make_unique<TestTechnique>());
    for (std::uint32_t key = 1; key < 64u; ++key) {
        const std::uint16_t g = mgr.InternDrawGroup(id.value, 0, key);
        EXPECT_LT(g, TechniquePacking::MAX_DRAW_GROUPS);
    }
}

TEST(DrawGroupTest, OutOfRangeGroupLookupsAreSafe) {
    TechniqueManager mgr;
    const auto id = mgr.Register(std::make_unique<TestTechnique>());
    (void)id;
    const std::uint16_t bogus = static_cast<std::uint16_t>(TechniquePacking::MAX_DRAW_GROUPS - 1);
    EXPECT_EQ(mgr.GetTechnique(mgr.GetDrawGroupTechnique(bogus)), nullptr);
}

TEST(DrawGroupTest, ShutdownResetsTheInternTable) {
    TechniqueManager mgr;
    (void)mgr.Register(std::make_unique<TestTechnique>());
    (void)mgr.InternDrawGroup(0, 0, 0x10u);
    EXPECT_GT(mgr.GetDrawGroupCount(), 0u);
    mgr.Shutdown();
    EXPECT_EQ(mgr.GetDrawGroupCount(), 0u);
    EXPECT_EQ(mgr.GetTechniqueCount(), 0u);
}

}  // namespace
