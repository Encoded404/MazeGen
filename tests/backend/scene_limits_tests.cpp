#include <gtest/gtest.h>

import std;

import VulkanEngine.SceneLimits;

namespace {

using namespace VulkanEngine::SceneLimits;

} // namespace

TEST(SceneLimitsTest, EntriesPerBlockMirrorsTheShaderConstant) {
    // The Slang mirror is shaders/scene_block_layout.slang. If this changes, the
    // seven submesh shaders must change with it.
    EXPECT_EQ(kEntriesPerBlock, 256u);
    EXPECT_EQ(kMaxSubmeshesCeiling, kMaxBlocksCeiling * kEntriesPerBlock);
}

TEST(SceneLimitsTest, PlanStorageBufferCountsMatchTheirLayouts) {
    constexpr std::uint32_t blocks = 8;

    // Every count is derived from the plan's named fields, so a wrong
    // StorageBuffersForPlan (or a plan row that moved) fails this instead of
    // agreeing with a copied literal.
    const auto field_derived = [&](const SetPlan& plan) {
        return plan.block_bindings * blocks + plan.single_storage_buffers;
    };

    // submesh-vertex: one block array.
    EXPECT_EQ(StorageBuffersForPlan(kSubmeshVertexPlan, blocks),
              field_derived(kSubmeshVertexPlan));
    // expand: 4 block arrays + 3 singles.
    EXPECT_EQ(StorageBuffersForPlan(kExpandPlan, blocks), field_derived(kExpandPlan));
    // occlusion: 4 block arrays + 4 singles.
    EXPECT_EQ(StorageBuffersForPlan(kOcclusionPlan, blocks), field_derived(kOcclusionPlan));
    // occluder-select: 3 block arrays + 5 singles.
    EXPECT_EQ(StorageBuffersForPlan(kOccluderSelectPlan, blocks),
              field_derived(kOccluderSelectPlan));
    // collect: 1 block array + 5 singles.
    EXPECT_EQ(StorageBuffersForPlan(kCollectPlan, blocks), field_derived(kCollectPlan));
}

TEST(SceneLimitsTest, MaxStorageBuffersPerSetIsTheMaxOfThePlanTable) {
    // MaxStorageBuffersPerSet documents "largest single-set storage-buffer
    // count across all plans", so the pin recomputes that max from
    // kSceneSetPlans: a plan row left out of the max, or a max that drifts
    // from the table, fails here. At any realistic block count the winner is
    // the depth-graphics stage row (submesh-vertex blocks plus the fixed
    // vertex/UV tables).
    constexpr std::uint32_t blocks = 16;
    std::uint32_t worst = 0;
    for (const SetPlan& plan : kSceneSetPlans) {
        worst = std::max(worst, StorageBuffersForPlan(plan, blocks));
    }
    EXPECT_EQ(MaxStorageBuffersPerSet(blocks), worst);
}

TEST(SceneLimitsTest, HugeLimitsLeaveTheCeilingInPlace) {
    const DeviceLimits limits{1u << 20, 1u << 20};
    EXPECT_EQ(MaxBlocksForDevice(limits), kMaxBlocksCeiling);
}

TEST(SceneLimitsTest, PerSetLimitClampsTheBudget) {
    // occlusion needs 4B + 4; with a per-set limit of 4*100 + 4 the largest
    // fitting block count is 100.
    const DeviceLimits limits{4u * 100u + 4u, 1u << 20};
    EXPECT_EQ(MaxBlocksForDevice(limits), 100u);
}

TEST(SceneLimitsTest, PerStageLimitClampsTheBudget) {
    // Same derivation through maxPerStageDescriptorStorageBuffers.
    const DeviceLimits limits{1u << 20, 4u * 50u + 4u};
    EXPECT_EQ(MaxBlocksForDevice(limits), 50u);
}

TEST(SceneLimitsTest, NeverReturnsZeroBlocks) {
    // A device that cannot fit the fixed bindings still gets one block so the
    // renderer can start and report the condition.
    const DeviceLimits limits{0, 0};
    EXPECT_EQ(MaxBlocksForDevice(limits), 1u);
}

TEST(SceneLimitsTest, SmallDeviceStillFitsOneBlock) {
    const DeviceLimits limits{4u + 4u, 4u + 4u};
    EXPECT_EQ(MaxBlocksForDevice(limits), 1u);
}
