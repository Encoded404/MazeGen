#include <gtest/gtest.h>

import std;

import VulkanEngine.GpuResources.TransientAllocator;

namespace {

using namespace VulkanEngine::GpuResources;

TransientRequirements Req(std::string name,
                          std::uint64_t size,
                          std::int32_t first,
                          std::int32_t last,
                          bool aliasable = true,
                          std::uint32_t heap_key = 0,
                          std::uint64_t alignment = 256) {
    return TransientRequirements{
        .name = std::move(name),
        .kind = TransientKind::Image,
        .heap_key = heap_key,
        .size = size,
        .alignment = alignment,
        .first_pass = first,
        .last_pass = last,
        .aliasable = aliasable,
    };
}

const TransientPlacement* Find(const TransientPlacementPlan& plan, std::uint32_t resource, std::uint32_t slot = 0) {
    for (const auto& placement : plan.placements) {
        if (placement.resource_index == resource && placement.fif_slot == slot) {
            return &placement;
        }
    }
    return nullptr;
}

// Invariants that must hold for any placement strategy: the offset honors the
// resource's alignment and the extent fits inside its heap for the FIF slot it
// was laid out for.
void ExpectValidPlacement(const TransientPlacementPlan& plan,
                          const std::vector<TransientRequirements>& requirements,
                          const TransientPlacement& placement) {
    ASSERT_LT(placement.resource_index, requirements.size());
    const TransientRequirements& requirement = requirements[placement.resource_index];
    EXPECT_EQ(placement.offset % requirement.alignment, 0u);
    const std::uint64_t slot_end =
        (static_cast<std::uint64_t>(placement.fif_slot) + 1) * plan.GetHeapCopySize(placement.heap_key);
    EXPECT_LE(placement.offset + placement.size, slot_end);
}

TEST(TransientAllocatorTest, DisjointLifetimesShareAnOffset) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 1),
        Req("b", 256, 2, 3),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    ASSERT_TRUE(plan.valid);
    ASSERT_EQ(plan.placements.size(), 2u);

    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ExpectValidPlacement(plan, requirements, *a);
    ExpectValidPlacement(plan, requirements, *b);
    // Disjoint lifetimes that may alias must land on one shared offset: read
    // both offsets from the plan instead of predicting where the planner puts
    // the shared range.
    EXPECT_EQ(a->offset, b->offset);
    EXPECT_EQ(plan.GetHeapCopySize(0), 256u);
    ASSERT_EQ(plan.aliases.size(), 1u);
    const PlannedAlias& alias = plan.aliases.front();
    // The alias names exactly the two disjoint resources, one taking over the
    // other's range.
    EXPECT_TRUE((alias.aliased_resource == 0u && alias.after_resource == 1u) ||
                (alias.aliased_resource == 1u && alias.after_resource == 0u));
    // The reuse pass sits strictly after the vacated range's last pass, and it
    // is the taker-over's first pass.
    EXPECT_EQ(alias.pass_index, requirements[alias.aliased_resource].first_pass);
    EXPECT_GT(alias.pass_index, requirements[alias.after_resource].last_pass);
}

TEST(TransientAllocatorTest, OverlappingLifetimesDoNotAlias) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 3),
        Req("b", 256, 1, 4),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    ASSERT_TRUE(plan.valid);
    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a->offset, b->offset);
    EXPECT_EQ(plan.GetHeapCopySize(0), 512u);
    EXPECT_TRUE(plan.aliases.empty());
}

TEST(TransientAllocatorTest, NonAliasableResourceIsNotReused) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 1, /*aliasable=*/false),
        Req("b", 256, 2, 3, /*aliasable=*/true),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ExpectValidPlacement(plan, requirements, *a);
    ExpectValidPlacement(plan, requirements, *b);
    // Even though the lifetimes are disjoint, non-aliasable memory keeps its
    // range private: b must not reclaim a's offset.
    EXPECT_NE(a->offset, b->offset);
    EXPECT_EQ(plan.GetHeapCopySize(0), 512u);
    EXPECT_TRUE(plan.aliases.empty());
}

TEST(TransientAllocatorTest, LargerRequirementDoesNotFitSmallerRange) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 1),
        Req("b", 512, 2, 3),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ExpectValidPlacement(plan, requirements, *a);
    ExpectValidPlacement(plan, requirements, *b);
    // The 512-byte requirement cannot be squeezed into the 256-byte range the
    // first resource vacated: the extents stay disjoint and the heap grows to
    // cover both sizes.
    const bool disjoint = a->offset + a->size <= b->offset || b->offset + b->size <= a->offset;
    EXPECT_TRUE(disjoint);
    EXPECT_EQ(plan.GetHeapCopySize(0), 768u);
    EXPECT_TRUE(plan.aliases.empty());
}

TEST(TransientAllocatorTest, DifferentHeapKeysAreIndependent) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 1, true, /*heap_key=*/0),
        Req("b", 256, 0, 1, true, /*heap_key=*/1),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(a->heap_key, 0u);
    EXPECT_EQ(b->heap_key, 1u);
    // Separate heaps own separate address spaces, so raw offsets are not
    // comparable across them; only the per-heap placement invariants are.
    ExpectValidPlacement(plan, requirements, *a);
    ExpectValidPlacement(plan, requirements, *b);
    EXPECT_EQ(plan.GetHeapCopySize(0), 256u);
    EXPECT_EQ(plan.GetHeapCopySize(1), 256u);
}

TEST(TransientAllocatorTest, LayoutIsReplicatedPerFifSlot) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 256, 0, 1),
        Req("b", 256, 2, 3),
    };

    const auto plan = PlanTransientPlacements(requirements, 3);
    ASSERT_EQ(plan.placements.size(), 6u);

    // Copy semantics: every FIF slot replays the identical layout, strided by
    // one heap copy per slot: placement offsets include the slot's base, so the
    // invariant is per resource -- (offset - fif_slot * copy_size) is identical
    // in every slot where the resource lives. (Slot containers themselves
    // legitimately differ: different resources are alive in different slots.)
    std::map<std::uint32_t, std::set<std::uint64_t>> copy_offsets_per_resource;
    for (const auto& placement : plan.placements) {
        const std::uint64_t copy_size = plan.GetHeapCopySize(placement.heap_key);
        ASSERT_GT(copy_size, 0u) << "unknown heap " << placement.heap_key;
        copy_offsets_per_resource[placement.resource_index].insert(
            placement.offset - static_cast<std::uint64_t>(placement.fif_slot) * copy_size);
    }
    ASSERT_EQ(copy_offsets_per_resource.size(), 2u);
    for (const auto& [resource, offsets] : copy_offsets_per_resource) {
        EXPECT_EQ(offsets.size(), 1u) << "resource " << resource
                                      << " changes its position within the FIF copy";
    }
}

TEST(TransientAllocatorTest, FreshPlacementRespectsAlignment) {
    const std::vector<TransientRequirements> requirements{
        Req("a", 100, 0, 3, true, 0, 128),
        Req("b", 100, 1, 4, true, 0, 128),
    };

    const auto plan = PlanTransientPlacements(requirements, 1);
    const auto* a = Find(plan, 0);
    const auto* b = Find(plan, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(a->offset % 128, 0u);
    EXPECT_EQ(b->offset % 128, 0u);
    ExpectValidPlacement(plan, requirements, *a);
    ExpectValidPlacement(plan, requirements, *b);
    // Freshly placed resources with overlapping lifetimes never share an
    // offset, whatever the alignment rounding does to the gaps between them.
    EXPECT_NE(a->offset, b->offset);
}

TEST(TransientAllocatorTest, GranularityPaddingSeparatesLinearAndNonLinear) {
    constexpr std::uint64_t granularity = 512;

    // A non-linear allocation at 0 occupies page 0.
    const std::uint64_t image_offset = 0;
    const std::uint64_t image_size = 100;

    // The next linear allocation must start on a fresh page.
    const std::uint64_t line_offset =
        PadForLinearity(image_offset + image_size, /*previous_is_linear=*/false, /*is_linear=*/true, granularity);
    EXPECT_EQ(line_offset, 512u);
    EXPECT_FALSE(SharesGranularityPage(image_offset, image_size, line_offset, 100, granularity));

    // Two allocations with the same linearity may share a page.
    const std::uint64_t same_linearity = PadForLinearity(0, true, true, granularity);
    EXPECT_EQ(same_linearity, 0u);

    // Crossing a page boundary is detected.
    EXPECT_TRUE(SharesGranularityPage(500, 100, 0, 100, granularity));
}

}  // namespace