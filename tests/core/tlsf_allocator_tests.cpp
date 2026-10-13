#include <gtest/gtest.h>

import std;

import VulkanEngine.GpuResources.TlsfAllocator;

namespace {

using VulkanEngine::GpuResources::TlsfAllocator;

constexpr std::uint64_t kInvalid = std::numeric_limits<std::uint64_t>::max();

TEST(TlsfAllocatorTest, BasicAllocateHonorsAlignmentAndTracksExtents) {
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    const std::uint64_t a = allocator.Allocate(10, 64);
    ASSERT_NE(a, kInvalid);
    EXPECT_EQ(a % 64, 0u);

    const std::uint64_t b = allocator.Allocate(10, 64);
    ASSERT_NE(b, kInvalid);
    EXPECT_EQ(b % 64, 0u);
    EXPECT_GE(b, a + 10);

    EXPECT_EQ(allocator.GetLiveAllocationCount(), 2u);
    EXPECT_TRUE(allocator.Free(a, 10));
    EXPECT_TRUE(allocator.Free(b, 10));
    EXPECT_EQ(allocator.GetLiveAllocationCount(), 0u);
}

TEST(TlsfAllocatorTest, AlignmentSlackFallsBackToAnotherBlock) {
    // The bitmap lookup keys off size + alignment - 1 and can come up empty even
    // though an aligned fit exists. The allocator must retry instead of failing,
    // no matter which valid block the retry eventually lands on.
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    const std::uint64_t a = allocator.Allocate(300, 1);
    ASSERT_NE(a, kInvalid);
    const std::uint64_t b = allocator.Allocate(200, 1);
    ASSERT_NE(b, kInvalid);
    ASSERT_EQ(allocator.GetLiveAllocationCount(), 2u);

    // Freeing the first allocation leaves a hole whose byte length covers the
    // 512-aligned request, but whose size-bucketed key does not: only the retry
    // path can return it. The request must still succeed somewhere in the heap.
    ASSERT_TRUE(allocator.Free(a, 300));
    ASSERT_EQ(allocator.GetLiveAllocationCount(), 1u);

    const std::uint64_t big = allocator.Allocate(256, 512);
    ASSERT_NE(big, kInvalid);
    EXPECT_EQ(big % 512, 0u);
    EXPECT_LE(big + 256, 1024u);
    ASSERT_EQ(allocator.GetLiveAllocationCount(), 2u);
}

TEST(TlsfAllocatorTest, FreeCoalescesBothNeighbours) {
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    const std::uint64_t a = allocator.Allocate(100, 1);
    const std::uint64_t b = allocator.Allocate(100, 1);
    const std::uint64_t c = allocator.Allocate(100, 1);
    ASSERT_NE(a, kInvalid);
    ASSERT_NE(b, kInvalid);
    ASSERT_NE(c, kInvalid);

    // Free the middle first, then rip out both ends around it: a single Free of
    // the middle must coalesce with both neighbours into one hole big enough
    // for the whole heap.
    ASSERT_TRUE(allocator.Free(b, 100));
    ASSERT_TRUE(allocator.Free(a, 100));
    ASSERT_TRUE(allocator.Free(c, 100));

    EXPECT_EQ(allocator.GetUsedSize(), 0u);
    EXPECT_EQ(allocator.GetFreeSize(), 1024u);
    EXPECT_EQ(allocator.GetLiveAllocationCount(), 0u);

    const std::uint64_t whole = allocator.Allocate(1024, 1);
    EXPECT_NE(whole, kInvalid);
    EXPECT_EQ(allocator.GetUsedSize(), 1024u);
    EXPECT_EQ(allocator.GetFreeSize(), 0u);
    EXPECT_EQ(allocator.GetLiveAllocationCount(), 1u);
}

TEST(TlsfAllocatorTest, FreeCoalescesThreeOrMoreNeighbours) {
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    const std::uint64_t a = allocator.Allocate(100, 1);
    const std::uint64_t b = allocator.Allocate(100, 1);
    const std::uint64_t c = allocator.Allocate(100, 1);
    const std::uint64_t d = allocator.Allocate(100, 1);
    ASSERT_NE(a, kInvalid);
    ASSERT_NE(b, kInvalid);
    ASSERT_NE(c, kInvalid);
    ASSERT_NE(d, kInvalid);

    // Free b, then c (merges b+c), then a (merges a+b+c): three adjacent holes
    // must end up as one region large enough for a single 300-byte allocation.
    ASSERT_TRUE(allocator.Free(b, 100));
    ASSERT_TRUE(allocator.Free(c, 100));
    ASSERT_TRUE(allocator.Free(a, 100));

    // Only d is still live; the merged region must be handed back whole, not
    // carved into pieces.
    EXPECT_EQ(allocator.GetUsedSize(), 100u);
    EXPECT_EQ(allocator.GetFreeSize(), 1024u - 100u);
    const std::uint64_t merged = allocator.Allocate(300, 1);
    EXPECT_NE(merged, kInvalid);
    EXPECT_EQ(allocator.GetLiveAllocationCount(), 2u);
    EXPECT_EQ(allocator.GetUsedSize(), 100u + 300u);
}

TEST(TlsfAllocatorTest, DoubleFreeIsDetectedAndRejected) {
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    const std::uint64_t a = allocator.Allocate(64, 1);
    ASSERT_NE(a, kInvalid);
    ASSERT_TRUE(allocator.Free(a, 64));
    EXPECT_FALSE(allocator.Free(a, 64));
    EXPECT_TRUE(allocator.HasDebugDoubleFree());
    EXPECT_FALSE(allocator.HasDebugOverlap());
}

TEST(TlsfAllocatorTest, ResetRestoresFullCapacityAndClearsTracking) {
    TlsfAllocator allocator;
    ASSERT_TRUE(allocator.Initialize(1024));

    ASSERT_NE(allocator.Allocate(512, 16), kInvalid);
    ASSERT_EQ(allocator.GetUsedSize(), 512u);

    allocator.Reset();
    EXPECT_EQ(allocator.GetFreeSize(), 1024u);
    EXPECT_EQ(allocator.GetUsedSize(), 0u);
    EXPECT_EQ(allocator.GetLiveAllocationCount(), 0u);

    EXPECT_EQ(allocator.Allocate(1024, 1), 0u);
}

}  // namespace