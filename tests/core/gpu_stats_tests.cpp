#include <gtest/gtest.h>

import std;

import VulkanEngine.GpuStats;

namespace {

using VulkanEngine::GpuStats::Accumulate;
using VulkanEngine::GpuStats::Counters;
using VulkanEngine::GpuStats::kAvailabilityWord;
using VulkanEngine::GpuStats::kComputeAvailabilityWord;
using VulkanEngine::GpuStats::kComputeQueryWords;
using VulkanEngine::GpuStats::kCounterCount;
using VulkanEngine::GpuStats::kQueryWords;
using VulkanEngine::GpuStats::QueryPoolSize;
using VulkanEngine::GpuStats::QuerySlot;
using VulkanEngine::GpuStats::TryParseComputeQueryResult;
using VulkanEngine::GpuStats::TryParseQueryResult;

TEST(GpuStatsTest, AccumulateSumsEveryCounter) {
    Counters a{};
    a.input_assembly_vertices = 1;
    a.input_assembly_primitives = 2;
    a.vertex_shader_invocations = 3;
    a.clipping_invocations = 4;
    a.clipping_primitives = 5;
    a.fragment_shader_invocations = 6;
    a.compute_shader_invocations = 7;

    Counters b{};
    b.input_assembly_vertices = 10;
    b.compute_shader_invocations = 100;

    const std::array<Counters, 3> runs{a, b, Counters{}};
    const Counters total = Accumulate(runs);
    EXPECT_EQ(total.input_assembly_vertices, 11u);
    EXPECT_EQ(total.input_assembly_primitives, 2u);
    EXPECT_EQ(total.vertex_shader_invocations, 3u);
    EXPECT_EQ(total.clipping_invocations, 4u);
    EXPECT_EQ(total.clipping_primitives, 5u);
    EXPECT_EQ(total.fragment_shader_invocations, 6u);
    EXPECT_EQ(total.compute_shader_invocations, 107u);
}

TEST(GpuStatsTest, AccumulateOfNoRunsIsZero) {
    const Counters total = Accumulate({});
    EXPECT_EQ(total.input_assembly_vertices, 0u);
    EXPECT_EQ(total.compute_shader_invocations, 0u);
}

TEST(GpuStatsTest, QuerySlotsAreUniquePerFifAndRun) {
    constexpr std::uint32_t kFrames = 3;
    constexpr std::uint32_t kSlots = 9;

    // Every (fif, run) pair maps to a distinct pool slot, and the fif wraps.
    std::set<std::uint32_t> seen;
    for (std::uint32_t fif = 0; fif < kFrames; ++fif) {
        for (std::uint32_t run = 0; run < kSlots; ++run) {
            const std::uint32_t slot = QuerySlot(fif, run, kFrames, kSlots);
            EXPECT_TRUE(seen.insert(slot).second) << "slot collision at fif=" << fif << " run=" << run;
            EXPECT_LT(slot, QueryPoolSize(kFrames, kSlots));
        }
    }
    EXPECT_EQ(seen.size(), kFrames * kSlots);

    // Sample 0 (the graphics preamble / single run) is distinct from the compute
    // run slots of the same frame; sharing one query across runs was the bug.
    EXPECT_NE(QuerySlot(0, 0, kFrames, kSlots), QuerySlot(0, 1, kFrames, kSlots));

    // The same run of the next frame reuse of the ring is the same slot.
    EXPECT_EQ(QuerySlot(kFrames, 1, kFrames, kSlots), QuerySlot(0, 1, kFrames, kSlots));
}

TEST(GpuStatsTest, QueryPoolSizeCoversEveryFifAndRun) {
    EXPECT_EQ(QueryPoolSize(1, 1), 1u);
    EXPECT_EQ(QueryPoolSize(3, 9), 27u);
}

TEST(GpuStatsTest, TryParseQueryResultReadsCountersWhenAvailable) {
    std::array<std::uint64_t, kQueryWords> words{};
    for (std::uint32_t i = 0; i < kCounterCount; ++i) {
        words[i] = i + 1;
    }
    words[kAvailabilityWord] = 1; // available

    Counters parsed{};
    ASSERT_TRUE(TryParseQueryResult(words.data(), parsed));
    EXPECT_EQ(parsed.input_assembly_vertices, 1u);
    EXPECT_EQ(parsed.input_assembly_primitives, 2u);
    EXPECT_EQ(parsed.vertex_shader_invocations, 3u);
    EXPECT_EQ(parsed.clipping_invocations, 4u);
    EXPECT_EQ(parsed.clipping_primitives, 5u);
    EXPECT_EQ(parsed.fragment_shader_invocations, 6u);
    EXPECT_EQ(parsed.compute_shader_invocations, 7u);
}

TEST(GpuStatsTest, ComputePoolQueryHasOnlyTheComputeCounter) {
    // A compute-only pool cannot enable graphics statistics, so its result is a
    // single counter plus availability.
    std::array<std::uint64_t, kComputeQueryWords> words{};
    words[0] = 123;
    words[kComputeAvailabilityWord] = 1;
    Counters counters{};
    counters.fragment_shader_invocations = 7;
    ASSERT_TRUE(TryParseComputeQueryResult(words.data(), counters));
    EXPECT_EQ(counters.compute_shader_invocations, 123u);
    EXPECT_EQ(counters.fragment_shader_invocations, 7u) << "graphics counters are not in this pool";

    words[kComputeAvailabilityWord] = 0;
    EXPECT_FALSE(TryParseComputeQueryResult(words.data(), counters));
    EXPECT_FALSE(TryParseComputeQueryResult(nullptr, counters));
}

TEST(GpuStatsTest, TryParseQueryResultRejectsUnavailableAndLeavesOutput) {
    std::array<std::uint64_t, kQueryWords> words{};
    words[0] = 999;
    words[kAvailabilityWord] = 0; // not available

    Counters parsed{};
    parsed.compute_shader_invocations = 42;
    EXPECT_FALSE(TryParseQueryResult(words.data(), parsed));
    EXPECT_EQ(parsed.compute_shader_invocations, 42u) << "unavailable results must not be consumed";
    EXPECT_FALSE(TryParseQueryResult(nullptr, parsed));
}

}  // namespace
