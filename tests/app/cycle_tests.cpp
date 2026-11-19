#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

import MazeGenerator.Cycle;

namespace {

using MazeGenerator::CycleConfig;
using MazeGenerator::MazeCycle;

CycleConfig EnabledConfig(std::uint32_t base_seed, float dwell_seconds) {
    return CycleConfig{
        .enabled = true,
        .base_seed = base_seed,
        .dwell_seconds = dwell_seconds,
    };
}

} // namespace

TEST(MazeCycleTest, StartsOnTheBaseSeedWithoutHavingCompletedACycle) {
    MazeCycle cycle{EnabledConfig(1234, 1.5f)};

    EXPECT_TRUE(cycle.Enabled());
    EXPECT_EQ(cycle.Seed(), 1234U);
    EXPECT_EQ(cycle.CompletedCycles(), 0U);
    EXPECT_FALSE(cycle.Dwelling());
}

TEST(MazeCycleTest, AdvancesOnceTheDwellElapses) {
    MazeCycle cycle{EnabledConfig(10, 1.5f)};
    cycle.OnSolveFinished();

    EXPECT_TRUE(cycle.Dwelling());
    EXPECT_FALSE(cycle.Advance(0.9f, false));
    EXPECT_FALSE(cycle.Advance(0.4f, false));
    EXPECT_TRUE(cycle.Advance(0.3f, false));

    EXPECT_EQ(cycle.CompletedCycles(), 1U);
    EXPECT_EQ(cycle.Seed(), 11U);
    EXPECT_FALSE(cycle.Dwelling());
    // The transition fires exactly once.
    EXPECT_FALSE(cycle.Advance(10.0f, false));
}

TEST(MazeCycleTest, DoesNotAdvanceWhileTheSolverRuns) {
    MazeCycle cycle{EnabledConfig(0, 1.5f)};
    cycle.OnSolveFinished();

    EXPECT_FALSE(cycle.Advance(10.0f, true));
    EXPECT_EQ(cycle.CompletedCycles(), 0U);
    EXPECT_TRUE(cycle.Advance(1.5f, false));
}

TEST(MazeCycleTest, RepeatedFinishSignalsDoNotRestartTheTimer) {
    MazeCycle cycle{EnabledConfig(0, 1.5f)};
    cycle.OnSolveFinished();

    EXPECT_FALSE(cycle.Advance(1.0f, false));
    // The app observes the terminal status every frame; that must not starve the dwell.
    cycle.OnSolveFinished();
    EXPECT_TRUE(cycle.Advance(0.6f, false));
}

TEST(MazeCycleTest, StartingASolveCancelsTheDwell) {
    MazeCycle cycle{EnabledConfig(0, 1.5f)};
    cycle.OnSolveFinished();
    EXPECT_FALSE(cycle.Advance(1.0f, false));

    cycle.OnSolveStarted();
    EXPECT_FALSE(cycle.Dwelling());
    EXPECT_FALSE(cycle.Advance(10.0f, false));
    EXPECT_EQ(cycle.CompletedCycles(), 0U);
}

TEST(MazeCycleTest, ZeroDwellAdvancesOnTheNextFrame) {
    MazeCycle cycle{EnabledConfig(0, 0.0f)};
    cycle.OnSolveFinished();

    EXPECT_TRUE(cycle.Advance(0.0f, false));
}

TEST(MazeCycleTest, NegativeDeltaDoesNotReverseTheTimer) {
    MazeCycle cycle{EnabledConfig(0, 1.5f)};
    cycle.OnSolveFinished();

    EXPECT_FALSE(cycle.Advance(-5.0f, false));
    EXPECT_TRUE(cycle.Advance(1.5f, false));
}

TEST(MazeCycleTest, SeedsAdvanceWithCompletedCycles) {
    MazeCycle cycle{EnabledConfig(100, 0.0f)};

    for (std::uint32_t expected = 101; expected <= 103; ++expected) {
        cycle.OnSolveFinished();
        ASSERT_TRUE(cycle.Advance(0.0f, false));
        EXPECT_EQ(cycle.Seed(), expected);
    }
    EXPECT_EQ(cycle.CompletedCycles(), 3U);
}

TEST(MazeCycleTest, SeedWrapsAfterTheLargestSeed) {
    MazeCycle cycle{EnabledConfig(std::numeric_limits<std::uint32_t>::max(), 0.0f)};
    cycle.OnSolveFinished();

    ASSERT_TRUE(cycle.Advance(0.0f, false));
    EXPECT_EQ(cycle.Seed(), 0U);
}

TEST(MazeCycleTest, DisabledCycleNeverAdvances) {
    MazeCycle cycle{CycleConfig{
        .enabled = false,
        .base_seed = 7,
        .dwell_seconds = 0.0f,
    }};
    cycle.OnSolveFinished();

    EXPECT_FALSE(cycle.Enabled());
    EXPECT_FALSE(cycle.Advance(100.0f, false));
    EXPECT_EQ(cycle.Seed(), 7U);
    EXPECT_EQ(cycle.CompletedCycles(), 0U);
}

TEST(MazeCycleTest, DisableStopsAnInFlightCycle) {
    MazeCycle cycle{EnabledConfig(1, 0.0f)};
    cycle.OnSolveFinished();
    cycle.Disable();

    EXPECT_FALSE(cycle.Enabled());
    EXPECT_FALSE(cycle.Advance(0.0f, false));
    EXPECT_FALSE(cycle.Dwelling());
}
