// Device-free FrameProfiler tests.
//
// The profiler is pure CPU state: stage attribution, interval quantiles, present
// cadence, summary cadence, and the ECS system breakdown. Ported from the
// SilhouetteGame fork, adapted to V5's single-window API and Runtime namespace.

#include <gtest/gtest.h>

import std;

import test_logging;
import Runtime.FrameProfiler;
import VulkanShared.SystemProfile;

namespace {

using Runtime::FrameProfileConfig;
using Runtime::FrameProfiler;
using Runtime::FrameStage;
using Runtime::FrameStageName;
using Runtime::IntervalStats;
using VulkanShared::SystemProfileKind;

[[maybe_unused]] const bool kLoggerInstalled = [] {
    TestLogging::InstallPerTestFileLogger();
    return true;
}();

// A profiler with a tiny cadence so a handful of synthetic iterations exercises
// every path without waiting on a clock.
FrameProfiler MakeProfiler(const double long_frame_ms = 20.0,
                           const std::uint64_t log_every = 1000) {
    FrameProfileConfig config{};
    config.long_frame_ms = long_frame_ms;
    config.log_every = log_every;
    FrameProfiler profiler{config};
    return profiler;
}

// Sleeps `ms` so a synthetic iteration produces a controllable interval.
void SleepMs(const int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

} // namespace

TEST(IntervalStatsTest, EmptyStatsAreZeroRatherThanGarbage) {
    IntervalStats stats{};
    EXPECT_TRUE(stats.Empty());
    EXPECT_EQ(stats.Count(), 0u);
    EXPECT_DOUBLE_EQ(stats.MeanMs(), 0.0);
    EXPECT_DOUBLE_EQ(stats.MaxMs(), 0.0);
    EXPECT_DOUBLE_EQ(stats.QuantileMs(0.5), 0.0);
    EXPECT_EQ(stats.Describe(), "no samples");
}

TEST(IntervalStatsTest, QuantilesAreExactAndNotBucketEdges) {
    IntervalStats stats{};
    for (int i = 1; i <= 100; ++i) {
        stats.Add(static_cast<double>(i));
    }
    EXPECT_EQ(stats.Count(), 100u);
    EXPECT_DOUBLE_EQ(stats.MinMs(), 1.0);
    EXPECT_DOUBLE_EQ(stats.MaxMs(), 100.0);
    // Nearest-rank: the p50 of 1..100 is the 50th value.
    EXPECT_DOUBLE_EQ(stats.QuantileMs(0.50), 50.0);
    EXPECT_DOUBLE_EQ(stats.QuantileMs(0.90), 90.0);
    EXPECT_DOUBLE_EQ(stats.QuantileMs(1.00), 100.0);
}

// The whole reason the statistics keep every sample: a cadence of "mostly 31 ms
// with the occasional 47 ms" is a stutter, and a mean alone hides it.
TEST(IntervalStatsTest, CountAtOrAboveSeparatesStutterFromASteadyCadence) {
    IntervalStats steady{};
    IntervalStats stuttery{};
    for (int i = 0; i < 100; ++i) {
        steady.Add(39.0);
        stuttery.Add(i % 2 == 0 ? 31.0 : 47.0);
    }
    EXPECT_DOUBLE_EQ(steady.MeanMs(), stuttery.MeanMs());
    EXPECT_EQ(steady.CountAtOrAbove(40.0), 0u);
    EXPECT_EQ(stuttery.CountAtOrAbove(40.0), 50u);
    EXPECT_DOUBLE_EQ(stuttery.QuantileMs(0.99), 47.0);
}

TEST(IntervalStatsTest, SamplesBeyondTheCapStillCountTowardsTotals) {
    IntervalStats stats{};
    for (std::size_t i = 0; i < IntervalStats::kMaxSamples + 10; ++i) {
        stats.Add(2.0);
    }
    EXPECT_EQ(stats.Count(), IntervalStats::kMaxSamples + 10);
    EXPECT_DOUBLE_EQ(stats.TotalMs(), 2.0 * static_cast<double>(IntervalStats::kMaxSamples + 10));
    // The overflow is disclosed rather than silently dropping out of the p99.
    EXPECT_NE(stats.Describe().find("not stored"), std::string::npos);
}

TEST(FrameProfilerTest, DisabledByDefaultAndMeasuresNothing) {
    FrameProfiler profiler{};
    EXPECT_FALSE(profiler.Enabled());

    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 5.0);
    profiler.EndIteration();
    profiler.BeginIteration();

    EXPECT_EQ(profiler.Iterations(), 0u);
    EXPECT_EQ(profiler.Stage(FrameStage::AppUpdate).samples, 0u);
}

TEST(FrameProfilerTest, StageTotalsAccumulate) {
    FrameProfiler profiler = MakeProfiler();
    for (int i = 0; i < 4; ++i) {
        profiler.BeginIteration();
        profiler.Add(FrameStage::AppRender, 1.5);
        profiler.Add(FrameStage::AppUpdate, 0.5);
    }

    const auto& render = profiler.Stage(FrameStage::AppRender);
    EXPECT_EQ(render.samples, 4u);
    EXPECT_DOUBLE_EQ(render.total_ms, 6.0);
    EXPECT_DOUBLE_EQ(render.max_ms, 1.5);
    EXPECT_DOUBLE_EQ(profiler.Stage(FrameStage::AppUpdate).total_ms, 2.0);
}

// The first iteration has no predecessor, so it cannot produce an interval; the
// iterations after it each close one.
TEST(FrameProfilerTest, IterationCountLagsTheBeginCallsByOne) {
    FrameProfiler profiler = MakeProfiler();
    for (int i = 0; i < 5; ++i) {
        profiler.BeginIteration();
    }
    EXPECT_EQ(profiler.Iterations(), 4u);
    EXPECT_EQ(profiler.TotalIterations(), 4u);
}

TEST(FrameProfilerTest, QuantilesTrackTheDistribution) {
    FrameProfiler profiler = MakeProfiler();
    profiler.BeginIteration();
    for (int i = 0; i < 9; ++i) {
        SleepMs(2);
        profiler.BeginIteration();
    }
    SleepMs(45);
    profiler.BeginIteration();

    EXPECT_EQ(profiler.Iterations(), 10u);
    EXPECT_GT(profiler.IterationInterval().QuantileMs(0.50), 0.0);
    EXPECT_GE(profiler.MaxIntervalMs(), 45.0);
    EXPECT_LE(profiler.IterationInterval().QuantileMs(0.50),
              profiler.IterationInterval().QuantileMs(0.90));
    EXPECT_LE(profiler.IterationInterval().QuantileMs(0.90), profiler.MaxIntervalMs());
}

// A long frame is charged to whichever single stage dominated it -- the whole
// purpose of the profiler, since a blocking call can dominate without any work
// counter noticing.
TEST(FrameProfilerTest, LongFrameIsBlamedOnItsLargestStage) {
    FrameProfiler profiler = MakeProfiler(20.0);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);
    profiler.Add(FrameStage::Acquire, 80.0);
    profiler.Add(FrameStage::AppRender, 2.0);
    SleepMs(90);
    profiler.BeginIteration();

    ASSERT_EQ(profiler.Iterations(), 1u);
    ASSERT_EQ(profiler.LongFrames(), 1u);
    EXPECT_EQ(profiler.Stage(FrameStage::Acquire).blame_count, 1u);
    EXPECT_DOUBLE_EQ(profiler.Stage(FrameStage::Acquire).blame_total_ms, 80.0);
    EXPECT_EQ(profiler.Stage(FrameStage::AppUpdate).blame_count, 0u);
    EXPECT_EQ(profiler.Stage(FrameStage::AppRender).blame_count, 0u);
}

// Time the loop spent blocked in a call no scope covers is charged to Idle, and
// Idle wins the blame when it outweighs every measured stage. This is the case
// that matters: it names an unwrapped stall instead of silently attributing the
// frame to whatever work happened to be largest.
TEST(FrameProfilerTest, UnaccountedTimeWinsTheBlameWhenItDominates) {
    FrameProfiler profiler = MakeProfiler(20.0);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);
    profiler.Add(FrameStage::AppRender, 2.0);
    SleepMs(60);
    profiler.BeginIteration();

    ASSERT_EQ(profiler.LongFrames(), 1u);
    EXPECT_EQ(profiler.Stage(FrameStage::Idle).blame_count, 1u);
    EXPECT_GT(profiler.Stage(FrameStage::Idle).blame_total_ms, 50.0);
    EXPECT_EQ(profiler.Stage(FrameStage::AppRender).blame_count, 0u);
}

TEST(FrameProfilerTest, ShortFramesAreNotBlamed) {
    FrameProfiler profiler = MakeProfiler(20.0);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);
    profiler.BeginIteration();

    EXPECT_EQ(profiler.LongFrames(), 0u);
    EXPECT_EQ(profiler.Stage(FrameStage::AppUpdate).blame_count, 0u);
}

TEST(FrameProfilerTest, PresentIntervalsAreTracked) {
    FrameProfiler profiler = MakeProfiler();
    // The first present has no predecessor, so it records nothing.
    profiler.NotePresent();
    EXPECT_TRUE(profiler.PresentInterval().Empty());

    SleepMs(5);
    profiler.NotePresent();
    SleepMs(7);
    profiler.NotePresent();

    EXPECT_EQ(profiler.PresentInterval().Count(), 2u);
    EXPECT_GE(profiler.PresentInterval().MaxMs(), 5.0);
    EXPECT_NE(profiler.Summary().find("present interval"), std::string::npos);
}

TEST(FrameProfilerTest, SummaryFiresOnceEveryCadence) {
    FrameProfiler profiler = MakeProfiler(20.0, 3);
    EXPECT_FALSE(profiler.TakeSummaryDue());
    for (int i = 0; i < 4; ++i) {
        profiler.BeginIteration();
    }
    // Four BeginIteration calls close three intervals, so the cadence is met.
    EXPECT_TRUE(profiler.TakeSummaryDue());
    // And only once.
    EXPECT_FALSE(profiler.TakeSummaryDue());
}

// ResetWindow is what keeps consecutive summaries disjoint: the cadence counter
// must survive it, or the next summary never comes.
TEST(FrameProfilerTest, ResetWindowClearsTheReportButKeepsTheCadence) {
    FrameProfiler profiler = MakeProfiler(20.0, 3);
    for (int i = 0; i < 4; ++i) {
        profiler.BeginIteration();
    }
    ASSERT_TRUE(profiler.TakeSummaryDue());
    ASSERT_EQ(profiler.Iterations(), 3u);
    ASSERT_EQ(profiler.TotalIterations(), 3u);

    profiler.ResetWindow();
    EXPECT_EQ(profiler.Iterations(), 0u);
    EXPECT_EQ(profiler.Stage(FrameStage::AppUpdate).samples, 0u);
    EXPECT_EQ(profiler.LongFrames(), 0u);
    // The cadence counter did not reset, so three more intervals fire again.
    EXPECT_EQ(profiler.TotalIterations(), 3u);
    for (int i = 0; i < 3; ++i) {
        profiler.BeginIteration();
    }
    EXPECT_TRUE(profiler.TakeSummaryDue());
    EXPECT_EQ(profiler.Iterations(), 3u);
}

// DiscardNextInterval exists so the summary's own log write is not measured as a
// stutter: the interval that would contain it never closes.
TEST(FrameProfilerTest, DiscardingAnIntervalDropsItFromTheHistogram) {
    FrameProfiler profiler = MakeProfiler(20.0);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);
    profiler.DiscardNextInterval();
    SleepMs(30);
    profiler.BeginIteration();

    EXPECT_EQ(profiler.Iterations(), 0u);
    EXPECT_EQ(profiler.LongFrames(), 0u);
    // The next interval is recorded again.
    profiler.BeginIteration();
    EXPECT_EQ(profiler.Iterations(), 1u);
}

TEST(FrameProfilerTest, ResetClearsEveryAccumulatorAndTheCadence) {
    FrameProfiler profiler = MakeProfiler();
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 3.0);
    profiler.AddSystemTiming(SystemProfileKind::Frame, "game.system", 1.0);
    SleepMs(2);
    profiler.BeginIteration();
    ASSERT_GT(profiler.Iterations(), 0u);

    profiler.Reset();
    EXPECT_EQ(profiler.Iterations(), 0u);
    EXPECT_EQ(profiler.TotalIterations(), 0u);
    EXPECT_EQ(profiler.LongFrames(), 0u);
    EXPECT_DOUBLE_EQ(profiler.MaxIntervalMs(), 0.0);
    EXPECT_EQ(profiler.Stage(FrameStage::AppUpdate).samples, 0u);
    EXPECT_TRUE(profiler.FrameSystems().empty());
    EXPECT_TRUE(profiler.ComponentSystems().empty());
    EXPECT_FALSE(profiler.TakeSummaryDue());
}

TEST(FrameProfilerTest, EveryStageHasAName) {
    for (std::size_t i = 0; i < Runtime::kFrameStageCount; ++i) {
        const auto name = FrameStageName(static_cast<FrameStage>(i));
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, "unknown") << "stage index " << i << " has no name";
    }
}

TEST(FrameProfilerTest, SummaryReportsTheHeadlineNumbers) {
    FrameProfiler profiler = MakeProfiler(20.0);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);
    SleepMs(30);
    profiler.BeginIteration();

    const std::string summary = profiler.Summary();
    EXPECT_NE(summary.find("1 iterations"), std::string::npos);
    EXPECT_NE(summary.find("1 long frames"), std::string::npos);
    EXPECT_NE(summary.find("app-update"), std::string::npos);
    EXPECT_NE(summary.find("unaccounted"), std::string::npos);
}

// ── ECS system breakdown ────────────────────────────────────────────────────

TEST(FrameProfilerTest, SystemTimingsAggregateByKindAndLabel) {
    FrameProfiler profiler = MakeProfiler();
    profiler.AddSystemTiming(SystemProfileKind::Frame, "frame.mesh_process", 1.5);
    profiler.AddSystemTiming(SystemProfileKind::Frame, "frame.mesh_process", 2.5);
    profiler.AddSystemTiming(SystemProfileKind::Frame, "frame.other", 4.0);
    profiler.AddSystemTiming(SystemProfileKind::Component, "physics.simulate", 0.5);
    profiler.AddSystemTiming(SystemProfileKind::Component, "physics.simulate", 1.5);

    ASSERT_EQ(profiler.FrameSystems().size(), 2u);
    const auto& mesh = profiler.FrameSystems().at("frame.mesh_process");
    EXPECT_EQ(mesh.samples, 2u);
    EXPECT_DOUBLE_EQ(mesh.total_ms, 4.0);
    EXPECT_DOUBLE_EQ(mesh.max_ms, 2.5);

    ASSERT_EQ(profiler.ComponentSystems().size(), 1u);
    const auto& physics = profiler.ComponentSystems().at("physics.simulate");
    EXPECT_EQ(physics.samples, 2u);
    EXPECT_DOUBLE_EQ(physics.total_ms, 2.0);
    EXPECT_DOUBLE_EQ(physics.max_ms, 1.5);
}

TEST(FrameProfilerTest, SystemTimingsAreNotRecordedWhileDisabled) {
    FrameProfiler profiler{};
    profiler.AddSystemTiming(SystemProfileKind::Frame, "game.system", 1.0);
    EXPECT_TRUE(profiler.FrameSystems().empty());
}

TEST(FrameProfilerTest, SummaryPrintsTheEcsBreakdownAsASubset) {
    FrameProfiler profiler = MakeProfiler();
    profiler.AddSystemTiming(SystemProfileKind::Frame, "frame.mesh_process", 2.0);
    profiler.AddSystemTiming(SystemProfileKind::Component, "physics.simulate", 1.0);

    const std::string summary = profiler.Summary();
    EXPECT_NE(summary.find("ecs systems"), std::string::npos);
    EXPECT_NE(summary.find("not additional time"), std::string::npos);
    EXPECT_NE(summary.find("frame.mesh_process"), std::string::npos);
    EXPECT_NE(summary.find("physics.simulate"), std::string::npos);
    // Larger total first, regardless of kind.
    EXPECT_LT(summary.find("frame.mesh_process"), summary.find("physics.simulate"));
}

TEST(FrameProfilerTest, ResetWindowClearsSystemsButKeepsTheCadence) {
    FrameProfiler profiler = MakeProfiler(20.0, 3);
    profiler.AddSystemTiming(SystemProfileKind::Frame, "a", 1.0);
    profiler.AddSystemTiming(SystemProfileKind::Component, "b", 1.0);
    for (int i = 0; i < 4; ++i) {
        profiler.BeginIteration();
    }
    ASSERT_TRUE(profiler.TakeSummaryDue());

    profiler.ResetWindow();
    EXPECT_TRUE(profiler.FrameSystems().empty());
    EXPECT_TRUE(profiler.ComponentSystems().empty());
    EXPECT_EQ(profiler.TotalIterations(), 3u);
}

TEST(FrameProfilerTest, HasSamplesCoversAOneIterationRun) {
    FrameProfiler profiler = MakeProfiler(20.0, 600);
    profiler.BeginIteration();
    profiler.Add(FrameStage::AppUpdate, 1.0);

    EXPECT_EQ(profiler.Iterations(), 0u);
    EXPECT_TRUE(profiler.HasSamples());

    profiler.Reset();
    EXPECT_FALSE(profiler.HasSamples());
}

TEST(FrameProfilerTest, SystemProfileKindNamesCoverEveryKind) {
    EXPECT_EQ(VulkanShared::SystemProfileKindName(SystemProfileKind::Frame), "frame");
    EXPECT_EQ(VulkanShared::SystemProfileKindName(SystemProfileKind::Component), "component");
}
