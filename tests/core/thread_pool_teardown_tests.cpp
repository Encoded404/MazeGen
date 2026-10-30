// Worker-pool shutdown latency (regression: EngineContext teardown stalls).
//
// A DynamicThreadPool worker parks in a timed wait for the pool's idle timeout
// (30 s by default) and only a stop notification ends it early. When the
// destructor published that notification outside the mutex the worker waits
// under, it could land between the worker's stop check and its park and be
// lost, so the join in the destructor paid the whole idle timeout. In practice
// that surfaced as FrameScheduleTest cases randomly reporting 30000 ms while
// their bodies do microseconds of work — one destroyed EngineContext per case,
// each owning a FileLoader pool it never used.
//
// These tests pin teardown to a budget far below the timeout, so the race fails
// here by name instead of as a mysteriously slow test somewhere else.

#include <gtest/gtest.h>

import std;

import FileLoader.DynamicThreadPool;

namespace {

using namespace FileLoader;
using Clock = std::chrono::steady_clock;

// A correct teardown is a thread join: tens of microseconds. The bug pays the
// entire 30 s idle timeout, so a one-second ceiling separates the two by orders
// of magnitude while staying indifferent to machine speed and CI load.
constexpr std::chrono::milliseconds kTeardownBudget{1000};

// The idle timeout these pools use. Short enough that a regression is obvious
// against kTeardownBudget, long enough to never expire inside the test.
constexpr std::chrono::milliseconds kIdleTimeout{30'000};

} // namespace

// No work was ever posted, so every worker sits in the idle wait when the pool
// goes out of scope; destroying it must not wait for that wait to time out.
// Repeated because the failure is a race: one iteration can get lucky.
TEST(DynamicThreadPoolTeardownTest, DestroyingAnIdlePoolDoesNotWaitForTheIdleTimeout) {
    constexpr int kIterations = 50;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        const auto start = Clock::now();
        {
            DynamicThreadPool pool(DynamicThreadPool::Config{
                .min_threads = 1,
                .max_threads = 2,
                .idle_timeout = kIdleTimeout,
            });
        }
        const auto elapsed = Clock::now() - start;
        ASSERT_LT(elapsed, kTeardownBudget)
            << "teardown took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            << " ms on iteration " << iteration << "; the pool's idle timeout is "
            << kIdleTimeout.count() << " ms, so the shutdown wakeup was lost";
    }
}

// The same pool with every worker idle after having run a task. A worker that
// has just finished a task re-enters the idle wait, which is the state right
// before the destructor's notification, so this covers the wakeup from the
// other side of the loop.
TEST(DynamicThreadPoolTeardownTest, DestroyingAPoolThatJustRanWorkDoesNotWaitForTheIdleTimeout) {
    constexpr int kIterations = 20;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        const auto start = Clock::now();
        {
            DynamicThreadPool pool(DynamicThreadPool::Config{
                .min_threads = 1,
                .max_threads = 2,
                .idle_timeout = kIdleTimeout,
            });
            auto done = pool.Enqueue([] { return 7; });
            ASSERT_EQ(done.get(), 7) << "iteration " << iteration;
        }
        const auto elapsed = Clock::now() - start;
        ASSERT_LT(elapsed, kTeardownBudget)
            << "teardown took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            << " ms on iteration " << iteration << "; the pool's idle timeout is "
            << kIdleTimeout.count() << " ms, so the shutdown wakeup was lost";
    }
}
