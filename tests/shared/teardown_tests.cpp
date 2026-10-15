#include <gtest/gtest.h>

import std;

import VulkanShared.Teardown;

namespace {

using VulkanShared::RunDetached;
using VulkanShared::TeardownScheduler;

// Order recorder safe to touch from any scheduler worker.
struct SafeOrder {
    std::mutex mutex;
    std::vector<int> order;

    void Push(int v) {
        std::lock_guard lock(mutex);
        order.push_back(v);
    }
};

TEST(TeardownScheduler, RunsAllTasks) {
    TeardownScheduler teardown{};
    std::atomic<int> count{0};
    for (int i = 0; i < 8; ++i) {
        teardown.Add("task" + std::to_string(i), [&] { count.fetch_add(1); });
    }
    teardown.Run();
    EXPECT_EQ(count.load(), 8);
}

TEST(TeardownScheduler, DependencyOrdering) {
    TeardownScheduler teardown{};
    SafeOrder order;

    // b depends on a; c depends on b -> execution must be a, b, c.
    const auto a = teardown.Add("a", [&] { order.Push(1); });
    const auto b = teardown.Add("b", [&] { order.Push(2); }, {a});
    const auto c = teardown.Add("c", [&] { order.Push(3); }, {b});

    (void)c;
    teardown.Run();

    std::lock_guard lock(order.mutex);
    ASSERT_EQ(order.order.size(), 3u);
    EXPECT_EQ(order.order[0], 1);
    EXPECT_EQ(order.order[1], 2);
    EXPECT_EQ(order.order[2], 3);
}

TEST(TeardownScheduler, IndependentTasksCanRunConcurrently) {
    TeardownScheduler teardown{};
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    std::atomic<int> completed{0};

    auto busy = [&] {
        const int now = ++active;
        int expected = max_active.load(std::memory_order_relaxed);
        while (expected < now &&
               !max_active.compare_exchange_weak(expected, now, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        --active;
        ++completed;
    };

    teardown.Add("t1", busy);
    teardown.Add("t2", busy);
    teardown.Add("t3", busy);
    teardown.Run();

    EXPECT_EQ(completed.load(), 3);
    EXPECT_GE(max_active.load(), 2);  // at least two overlapped
}

TEST(TeardownScheduler, ExceptionCapturedAndRethrown) {
    TeardownScheduler teardown{};
    std::atomic<int> ran{0};
    teardown.Add("ok", [&] { ran.fetch_add(1); });
    teardown.Add("boom", [] { throw std::runtime_error("boom"); });
    teardown.Add("also-ok", [&] { ran.fetch_add(1); });

    EXPECT_THROW(teardown.Run(), std::runtime_error);
    // All tasks were attempted even though one threw.
    EXPECT_EQ(ran.load(), 2);
}

TEST(TeardownScheduler, AddAfterRunThrows) {
    TeardownScheduler teardown{};
    teardown.Add("a", [] {});
    teardown.Run();
    EXPECT_THROW(teardown.Add("late", [] {}), std::logic_error);
}

TEST(TeardownScheduler, OutOfRangeDependencyThrows) {
    TeardownScheduler teardown{};
    teardown.Add("a", [] {});
    // Dependencies must refer to already-added tasks.
    EXPECT_THROW(teardown.Add("b", [] {}, {5}), std::out_of_range);
}

TEST(TeardownScheduler, RunIsIdempotent) {
    TeardownScheduler teardown{};
    std::atomic<int> ran{0};
    teardown.Add("a", [&] { ran.fetch_add(1); });
    teardown.Run();
    teardown.Run();  // second call is a no-op
    EXPECT_EQ(ran.load(), 1);
}

TEST(RunDetached, ExecutesWork) {
    std::atomic<bool> done{false};
    RunDetached([&] { done.store(true); });

    // Poll briefly for the detached thread to run.
    for (int i = 0; i < 1000 && !done.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(done.load());
}

TEST(RunDetached, SwallowsExceptions) {
    // Must not terminate or propagate; just reaching here is a pass.
    RunDetached([] { throw std::runtime_error("ignored"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

} // namespace
