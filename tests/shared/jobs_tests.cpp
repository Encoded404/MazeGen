#include <gtest/gtest.h>

import std;

import VulkanShared.Jobs;

namespace {

using VulkanShared::IoExecutor;
using VulkanShared::JobBudget;
using VulkanShared::JobHandle;
using VulkanShared::JobScheduler;
using VulkanShared::Tier;

// A scheduler whose worker count is fixed and small, so ordering assertions do
// not depend on the machine's core count. `workers = 0` is fully
// single-threaded: nothing runs until the test pops it, which makes priority
// selection deterministic.
JobBudget Budget(unsigned int workers, unsigned int starvation_limit = 64) {
    JobBudget budget;
    budget.workers = workers;
    budget.background_starvation_limit = starvation_limit;
    return budget;
}

TEST(JobScheduler, AutoBudgetPicksHardwareMinusOneButNeverZero) {
    JobScheduler scheduler{Budget(JobBudget::kAuto)};
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    EXPECT_EQ(scheduler.WorkerCount(), std::max(1u, hardware - 1u));
}

TEST(JobScheduler, ConfigureResizesThePool) {
    JobScheduler scheduler{Budget(2)};
    EXPECT_EQ(scheduler.WorkerCount(), 2u);
    scheduler.Configure(Budget(5));
    EXPECT_EQ(scheduler.WorkerCount(), 5u);
    scheduler.Configure(Budget(0));
    EXPECT_EQ(scheduler.WorkerCount(), 0u);
}

TEST(JobScheduler, DependencyJobQueuesWhenLastDependencyIsReleased) {
    JobScheduler scheduler{Budget(2)};
    std::atomic<int> ran{0};

    auto job = scheduler.CreateJob(
        JobScheduler::JobDesc{Tier::Compute, "dependent", [&] { ran.fetch_add(1); }}, 1);
    scheduler.QueueJob(job);

    // One dependency outstanding: queued, but not runnable.
    EXPECT_EQ(ran.load(), 0);
    EXPECT_FALSE(job.IsDone());

    scheduler.RemoveDependency(job);
    scheduler.Wait(job);

    EXPECT_EQ(ran.load(), 1);
    EXPECT_TRUE(job.IsDone());
    scheduler.FreeJob(job);
}

TEST(JobScheduler, DependencyReleaseBeforeQueueStillRunsTheJob) {
    JobScheduler scheduler{Budget(2)};
    std::atomic<int> ran{0};

    auto job = scheduler.CreateJob(
        JobScheduler::JobDesc{Tier::Compute, "late-queue", [&] { ran.fetch_add(1); }}, 1);
    // Release first: the count reaches zero, but the job has not been queued.
    scheduler.RemoveDependency(job);
    EXPECT_EQ(ran.load(), 0);

    scheduler.QueueJob(job);
    scheduler.Wait(job);

    EXPECT_EQ(ran.load(), 1);
    scheduler.FreeJob(job);
}

TEST(JobScheduler, WaitExecutesQueuedWorkOnTheCallingThread) {
    // No workers at all: `Wait` must run the job itself rather than block.
    JobScheduler scheduler{Budget(0)};
    std::atomic<int> ran{0};
    const std::thread::id caller = std::this_thread::get_id();
    std::thread::id runner{};

    auto job = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "inline", [&] {
        ran.fetch_add(1);
        runner = std::this_thread::get_id();
    }}, 0);
    scheduler.QueueJob(job);
    scheduler.Wait(job);

    EXPECT_EQ(ran.load(), 1);
    EXPECT_EQ(runner, caller);
    scheduler.FreeJob(job);
}

TEST(JobScheduler, WaitNeverDeadlocksWithASingleWorker) {
    JobScheduler scheduler{Budget(1)};
    std::atomic<int> ran{0};

    std::vector<JobHandle> jobs;
    for (int i = 0; i < 16; ++i) {
        jobs.push_back(scheduler.CreateJob(
            JobScheduler::JobDesc{Tier::Compute, "burst", [&] { ran.fetch_add(1); }}, 0));
    }
    for (JobHandle& job : jobs) scheduler.QueueJob(job);
    for (JobHandle& job : jobs) scheduler.Wait(job);

    EXPECT_EQ(ran.load(), 16);
    for (JobHandle& job : jobs) scheduler.FreeJob(job);
}

TEST(JobScheduler, DependencyChainRunsInOrder) {
    JobScheduler scheduler{Budget(3)};
    std::vector<int> order;
    std::mutex mutex;

    auto second = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "second", [&] {
        std::lock_guard lock(mutex);
        order.push_back(2);
    }}, 1);

    // `first` has no dependencies and releases `second` when it finishes.
    auto first = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "first", [&, second] {
        {
            std::lock_guard lock(mutex);
            order.push_back(1);
        }
        scheduler.RemoveDependency(second);
    }}, 0);

    scheduler.QueueJob(first);
    scheduler.QueueJob(second);
    scheduler.Wait(second);
    scheduler.Wait(first);

    const std::vector<int> expected{1, 2};
    EXPECT_EQ(order, expected);
    scheduler.FreeJob(first);
    scheduler.FreeJob(second);
}

TEST(JobScheduler, BarrierWaitsForEveryAddedJob) {
    JobScheduler scheduler{Budget(3)};
    auto* barrier = scheduler.CreateBarrier();
    std::atomic<int> ran{0};

    std::vector<JobHandle> jobs;
    for (int i = 0; i < 8; ++i) {
        jobs.push_back(scheduler.CreateJob(
            JobScheduler::JobDesc{Tier::Compute, "barrier", [&] { ran.fetch_add(1); }}, 0));
    }
    barrier->AddJobs(jobs.data(), jobs.size());
    for (JobHandle& job : jobs) scheduler.QueueJob(job);

    scheduler.WaitForJobs(barrier);
    EXPECT_EQ(ran.load(), 8);
    EXPECT_EQ(barrier->Pending(), 0u);

    for (JobHandle& job : jobs) scheduler.FreeJob(job);
    EXPECT_NO_THROW(scheduler.DestroyBarrier(barrier));
}

TEST(JobScheduler, BarrierAcceptsJobsAddedWhileItIsBeingWaitedOn) {
    JobScheduler scheduler{Budget(2)};
    auto* barrier = scheduler.CreateBarrier();
    std::atomic<int> ran{0};

    auto child = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "child", [&] {
        ran.fetch_add(1);
    }}, 0);

    // The spawner adds its child to the barrier while the barrier is being
    // waited on — the pattern Jolt's solver uses for spawned child jobs.
    auto spawner = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "spawner", [&] {
        barrier->AddJob(child);
        scheduler.QueueJob(child);
        ran.fetch_add(1);
    }}, 0);

    barrier->AddJob(spawner);
    scheduler.QueueJob(spawner);
    scheduler.WaitForJobs(barrier);

    EXPECT_EQ(ran.load(), 2); // spawner + child
    EXPECT_EQ(barrier->Pending(), 0u);

    scheduler.FreeJob(child);
    scheduler.FreeJob(spawner);
    scheduler.DestroyBarrier(barrier);
}

TEST(JobScheduler, DestroyingANonEmptyBarrierIsAnError) {
    JobScheduler scheduler{Budget(0)};
    auto* barrier = scheduler.CreateBarrier();
    auto job = scheduler.CreateJob(
        JobScheduler::JobDesc{Tier::Compute, "pending", [] {}}, 0);
    barrier->AddJob(job);
    EXPECT_THROW(scheduler.DestroyBarrier(barrier), std::logic_error);

    // Drain it so the barrier can go back to the pool.
    scheduler.QueueJob(job);
    scheduler.WaitForJobs(barrier);
    scheduler.FreeJob(job);
    EXPECT_NO_THROW(scheduler.DestroyBarrier(barrier));
}

TEST(JobScheduler, RealtimeWorkIsServedBeforeQueuedComputeWork) {
    // Single-threaded: nothing runs until the test pops it, so the choice
    // between two ready tiers is observable rather than a scheduling race.
    JobScheduler scheduler{Budget(0)};
    std::vector<std::string> order;

    scheduler.Submit(Tier::Compute, "compute", [&] { order.push_back("compute"); });
    scheduler.Submit(Tier::Realtime, "realtime", [&] { order.push_back("realtime"); });

    ASSERT_TRUE(scheduler.TryExecuteOne());
    EXPECT_EQ(order, std::vector<std::string>{"realtime"});
    ASSERT_TRUE(scheduler.TryExecuteOne());
    EXPECT_EQ(order, (std::vector<std::string>{"realtime", "compute"}));
    EXPECT_FALSE(scheduler.TryExecuteOne());
}

TEST(JobScheduler, BackgroundWorkIsStarvedButNotForever) {
    constexpr unsigned int kLimit = 3;
    JobScheduler scheduler{Budget(0, kLimit)};
    std::vector<std::string> order;

    scheduler.Submit(Tier::Background, "background", [&] { order.push_back("background"); });
    for (int i = 0; i < 8; ++i) {
        scheduler.Submit(Tier::Compute, "compute", [&] { order.push_back("compute"); });
    }
    while (scheduler.TryExecuteOne()) {}

    const auto position = std::ranges::find(order, "background");
    ASSERT_NE(position, order.end()) << "background work was starved for the whole run";
    EXPECT_LE(std::distance(order.begin(), position), static_cast<std::ptrdiff_t>(kLimit));
}

TEST(JobScheduler, ParallelForVisitsEveryIndexExactlyOnce) {
    JobScheduler scheduler{Budget(3)};
    constexpr std::size_t kCount = 2000;
    std::vector<std::atomic<int>> hits(kCount);

    scheduler.ParallelFor(kCount, [&](const std::size_t index) { hits[index].fetch_add(1); });

    for (std::size_t i = 0; i < kCount; ++i) {
        ASSERT_EQ(hits[i].load(), 1) << "index " << i;
    }
}

TEST(JobScheduler, ParallelForIsInlineAndOrderedWhenSingleThreaded) {
    JobScheduler scheduler{Budget(0)};
    const std::thread::id caller = std::this_thread::get_id();
    std::vector<std::size_t> visited;
    std::vector<std::thread::id> threads;

    scheduler.ParallelFor(5, [&](const std::size_t index) {
        visited.push_back(index);
        threads.push_back(std::this_thread::get_id());
    });

    const std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    EXPECT_EQ(visited, expected);
    for (const std::thread::id id : threads) EXPECT_EQ(id, caller);
}

TEST(JobScheduler, NestedParallelForFromAJobDoesNotDeadlock) {
    // One worker: the outer job occupies the only worker and then waits on its
    // own nested fan-out, which can only make progress because the wait
    // participates.
    JobScheduler scheduler{Budget(1)};
    std::atomic<int> inner{0};

    scheduler.Submit(Tier::Compute, "outer", [&] {
        scheduler.ParallelFor(32, [&](std::size_t) { inner.fetch_add(1); });
    });
    scheduler.WaitForIdle();

    EXPECT_EQ(inner.load(), 32);
}

TEST(JobScheduler, ParallelForRethrowsTheFirstFailure) {
    JobScheduler scheduler{Budget(3)};
    std::atomic<int> visited{0};
    EXPECT_THROW(
        scheduler.ParallelFor(64, [&](const std::size_t index) {
            visited.fetch_add(1);
            if (index == 7) throw std::runtime_error("boom");
        }),
        std::runtime_error);
    // Every index is still attempted: one failure does not cancel the rest.
    EXPECT_EQ(visited.load(), 64);
}

TEST(JobScheduler, WaitRethrowsAJobFailure) {
    JobScheduler scheduler{Budget(2)};
    auto job = scheduler.CreateJob(JobScheduler::JobDesc{Tier::Compute, "throwing", [] {
        throw std::runtime_error("job failed");
    }}, 0);
    scheduler.QueueJob(job);
    EXPECT_THROW(scheduler.Wait(job), std::runtime_error);
    EXPECT_TRUE(job.IsDone());
    scheduler.FreeJob(job);
}

TEST(IoExecutor, RunsPostedWork) {
    IoExecutor io(2, 8);
    std::atomic<int> ran{0};
    std::promise<void> done;
    auto future = done.get_future();
    for (int i = 0; i < 4; ++i) {
        io.Post([&] { ran.fetch_add(1); });
    }
    io.Post([&] { done.set_value(); });
    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(ran.load(), 4);
}

TEST(IoExecutor, AppliesBackpressureInsteadOfGrowingWithoutBound) {
    IoExecutor io(1, 1); // one worker, one queued slot

    std::mutex mutex;
    std::condition_variable gate;
    bool release = false;
    std::atomic<bool> worker_busy{false};

    // Occupy the worker so the queue can be filled deterministically.
    io.Post([&] {
        worker_busy.store(true);
        std::unique_lock lock(mutex);
        gate.wait(lock, [&] { return release; });
    });
    while (!worker_busy.load()) std::this_thread::yield();

    EXPECT_TRUE(io.TryPost([] {}));   // fills the single queued slot
    EXPECT_FALSE(io.TryPost([] {}));  // full: refused, not dropped silently
    EXPECT_EQ(io.QueueDepth(), 1u);

    // A blocking Post waits for room rather than spawning a thread.
    auto third_done = std::make_shared<std::promise<void>>();
    auto third_future = third_done->get_future();
    std::thread third([&] { io.Post([third_done] { third_done->set_value(); }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(third_future.wait_for(std::chrono::milliseconds(0)),
              std::future_status::timeout);

    {
        std::lock_guard lock(mutex);
        release = true;
    }
    gate.notify_all();
    third.join();

    ASSERT_EQ(third_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(io.QueueDepth(), 0u);
}

TEST(IoExecutor, GlobalLaneIsUsable) {
    std::promise<void> done;
    auto future = done.get_future();
    IoExecutor::Global().Post([&] { done.set_value(); });
    EXPECT_EQ(future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
}

} // namespace
