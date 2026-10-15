// The engine's single compute scheduler.
//
// One process-wide pool owns `hardware_concurrency - 1` worker threads (or a
// configured count) and every compute-hungry system borrows from it: the ECS
// object-component update, texture decode/transcode, shader compilation, glyph
// rasterization, teardown, and the physics update burst. File/network/device IO
// does NOT come here — that is `IoExecutor`, deliberately a separate, bounded
// pool, because a 500-asset async scene load must not create 500 threads and
// must not compete with simulation work.
//
// The scheduler speaks the job model Jolt needs natively, so the physics bridge
// is a thin adapter rather than an emulation layer built out of packaged tasks:
//
//   * dependency-counted jobs that queue themselves the moment their last
//     dependency is released (`CreateJob` / `RemoveDependency` / `QueueJob`),
//   * waitable barriers that can grow while they are being waited on
//     (`Barrier` / `WaitForJobs`),
//   * and participating waits: `Wait`, `WaitForJobs`, and `ParallelFor` execute
//     queued work on the waiting thread instead of sleeping, so a simulation
//     burst never leaves pool capacity idle behind a blocked main thread.
//
// Priority is explicit rather than accidental. A decode job is `Compute` tier
// and yields to any queued `Realtime` job, which is exactly the guarantee the
// texture uploader was previously buying by refusing to share
// `ThreadPool::Global()` at all. `Background` work (asset cooking, warmups)
// preempts nothing and is served last, with a starvation guard so it cannot be
// postponed forever.
//
// Deterministic *scheduling* is explicitly not a goal: Jolt's simulation is
// deterministic irrespective of job order, and the only observable order that
// needs to be stable (contact/activation replay) is sorted in the facade.

module;

export module VulkanShared.Jobs;

import std;

export namespace VulkanShared {

/// Execution tier. Workers pick the highest non-empty tier first.
enum class Tier : std::uint8_t {
    Realtime = 0,   // simulation bursts: the physics update, per-step sweeps
    Compute = 1,    // ECS updates, decode/transcode, shader compile, glyphs
    Background = 2, // asset cooking, warmups; preempts nothing
    Count = 3,
};

/// Diagnostic spelling of a tier.
[[nodiscard]] constexpr std::string_view TierName(Tier tier) noexcept {
    switch (tier) {
        case Tier::Realtime:   return "realtime";
        case Tier::Compute:    return "compute";
        case Tier::Background: return "background";
        case Tier::Count:      break;
    }
    return "<unknown>";
}

/// Pool sizing policy. One policy for every compute-hungry system.
struct JobBudget {
    /// "Pick a sensible default": `hardware_concurrency - 1`, at least 1.
    static constexpr unsigned int kAuto = std::numeric_limits<unsigned int>::max();

    /// Worker threads. `kAuto` means `hw - 1`; `0` collapses the scheduler to
    /// fully single-threaded, which makes every parallel path run in submission
    /// order on the calling thread — the deterministic debugging mode.
    unsigned int workers = kAuto;

    /// After this many non-Background jobs served while a Background job waits,
    /// one Background job is served. Bounds starvation, not priority.
    unsigned int background_starvation_limit = 64;
};

namespace detail {
struct JobControl;
} // namespace detail

class JobScheduler;

/// Handle to a dependency-counted job.
///
/// The handle keeps the job alive; the job queues itself when its dependency
/// count reaches zero *and* it has been queued. `FreeJob` releases the handle's
/// reference — after that the handle no longer names the job.
class JobHandle {
public:
    JobHandle() noexcept = default;

    [[nodiscard]] bool IsValid() const noexcept;
    /// True once the job function has run to completion (or thrown; see Wait).
    [[nodiscard]] bool IsDone() const noexcept;

private:
    friend class JobScheduler;
    explicit JobHandle(std::shared_ptr<detail::JobControl> control) noexcept;

    std::shared_ptr<detail::JobControl> control_;
};

/// The engine's shared compute scheduler.
///
/// Instances are usable standalone (tests construct their own); `Global()` is
/// the process-wide pool every engine subsystem borrows from.
class JobScheduler {
public:
    using JobFn = std::function<void()>;

    /// A job's identity for diagnostics plus its tier and body.
    struct JobDesc {
        Tier tier = Tier::Compute;
        const char* name = ""; // profiler label; never an identity key
        JobFn fn;
    };

    /// A waitable set of jobs. Jobs may be added while the barrier is being
    /// waited on (Jolt's solver does exactly this as it spawns child jobs).
    class Barrier {
    public:
        Barrier() = default;
        Barrier(const Barrier&) = delete;
        Barrier& operator=(const Barrier&) = delete;

        void AddJob(const JobHandle& job);
        void AddJobs(const JobHandle* jobs, std::size_t count);

        [[nodiscard]] std::size_t Pending() const noexcept;

    private:
        friend class JobScheduler;
        void OnJobFinished() noexcept;
        void Reset() noexcept;

        mutable std::mutex mutex_;
        std::size_t pending_ = 0;
    };

    JobScheduler() : JobScheduler(JobBudget{}) {}
    explicit JobScheduler(JobBudget budget);
    ~JobScheduler();

    JobScheduler(const JobScheduler&) = delete;
    JobScheduler& operator=(const JobScheduler&) = delete;
    JobScheduler(JobScheduler&&) = delete;
    JobScheduler& operator=(JobScheduler&&) = delete;

    /// (Re)size the pool. Intended for startup and tests: existing workers are
    /// stopped and joined, and any queued work is drained first.
    void Configure(JobBudget budget);

    /// The process-wide pool. Every engine subsystem borrows from it; the
    /// budget comes from `GameConfig` via `EngineBootstrap`.
    static JobScheduler& Global();

    // ── Dependency-counted jobs ──
    // A job runs once it has been queued *and* its dependency count reaches
    // zero, whichever happens last. This is Jolt's model, so its bridge is a
    // forwarding adapter rather than a packaged-task emulation.
    [[nodiscard]] JobHandle CreateJob(JobDesc desc, std::uint32_t dependencies = 0);
    /// Decrement a dependency; the last decrement queues the job.
    void RemoveDependency(const JobHandle& job);
    /// Mark the job as runnable; it waits for its dependencies if it has any.
    void QueueJob(const JobHandle& job);
    void QueueJobs(const JobHandle* jobs, std::size_t count);
    /// Release this handle's reference to the job.
    void FreeJob(JobHandle& job);

    // ── Barriers ──
    [[nodiscard]] Barrier* CreateBarrier();
    void DestroyBarrier(Barrier* barrier);
    /// Wait for every job added to `barrier`, executing queued work (including
    /// the barrier's own jobs) on this thread while waiting.
    void WaitForJobs(Barrier* barrier);
    /// Wait for one job, participating the same way.
    void Wait(const JobHandle& job);

    // ── Fire-and-forget and bulk work ──
    /// Enqueue a job body. Exceptions are swallowed: a fire-and-forget job must
    /// not kill a worker thread.
    void Submit(Tier tier, const char* name, JobFn fn);

    /// Run `count` indices on the pool, participating on the calling thread.
    /// Exceptions are rethrown on the calling thread after every index has been
    /// attempted.
    template<typename Fn>
    void ParallelFor(std::size_t count, Fn&& fn, Tier tier = Tier::Compute) {
        if (count == 0) return;
        if (count == 1 || worker_count_ == 0) {
            for (std::size_t i = 0; i < count; ++i) fn(i);
            return;
        }

        auto next = std::make_shared<std::atomic<std::size_t>>(0);
        const std::size_t spawned = std::min<std::size_t>(worker_count_, count);
        auto remaining = std::make_shared<std::atomic<std::size_t>>(spawned + 1);
        auto error_mutex = std::make_shared<std::mutex>();
        auto first_error = std::make_shared<std::exception_ptr>();

        auto run = [next, remaining, count, error_mutex, first_error, &fn] {
            for (;;) {
                const std::size_t index = next->fetch_add(1, std::memory_order_acq_rel);
                if (index >= count) break;
                try {
                    fn(index);
                } catch (...) {
                    const std::scoped_lock lock(*error_mutex);
                    if (!*first_error) *first_error = std::current_exception();
                }
            }
            remaining->fetch_sub(1, std::memory_order_acq_rel);
        };

        for (std::size_t i = 0; i < spawned; ++i) Submit(tier, "parallel-for", run);
        run(); // the calling thread is a participant, never a spectator
        while (remaining->load(std::memory_order_acquire) != 0) {
            if (!TryExecuteOne()) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (*first_error) std::rethrow_exception(*first_error);
    }

    /// Execute one queued job on this thread. Returns false when none is queued.
    /// This is the primitive behind every participating wait.
    [[nodiscard]] bool TryExecuteOne();

    /// Block until every queued job has been executed and no job is running.
    void WaitForIdle();

    [[nodiscard]] unsigned int WorkerCount() const noexcept { return worker_count_; }
    [[nodiscard]] std::size_t QueuedCount() const noexcept;

private:
    struct WorkItem {
        Tier tier = Tier::Compute;
        const char* name = "";
        JobFn fn;
        std::uint64_t sequence = 0;
    };

    [[nodiscard]] static std::uint32_t ResolveWorkers(unsigned int requested) noexcept;

    void StartWorkers();
    void StopWorkers();
    void WorkerMain();
    static void MaybeQueue(const std::shared_ptr<detail::JobControl>& control);
    void SubmitLocked(Tier tier, const char* name, JobFn fn);
    [[nodiscard]] bool HasWorkLocked() const noexcept;
    /// Pop the next item by tier priority (under `mutex_`), counting it in
    /// flight so `WaitForIdle` cannot observe an idle pool with work in hand.
    [[nodiscard]] bool PopLocked(WorkItem& out);
    void RunItem(WorkItem& item) noexcept;

    std::array<std::deque<WorkItem>, static_cast<std::size_t>(Tier::Count)> queues_;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::vector<std::jthread> workers_;
    unsigned int worker_count_ = 0;
    unsigned int background_starvation_limit_ = 64;
    unsigned int served_since_background_ = 0;
    std::uint64_t next_sequence_ = 1;
    std::atomic<std::size_t> in_flight_{0};
    bool stopping_ = false;

    std::mutex barrier_pool_mutex_;
    std::vector<std::unique_ptr<Barrier>> barrier_free_list_;
};

/// A bounded IO executor: file loads, network fetches, device reads.
///
/// Deliberately NOT the compute pool. It exists to bound
/// one-thread-per-async-load and to keep IO latency out of the compute budget:
/// `Post` applies backpressure (it blocks) instead of growing without limit.
/// File watchers, camera capture, and networking keep their own native threads
/// and never touch this either.
class IoExecutor {
public:
    using IoFn = std::function<void()>;

    IoExecutor() : IoExecutor(4, 128) {}
    explicit IoExecutor(std::size_t workers, std::size_t capacity = 128);
    ~IoExecutor();

    IoExecutor(const IoExecutor&) = delete;
    IoExecutor& operator=(const IoExecutor&) = delete;

    /// Enqueue work, blocking while the bounded queue is full.
    void Post(IoFn fn);
    /// Enqueue work; returns false (and drops nothing) when the queue is full.
    [[nodiscard]] bool TryPost(IoFn fn);

    [[nodiscard]] std::size_t QueueDepth() const;
    [[nodiscard]] std::size_t Capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t WorkerCount() const noexcept { return worker_count_; }

    /// The process-wide IO lane.
    static IoExecutor& Global();

private:
    void WorkerMain();

    const std::size_t worker_count_;
    const std::size_t capacity_;
    std::vector<std::jthread> workers_;
    mutable std::mutex mutex_;
    std::condition_variable space_cv_;
    std::condition_variable work_cv_;
    std::deque<IoFn> queue_;
    bool stopping_ = false;
};

} // namespace VulkanShared
