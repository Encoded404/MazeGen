module;

module VulkanShared.Jobs;

import std;

namespace VulkanShared {

namespace detail {

/// The control block behind a dependency-counted job.
///
/// Kept alive by a `JobHandle` and, once queued, by the queued work item, so a
/// `FreeJob` on a still-running job is safe. `done`/`error`/`barrier` are
/// guarded by `mutex`; `pending` and the queue flags are atomics because the
/// dependency path (Jolt's solver, teardown graphs) is lock-free by design.
struct JobControl {
    JobScheduler* owner = nullptr;
    Tier tier = Tier::Compute;
    const char* name = "";
    JobScheduler::JobFn fn;

    std::atomic<std::uint32_t> pending{0};
    std::atomic<bool> queued{false};
    std::atomic<bool> submitted{false};
    std::atomic<bool> done{false};

    std::mutex mutex;
    std::condition_variable cv;
    JobScheduler::Barrier* barrier = nullptr; // guarded by mutex
    std::exception_ptr error;                 // guarded by mutex
};

} // namespace detail

// ── JobHandle ───────────────────────────────────────────────────────────────

JobHandle::JobHandle(std::shared_ptr<detail::JobControl> control) noexcept
    : control_(std::move(control)) {}

bool JobHandle::IsValid() const noexcept {
    return static_cast<bool>(control_);
}

bool JobHandle::IsDone() const noexcept {
    return control_ && control_->done.load(std::memory_order_acquire);
}

// ── JobScheduler::Barrier ───────────────────────────────────────────────────

void JobScheduler::Barrier::AddJobs(const JobHandle* jobs, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const std::shared_ptr<detail::JobControl>& control = jobs[i].control_;
        if (!control) continue;

        // Taking the job's own lock orders this increment against the job's
        // completion, which reads `barrier` under the same lock before calling
        // OnJobFinished. Without it a job that finished between the check and
        // the increment would decrement a count that was never incremented.
        const std::scoped_lock control_lock(control->mutex);
        if (control->done.load(std::memory_order_acquire)) continue;
        if (control->barrier != nullptr) continue; // a job belongs to one barrier
        control->barrier = this;
        const std::scoped_lock barrier_lock(mutex_);
        ++pending_;
    }
}

void JobScheduler::Barrier::AddJob(const JobHandle& job) {
    AddJobs(&job, 1);
}

std::size_t JobScheduler::Barrier::Pending() const noexcept {
    const std::scoped_lock lock(mutex_);
    return pending_;
}

void JobScheduler::Barrier::OnJobFinished() noexcept {
    const std::scoped_lock lock(mutex_);
    if (pending_ > 0) --pending_;
}

void JobScheduler::Barrier::Reset() noexcept {
    const std::scoped_lock lock(mutex_);
    pending_ = 0;
}

// ── JobScheduler ────────────────────────────────────────────────────────────

JobScheduler::JobScheduler(JobBudget budget) {
    Configure(budget);
}

JobScheduler::~JobScheduler() {
    WaitForIdle();
    StopWorkers();
}

std::uint32_t JobScheduler::ResolveWorkers(unsigned int requested) noexcept {
    if (requested != JobBudget::kAuto) return requested; // 0 == single-threaded
    return std::max(1u, std::thread::hardware_concurrency() - 1u);
}

void JobScheduler::Configure(JobBudget budget) {
    StopWorkers();
    {
        const std::scoped_lock lock(mutex_);
        background_starvation_limit_ = std::max(1u, budget.background_starvation_limit);
        served_since_background_ = 0;
    }
    worker_count_ = ResolveWorkers(budget.workers);
    StartWorkers();
}

void JobScheduler::StartWorkers() {
    workers_.reserve(worker_count_);
    for (unsigned int i = 0; i < worker_count_; ++i) {
        workers_.emplace_back(&JobScheduler::WorkerMain, this);
    }
}

void JobScheduler::StopWorkers() {
    {
        const std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    workers_.clear(); // joins (jthread), draining what is already queued
    const std::scoped_lock lock(mutex_);
    stopping_ = false;
}

JobScheduler& JobScheduler::Global() {
    static JobScheduler scheduler{};
    return scheduler;
}

bool JobScheduler::HasWorkLocked() const noexcept {
    for (const auto& queue : queues_) {
        if (!queue.empty()) return true;
    }
    return false;
}

bool JobScheduler::PopLocked(WorkItem& out) {
    std::size_t tier = static_cast<std::size_t>(Tier::Count);
    for (std::size_t i = 0; i < tier; ++i) {
        if (!queues_[i].empty()) {
            tier = i;
            break;
        }
    }
    if (tier == static_cast<std::size_t>(Tier::Count)) return false;

    // Strict priority, with a starvation guard: once `limit` higher-tier jobs
    // have been served while Background work waits, serve one Background job.
    constexpr auto background = static_cast<std::size_t>(Tier::Background);
    if (tier != background && !queues_[background].empty() &&
        served_since_background_ >= background_starvation_limit_) {
        out = std::move(queues_[background].front());
        queues_[background].pop_front();
        served_since_background_ = 0;
    } else {
        out = std::move(queues_[tier].front());
        queues_[tier].pop_front();
        if (tier == background) {
            served_since_background_ = 0;
        } else {
            ++served_since_background_;
        }
    }
    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

void JobScheduler::RunItem(WorkItem& item) noexcept {
    try {
        if (item.fn) item.fn();
    } catch (...) {
        // Fire-and-forget work must never kill a worker thread.
    }
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    idle_cv_.notify_all();
}

void JobScheduler::WorkerMain() {
    for (;;) {
        WorkItem item;
        {
            std::unique_lock lock(mutex_);
            work_cv_.wait(lock, [this] { return stopping_ || HasWorkLocked(); });
            if (!HasWorkLocked()) {
                if (stopping_) return;
                continue;
            }
            if (!PopLocked(item)) continue;
        }
        RunItem(item);
    }
}

bool JobScheduler::TryExecuteOne() {
    WorkItem item;
    {
        const std::scoped_lock lock(mutex_);
        if (!PopLocked(item)) return false;
    }
    RunItem(item);
    return true;
}

void JobScheduler::SubmitLocked(Tier tier, const char* name, JobFn fn) {
    WorkItem item;
    item.tier = tier;
    item.name = name != nullptr ? name : "";
    item.fn = std::move(fn);
    item.sequence = next_sequence_++;
    queues_[static_cast<std::size_t>(tier)].push_back(std::move(item));
}

void JobScheduler::Submit(Tier tier, const char* name, JobFn fn) {
    if (!fn) return;
    {
        const std::scoped_lock lock(mutex_);
        SubmitLocked(tier, name, std::move(fn));
    }
    work_cv_.notify_one();
}

std::size_t JobScheduler::QueuedCount() const noexcept {
    const std::scoped_lock lock(mutex_);
    std::size_t total = 0;
    for (const auto& queue : queues_) total += queue.size();
    return total;
}

void JobScheduler::WaitForIdle() {
    for (;;) {
        {
            const std::scoped_lock lock(mutex_);
            if (!HasWorkLocked() && in_flight_.load(std::memory_order_acquire) == 0) return;
        }
        if (!TryExecuteOne()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// ── Dependency-counted jobs ─────────────────────────────────────────────────

JobHandle JobScheduler::CreateJob(JobDesc desc, std::uint32_t dependencies) {
    auto control = std::make_shared<detail::JobControl>();
    control->owner = this;
    control->tier = desc.tier;
    control->name = desc.name != nullptr ? desc.name : "";
    control->fn = std::move(desc.fn);
    control->pending.store(dependencies, std::memory_order_relaxed);
    return JobHandle{std::move(control)};
}

void JobScheduler::MaybeQueue(const std::shared_ptr<detail::JobControl>& control) {
    if (!control) return;
    if (!control->queued.load(std::memory_order_acquire)) return;
    if (control->pending.load(std::memory_order_acquire) != 0) return;
    if (control->submitted.exchange(true, std::memory_order_acq_rel)) return;

    JobScheduler* owner = control->owner;
    if (owner == nullptr) return;

    owner->Submit(control->tier, control->name, [control] {
        try {
            if (control->fn) control->fn();
        } catch (...) {
            const std::scoped_lock lock(control->mutex);
            if (!control->error) control->error = std::current_exception();
        }

        Barrier* barrier = nullptr;
        {
            const std::scoped_lock lock(control->mutex);
            control->done.store(true, std::memory_order_release);
            barrier = control->barrier;
            control->barrier = nullptr;
        }
        control->cv.notify_all();
        if (barrier != nullptr) barrier->OnJobFinished();
    });
}

void JobScheduler::RemoveDependency(const JobHandle& job) {
    const std::shared_ptr<detail::JobControl>& control = job.control_;
    if (!control) return;
    if (control->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        MaybeQueue(control);
    }
}

void JobScheduler::QueueJob(const JobHandle& job) {
    const std::shared_ptr<detail::JobControl>& control = job.control_;
    if (!control) return;
    control->queued.store(true, std::memory_order_release);
    MaybeQueue(control);
}

void JobScheduler::QueueJobs(const JobHandle* jobs, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) QueueJob(jobs[i]);
}

void JobScheduler::FreeJob(JobHandle& job) {
    job.control_.reset();
}

void JobScheduler::Wait(const JobHandle& job) {
    const std::shared_ptr<detail::JobControl>& control = job.control_;
    if (!control) return;
    while (!control->done.load(std::memory_order_acquire)) {
        if (!TryExecuteOne()) {
            std::unique_lock lock(control->mutex);
            control->cv.wait_for(lock, std::chrono::microseconds(200), [&control] {
                return control->done.load(std::memory_order_acquire);
            });
        }
    }
    std::exception_ptr error;
    {
        const std::scoped_lock lock(control->mutex);
        error = control->error;
    }
    if (error) std::rethrow_exception(error);
}

// ── Barriers ────────────────────────────────────────────────────────────────

JobScheduler::Barrier* JobScheduler::CreateBarrier() {
    const std::scoped_lock lock(barrier_pool_mutex_);
    if (!barrier_free_list_.empty()) {
        std::unique_ptr<Barrier> barrier = std::move(barrier_free_list_.back());
        barrier_free_list_.pop_back();
        return barrier.release();
    }
    return new Barrier();
}

void JobScheduler::DestroyBarrier(Barrier* barrier) {
    if (barrier == nullptr) return;
    if (barrier->Pending() != 0) {
        throw std::logic_error("JobScheduler: DestroyBarrier called on a barrier that is not empty");
    }
    barrier->Reset();
    const std::scoped_lock lock(barrier_pool_mutex_);
    barrier_free_list_.emplace_back(barrier);
}

void JobScheduler::WaitForJobs(Barrier* barrier) {
    if (barrier == nullptr) return;
    while (barrier->Pending() != 0) {
        if (!TryExecuteOne()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// ── IoExecutor ──────────────────────────────────────────────────────────────

IoExecutor::IoExecutor(std::size_t workers, std::size_t capacity)
    : worker_count_(std::max<std::size_t>(1, workers)),
      capacity_(std::max<std::size_t>(1, capacity)) {
    workers_.reserve(worker_count_);
    for (std::size_t i = 0; i < worker_count_; ++i) {
        workers_.emplace_back(&IoExecutor::WorkerMain, this);
    }
}

IoExecutor::~IoExecutor() {
    {
        const std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    space_cv_.notify_all();
    workers_.clear(); // joins; already-queued IO still runs to completion
}

IoExecutor& IoExecutor::Global() {
    static IoExecutor executor{};
    return executor;
}

void IoExecutor::Post(IoFn fn) {
    if (!fn) return;
    std::unique_lock lock(mutex_);
    space_cv_.wait(lock, [this] { return queue_.size() < capacity_ || stopping_; });
    if (stopping_) return;
    queue_.push_back(std::move(fn));
    lock.unlock();
    work_cv_.notify_one();
}

bool IoExecutor::TryPost(IoFn fn) {
    if (!fn) return false;
    const std::scoped_lock lock(mutex_);
    if (stopping_ || queue_.size() >= capacity_) return false;
    queue_.push_back(std::move(fn));
    work_cv_.notify_one();
    return true;
}

std::size_t IoExecutor::QueueDepth() const {
    const std::scoped_lock lock(mutex_);
    return queue_.size();
}

void IoExecutor::WorkerMain() {
    for (;;) {
        IoFn fn;
        {
            std::unique_lock lock(mutex_);
            work_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stopping_) return;
                continue;
            }
            fn = std::move(queue_.front());
            queue_.pop_front();
        }
        space_cv_.notify_all();
        try {
            if (fn) fn();
        } catch (...) {
            // IO work is fire-and-forget by contract.
        }
    }
}

} // namespace VulkanShared
