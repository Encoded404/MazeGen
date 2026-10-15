module;

export module VulkanShared.Teardown;

import std;

import VulkanShared.Jobs;

export namespace VulkanShared {

/// Identifier for a task registered with a TeardownScheduler.
using TeardownId = std::size_t;

/// Executes teardown tasks concurrently, honoring explicit dependencies.
///
/// Intended for the shutdown path: after the GPU has been drained
/// (`waitIdle`), the resource destruction of independent subsystems can
/// proceed in parallel. A task starts only after every task in its
/// `dependencies` list has completed, so ownership/shared-object ordering can
/// be encoded (e.g. "destroy the shared backend only after the system that
/// holds the last reference has been torn down").
///
/// The graph runs on the engine's shared compute pool, so the shutdown path no
/// longer spawns a second, ephemeral thread population with its own sizing
/// policy: a task is a dependency-counted job plus an edge to each dependent,
/// and `Run()` drains the graph with participating waits.
///
/// Rules:
///   - Add all tasks before calling `Run()` (Add after Run throws).
///   - `Run()` returns only once every task has been attempted, and rethrows the
///     first task exception, so the caller knows every destruction finished
///     before the Vulkan device/instance are released.
class TeardownScheduler {
public:
    TeardownScheduler() = default;

    TeardownScheduler(const TeardownScheduler&) = delete;
    TeardownScheduler& operator=(const TeardownScheduler&) = delete;

    ~TeardownScheduler() {
        try {
            Run();
        } catch (...) {
            // Destructors must not throw; the caller that used Run() explicitly
            // has already seen any exception.
        }
    }

    TeardownId Add(std::string name, std::function<void()> task = {},
                   std::vector<TeardownId> dependencies = {}) {
        std::lock_guard lock(mutex_);
        if (run_) {
            throw std::logic_error("TeardownScheduler: Add called after Run");
        }
        const TeardownId id = tasks_.size();
        Task t;
        t.name = std::move(name);
        t.fn = std::move(task);
        t.pending_deps = dependencies.size();
        for (const TeardownId dep : dependencies) {
            if (dep >= id) {
                throw std::out_of_range(
                    "TeardownScheduler: dependency refers to a task that has not been added yet");
            }
            // One edge per declared dependency (duplicates are honored).
            tasks_[dep].dependents.push_back(id);
        }
        tasks_.push_back(std::move(t));
        return id;
    }

    void Run() {
        {
            std::lock_guard lock(mutex_);
            if (run_) {
                return;
            }
            run_ = true;
            if (HasCycle()) {
                cycle_ = true;
            }
        }
        if (cycle_) {
            throw std::logic_error("TeardownScheduler: dependency cycle detected");
        }
        if (tasks_.empty()) {
            return;
        }

        JobScheduler& scheduler = JobScheduler::Global();

        // The batch is shared, not captured per job: a task must be able to
        // release the dependencies of the tasks that follow it by handle, and
        // those handles do not exist until every job has been created.
        struct Batch {
            std::vector<JobHandle> handles;
            std::mutex mutex;
            std::exception_ptr first_error;
        };
        auto batch = std::make_shared<Batch>();
        batch->handles.resize(tasks_.size());

        for (std::size_t i = 0; i < tasks_.size(); ++i) {
            const auto dependencies = static_cast<std::uint32_t>(tasks_[i].pending_deps);
            batch->handles[i] = scheduler.CreateJob(
                JobScheduler::JobDesc{
                    Tier::Compute, "teardown",
                    [this, batch, i, &scheduler] {
                        try {
                            if (tasks_[i].fn) tasks_[i].fn();
                        } catch (...) {
                            const std::lock_guard lock(batch->mutex);
                            if (batch->first_error == nullptr) {
                                batch->first_error = std::current_exception();
                            }
                        }
                        // Release dependents even when the task failed: a
                        // teardown batch must never deadlock on a broken task.
                        for (const TeardownId dependent : tasks_[i].dependents) {
                            scheduler.RemoveDependency(batch->handles[dependent]);
                        }
                    }},
                dependencies);
        }

        for (JobHandle& handle : batch->handles) scheduler.QueueJob(handle);
        for (JobHandle& handle : batch->handles) scheduler.Wait(handle);
        for (JobHandle& handle : batch->handles) scheduler.FreeJob(handle);

        std::exception_ptr error;
        {
            const std::lock_guard lock(batch->mutex);
            error = batch->first_error;
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    struct Task {
        std::string name;
        std::function<void()> fn;
        std::size_t pending_deps = 0;
        std::vector<TeardownId> dependents{};
    };

    [[nodiscard]] bool HasCycle() const {
        std::vector<std::size_t> indegree(tasks_.size());
        for (std::size_t i = 0; i < tasks_.size(); ++i) {
            indegree[i] = tasks_[i].pending_deps;
        }
        std::vector<TeardownId> ready;
        for (std::size_t i = 0; i < tasks_.size(); ++i) {
            if (indegree[i] == 0) {
                ready.push_back(i);
            }
        }
        std::size_t processed = 0;
        while (!ready.empty()) {
            const TeardownId u = ready.back();
            ready.pop_back();
            ++processed;
            for (const TeardownId v : tasks_[u].dependents) {
                if (--indegree[v] == 0) {
                    ready.push_back(v);
                }
            }
        }
        return processed != tasks_.size();
    }

    std::vector<Task> tasks_;
    std::mutex mutex_;
    bool run_ = false;
    bool cycle_ = false;
};

/// Runs `fn` on a detached worker thread (fire-and-forget).
///
/// ONLY use this for work that (a) owns everything it touches and (b) may be
/// abandoned if the process exits while it is still running - e.g. joining a
/// file-watcher thread that blocks for its polling interval. Never use it for
/// GPU resources that must be destroyed before the Vulkan device/instance are
/// torn down; those must go through a TeardownScheduler so the caller can wait
/// for them.
template<typename Fn>
void RunDetached(Fn&& fn) {
    std::thread{[fn = std::forward<Fn>(fn)]() mutable {
        try {
            fn();
        } catch (...) {
            // Fire-and-forget: never let a background teardown kill the process.
        }
    }}.detach();
}

} // namespace VulkanShared
