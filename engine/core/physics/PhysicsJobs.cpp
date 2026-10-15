// The Jolt job-system bridge.
//
// Thin by construction: the engine's scheduler already speaks Jolt's job model
// (dependency-counted jobs, barriers, participating waits), so this file only
// translates Jolt's interface onto it. No `JobSystemThreadPool`, no second thread
// population: every physics job is a Realtime-tier job on
// `VulkanShared::JobScheduler::Global()`, and `WaitForJobs` executes queued work
// on the calling thread instead of sleeping on a semaphore.

module;

#include <Jolt/Jolt.h>

#include <Jolt/Core/Color.h>
#include <Jolt/Core/JobSystemWithBarrier.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsUpdateContext.h>

module VulkanEngine.Physics.World;

import std;

import VulkanShared.Jobs;

namespace VulkanEngine::Physics::detail {

namespace {

/// Jolt's `JobSystem` over the engine's shared scheduler.
///
/// Derives from `JPH::JobSystemWithBarrier`, not `JPH::JobSystem`: the barrier
/// half of the interface hands a `Job*` to `OnJobFinished`, and `Job` is a
/// protected nested type of `JobSystem`, so a barrier implementation can only be
/// written from inside Jolt. `JobSystemWithBarrier` is the base class Jolt
/// documents for exactly this purpose — "make it easier to create a new JobSystem
/// implementation that integrates with your own job system" — and it owns no
/// threads. What is left to this subclass is the four functions that touch the
/// pool: GetMaxConcurrency, CreateJob, FreeJob and QueueJob/QueueJobs.
///
/// `WaitForJobs` is deliberately *not* overridden. Jolt's barrier is stateful —
/// its read index and its per-job references are advanced only by the base class
/// `BarrierImpl::Wait()` — so a custom wait that ignores the barrier fills the
/// 2048-slot ring and stalls ("Barrier full, stalling!") after enough steps.
/// The base implementation is already the participating wait this design wants:
/// it executes the barrier's own executable jobs on the calling thread while the
/// pool runs the rest, which is exactly what Jolt's documentation prescribes
/// (and better than running unrelated engine work while inside the solver).
class EngineJobSystem final : public JPH::JobSystemWithBarrier {
public:
    explicit EngineJobSystem(VulkanShared::JobScheduler& scheduler)
        : JPH::JobSystemWithBarrier(JPH::cMaxPhysicsBarriers), scheduler_(scheduler) {}

    [[nodiscard]] int GetMaxConcurrency() const override {
        const int workers = static_cast<int>(scheduler_.WorkerCount());
        return std::clamp(workers, 1, JPH::PhysicsUpdateContext::cMaxConcurrency);
    }

    JPH::JobHandle CreateJob(const char* inName, JPH::ColorArg inColor,
                             const JPH::JobSystem::JobFunction& inJobFunction,
                             JPH::uint32 inNumDependencies) override {
        Job* job = new Job(inName, inColor, this, inJobFunction, inNumDependencies);
        JPH::JobHandle handle(job);
        // Jolt's contract: a job with no dependencies is runnable immediately.
        if (inNumDependencies == 0) QueueJob(job);
        return handle;
    }

protected:
    void QueueJob(Job* inJob) override { QueueJobs(&inJob, 1); }

    void QueueJobs(Job** inJobs, JPH::uint inNumJobs) override {
        for (JPH::uint i = 0; i < inNumJobs; ++i) {
            Job* job = inJobs[i];
            // The queued item holds a reference, exactly as Jolt's thread pool
            // does: the handle may be released before the body runs.
            job->AddRef();
            scheduler_.Submit(VulkanShared::Tier::Realtime, "jolt", [job] {
                // Execute is idempotent (the dependency counter guards it), so
                // this is safe even when the barrier's wait already ran the job
                // on the calling thread.
                job->Execute();
                job->Release();
            });
        }
    }

    void FreeJob(Job* inJob) override { delete inJob; }

private:
    VulkanShared::JobScheduler& scheduler_;
};

class EngineJobBridge final : public JobBridge {
public:
    explicit EngineJobBridge(VulkanShared::JobScheduler& scheduler)
        : scheduler_(scheduler), jobs_(scheduler) {}

    /// Drain the pool before the job system is destroyed.
    ///
    /// Jolt's barrier releases a job the moment `Job::IsDone()` turns true —
    /// which is inside `Job::Execute()`, before this bridge's own queue
    /// reference is released. `PhysicsSystem::Update` can therefore return while
    /// a queued wrapper lambda is still between `Execute()` and `Release()`.
    /// Destroying the job system then leaves that lambda calling
    /// `mJobSystem->FreeJob` through freed memory (a heap-use-after-free under
    /// ASan). `WaitForIdle` runs on this thread, executes whatever is left
    /// (`Execute` is idempotent, so the leftover is just the release) and
    /// returns only when the pool is empty, so `jobs_` below is destroyed after
    /// every wrapper has finished.
    ~EngineJobBridge() override { scheduler_.WaitForIdle(); }

    [[nodiscard]] void* NativeJobSystem() override { return &jobs_; }

private:
    VulkanShared::JobScheduler& scheduler_;
    EngineJobSystem jobs_;
};

} // namespace

std::unique_ptr<JobBridge> CreateJobBridge() {
    return std::make_unique<EngineJobBridge>(VulkanShared::JobScheduler::Global());
}

} // namespace VulkanEngine::Physics::detail
