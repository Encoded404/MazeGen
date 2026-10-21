// Wall-clock frame-loop profiling.
//
// The engine's own frame accounting answers "how long did the CPU spend", which
// is not the question a stutter asks. A loop that renders in 2 ms and still
// stutters is spending its time *blocked* -- inside a fence wait, an image
// acquire, a present, or off-CPU entirely -- and none of that shows up in a
// work counter.
//
// So this measures the wall-clock interval between iterations and attributes
// every long one to the single stage that dominated it. A stage that is
// blocking on the swapchain, on the device, or on the scheduler is then named
// directly instead of inferred. It also tracks the interval between present
// calls: in the sequential path that is the display cadence, while DiscardStale
// drains several completed frames at once and the compositor drops the stale
// ones, so there it is present throughput rather than scanout cadence.
//
// It additionally implements VulkanShared::ISystemProfileSink so the ECS
// schedules can report per-system CPU timings. Those rows are a subset of the
// app-update / app-render stages, never additional time.
//
// Cost when enabled: one steady_clock read per stage and a few dozen arithmetic
// operations per iteration. When disabled, FrameStageScope reads no clock at
// all.

export module Runtime.FrameProfiler;

import std;

import VulkanShared.SystemProfile;

export namespace Runtime {

// ── Interval statistics ─────────────────────────────────────────────────
//
// A cadence is not a mean. "29 fps" describes a mean interval of 34.5 ms and is
// consistent with anything from a metronome to a freeze-and-burst pattern, and
// the two feel completely different. Since the whole question here is whether
// something *stutters*, the statistics keep every sample up to kMaxSamples:
// quantiles are exact rather than bucket edges, and threshold counts are exact
// over the stored samples. Past the cap the totals stay exact and Describe()
// discloses that the distribution covers only the stored prefix.
class IntervalStats {
public:
    // A window that logs every 30000 frames stores a few thousand samples; the
    // cap only exists so a pathological cadence cannot grow without bound. Past
    // it, the count/total/max keep accumulating but the samples do not, and
    // Describe() says so rather than quietly reporting a truncated quantile.
    static constexpr std::size_t kMaxSamples = 100000;

    void Add(const double milliseconds) {
        ++count_;
        total_ms_ += milliseconds;
        max_ms_ = std::max(max_ms_, milliseconds);
        min_ms_ = std::min(min_ms_, milliseconds);
        if (samples_.size() < kMaxSamples) {
            samples_.push_back(milliseconds);
        } else {
            ++unsampled_;
        }
    }

    void Reset() {
        samples_.clear();
        count_ = 0;
        unsampled_ = 0;
        total_ms_ = 0.0;
        max_ms_ = 0.0;
        min_ms_ = std::numeric_limits<double>::max();
    }

    [[nodiscard]] bool Empty() const { return count_ == 0; }
    [[nodiscard]] std::uint64_t Count() const { return count_; }
    [[nodiscard]] double TotalMs() const { return total_ms_; }
    [[nodiscard]] double MeanMs() const {
        return count_ == 0 ? 0.0 : total_ms_ / static_cast<double>(count_);
    }
    [[nodiscard]] double MinMs() const { return count_ == 0 ? 0.0 : min_ms_; }
    [[nodiscard]] double MaxMs() const { return max_ms_; }

    // Nearest-rank quantile over the stored samples, so a p99 that is really a
    // freeze shows up as the freeze and not as a bucket edge near it.
    [[nodiscard]] double QuantileMs(const double quantile) const {
        if (samples_.empty()) {
            return 0.0;
        }
        std::vector<double> sorted = samples_;
        const std::size_t rank = static_cast<std::size_t>(std::clamp(
            std::ceil(quantile * static_cast<double>(sorted.size())), 1.0,
            static_cast<double>(sorted.size())));
        std::nth_element(sorted.begin(),
                         sorted.begin() + static_cast<std::ptrdiff_t>(rank - 1),
                         sorted.end());
        return sorted[rank - 1];
    }

    // Exact over the stored samples; once the cap is reached it covers the first
    // kMaxSamples, like QuantileMs(), rather than silently truncating.
    [[nodiscard]] std::uint64_t CountAtOrAbove(const double milliseconds) const {
        std::uint64_t hits = 0;
        for (const double sample : samples_) {
            if (sample >= milliseconds) {
                ++hits;
            }
        }
        return hits;
    }

    [[nodiscard]] std::string Describe() const {
        if (count_ == 0) {
            return "no samples";
        }
        std::string out = std::format(
            "n={} mean {:.2f} ms, p50 {:.2f}, p90 {:.2f}, p99 {:.2f}, min {:.2f}, max {:.2f}",
            count_, MeanMs(), QuantileMs(0.50), QuantileMs(0.90), QuantileMs(0.99), MinMs(),
            MaxMs());
        if (unsampled_ != 0) {
            out += std::format(" ({} samples not stored; quantiles and threshold counts cover "
                               "the first {})",
                               unsampled_, samples_.size());
        }
        return out;
    }

private:
    std::vector<double> samples_{};
    std::uint64_t count_ = 0;
    std::uint64_t unsampled_ = 0;
    double total_ms_ = 0.0;
    double max_ms_ = 0.0;
    // Max, not zero: a minimum seeded at zero can never be beaten by a positive
    // interval, so MinMs() would report zero forever.
    double min_ms_ = std::numeric_limits<double>::max();
};

// ── Frame profiling ─────────────────────────────────────────────────────

enum class FrameStage : std::uint8_t {
    PollEvents = 0,
    PreInput,
    InputProcess,
    RuntimeBegin,
    PresentDrain,
    BootstrapBegin,
    SwapchainRecreate,
    Acquire,
    AppUpdate,
    AppRender,
    Submit,
    Present,
    RuntimeEnd,
    BootstrapEnd,
    // Interval time no scope accounted for: a blocking wait that is not wrapped,
    // the log write itself, or the OS descheduling this thread.
    Idle,
    Count,
};

inline constexpr std::size_t kFrameStageCount = static_cast<std::size_t>(FrameStage::Count);

[[nodiscard]] inline std::string_view FrameStageName(const FrameStage stage) {
    switch (stage) {
        case FrameStage::PollEvents: return "poll-events";
        case FrameStage::PreInput: return "pre-input";
        case FrameStage::InputProcess: return "input-process";
        case FrameStage::RuntimeBegin: return "runtime-begin";
        case FrameStage::PresentDrain: return "present-drain";
        case FrameStage::BootstrapBegin: return "bootstrap-begin";
        case FrameStage::SwapchainRecreate: return "swapchain-recreate";
        case FrameStage::Acquire: return "acquire-image";
        case FrameStage::AppUpdate: return "app-update";
        case FrameStage::AppRender: return "app-render";
        case FrameStage::Submit: return "submit";
        case FrameStage::Present: return "present";
        case FrameStage::RuntimeEnd: return "runtime-end";
        case FrameStage::BootstrapEnd: return "bootstrap-end";
        case FrameStage::Idle: return "unaccounted";
        case FrameStage::Count: break;
    }
    return "unknown";
}

struct StageProfile {
    // Total and worst time this stage contributed, over the whole window.
    std::uint64_t samples = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    // Long frames (interval at or above the threshold) in which this stage was
    // the largest single contributor, and what it contributed to them.
    std::uint64_t blame_count = 0;
    double blame_total_ms = 0.0;
};

struct FrameProfileConfig {
    // An iteration whose wall-clock interval reaches this counts as a long frame
    // and gets its dominant stage blamed.
    double long_frame_ms = 20.0;
    // Emit one summary per this many iterations.
    std::uint64_t log_every = 600;
};

// One ECS system's contribution over the current summary window.
struct SystemProfile {
    std::uint64_t samples = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
};

class FrameProfiler final : public VulkanShared::ISystemProfileSink {
public:
    FrameProfiler() = default;

    explicit FrameProfiler(const FrameProfileConfig& config) { Configure(config); }

    // Off by default: an unconfigured profiler adds no measurement at all.
    void Configure(const FrameProfileConfig& config) {
        config_ = config;
        enabled_ = true;
        Reset();
    }

    void SetEnabled(const bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool Enabled() const { return enabled_; }
    [[nodiscard]] const FrameProfileConfig& Config() const { return config_; }

    // Full reset, including the summary cadence.
    void Reset() {
        stages_ = {};
        present_interval_.Reset();
        iteration_interval_.Reset();
        total_iterations_ = 0;
        long_frames_ = 0;
        have_previous_ = false;
        iteration_stage_ms_ = 0.0;
        worst_stage_ = FrameStage::Count;
        worst_stage_ms_ = 0.0;
        summary_pending_ = false;
        have_presented_ = false;
        frame_systems_.clear();
        component_systems_.clear();
    }

    // Clears everything a summary reports but keeps counting towards the next
    // cadence, so consecutive summaries describe disjoint windows instead of
    // re-reporting the run's history.
    void ResetWindow() {
        stages_ = {};
        present_interval_.Reset();
        iteration_interval_.Reset();
        long_frames_ = 0;
        have_presented_ = false;
        frame_systems_.clear();
        component_systems_.clear();
    }

    // Call at the top of every iteration. Closes the interval that just ended.
    void BeginIteration() {
        if (!enabled_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (have_previous_) {
            RecordInterval(
                std::chrono::duration<double, std::milli>(now - previous_start_).count());
        }
        previous_start_ = now;
        have_previous_ = true;
        iteration_stage_ms_ = 0.0;
        worst_stage_ = FrameStage::Count;
        worst_stage_ms_ = 0.0;
    }

    // Called on each successful present. Records the interval between present
    // calls. Under PresentPolicy::DiscardStale several frames may be drained
    // back to back, so that policy measures present throughput, not the
    // cadence the compositor actually scans out; the sequential path is
    // one-present-per-frame.
    void NotePresent() {
        if (!enabled_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (have_presented_) {
            present_interval_.Add(
                std::chrono::duration<double, std::milli>(now - last_present_).count());
        }
        last_present_ = now;
        have_presented_ = true;
    }

    void EndIteration() {}

    void Add(const FrameStage stage, const double milliseconds) {
        if (!enabled_) {
            return;
        }
        const auto index = static_cast<std::size_t>(stage);
        if (index >= kFrameStageCount) {
            return;
        }
        auto& profile = stages_[index];
        ++profile.samples;
        profile.total_ms += milliseconds;
        profile.max_ms = std::max(profile.max_ms, milliseconds);
        iteration_stage_ms_ += milliseconds;
        if (milliseconds > worst_stage_ms_) {
            worst_stage_ms_ = milliseconds;
            worst_stage_ = stage;
        }
    }

    // ── ECS system breakdown ──
    //
    // Single-threaded by construction: SystemSchedule runs on the loop thread
    // and object-component async updates are not sampled, so no atomics. The
    // label is copied only on first sight; the hot path is a heterogeneous
    // string_view lookup.
    void AddSystemTiming(const VulkanShared::SystemProfileKind kind,
                         const std::string_view label, const double milliseconds) override {
        if (!enabled_) {
            return;
        }
        auto& systems = kind == VulkanShared::SystemProfileKind::Frame ? frame_systems_
                                                                      : component_systems_;
        auto it = systems.find(label);
        if (it == systems.end()) {
            it = systems.try_emplace(std::string(label)).first;
        }
        ++it->second.samples;
        it->second.total_ms += milliseconds;
        it->second.max_ms = std::max(it->second.max_ms, milliseconds);
    }

    [[nodiscard]] const std::map<std::string, SystemProfile, std::less<>>& FrameSystems()
        const {
        return frame_systems_;
    }
    [[nodiscard]] const std::map<std::string, SystemProfile, std::less<>>& ComponentSystems()
        const {
        return component_systems_;
    }

    // Drops the next interval (the one that will close at the following
    // BeginIteration) from the histogram. Used when the caller is about to do
    // something it knows is slow and one-off -- writing the summary to the
    // console, for instance -- so the instrument does not appear in its own
    // measurement.
    void DiscardNextInterval() { have_previous_ = false; }

    // True once the cadence has come round; clears the flag.
    [[nodiscard]] bool TakeSummaryDue() {
        const bool due = summary_pending_;
        summary_pending_ = false;
        return due;
    }

    [[nodiscard]] std::uint64_t Iterations() const { return iteration_interval_.Count(); }
    // True when any measurement exists -- stages, present cadence, or ECS rows.
    // Unlike Iterations(), this is also true for a one-iteration run, whose
    // stage samples exist even though no interval has closed yet.
    [[nodiscard]] bool HasSamples() const {
        if (!present_interval_.Empty() || !frame_systems_.empty() ||
            !component_systems_.empty()) {
            return true;
        }
        for (const auto& stage : stages_) {
            if (stage.samples != 0) {
                return true;
            }
        }
        return false;
    }
    [[nodiscard]] std::uint64_t TotalIterations() const { return total_iterations_; }
    [[nodiscard]] std::uint64_t LongFrames() const { return long_frames_; }
    [[nodiscard]] double MeanIntervalMs() const { return iteration_interval_.MeanMs(); }
    [[nodiscard]] double MaxIntervalMs() const { return iteration_interval_.MaxMs(); }
    [[nodiscard]] const IntervalStats& IterationInterval() const { return iteration_interval_; }
    [[nodiscard]] const IntervalStats& PresentInterval() const { return present_interval_; }
    [[nodiscard]] const StageProfile& Stage(const FrameStage stage) const {
        return stages_[static_cast<std::size_t>(stage)];
    }

    [[nodiscard]] std::string Summary() const {
        std::string out = std::format(
            "Frame profile over {} iterations: interval {}; {} long frames (>= {:.1f} ms)",
            Iterations(), iteration_interval_.Describe(), long_frames_, config_.long_frame_ms);
        if (!present_interval_.Empty()) {
            out += std::format("\n  present interval: {}", present_interval_.Describe());
        }
        // Stages in descending total time: what the loop spends its wall clock
        // on, which is the whole point.
        std::array<std::size_t, kFrameStageCount> order{};
        for (std::size_t i = 0; i < kFrameStageCount; ++i) {
            order[i] = i;
        }
        std::ranges::sort(order, [this](std::size_t a, std::size_t b) {
            return stages_[a].total_ms > stages_[b].total_ms;
        });
        double stage_total_ms = 0.0;
        for (const auto& profile : stages_) {
            stage_total_ms += profile.total_ms;
        }
        out += "\n  stages (total ms, avg ms, max ms):";
        for (const std::size_t index : order) {
            const auto& profile = stages_[index];
            if (profile.samples == 0) {
                continue;
            }
            out += std::format("\n    {:<18} {:10.1f}  {:7.3f}  {:8.2f}",
                               FrameStageName(static_cast<FrameStage>(index)), profile.total_ms,
                               profile.total_ms / static_cast<double>(profile.samples),
                               profile.max_ms);
        }
        // Stage samples from deliberately discarded intervals (swapchain
        // recreation, the summary write) stay in the table, so in a window that
        // contains one this sum can exceed the interval total. The rows are
        // still correct; only the remainder is not a precise unaccounted figure
        // for such a window.
        out += std::format("\n    {:<18} {:10.1f}  (sum of stages vs {:.1f} ms of interval time)",
                           "(measured)", stage_total_ms, iteration_interval_.TotalMs());

        // Per-ECS-system rows. Explicitly a subset of app-update/app-render so
        // nobody adds them to the stage total.
        struct SystemRow {
            VulkanShared::SystemProfileKind kind;
            std::string_view label;
            const SystemProfile* profile;
        };
        std::vector<SystemRow> rows;
        rows.reserve(frame_systems_.size() + component_systems_.size());
        for (const auto& [label, profile] : frame_systems_) {
            rows.push_back({VulkanShared::SystemProfileKind::Frame, label, &profile});
        }
        for (const auto& [label, profile] : component_systems_) {
            rows.push_back({VulkanShared::SystemProfileKind::Component, label, &profile});
        }
        if (!rows.empty()) {
            std::ranges::sort(rows, [](const SystemRow& a, const SystemRow& b) {
                if (a.profile->total_ms != b.profile->total_ms) {
                    return a.profile->total_ms > b.profile->total_ms;
                }
                return a.label < b.label;
            });
            out += "\n  ecs systems (a breakdown of app-update / app-render; not additional "
                   "time):";
            for (const SystemRow& row : rows) {
                out += std::format(
                    "\n    {:<10} {:<26} n={}  total {:8.1f} ms  avg {:.3f} ms  max {:.2f} ms",
                    VulkanShared::SystemProfileKindName(row.kind), row.label,
                    row.profile->samples, row.profile->total_ms,
                    row.profile->total_ms / static_cast<double>(row.profile->samples),
                    row.profile->max_ms);
            }
        }

        if (long_frames_ != 0) {
            out += std::format("\n  blame for the {} long frames:", long_frames_);
            std::ranges::sort(order, [this](std::size_t a, std::size_t b) {
                return stages_[a].blame_count > stages_[b].blame_count;
            });
            bool any = false;
            for (const std::size_t index : order) {
                const auto& profile = stages_[index];
                if (profile.blame_count == 0) {
                    continue;
                }
                any = true;
                out += std::format(
                    "\n    {:<18} {:6} frames, {:10.1f} ms total, {:8.2f} ms avg",
                    FrameStageName(static_cast<FrameStage>(index)), profile.blame_count,
                    profile.blame_total_ms,
                    profile.blame_total_ms / static_cast<double>(profile.blame_count));
            }
            if (!any) {
                out += "\n    (none attributed)";
            }
        }
        return out;
    }

private:
    void RecordInterval(const double interval_ms) {
        ++total_iterations_;
        iteration_interval_.Add(interval_ms);

        if (interval_ms >= config_.long_frame_ms) {
            ++long_frames_;
            // Whatever the loop was doing when it ran long: the largest stage it
            // measured, or the unaccounted remainder if that is bigger. The
            // remainder is the interesting one -- a fence wait or a present that
            // no scope covers -- so it is charged to Idle and can win the blame.
            const double idle_ms = std::max(0.0, interval_ms - iteration_stage_ms_);
            if (idle_ms > worst_stage_ms_) {
                worst_stage_ = FrameStage::Idle;
                worst_stage_ms_ = idle_ms;
            }
            const auto index = static_cast<std::size_t>(worst_stage_);
            if (worst_stage_ != FrameStage::Count && index < kFrameStageCount) {
                ++stages_[index].blame_count;
                stages_[index].blame_total_ms += worst_stage_ms_;
            }
        }

        const std::uint64_t cadence = std::max<std::uint64_t>(1, config_.log_every);
        if (total_iterations_ % cadence == 0) {
            summary_pending_ = true;
        }
    }

    FrameProfileConfig config_{};
    std::array<StageProfile, kFrameStageCount> stages_{};
    IntervalStats present_interval_{};
    std::chrono::steady_clock::time_point last_present_{};
    IntervalStats iteration_interval_{};
    bool enabled_ = false;
    bool have_previous_ = false;
    bool summary_pending_ = false;
    bool have_presented_ = false;
    std::chrono::steady_clock::time_point previous_start_{};
    std::uint64_t total_iterations_ = 0;
    std::uint64_t long_frames_ = 0;
    double iteration_stage_ms_ = 0.0;
    FrameStage worst_stage_ = FrameStage::Count;
    double worst_stage_ms_ = 0.0;
    std::map<std::string, SystemProfile, std::less<>> frame_systems_{};
    std::map<std::string, SystemProfile, std::less<>> component_systems_{};
};

// RAII timing for one stage. Reads no clock when the profiler is disabled.
class FrameStageScope {
public:
    FrameStageScope(FrameProfiler& profiler, const FrameStage stage)
        : profiler_(&profiler)
        , stage_(stage)
        , active_(profiler.Enabled()) {
        if (active_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~FrameStageScope() {
        if (active_) {
            profiler_->Add(stage_, std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - start_)
                                       .count());
        }
    }

    FrameStageScope(const FrameStageScope&) = delete;
    FrameStageScope& operator=(const FrameStageScope&) = delete;
    FrameStageScope(FrameStageScope&&) = delete;
    FrameStageScope& operator=(FrameStageScope&&) = delete;

private:
    FrameProfiler* profiler_;
    FrameStage stage_;
    bool active_ = false;
    std::chrono::steady_clock::time_point start_{};
};

} // namespace Runtime
