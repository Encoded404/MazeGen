// Pacing for the automatic generate -> solve -> repeat loop behind --cycle.
//
// The app owns generation and solver stepping; this class only decides when the next maze is due.
// It is deliberately device-free so the timing and seed progression can be pinned by the headless app tests.
//
// Seeds are deterministic: the first maze uses the configured base seed and every later maze uses base + completed cycles (wrapping).
// The UI shows the current seed, so a relaunch can reproduce any cycle.

module;

export module MazeGenerator.Cycle;

import std;

export namespace MazeGenerator {

struct CycleConfig {
    bool enabled = false;
    std::uint32_t base_seed = 0;
    // How long a finished solve stays on screen before the next maze begins.
    float dwell_seconds = 1.5f;
};

class MazeCycle {
public:
    MazeCycle() = default;
    explicit MazeCycle(CycleConfig config)
        : config_(config), seed_(config.base_seed) {
        // A negative dwell means "advance on the next frame", which is also what zero means; folding it to zero keeps Advance() monotonic.
        config_.dwell_seconds = std::max(config_.dwell_seconds, 0.0f);
    }

    [[nodiscard]] bool Enabled() const noexcept { return config_.enabled; }
    void Disable() noexcept {
        config_.enabled = false;
        dwelling_ = false;
        dwell_elapsed_ = 0.0f;
    }

    // Seed for the maze currently on screen (or about to be generated).
    [[nodiscard]] std::uint32_t Seed() const noexcept { return seed_; }
    // How many generate -> solve cycles have completed since startup.
    [[nodiscard]] std::uint32_t CompletedCycles() const noexcept {
        return completed_cycles_;
    }
    [[nodiscard]] bool Dwelling() const noexcept { return dwelling_; }

    // The solver for the current maze reached Found/Exhausted.
    // Idempotent: a second call while already dwelling keeps the running dwell timer, so a caller that observes the terminal status every frame cannot starve it.
    void OnSolveFinished() noexcept {
        if (!config_.enabled || dwelling_) {
            return;
        }
        dwelling_ = true;
        dwell_elapsed_ = 0.0f;
    }

    // A solve started (auto-cycle, the UI's Solve button, or a reset).
    // Cancels the dwell so a stale timer cannot trigger a generation.
    void OnSolveStarted() noexcept {
        dwelling_ = false;
        dwell_elapsed_ = 0.0f;
    }

    // Advances the dwell timer.
    // Returns true exactly once, on the frame the next maze is due; Seed() then holds the next maze's seed and CompletedCycles() includes the solve just finished.
    //
    // `solver_running` keeps a solve in progress from being cut short, so the app can call this every frame without tracking transitions.
    bool Advance(float dt, bool solver_running) noexcept {
        if (!config_.enabled || solver_running || !dwelling_) {
            return false;
        }
        dwell_elapsed_ += std::max(dt, 0.0f);
        if (dwell_elapsed_ < config_.dwell_seconds) {
            return false;
        }
        dwelling_ = false;
        dwell_elapsed_ = 0.0f;
        ++completed_cycles_;
        seed_ = config_.base_seed + completed_cycles_;
        return true;
    }

private:
    CycleConfig config_{};
    std::uint32_t seed_ = 0;
    std::uint32_t completed_cycles_ = 0;
    float dwell_elapsed_ = 0.0f;
    bool dwelling_ = false;
};

} // namespace MazeGenerator
