module;

#include <CLI/CLI.hpp>

export module Runtime.ExampleCli;

import std;

import VulkanEngine.Application;
import Runtime.Overrides;

export namespace Runtime {

// CLI builder for example executables.
//
// The runtime owns the CLI11 app, the --help formatter, all engine-standard
// options (-l/--log-level, --validation/--no-validation, --overwrite,
// --force-disable-ext) and the construction of ApplicationConfig. Examples
// register their own options on App() and their own --overwrite keys via
// RegisterOverride(), then call Parse() and build their game with the parsed
// values.
class Cli {
public:
    // `app_name` is the user-facing title (window and --help). `org_id` and
    // `app_id` are the storage identity: two path components under the per-user
    // root, which must stay stable across releases. build systems normally
    // supply them (see add_engine_example) so that every executable gets its own
    // directory; passing the title as app_id would make all of them share one.
    explicit Cli(std::string_view app_name, std::string_view org_id, std::string_view app_id);

    // Register an example-defined --overwrite key. Must be called before
    // Parse() so the option validation and the --help footer see it.
    void RegisterOverride(OverrideSpec spec);

    // Underlying CLI11 app — register example-specific options on it.
    CLI::App& App() { return app_; }

    // Parses argv; returns false when the app should exit (--help, bad args).
    // On failure, ExitCode() holds the process return code.
    bool Parse(int argc, char* const argv[]);
    [[nodiscard]] int ExitCode() const { return exit_code_; }

    [[nodiscard]] const Overrides& GetOverrides() const { return overrides_; }
    [[nodiscard]] const std::filesystem::path& ExecutablePath() const { return executable_path_; }

    // Effective validation choice after explicit CLI overrides: true only when
    // the user passed --no-validation and did not also pass --validation.
    [[nodiscard]] bool NoValidation() const {
        return force_no_validation_ && !force_validation_;
    }

    // Effective portable-mode choice: --portable wins when both flags are given.
    [[nodiscard]] bool DisablePortable() const {
        return force_no_portable_ && !force_portable_;
    }

    // Frames to render before exiting cleanly; 0 means run until the user quits.
    [[nodiscard]] std::uint32_t MaxFrames() const { return max_frames_; }

    // Application config assembled from the standard options and the overrides.
    [[nodiscard]] VulkanEngine::Application::ApplicationConfig MakeConfig() const;

private:
    CLI::App app_{};
    std::string app_name_{};
    std::string org_id_{};
    std::string app_id_{};
    Overrides overrides_{};
    std::vector<std::string> overwrite_values_{};
    std::vector<std::string> force_disabled_extensions_{};
    std::string log_level_ = "info";
    std::string user_dir_{};
    std::uint32_t max_frames_ = 0;
    // Wall-clock frame profiling (--frame-profile et al.).
    bool frame_profile_ = false;
    double frame_profile_long_ms_ = 20.0;
    std::uint64_t frame_profile_every_ = 600;
    // Per-pass GPU timestamps (--gpu-profile et al.).
    bool gpu_profile_ = false;
    std::uint64_t gpu_profile_every_ = 240;
    // Fixed simulation rate in Hz; 0 keeps the variable-step default.
    double fixed_timestep_hz_ = 0.0;
    bool force_validation_ = false;
    bool force_no_validation_ = false;
    bool force_portable_ = false;
    bool force_no_portable_ = false;
    int exit_code_ = 0;
    std::filesystem::path executable_path_{};
};

} // namespace Runtime