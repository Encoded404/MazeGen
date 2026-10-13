module;

#include <CLI/CLI.hpp>

#include <logging/logging_macros.hpp>

module Runtime.ExampleCli;

import std;

import logiface;

import VulkanEngine.Application;
import VulkanEngine.Startup;
import Runtime.Overrides;

namespace Runtime {

Cli::Cli(std::string_view app_name, std::string_view org_id, std::string_view app_id)
    : app_(std::string{app_name})
    , app_name_(app_name)
    , org_id_(org_id)
    , app_id_(app_id) {
    // CLI11 paragraph-wraps the footer by default, which strips leading
    // whitespace and collapses multi-space runs — destroying the aligned
    // overrides table. Turn footer formatting off; everything else matches
    // the default Formatter.
    auto formatter = std::make_shared<CLI::Formatter>();
    formatter->enable_footer_formatting(false);
    app_.formatter(formatter);

    app_.add_option("-l,--log-level", log_level_,
                    "Console output log level (trace, debug, info, warn, error, critical)")
       ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical"}));

    app_.add_flag("--validation", force_validation_, "Enable the Vulkan validation layer");
    app_.add_flag("--no-validation", force_no_validation_, "Disable the Vulkan validation layer");

    // Deterministic exit for automated smoke runs; 0 keeps the interactive loop.
    app_.add_option("--max-frames", max_frames_,
                    "Exit cleanly after this many rendered frames (0 = run until quit)")
       ->type_name("N");

    // Simulation cadence. 0 keeps the variable-step loop, which is the default:
    // on_fixed_update then runs once per frame carrying the variable delta.
    app_.add_option("--fixed-timestep", fixed_timestep_hz_,
                    "Run on_fixed_update at this rate in Hz (0 = variable step)")
       ->type_name("HZ");

    // Storage options. Everything the application writes (settings, saves,
    // pipeline caches, logs) lives under the resolved root; these only decide
    // where that root is. --user-dir wins over --portable.
    app_.add_option("--user-dir", user_dir_,
                    "Store settings, saves, cache and logs under this directory instead of the "
                    "per-user platform location (also VKENGINE_USER_DIR)")
       ->type_name("DIR");
    app_.add_flag("--portable", force_portable_,
                  "Store user data next to the executable (implies a writable install directory)");
    app_.add_flag("--no-portable", force_no_portable_,
                  "Ignore a portable marker next to the executable");

    app_.add_option("--overwrite", overwrite_values_,
                    "Testing overrides; run --help for the full key list")
       ->transform([this](std::string kv) {            // CLI11 2.5.0: the transform's
           try {                                       // return value REPLACES the stored
               overrides_.Apply(kv);                   // value, so errors must be
               return kv;                              // thrown, not returned
           } catch (const OverrideError& e) {
               throw CLI::ValidationError(std::string{e.what()});
           }
       });

    app_.add_option("--force-disable-ext", force_disabled_extensions_,
                    "[deprecated] Force-disable a Vulkan extension for testing fallback paths; use --overwrite ext.disable=<name>");
}

void Cli::RegisterOverride(OverrideSpec spec) {
    overrides_.Register(std::move(spec));
}

bool Cli::Parse(int argc, char* const argv[]) {
    executable_path_ = std::filesystem::absolute(std::filesystem::path(argv[0]));

    // Help fires during parse, so the footer must be set before; the pristine
    // registry makes "default" and "current" show the programmatic defaults.
    app_.footer(overrides_.GenerateHelp());

    try {
        app_.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        exit_code_ = app_.exit(e);
        return false;
    }
    exit_code_ = 0;

    // Fold the deprecated flag into the registry sink, then log the effective
    // overrides BEFORE any move leaves the vector empty. The logger is
    // initialized here (idempotent; RunApplication re-applies the level).
    overrides_.force_disabled_extensions.insert(overrides_.force_disabled_extensions.end(),
                                               force_disabled_extensions_.begin(),
                                               force_disabled_extensions_.end());
    VulkanEngine::Startup::InitializeLogger(log_level_);
    LOGIFACE_LOG(info, "effective overrides: " + overrides_.DescribeEffective());

    return true;
}

VulkanEngine::Application::ApplicationConfig Cli::MakeConfig() const {
    VulkanEngine::Application::ApplicationConfig config{};
    config.app_name = app_name_;
    config.org_id = org_id_;
    config.app_id = app_id_;
    config.executable_path = executable_path_;
    config.user_dir = user_dir_;
    config.force_portable = force_portable_;
    config.disable_portable = DisablePortable();
    config.log_level = log_level_;
    config.max_frames = max_frames_;
    // Hz at the CLI, seconds in the config: the loop integrates seconds.
    config.fixed_timestep =
        fixed_timestep_hz_ > 0.0 ? static_cast<float>(1.0 / fixed_timestep_hz_) : 0.0f;
    // The config's default is build-type aware (debug: on, optimized: off);
    // explicit CLI flags override it in either direction. Both given: --validation wins.
    config.bootstrap_config.enable_validation =
        force_validation_ ||
        (config.bootstrap_config.enable_validation && !force_no_validation_);
    config.bootstrap_config.force_disabled_extensions = overrides_.force_disabled_extensions;
    config.draw_mode = overrides_.draw_mode;
    return config;
}

} // namespace Runtime