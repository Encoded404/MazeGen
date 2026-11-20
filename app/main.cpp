#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import MazeGenerator.App;
import MazeGenerator.GeneratorCli;
import MazeGenerator.Generators;
import MazeGenerator.Solvers;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    MazeGenerator::App::StartupOptions options{};

    // One settings value per generator, created before the CLI options bind to it.
    // CLI11 stores references into these `std::any` objects, so the vector is reserved once here and never reallocated until parsing is finished.
    std::vector<std::any> generator_settings;
    generator_settings.reserve(std::size(MazeGenerator::kMazeGenerators));
    for (const MazeGenerator::MazeGeneratorType& generator :
         MazeGenerator::kMazeGenerators) {
        generator_settings.push_back(generator.make_defaults());
    }

    std::vector<std::string> generator_names;
    generator_names.reserve(std::size(MazeGenerator::kMazeGenerators));
    for (const MazeGenerator::MazeGeneratorType& generator :
         MazeGenerator::kMazeGenerators) {
        generator_names.emplace_back(generator.name);
    }

    std::vector<std::string> solver_names;
    solver_names.reserve(std::size(MazeGenerator::kSolverTypes));
    for (const MazeGenerator::SolverType& solver : MazeGenerator::kSolverTypes) {
        solver_names.emplace_back(solver.name);
    }

    std::string generator_name = generator_names.front();
    std::string solver_name = solver_names.front();

    cli.App()
        .add_option("--generator", generator_name, "Maze generator to use")
        ->check(CLI::IsMember(generator_names))
        ->capture_default_str();
    cli.App()
        .add_option("--solver", solver_name, "Maze solver to use")
        ->check(CLI::IsMember(solver_names))
        ->capture_default_str();

    cli.App()
        .add_option("--width", options.width, "Initial maze width in cells")
        ->type_name("N")
        ->check(CLI::Range(MazeGenerator::App::kMinMazeSide,
                           MazeGenerator::App::kMaxMazeSide))
        ->capture_default_str();
    cli.App()
        .add_option("--height", options.height, "Initial maze height in cells")
        ->type_name("N")
        ->check(CLI::Range(MazeGenerator::App::kMinMazeSide,
                           MazeGenerator::App::kMaxMazeSide))
        ->capture_default_str();
    cli.App()
        .add_option("--seed", options.seed,
                    "Seed for the first maze; each cycle advances it")
        ->type_name("N")
        ->capture_default_str();

    cli.App()
        .add_option("--steps-per-frame", options.steps_per_frame,
                    "Solver expansions per frame (the maximum means unlimited)")
        ->type_name("N")
        ->check(CLI::Range(1, MazeGenerator::App::kMaxStepsPerFrame))
        ->capture_default_str();

    cli.App()
        .add_flag("--reset-view-on-generate", options.reset_view_on_generate,
                  "Recenter and re-fit the view after each maze generation");
    cli.App()
        .add_flag("--cycle", options.cycle,
                  "Keep cycling: generate a maze, solve it, hold the result, then repeat");
    cli.App()
        .add_option("--cycle-dwell", options.cycle_dwell_seconds,
                    "Seconds to hold a finished solve on screen before the next maze")
        ->type_name("S")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();

    // Every registry setting becomes --<generator>-<key>, with its range and help text, e.g. --random-open-probability 0.6.
    MazeGenerator::RegisterGeneratorOptions(
        cli.App(), MazeGenerator::kMazeGenerators, generator_settings);

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    // The membership checks above already guarantee these succeed; reaching the error means the registry and the validator disagree.
    const std::optional<std::size_t> generator_index =
        MazeGenerator::FindGeneratorIndex(generator_name);
    const std::optional<std::size_t> solver_index =
        MazeGenerator::FindSolverIndex(solver_name);
    if (!generator_index.has_value() || !solver_index.has_value()) {
        std::cerr << "MazeGenerator: unknown generator or solver\n";
        return 2;
    }
    options.generator_index = *generator_index;
    options.solver_index = *solver_index;
    options.generator_settings = std::move(generator_settings);

    auto app = std::make_unique<MazeGenerator::App::MazeGeneratorApp>(
        cli.ExecutablePath(), std::move(options));

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), app->GetHooks());
}
