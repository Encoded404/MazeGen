#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Examples.PhysicsVehicle.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"VulkanEngineV5 Physics Vehicle", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    auto game = std::make_unique<Examples::PhysicsVehicle::Game::Game>(cli.ExecutablePath());

    // The vehicle rides on the engine's fixed-step frame contract: its
    // suspension and tires run in the physics systems of the fixed update. A
    // stable simulation rate is required, so default to 60 Hz unless the user
    // supplied one.
    auto config = cli.MakeConfig();
    if (config.fixed_timestep <= 0.0f) {
        config.fixed_timestep = 1.0f / 60.0f;
    }

    return VulkanEngine::Application::RunApplication(config, game->GetHooks());
}