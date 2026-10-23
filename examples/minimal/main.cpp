#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Examples.Minimal.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator Minimal", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    auto game = std::make_unique<Examples::Minimal::Game::Game>(cli.ExecutablePath());

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), game->GetHooks());
}
