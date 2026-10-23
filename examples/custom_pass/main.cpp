#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Examples.CustomPass.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator Custom Pass", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    auto game = std::make_unique<Examples::CustomPass::Game::CustomPassGame>(cli.ExecutablePath());

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), game->GetHooks());
}
