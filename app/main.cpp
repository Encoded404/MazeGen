#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import MazeGenerator.App;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    auto app = std::make_unique<MazeGenerator::App::MazeGeneratorApp>(cli.ExecutablePath());

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), app->GetHooks());
}
