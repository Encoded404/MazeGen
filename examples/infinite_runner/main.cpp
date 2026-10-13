#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Examples.InfiniteRunner.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    // Runtime::Cli is a engine wrapper around CLI11 which provides a simple way to parse command line arguments.
    Runtime::Cli cli{"VulkanEngineV5 Infinite Runner", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    std::string leaderboard_host;
    cli.App().add_option("--leaderboard-host", leaderboard_host,
                         "Override the sealed leaderboard host (IP or hostname)");

    int leaderboard_port = 0;
    cli.App().add_option("--leaderboard-port", leaderboard_port,
                         "Override the sealed leaderboard port (0 keeps the sealed value)")
    // the value must be in the valid port range.
       ->check(CLI::Range(0, 65535));

    // if parsing fails, exit with the appropriate error code
    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    // the infinite runner's network endpoint
    Examples::InfiniteRunner::Game::EndpointOverride endpoint{};
    endpoint.host = std::move(leaderboard_host);
    endpoint.port = static_cast<std::uint16_t>(leaderboard_port);

    // create the game instance
    auto game = std::make_unique<Examples::InfiniteRunner::Game::Game>(cli.ExecutablePath(),
                                                                       std::move(endpoint));

    // use the engine side game runner to run the game.
    // this is how all the game instance's functions like OnFrameUpdate are called.
    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), game->GetHooks());
}
