#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Examples.TextDemo.Config;
import Examples.TextDemo.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator Text Demo", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    // The world-text backend is the one choice the example makes at startup.
    // msdf is the engine default; slug selects the analytic hb-gpu rasterizer.
    std::string backend_name = "msdf";
    cli.App()
        .add_option("--text-backend", backend_name,
                    "World-text rasterizer: 'msdf' (default) or 'slug'")
        ->default_val("msdf");

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    // ParseTextBackend is the single place the CLI spelling becomes the engine
    // enum, and it is what the device-free test pins.
    const std::optional<VulkanEngine::SceneRenderer::TextBackend> backend =
        Examples::TextDemo::ParseTextBackend(backend_name);
    if (!backend.has_value()) {
        std::cerr << "text_demo: unknown --text-backend '" << backend_name
                  << "' (expected 'msdf' or 'slug')\n";
        return 2;
    }

    auto game = std::make_unique<Examples::TextDemo::TextDemoGame>(*backend, cli.ExecutablePath());

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), game->GetHooks());
}
