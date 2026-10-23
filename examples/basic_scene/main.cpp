#include <CLI/CLI.hpp>

#include "engine/core/bootstrap/EntryPoint.hpp"

import std;

import Runtime.Application;
import Runtime.ExampleCli;
import Runtime.Overrides;
import VulkanEngine.Application;
import Examples.BasicScene.Game;

int VulkanEngine::AppMain(int argc, char* const argv[]) {
    Runtime::Cli cli{"MazeGenerator Basic Scene", VKENGINE_ORG_ID, VKENGINE_APP_ID};

    Examples::BasicScene::Game::RenderMode render_mode = Examples::BasicScene::Game::RenderMode::Normal;
    cli.App().add_option("-m,--mode", render_mode, "Rendering mode")
       ->transform(CLI::CheckedTransformer(std::map<std::string, Examples::BasicScene::Game::RenderMode>{
           {"normal", Examples::BasicScene::Game::RenderMode::Normal},
           {"normals", Examples::BasicScene::Game::RenderMode::Normals},
           {"no-textures", Examples::BasicScene::Game::RenderMode::NoTextures}
       }, CLI::ignore_case));

    std::string model_path;
    cli.App().add_option("-M,--model", model_path, "Path to a .bin model file to load");

    std::string texture_path;
    cli.App().add_option("-T,--texture", texture_path, "Path to a texture file to load (.png, .jpg, .ktx)");

    if (!cli.Parse(argc, argv)) {
        return cli.ExitCode();
    }

    auto game = std::make_unique<Examples::BasicScene::Game::DemoGame>(
        render_mode, cli.ExecutablePath(),
        model_path.empty() ? std::filesystem::path{} : std::filesystem::path{model_path},
        texture_path.empty() ? std::filesystem::path{} : std::filesystem::path{texture_path},
        cli.GetOverrides().gpl_policy, cli.GetOverrides().gpl_structure);

    return VulkanEngine::Application::RunApplication(cli.MakeConfig(), game->GetHooks());
}
