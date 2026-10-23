module;

export module Examples.Minimal.Game;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

export namespace Examples::Minimal::Game {

// Smallest possible engine example: setup engine, register one shader, render
// one mesh with a camera. The primary app is app/; this is a compact reference
// for adding more examples.
class Game {
public:
    explicit Game(const std::filesystem::path& executable_path);
    ~Game();

    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

    [[nodiscard]] const VulkanEngine::Application::ApplicationHooks& GetHooks() const { return hooks_; }

private:
    bool OnSetup(VulkanEngine::Application::ApplicationContext& ctx);
    void OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnShutdown(VulkanEngine::Application::ApplicationContext& ctx);

    VulkanEngine::Application::ApplicationHooks hooks_{};

    VulkanShared::ScopedHandle<bool(VulkanEngine::Application::ApplicationContext&)> setup_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> fixed_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_render_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> shutdown_token_{};

    std::filesystem::path exe_dir_{};
    VulkanEngine::GameEngine engine_game_{};
};

} // namespace Examples::Minimal::Game
