module;

export module MazeGenerator.App;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

export namespace MazeGenerator::App {

// Smallest possible MazeGenerator app: set up the engine, register one shader,
// render one mesh with a camera. Maze generation and gameplay hang off the
// update/render hooks below.
class MazeGeneratorApp {
public:
    explicit MazeGeneratorApp(const std::filesystem::path& executable_path);
    ~MazeGeneratorApp();

    MazeGeneratorApp(const MazeGeneratorApp&) = delete;
    MazeGeneratorApp& operator=(const MazeGeneratorApp&) = delete;

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

} // namespace MazeGenerator::App
