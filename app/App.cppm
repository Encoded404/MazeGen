module;

export module MazeGenerator.App;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

import VulkanEngine.GpuResources;
import VulkanEngine.Input;

import MazeGenerator.MazeRenderPass;
import MazeGenerator.TiledMaze;

export namespace MazeGenerator::App {

// Fullscreen maze viewer: one random TiledMaze, one render pass reading the
// packed wall bits, and a 2D pan/zoom view.
class MazeGeneratorApp {
public:
    explicit MazeGeneratorApp(const std::filesystem::path& executable_path);
    ~MazeGeneratorApp();

    MazeGeneratorApp(const MazeGeneratorApp&) = delete;
    MazeGeneratorApp& operator=(const MazeGeneratorApp&) = delete;

    [[nodiscard]] const VulkanEngine::Application::ApplicationHooks& GetHooks() const { return hooks_; }

private:
    enum class Tier : std::uint8_t {
        Exact = 0,
        Analytic = 1,
        Density = 2,
    };

    bool OnSetup(VulkanEngine::Application::ApplicationContext& ctx);
    void OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnShutdown(VulkanEngine::Application::ApplicationContext& ctx);

    void ReleaseGpuBuffers() noexcept;
    void SyncCamera(std::uint32_t height) noexcept;
    void FillConstants(std::uint32_t width, std::uint32_t height);

    VulkanEngine::Application::ApplicationHooks hooks_{};

    VulkanShared::ScopedHandle<bool(VulkanEngine::Application::ApplicationContext&)> setup_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> fixed_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_render_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> shutdown_token_{};

    std::filesystem::path exe_dir_{};
    VulkanEngine::GameEngine engine_game_{};

    TiledMaze maze_{};
    float center_x_ = 0.0f;
    float center_y_ = 0.0f;
    float pixels_per_cell_ = 3.0f;
    Tier tier_ = Tier::Exact;
    std::uint32_t density_mip_count_ = 0;

    VulkanEngine::Input::Action<2> pan_action_{};
    VulkanEngine::Input::ActionHandle zoom_action_{};

    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> maze_buffer_{};
    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> density_buffer_{};
    MazeRenderPass* maze_pass_ = nullptr;
    MazeConstants constants_{};
};

} // namespace MazeGenerator::App
