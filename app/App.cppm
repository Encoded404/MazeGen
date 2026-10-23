module;

export module MazeGenerator.App;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuResources;
import VulkanEngine.Input;
import VulkanEngine.PipelinePass;

import MazeGenerator.DebugGrid;
import MazeGenerator.DebugOverlayPass;
import MazeGenerator.Generation;
import MazeGenerator.Generators;
import MazeGenerator.MazeRenderPass;
import MazeGenerator.Solver;
import MazeGenerator.TiledMaze;

export namespace MazeGenerator::App {

// Fullscreen maze viewer with a debug overlay and incremental solvers:
//   * one TiledMaze, rendered from its packed wall bits at three LoD tiers;
//   * one debug overlay pass reading the solver's per-cell state;
//   * an ImGui panel that regenerates the maze, drives the solver, and toggles
//     the debug layers.
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

    // The UI's editable state. The ImGui callback runs during rendering, so it
    // only writes these fields; ApplyUiRequests() consumes the pending flags at
    // the start of the next OnFrameUpdate.
    struct UiState {
        int width = 4096;
        int height = 4096;
        int seed = 1234;
        int generator_index = 0;
        int solver_index = 0;
        int steps_per_frame = 500;
        bool debug_draw = true;
        unsigned debug_categories = 0x7u; // open | closed | path
        bool pending_generate = false;
        bool pending_solve = false;
        bool pending_stop = false;
        bool pending_reset = false;
    };

    bool OnSetup(VulkanEngine::Application::ApplicationContext& ctx);
    void OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnShutdown(VulkanEngine::Application::ApplicationContext& ctx);

    void DrawUi();
    void DrawSizeField(const char* label, int& value);
    void ApplyUiRequests(const VulkanEngine::Application::ApplicationContext& ctx);
    void RegenerateMaze(const VulkanEngine::Application::ApplicationContext& ctx);
    void ResetSolver();
    void StartSolver();
    void AdvanceSolver();
    void FindOpenings();
    void ResetViewForMaze();

    void FlushDebugGrid();
    bool CreateDebugBuffers(VulkanBackend::Vulkan::IVulkanBootstrap& backend);
    void ReleaseDebugBuffers() noexcept;
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

    // Maze + generation.
    TiledMaze maze_{};
    std::array<MazeGenSettings, std::size(kMazeGenerators)> gen_settings_{};
    std::uint32_t density_mip_count_ = 0;
    // Largest side the selected device can bind for the debug cell buffer,
    // derived once at setup from its storage-range/allocation limits.
    std::uint32_t max_maze_size_ = 4096;
    UiState ui_{};

    // View.
    float center_x_ = 0.0f;
    float center_y_ = 0.0f;
    float pixels_per_cell_ = 3.0f;
    Tier tier_ = Tier::Exact;
    std::uint32_t viewport_width_ = 1;
    std::uint32_t viewport_height_ = 1;

    // Solver + debug state.
    std::unique_ptr<IMazeSolver> solver_{};
    SolverStatus solver_status_ = SolverStatus::Running;
    SolverStats solver_stats_{};
    bool solver_running_ = false;
    CellRef start_{};
    CellRef goal_{};
    DebugGrid debug_grid_{};

    VulkanEngine::Input::Action<2> pan_action_{};
    VulkanEngine::Input::ActionHandle zoom_action_{};

    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> maze_buffer_{};
    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> density_buffer_{};
    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> debug_cells_buffer_{};
    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> debug_blocks_buffer_{};
    std::uint8_t* debug_cells_mapped_ = nullptr;
    std::uint32_t* debug_blocks_mapped_ = nullptr;

    MazeRenderPass* maze_pass_ = nullptr;
    DebugOverlayPass* debug_pass_ = nullptr;
    VulkanEngine::RenderGraph::PassHandle maze_pass_handle_{};
    VulkanEngine::RenderGraph::PassHandle debug_pass_handle_{};
    bool debug_pass_enabled_ = true;
    MazeConstants constants_{};
    DebugConstants debug_constants_{};

    VulkanShared::ScopedHandle<void()> ui_token_{};
};

} // namespace MazeGenerator::App
