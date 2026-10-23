module;
#include <SDL3/SDL_keycode.h>

module MazeGenerator.App;

import std;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GameEngine;
import VulkanEngine.GpuResources;
import VulkanEngine.PipelinePass;
import VulkanEngine.RenderPipeline;
import VulkanEngine.ShaderManager;

import Shaders.MazeGenerator.MazeFrag;
import Shaders.MazeGenerator.MazeFullscreenVert;

import MazeGenerator.MazeDensity;
import MazeGenerator.MazeRenderPass;
import MazeGenerator.TiledMaze;

namespace MazeGenerator::App {

namespace {

constexpr std::uint32_t kMazeWidth = 4096;
constexpr std::uint32_t kMazeHeight = 4096;
constexpr double kOpenProbability = 0.45;
constexpr float kWallThicknessCells = 0.12f;
constexpr float kInitialPixelsPerCell = 3.0f;
constexpr float kMaxPixelsPerCell = 64.0f;
constexpr float kPanPixelsPerSecond = 700.0f;
constexpr float kZoomStep = 1.2f;
constexpr float kMaxFrameDelta = 0.1f;

[[nodiscard]] std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> CreateUploadedBuffer(
    VulkanEngine::GameEngine& engine,
    VulkanBackend::Vulkan::IVulkanBootstrap& backend,
    const void* data,
    std::uint64_t size) {
    if (data == nullptr || size == 0) {
        return nullptr;
    }

    auto buffer = std::make_unique<VulkanEngine::GpuResources::GpuBuffer>(
        VulkanEngine::GpuResources::GpuBuffer::Create(
            backend, size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal));
    if (!buffer->IsValid()) {
        std::cerr << "MazeGenerator: failed to create a device-local maze buffer\n";
        return nullptr;
    }

    auto& staging_pool = engine.GetStagingPool();
    auto staging = staging_pool.Allocate(size, 256);
    if (!staging.has_value()) {
        std::cerr << "MazeGenerator: staging allocation failed (size " << size << ")\n";
        return nullptr;
    }

    std::memcpy(staging->mapped_ptr, data, static_cast<std::size_t>(size));
    staging_pool.RecordBufferCopy(*staging, *buffer->GetBuffer(), 0);
    return buffer;
}

} // namespace

MazeGeneratorApp::MazeGeneratorApp(const std::filesystem::path& executable_path)
    : exe_dir_(executable_path.parent_path()) {
    setup_token_ = hooks_.on_setup.Register([this](VulkanEngine::Application::ApplicationContext& ctx) -> bool {
        return OnSetup(ctx);
    });
    fixed_update_token_ = hooks_.on_fixed_update.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFixedUpdate(ctx);
    });
    frame_update_token_ = hooks_.on_frame_update.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFrameUpdate(ctx);
    });
    frame_render_token_ = hooks_.on_frame_render.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFrameRender(ctx);
    });
    shutdown_token_ = hooks_.on_shutdown.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnShutdown(ctx);
    });
}

MazeGeneratorApp::~MazeGeneratorApp() = default;

bool MazeGeneratorApp::OnSetup(VulkanEngine::Application::ApplicationContext& ctx) {
    VulkanEngine::GameConfig config{};
    config.enable_imgui = false;
    config.renderer_config.enable_imgui = false;
    config.renderer_config.enable_text = false;
    config.renderer_config.clear_color = {0.03f, 0.03f, 0.05f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();

    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    const auto fullscreen_vert =
        Shaders::MazeGenerator::MazeFullscreenVert::Register(shader_mgr, config.shader_data_dir);
    const auto maze_frag =
        Shaders::MazeGenerator::MazeFrag::Register(shader_mgr, config.shader_data_dir);

    const auto& shader_ids = engine_game_.GetContext().GetShaderIds();
    if (!engine_game_.InitRenderer(ctx, shader_ids.main_indir_vert, shader_ids.standard_mesh_frag,
                                   &shader_mgr)) {
        return false;
    }
    (void)engine_game_.AddShaderDirectory(config.shader_data_dir);

    maze_ = TiledMaze::CreateRandom(kMazeWidth, kMazeHeight, kOpenProbability);
    const MazeDensityData density = BuildMazeDensity(maze_);
    if (density.values.empty()) {
        std::cerr << "MazeGenerator: density pyramid is empty\n";
        return false;
    }

    const std::span<const std::byte> raw = maze_.RawBytes();
    const std::size_t expected_bytes =
        static_cast<std::size_t>(maze_.TilesX()) * maze_.TilesY() * 64;
    if (raw.size() != expected_bytes || raw.size() % 4 != 0) {
        std::cerr << "MazeGenerator: maze byte layout does not match the tile layout\n";
        return false;
    }

    auto& backend = ctx.bootstrap->GetBackend();
    auto maze_buffer = CreateUploadedBuffer(engine_game_, backend, raw.data(), raw.size());
    if (maze_buffer == nullptr) {
        return false;
    }
    auto density_buffer = CreateUploadedBuffer(
        engine_game_, backend, density.values.data(),
        static_cast<std::uint64_t>(density.values.size() * sizeof(float)));
    if (density_buffer == nullptr) {
        return false;
    }
    if (!engine_game_.GetStagingPool().FlushImmediate()) {
        std::cerr << "MazeGenerator: maze upload failed\n";
        return false;
    }

    maze_buffer_ = std::move(maze_buffer);
    density_buffer_ = std::move(density_buffer);
    density_mip_count_ = density.mip_count;

    auto& pipeline = engine_game_.GetRenderPipeline();
    pipeline.RegisterBufferResolver("maze-bits", [this](std::uint32_t) -> vk::Buffer {
        return maze_buffer_ != nullptr
                   ? static_cast<vk::Buffer>(*maze_buffer_->GetBuffer())
                   : vk::Buffer{};
    });
    pipeline.RegisterBufferResolver("maze-density", [this](std::uint32_t) -> vk::Buffer {
        return density_buffer_ != nullptr
                   ? static_cast<vk::Buffer>(*density_buffer_->GetBuffer())
                   : vk::Buffer{};
    });

    auto pass = std::make_unique<MazeRenderPass>(
        static_cast<std::uint64_t>(fullscreen_vert), static_cast<std::uint64_t>(maze_frag));
    maze_pass_ = pass.get();
    auto registered = pipeline.RegisterPass(std::move(pass));
    if (!registered.has_value()) {
        std::cerr << "MazeGenerator: failed to register the maze pass: "
                  << registered.error().message << '\n';
        maze_pass_ = nullptr;
        ReleaseGpuBuffers();
        return false;
    }

    auto& registry = engine_game_.GetContext().GetComponentRegistry();
    engine_game_.CreateCamera(registry);
    if (engine_game_.GetCamera() == nullptr) {
        std::cerr << "MazeGenerator: failed to create the camera\n";
        ReleaseGpuBuffers();
        return false;
    }

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    (void)backend.GetSwapchainExtent(width, height);
    width = std::max(width, 1u);
    height = std::max(height, 1u);

    center_x_ = 0.5f * static_cast<float>(kMazeWidth);
    center_y_ = 0.5f * static_cast<float>(kMazeHeight);
    const float fit = std::min(static_cast<float>(width) / static_cast<float>(kMazeWidth),
                               static_cast<float>(height) / static_cast<float>(kMazeHeight));
    pixels_per_cell_ = std::min(std::max(kInitialPixelsPerCell, fit), kMaxPixelsPerCell);
    FillConstants(width, height);
    SyncCamera(height);

    pan_action_ = ctx.input_system->BindAction<2>(
        "maze-pan",
        VulkanEngine::Input::ActionConfig{
            .processing = VulkanEngine::Input::ActionProcessing::ClampLength,
            .max_length = 1.0f,
        });
    pan_action_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_A).Component(0, -1.0f));
    pan_action_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_D).Component(0, 1.0f));
    pan_action_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_W).Component(1, -1.0f));
    pan_action_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_S).Component(1, 1.0f));
    zoom_action_ = ctx.input_system->BindAction(
        "maze-zoom", VulkanEngine::Input::InputBinding::MouseWheel(1));

    ctx.quit_action_handle = ctx.input_system->BindAction(
        "quit", VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));

    engine_game_.MarkSceneValid();
    return true;
}

void MazeGeneratorApp::OnFixedUpdate(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FixedUpdate(ctx);
}

void MazeGeneratorApp::OnFrameUpdate(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    if (ctx.bootstrap != nullptr) {
        (void)ctx.bootstrap->GetBackend().GetSwapchainExtent(width, height);
    }
    if (width == 0 || height == 0) {
        engine_game_.FrameUpdate(ctx);
        return;
    }

    const float dt = std::clamp(ctx.frame.delta_time, 0.0f, kMaxFrameDelta);
    float p = std::max(pixels_per_cell_, 1.0e-4f);

    if (pan_action_.Valid()) {
        const auto pan = pan_action_.Value();
        const float cells_per_second = kPanPixelsPerSecond / p;
        center_x_ += pan[0] * cells_per_second * dt;
        center_y_ += pan[1] * cells_per_second * dt;
    }

    const float fit = std::max(
        std::min(static_cast<float>(width) / static_cast<float>(kMazeWidth),
                 static_cast<float>(height) / static_cast<float>(kMazeHeight)),
        1.0e-4f);

    const float wheel = ctx.input_system != nullptr
                            ? ctx.input_system->GetActionValue(zoom_action_)
                            : 0.0f;
    if (wheel != 0.0f) {
        const float new_p =
            std::clamp(p * std::pow(kZoomStep, wheel), fit, kMaxPixelsPerCell);

        bool cursor_inside = false;
        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        if (ctx.input_system != nullptr && ctx.platform_state != nullptr &&
            ctx.platform_state->content_scale > 0.0f) {
            const auto& raw = ctx.input_system->GetRawState();
            const float scale = ctx.platform_state->content_scale;
            mouse_x = raw.mouse_x * scale;
            mouse_y = raw.mouse_y * scale;
            cursor_inside = mouse_x >= 0.0f && mouse_x < static_cast<float>(width) &&
                            mouse_y >= 0.0f && mouse_y < static_cast<float>(height);
        }

        if (cursor_inside) {
            const float cell_x = center_x_ + (mouse_x - 0.5f * static_cast<float>(width)) / p;
            const float cell_y = center_y_ + (mouse_y - 0.5f * static_cast<float>(height)) / p;
            center_x_ = cell_x - (mouse_x - 0.5f * static_cast<float>(width)) / new_p;
            center_y_ = cell_y - (mouse_y - 0.5f * static_cast<float>(height)) / new_p;
        }
        p = new_p;
    }

    p = std::max(p, fit);
    p = std::min(p, kMaxPixelsPerCell);
    pixels_per_cell_ = p;

    const float half_x = 0.5f * static_cast<float>(width) / p;
    const float half_y = 0.5f * static_cast<float>(height) / p;
    if (2.0f * half_x < static_cast<float>(kMazeWidth)) {
        center_x_ = std::clamp(center_x_, half_x, static_cast<float>(kMazeWidth) - half_x);
    } else {
        center_x_ = 0.5f * static_cast<float>(kMazeWidth);
    }
    if (2.0f * half_y < static_cast<float>(kMazeHeight)) {
        center_y_ = std::clamp(center_y_, half_y, static_cast<float>(kMazeHeight) - half_y);
    } else {
        center_y_ = 0.5f * static_cast<float>(kMazeHeight);
    }

    const float density_entry = std::max(1.0f / 3.0f, fit);
    const float density_exit = density_entry + 0.02f;
    if (tier_ == Tier::Exact) {
        if (p < 2.0f) {
            tier_ = p <= density_entry ? Tier::Density : Tier::Analytic;
        }
    } else if (tier_ == Tier::Analytic) {
        if (p >= 2.05f) {
            tier_ = Tier::Exact;
        } else if (p <= density_entry) {
            tier_ = Tier::Density;
        }
    } else if (p > density_exit) {
        tier_ = Tier::Analytic;
    }

    FillConstants(width, height);
    SyncCamera(height);
    engine_game_.FrameUpdate(ctx);
}

void MazeGeneratorApp::OnFrameRender(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void MazeGeneratorApp::OnShutdown(
    VulkanEngine::Application::ApplicationContext& /*ctx*/) {
    ReleaseGpuBuffers();
    engine_game_.Shutdown();
}

void MazeGeneratorApp::ReleaseGpuBuffers() noexcept {
    density_buffer_.reset();
    maze_buffer_.reset();
    maze_pass_ = nullptr;
}

void MazeGeneratorApp::SyncCamera(std::uint32_t height) noexcept {
    auto* camera = engine_game_.GetCamera();
    if (camera == nullptr) {
        return;
    }
    camera->projection_mode = VulkanEngine::Components::ProjectionMode::Orthographic;
    camera->position = {center_x_, center_y_, 3.0f};
    camera->target = {center_x_, center_y_, 0.0f};
    camera->up = {0.0f, 1.0f, 0.0f};
    camera->near_plane = 0.1f;
    camera->far_plane = 100.0f;
    camera->orthographic_size =
        static_cast<float>(height) / std::max(pixels_per_cell_, 1.0e-4f);
}

void MazeGeneratorApp::FillConstants(std::uint32_t width, std::uint32_t height) {
    constants_.view = {center_x_, center_y_, pixels_per_cell_, kWallThicknessCells};
    constants_.screen = {static_cast<float>(width), static_cast<float>(height),
                         static_cast<float>(static_cast<std::uint32_t>(tier_)), 4.0f};
    constants_.maze = {maze_.Width(), maze_.Height(), maze_.TilesX(), density_mip_count_};
    if (maze_pass_ != nullptr) {
        maze_pass_->SetConstants(constants_);
    }
}

} // namespace MazeGenerator::App
