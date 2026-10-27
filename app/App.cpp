module;
#include <SDL3/SDL_keycode.h>
#include <imgui.h>
#include <logging/logging_macros.hpp>

module MazeGenerator.App;

import std;

import logiface;
import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GameEngine;
import VulkanEngine.GpuResources;
import VulkanEngine.PipelinePass;
import VulkanEngine.RenderPipeline;
import VulkanEngine.ShaderManager;
import VulkanShared.Timer;

import Shaders.MazeGenerator.DebugOverlayFrag;
import Shaders.MazeGenerator.MazeFrag;
import Shaders.MazeGenerator.MazeFullscreenVert;

import MazeGenerator.DebugGrid;
import MazeGenerator.DebugOverlayPass;
import MazeGenerator.Generation;
import MazeGenerator.Generators;
import MazeGenerator.MazeDensity;
import MazeGenerator.MazeRenderPass;
import MazeGenerator.Solver;
import MazeGenerator.Solvers;
import MazeGenerator.TiledMaze;

namespace MazeGenerator::App {

namespace {

constexpr std::uint32_t kDefaultMazeSize = 4096;
constexpr std::uint32_t kMinMazeSize = 64;
// Soft limit: dragging a size field stops here. Ctrl+Click (or double-click)
// text entry can go past it, up to the device-derived hard cap.
constexpr std::uint32_t kDragMaxMazeSize = 4096;
// 65536^2 is exactly 4 GiB, one byte past the largest possible storage-buffer
// binding range (uint32 max), so no side can exceed 65535.
constexpr std::uint32_t kAbsoluteMaxMazeSize = 65535;
// Above this the synchronous generator allocates several GiB and can take
// minutes; the UI still allows it, but warns.
constexpr std::uint32_t kLargeMazeWarningSize = 16384;
constexpr float kWallThicknessCells = 0.12f;
constexpr float kInitialPixelsPerCell = 3.0f;
constexpr float kMaxPixelsPerCell = 64.0f;
// Full zoom-out sits this many times further out than the fit-to-window zoom,
// so the entire maze is visible and covers 1 / kMaxZoomOutFactor of the
// limiting screen axis (2/3 at 1.5), with the remaining third split into
// padding on both sides. `fit` picks the limiting axis: a square maze on 16:9
// is padded on height, a 16K x 1K maze on the same screen on width.
constexpr float kMaxZoomOutFactor = 1.5f;
static_assert(kMaxZoomOutFactor >= 1.0f,
              "full zoom-out cannot be tighter than the fit-to-window zoom");
constexpr float kPanPixelsPerSecond = 700.0f;
// Panning may push the view past a maze edge by at most this fraction of the
// viewport, so that much background can show on one side at the extreme.
constexpr float kPanOverscrollFraction = 0.4f;
constexpr float kZoomStep = 1.2f;
constexpr float kMaxFrameDelta = 0.1f;
// "Steps / frame" slider: the far-right position is a sentinel for an
// unlimited expansion budget, so the largest finite value sits just below it.
constexpr int kMaxStepsPerFrame = 500000;
constexpr std::uint32_t kDebugCategoryOpen = 1u;
constexpr std::uint32_t kDebugCategoryClosed = 2u;
constexpr std::uint32_t kDebugCategoryPath = 4u;

// Clamps one camera axis so no more than kPanOverscrollFraction of the viewport
// shows background past a maze edge. `half_view` is half the viewport and
// `extent` the maze's size, both in cells. Both axes use this same rule at
// every zoom, so neither direction is pinned while the other pans. When the
// viewport dwarfs the maze the two caps cannot hold together; centering is the
// limit of the range closing, so the rule stays continuous.
void ClampAxisCenter(float& center, float half_view, float extent) {
    const float margin = kPanOverscrollFraction * 2.0f * half_view;
    const float lo = half_view - margin;
    const float hi = extent - half_view + margin;
    if (lo <= hi) {
        center = std::clamp(center, lo, hi);
    } else {
        center = 0.5f * extent;
    }
}

// The debug grid stores one byte per cell in a single storage buffer, so a
// width x height maze needs width*height bytes in one binding. Bound each side
// by the device's largest bindable storage range, largest single allocation,
// and largest buffer, with the host-visible heap (where the debug buffers live)
// supplying headroom. Round down to a whole 16x16 tile.
[[nodiscard]] std::uint32_t ComputeMaxMazeSize(
    VulkanBackend::Vulkan::IVulkanBootstrap& backend) {
    const auto& physical_device = backend.GetPhysicalDevice();
    const auto props = physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMaintenance3Properties,
        vk::PhysicalDeviceMaintenance4Properties>();
    const auto& limits = props.get<vk::PhysicalDeviceProperties2>().properties.limits;
    const auto& maintenance3 = props.get<vk::PhysicalDeviceMaintenance3Properties>();
    const auto& maintenance4 = props.get<vk::PhysicalDeviceMaintenance4Properties>();

    std::uint64_t cell_bytes = limits.maxStorageBufferRange;
    if (maintenance3.maxMemoryAllocationSize != 0) {
        cell_bytes = std::min(cell_bytes, maintenance3.maxMemoryAllocationSize);
    }
    if (maintenance4.maxBufferSize != 0) {
        cell_bytes = std::min(cell_bytes, maintenance4.maxBufferSize);
    }

    // The cell buffer and the 8x8 block summary are host-visible and written
    // every frame by the CPU. Give the cells at most half of the largest
    // host-visible heap so the maze/density buffers and the rest of the frame
    // still have room.
    const auto memory = physical_device.getMemoryProperties();
    const vk::MemoryPropertyFlags host_visible =
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent;
    std::uint64_t host_heap_bytes = 0;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((memory.memoryTypes[i].propertyFlags & host_visible) == host_visible) {
            const auto& heap = memory.memoryHeaps[memory.memoryTypes[i].heapIndex];
            host_heap_bytes = std::max(host_heap_bytes,
                                       static_cast<std::uint64_t>(heap.size));
        }
    }
    if (host_heap_bytes > 0) {
        cell_bytes = std::min(cell_bytes, host_heap_bytes / 2);
    }

    std::uint64_t side = static_cast<std::uint64_t>(
        std::sqrt(static_cast<double>(cell_bytes)));
    // The square root of a huge value can round up by one; walk back until the
    // product provably fits.
    while (side > 0 && side * side > cell_bytes) {
        --side;
    }
    side = std::min(side, static_cast<std::uint64_t>(kAbsoluteMaxMazeSize));
    side -= side % 16;
    return static_cast<std::uint32_t>(
        std::max(side, static_cast<std::uint64_t>(kMinMazeSize)));
}

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
        std::cerr << "MazeGenerator: failed to create a device-local buffer\n";
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

// A persistently mapped host-visible storage buffer. The debug grid is a debug
// surface: it is written by the CPU and read by the overlay shader, so the
// host-visible path avoids staging copies and barriers entirely.
[[nodiscard]] std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> CreateMappedBuffer(
    VulkanBackend::Vulkan::IVulkanBootstrap& backend,
    std::uint64_t size,
    void*& mapped) {
    mapped = nullptr;
    if (size == 0) {
        return nullptr;
    }

    auto buffer = std::make_unique<VulkanEngine::GpuResources::GpuBuffer>(
        VulkanEngine::GpuResources::GpuBuffer::Create(
            backend, size,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent));
    if (!buffer->IsValid()) {
        return nullptr;
    }
    mapped = buffer->Map(0, size);
    if (mapped == nullptr) {
        return nullptr;
    }
    return buffer;
}

[[nodiscard]] const char* SolverStatusLabel(SolverStatus status) {
    switch (status) {
        case SolverStatus::Running: return "running";
        case SolverStatus::Found: return "found";
        case SolverStatus::Exhausted: return "no path";
    }
    return "unknown";
}

// ImGui's Combo getter for EnumOption spans. `data` points at the span passed
// to Enum() and only lives for the duration of the Combo call.
[[nodiscard]] const char* EnumOptionsGetter(void* data, int index) {
    const auto* options = static_cast<const std::span<const EnumOption>*>(data);
    if (options == nullptr || index < 0 ||
        index >= static_cast<int>(options->size())) {
        return nullptr;
    }
    return (*options)[static_cast<std::size_t>(index)].label;
}

// Renders a generator's settings through ImGui. This is the only place the
// settings system touches ImGui; the generator modules stay UI-free and their
// settings remain plain structs.
class ImGuiSettingsVisitor final : public SettingsVisitor {
public:
    void Float(const char* label, float& value, float min, float max,
               const char* format, FieldHint hint) override {
        ImGui::SetNextItemWidth(240.0f);
        switch (hint) {
            case FieldHint::Slider:
                ImGui::SliderFloat(label, &value, min, max, format);
                break;
            case FieldHint::Drag:
                ImGui::DragFloat(label, &value, 0.01f, min, max, format);
                break;
            case FieldHint::Input:
                ImGui::InputFloat(label, &value, 0.0f, 0.0f, format);
                break;
        }
    }

    void Int(const char* label, int& value, int min, int max,
             FieldHint hint) override {
        ImGui::SetNextItemWidth(240.0f);
        switch (hint) {
            case FieldHint::Slider:
                ImGui::SliderInt(label, &value, min, max);
                break;
            case FieldHint::Drag:
                ImGui::DragInt(label, &value, 1.0f, min, max);
                break;
            case FieldHint::Input:
                ImGui::InputInt(label, &value);
                break;
        }
    }

    void Bool(const char* label, bool& value) override {
        ImGui::Checkbox(label, &value);
    }

    void Enum(const char* label, int& value,
              std::span<const EnumOption> options) override {
        int index = 0;
        for (int i = 0; i < static_cast<int>(options.size()); ++i) {
            if (options[static_cast<std::size_t>(i)].value == value) {
                index = i;
                break;
            }
        }
        ImGui::SetNextItemWidth(240.0f);
        if (ImGui::Combo(label, &index, &EnumOptionsGetter, &options,
                         static_cast<int>(options.size())) &&
            index >= 0 && index < static_cast<int>(options.size())) {
            value = options[static_cast<std::size_t>(index)].value;
        }
    }

    void Path(const char* label, std::string& value) override {
        std::array<char, 1024> buffer{};
        const std::size_t count = std::min(value.size(), buffer.size() - 1);
        std::memcpy(buffer.data(), value.data(), count);
        ImGui::SetNextItemWidth(240.0f);
        if (ImGui::InputText(label, buffer.data(), buffer.size())) {
            value = buffer.data();
        }
    }

    bool BeginGroup(const char* label, bool default_open) override {
        const ImGuiTreeNodeFlags flags =
            default_open ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None;
        if (!ImGui::CollapsingHeader(label, flags)) {
            return false;
        }
        ImGui::PushID(label);
        ImGui::Indent();
        return true;
    }

    void EndGroup() override {
        ImGui::Unindent();
        ImGui::PopID();
    }

    void Help(std::string_view text) override {
        if (!text.empty() && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%.*s", static_cast<int>(text.size()), text.data());
        }
    }
};

// Runs a registry generator and logs how long it took, tagging the entry with
// the generator's own name. This lives in the app instead of the generator
// module so generators stay free of Vulkan/logging dependencies and remain
// usable from the device-free tests.
[[nodiscard]] std::optional<TiledMaze> GenerateMazeLogged(
    const MazeGeneratorType& generator, const std::any& settings,
    const MazeSpec& spec) {
    VulkanShared::Timer timer(true);
    std::optional<TiledMaze> maze = generator.generate(spec, settings);
    timer.Stop();
    if (maze.has_value()) {
        LOGIFACE_LOG(info, std::format("{} made {}x{} maze in {:.1f} ms",
                                       generator.name, spec.width, spec.height,
                                       timer.ElapsedMs()));
    }
    return maze;
}

} // namespace

MazeGeneratorApp::MazeGeneratorApp(const std::filesystem::path& executable_path)
    : exe_dir_(executable_path.parent_path()) {
    // Start every generator with its own defaults; the registry manufactures
    // the concrete settings type so the app only stores type-erased values.
    for (std::size_t i = 0; i < std::size(kMazeGenerators); ++i) {
        gen_settings_[i] = kMazeGenerators[i].make_defaults();
    }

    // The ImGui overlay consumes mouse/keyboard events when it is hovered or
    // focused. Without this the maze also pans/zooms while the panel is used.
    hooks_.should_filter_mouse_input = []() -> bool {
        return ImGui::GetCurrentContext() != nullptr &&
               ImGui::GetIO().WantCaptureMouse;
    };
    hooks_.should_filter_keyboard_input = []() -> bool {
        return ImGui::GetCurrentContext() != nullptr &&
               ImGui::GetIO().WantCaptureKeyboard;
    };

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
    config.enable_imgui = true;
    config.renderer_config.enable_imgui = true;
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
    const auto debug_frag =
        Shaders::MazeGenerator::DebugOverlayFrag::Register(shader_mgr, config.shader_data_dir);

    const auto& shader_ids = engine_game_.GetContext().GetShaderIds();
    if (!engine_game_.InitRenderer(ctx, shader_ids.main_indir_vert, shader_ids.standard_mesh_frag,
                                   &shader_mgr)) {
        return false;
    }
    (void)engine_game_.AddShaderDirectory(config.shader_data_dir);

    // ── Initial maze, through the generator registry ──
    ui_.width = static_cast<int>(kDefaultMazeSize);
    ui_.height = static_cast<int>(kDefaultMazeSize);

    const MazeGeneratorType& generator = kMazeGenerators[ui_.generator_index];
    const MazeSpec spec{
        .width = kDefaultMazeSize,
        .height = kDefaultMazeSize,
        .seed = static_cast<std::uint32_t>(ui_.seed),
    };
    std::optional<TiledMaze> generated =
        GenerateMazeLogged(generator, gen_settings_[ui_.generator_index], spec);
    if (!generated.has_value()) {
        std::cerr << "MazeGenerator: initial generator failed\n";
        return false;
    }
    maze_ = std::move(*generated);

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
    max_maze_size_ = ComputeMaxMazeSize(backend);
    std::cout << "MazeGenerator: max maze side " << max_maze_size_
              << " (device storage-buffer limit)\n";

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

    // ── Debug state ──
    if (!CreateDebugBuffers(backend)) {
        std::cerr << "MazeGenerator: failed to create the debug buffers\n";
        ReleaseGpuBuffers();
        return false;
    }
    FindOpenings();
    ResetSolver();

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
    pipeline.RegisterBufferResolver("debug-cells", [this](std::uint32_t) -> vk::Buffer {
        return debug_cells_buffer_ != nullptr
                   ? static_cast<vk::Buffer>(*debug_cells_buffer_->GetBuffer())
                   : vk::Buffer{};
    });
    pipeline.RegisterBufferResolver("debug-blocks", [this](std::uint32_t) -> vk::Buffer {
        return debug_blocks_buffer_ != nullptr
                   ? static_cast<vk::Buffer>(*debug_blocks_buffer_->GetBuffer())
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
    maze_pass_handle_ = *registered;

    auto debug_pass = std::make_unique<DebugOverlayPass>(
        static_cast<std::uint64_t>(fullscreen_vert),
        static_cast<std::uint64_t>(debug_frag));
    debug_pass_ = debug_pass.get();
    auto debug_registered = pipeline.RegisterPass(std::move(debug_pass));
    if (!debug_registered.has_value()) {
        std::cerr << "MazeGenerator: failed to register the debug pass: "
                  << debug_registered.error().message << '\n';
        debug_pass_ = nullptr;
        ReleaseGpuBuffers();
        return false;
    }
    debug_pass_handle_ = *debug_registered;
    if (!pipeline.AddDependency(maze_pass_handle_, debug_pass_handle_)) {
        std::cerr << "MazeGenerator: failed to order the debug pass after the maze pass\n";
        ReleaseGpuBuffers();
        return false;
    }
    (void)pipeline.SetPassEnabled(debug_pass_handle_, ui_.debug_draw);
    debug_pass_enabled_ = ui_.debug_draw;

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
    viewport_width_ = std::max(width, 1u);
    viewport_height_ = std::max(height, 1u);
    ResetViewForMaze();
    FillConstants(viewport_width_, viewport_height_);
    SyncCamera(viewport_height_);

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

    if (auto* imgui = engine_game_.GetImGuiSystem()) {
        ui_token_ = imgui->draw_callbacks.Register([this] { DrawUi(); });
    }

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
    viewport_width_ = width;
    viewport_height_ = height;

    // UI requests were recorded during last frame's render; apply them before
    // anything reads the maze or the debug grid.
    ApplyUiRequests(ctx);

    const float dt = std::clamp(ctx.frame.delta_time, 0.0f, kMaxFrameDelta);
    float p = std::max(pixels_per_cell_, 1.0e-4f);
    const float maze_width = static_cast<float>(maze_.Width());
    const float maze_height = static_cast<float>(maze_.Height());

    if (pan_action_.Valid()) {
        const auto pan = pan_action_.Value();
        const float cells_per_second = kPanPixelsPerSecond / p;
        center_x_ += pan[0] * cells_per_second * dt;
        center_y_ += pan[1] * cells_per_second * dt;
    }

    const float fit = std::max(
        std::min(static_cast<float>(width) / maze_width,
                 static_cast<float>(height) / maze_height),
        1.0e-4f);
    // Full zoom-out leaves the whole maze visible at 1 / kMaxZoomOutFactor of
    // the limiting axis, so the limit follows the window and maze size.
    const float min_p = fit / kMaxZoomOutFactor;

    const float wheel = ctx.input_system != nullptr
                            ? ctx.input_system->GetActionValue(zoom_action_)
                            : 0.0f;
    if (wheel != 0.0f) {
        const float new_p =
            std::clamp(p * std::pow(kZoomStep, wheel), min_p, kMaxPixelsPerCell);

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

    p = std::max(p, min_p);
    p = std::min(p, kMaxPixelsPerCell);
    pixels_per_cell_ = p;

    const float half_x = 0.5f * static_cast<float>(width) / p;
    const float half_y = 0.5f * static_cast<float>(height) / p;
    // Half the viewport is half_x/half_y cells, so the camera center is bounded
    // by the maze's cell range plus the allowed overscroll on each side.
    ClampAxisCenter(center_x_, half_x, maze_width);
    ClampAxisCenter(center_y_, half_y, maze_height);

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

    AdvanceSolver();

    FillConstants(width, height);
    SyncCamera(height);
    FlushDebugGrid();
    engine_game_.FrameUpdate(ctx);
}

void MazeGeneratorApp::OnFrameRender(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void MazeGeneratorApp::OnShutdown(
    VulkanEngine::Application::ApplicationContext& /*ctx*/) {
    // The ImGui system owns the callback list this token unregisters from, and
    // engine_game_.Shutdown() destroys it. Drop the token first, or its
    // destructor touches freed memory.
    ui_token_ = {};
    ReleaseGpuBuffers();
    engine_game_.Shutdown();
}

// ── UI ─────────────────────────────────────────────────────────────────────

void MazeGeneratorApp::DrawSizeField(const char* label, int& value) {
    ImGui::SetNextItemWidth(150.0f);
    // Dragging clamps to the soft range; Ctrl+Click (or double-click) manual
    // entry is not clamped by ImGui unless ImGuiSliderFlags_AlwaysClamp is
    // passed, which is exactly the soft-limit behavior we want.
    ImGui::DragInt(label, &value, 16.0f, static_cast<int>(kMinMazeSize),
                   static_cast<int>(kDragMaxMazeSize));
    // Apply the device hard cap only once the edit finishes, so typing "8192"
    // is not clamped at the intermediate "8".
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        value = std::clamp(value, static_cast<int>(kMinMazeSize),
                           static_cast<int>(max_maze_size_));
    }
}

void MazeGeneratorApp::DrawUi() {
    ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Maze", nullptr)) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Generation");

    DrawSizeField("Width", ui_.width);
    ImGui::SameLine();
    DrawSizeField("Height", ui_.height);

    ImGui::SetNextItemWidth(150.0f);
    ImGui::InputInt("Seed", &ui_.seed);
    ImGui::SameLine();
    if (ImGui::Button("Randomize")) {
        ui_.seed = static_cast<int>(std::random_device{}());
    }
    ui_.seed = std::max(ui_.seed, 0);

    std::array<const char*, std::size(kMazeGenerators)> generator_names{};
    for (std::size_t i = 0; i < generator_names.size(); ++i) {
        generator_names[i] = kMazeGenerators[i].name.data();
    }
    ImGui::SetNextItemWidth(240.0f);
    ImGui::Combo("Generator", &ui_.generator_index, generator_names.data(),
                 static_cast<int>(generator_names.size()));
    ui_.generator_index = std::clamp(ui_.generator_index, 0,
                                     static_cast<int>(generator_names.size()) - 1);

    const MazeGeneratorType& generator = kMazeGenerators[ui_.generator_index];
    ImGui::PushID(ui_.generator_index);
    if (ImGui::CollapsingHeader(generator.name.data(),
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGuiSettingsVisitor visitor;
        generator.visit(gen_settings_[ui_.generator_index], visitor);
    }
    ImGui::PopID();

    if (ImGui::Button("Generate", ImVec2(-1.0f, 0.0f))) {
        ui_.pending_generate = true;
    }
    ImGui::TextDisabled("Current: %u x %u", maze_.Width(), maze_.Height());
    ImGui::TextDisabled("Max side: %u (device limit)", max_maze_size_);
    if (ui_.width > static_cast<int>(kLargeMazeWarningSize) ||
        ui_.height > static_cast<int>(kLargeMazeWarningSize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f),
                           "Large mazes can take many minutes and several GiB");
    }

    ImGui::SeparatorText("Solver");

    std::array<const char*, std::size(kSolverTypes)> solver_names{};
    for (std::size_t i = 0; i < solver_names.size(); ++i) {
        solver_names[i] = kSolverTypes[i].name.data();
    }
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::Combo("Algorithm", &ui_.solver_index, solver_names.data(),
                     static_cast<int>(solver_names.size()))) {
        ui_.pending_stop = true;
    }
    ui_.solver_index = std::clamp(ui_.solver_index, 0,
                                  static_cast<int>(solver_names.size()) - 1);

    ImGui::Text("Start (%u, %u)   Goal (%u, %u)",
                static_cast<unsigned>(start_.x), static_cast<unsigned>(start_.y),
                static_cast<unsigned>(goal_.x), static_cast<unsigned>(goal_.y));

    if (solver_running_) {
        if (ImGui::Button("Stop")) {
            ui_.pending_stop = true;
        }
    } else {
        if (ImGui::Button("Solve")) {
            ui_.pending_solve = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        ui_.pending_reset = true;
    }

    ImGui::SetNextItemWidth(240.0f);
    // The far-right position means "no budget"; show that instead of the
    // finite sentinel value.
    const char* steps_format =
        ui_.steps_per_frame >= kMaxStepsPerFrame ? "Unlimited" : "%d";
    ImGui::SliderInt("Steps / frame", &ui_.steps_per_frame, 1,
                     kMaxStepsPerFrame, steps_format,
                     ImGuiSliderFlags_Logarithmic);

    ImGui::Text("Status: %s",
                solver_ == nullptr ? "idle" : SolverStatusLabel(solver_status_));
    ImGui::Text("expanded %llu   frontier %llu   path %llu",
                static_cast<unsigned long long>(solver_stats_.expanded),
                static_cast<unsigned long long>(solver_stats_.frontier),
                static_cast<unsigned long long>(solver_stats_.path_length));

    ImGui::SeparatorText("Debug draw");
    ImGui::Checkbox("Enabled", &ui_.debug_draw);
    ImGui::BeginDisabled(!ui_.debug_draw);
    ImGui::CheckboxFlags("Frontier", &ui_.debug_categories, kDebugCategoryOpen);
    ImGui::CheckboxFlags("Visited", &ui_.debug_categories, kDebugCategoryClosed);
    ImGui::CheckboxFlags("Path", &ui_.debug_categories, kDebugCategoryPath);
    ImGui::EndDisabled();

    ImGui::End();
}

void MazeGeneratorApp::ApplyUiRequests(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    if (ui_.pending_generate) {
        ui_.pending_generate = false;
        RegenerateMaze(ctx);
    }
    if (ui_.pending_reset) {
        ui_.pending_reset = false;
        ResetSolver();
    }
    if (ui_.pending_stop) {
        ui_.pending_stop = false;
        solver_running_ = false;
    }
    if (ui_.pending_solve) {
        ui_.pending_solve = false;
        StartSolver();
    }

    // The pass cannot be enabled without the buffers its descriptors bind.
    if (ui_.debug_draw &&
        (debug_cells_buffer_ == nullptr || debug_blocks_buffer_ == nullptr)) {
        ui_.debug_draw = false;
    }
    if (ui_.debug_draw != debug_pass_enabled_) {
        (void)engine_game_.GetRenderPipeline().SetPassEnabled(debug_pass_handle_,
                                                              ui_.debug_draw);
        debug_pass_enabled_ = ui_.debug_draw;
    }
}

void MazeGeneratorApp::RegenerateMaze(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    const std::uint32_t width = std::clamp(
        static_cast<std::uint32_t>(std::max(ui_.width, 0)),
        kMinMazeSize, max_maze_size_);
    const std::uint32_t height = std::clamp(
        static_cast<std::uint32_t>(std::max(ui_.height, 0)),
        kMinMazeSize, max_maze_size_);
    ui_.width = static_cast<int>(width);
    ui_.height = static_cast<int>(height);
    ui_.generator_index = std::clamp(
        ui_.generator_index, 0, static_cast<int>(std::size(kMazeGenerators)) - 1);

    const MazeGeneratorType& generator = kMazeGenerators[ui_.generator_index];
    const MazeSpec spec{
        .width = width,
        .height = height,
        .seed = static_cast<std::uint32_t>(std::max(ui_.seed, 0)),
    };

    std::optional<TiledMaze> generated =
        GenerateMazeLogged(generator, gen_settings_[ui_.generator_index], spec);
    if (!generated.has_value()) {
        std::cerr << "MazeGenerator: generator '" << generator.name
                  << "' rejected the current settings\n";
        return;
    }

    MazeDensityData new_density = BuildMazeDensity(*generated);
    if (new_density.values.empty()) {
        std::cerr << "MazeGenerator: density pyramid is empty\n";
        return;
    }

    auto& backend = ctx.bootstrap->GetBackend();
    auto new_maze_buffer = CreateUploadedBuffer(
        engine_game_, backend, generated->RawBytes().data(),
        generated->RawBytes().size());
    auto new_density_buffer = CreateUploadedBuffer(
        engine_game_, backend, new_density.values.data(),
        static_cast<std::uint64_t>(new_density.values.size() * sizeof(float)));
    if (new_maze_buffer == nullptr || new_density_buffer == nullptr) {
        // Flush any half-recorded immediate copies before returning.
        (void)engine_game_.GetStagingPool().FlushImmediate();
        return;
    }
    if (!engine_game_.GetStagingPool().FlushImmediate()) {
        std::cerr << "MazeGenerator: maze upload failed\n";
        return;
    }

    // The immediate flush submitted after all in-flight frames and waited, so
    // replacing the old buffers here is safe.
    maze_ = std::move(*generated);
    density_mip_count_ = new_density.mip_count;
    maze_buffer_ = std::move(new_maze_buffer);
    density_buffer_ = std::move(new_density_buffer);

    if (!CreateDebugBuffers(backend)) {
        std::cerr << "MazeGenerator: failed to recreate the debug buffers\n";
        ui_.debug_draw = false;
        (void)engine_game_.GetRenderPipeline().SetPassEnabled(debug_pass_handle_, false);
        debug_pass_enabled_ = false;
    }

    ResetSolver();
    FindOpenings();
    ResetViewForMaze();
    FillConstants(viewport_width_, viewport_height_);
}

void MazeGeneratorApp::ResetSolver() {
    solver_.reset();
    solver_running_ = false;
    solver_status_ = SolverStatus::Running;
    solver_stats_ = {};
    debug_grid_.Clear();
    debug_grid_.FlushTo(debug_cells_mapped_, debug_blocks_mapped_);
}

void MazeGeneratorApp::StartSolver() {
    solver_.reset();
    solver_ = CreateSolver(static_cast<std::size_t>(std::max(ui_.solver_index, 0)));
    if (solver_ == nullptr) {
        return;
    }
    if (start_.x >= maze_.Width() || start_.y >= maze_.Height() ||
        goal_.x >= maze_.Width() || goal_.y >= maze_.Height()) {
        solver_.reset();
        return;
    }

    debug_grid_.Clear();
    for (const SolverCellChange& change : solver_->Reset(maze_, start_, goal_)) {
        debug_grid_.Apply(change);
    }
    solver_running_ = true;
    solver_status_ = SolverStatus::Running;
    solver_stats_ = {};
}

void MazeGeneratorApp::AdvanceSolver() {
    if (!solver_running_ || solver_ == nullptr) {
        return;
    }

    // The slider's far-right position asks for an unlimited budget: keep
    // stepping until the solver reports Found/Exhausted. Step() only needs the
    // budget to exceed the cell count, and even the largest allowed maze
    // (65535^2 cells) is below UINT32_MAX.
    const std::uint32_t budget =
        ui_.steps_per_frame >= kMaxStepsPerFrame
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(std::max(ui_.steps_per_frame, 1));
    const auto result = solver_->Step(maze_, budget);
    for (const SolverCellChange& change : result.changes) {
        debug_grid_.Apply(change);
    }
    solver_stats_ = result.stats;
    solver_status_ = result.status;
    if (result.status != SolverStatus::Running) {
        solver_running_ = false; // keep the finished visualization on screen
    }
}

void MazeGeneratorApp::FindOpenings() {
    start_ = {0, 0};
    goal_ = {maze_.Width() - 1, maze_.Height() - 1};

    for (std::uint32_t x = 0; x < maze_.Width(); ++x) {
        if (!maze_.North(x, 0)) {
            start_ = {x, 0};
            break;
        }
    }
    for (std::uint32_t x = 0; x < maze_.Width(); ++x) {
        if (!maze_.South(x, maze_.Height() - 1)) {
            goal_ = {x, maze_.Height() - 1};
            break;
        }
    }
}

void MazeGeneratorApp::ResetViewForMaze() {
    center_x_ = 0.5f * static_cast<float>(maze_.Width());
    center_y_ = 0.5f * static_cast<float>(maze_.Height());

    const float width = static_cast<float>(std::max(viewport_width_, 1u));
    const float height = static_cast<float>(std::max(viewport_height_, 1u));
    const float fit = std::max(
        std::min(width / static_cast<float>(maze_.Width()),
                 height / static_cast<float>(maze_.Height())),
        1.0e-4f);
    pixels_per_cell_ = std::min(std::max(kInitialPixelsPerCell, fit),
                                kMaxPixelsPerCell);
    // The frame's hysteresis will lower this immediately if p is below the
    // exact-tier threshold.
    tier_ = Tier::Exact;
}

// ── Debug buffers ──────────────────────────────────────────────────────────

bool MazeGeneratorApp::CreateDebugBuffers(
    VulkanBackend::Vulkan::IVulkanBootstrap& backend) {
    ReleaseDebugBuffers();
    debug_grid_.Reset(maze_.Width(), maze_.Height());

    const std::size_t cell_count =
        static_cast<std::size_t>(debug_grid_.Width()) * debug_grid_.Height();
    // The shader reads four cells at a time (one uint), so round the buffer up
    // to a whole word; the tail bytes stay zero.
    const std::size_t cell_bytes = ((cell_count + 3u) / 4u) * 4u;
    const std::size_t block_bytes =
        static_cast<std::size_t>(debug_grid_.BlocksX()) * debug_grid_.BlocksY() *
        sizeof(std::uint32_t);

    void* cells = nullptr;
    void* blocks = nullptr;
    debug_cells_buffer_ = CreateMappedBuffer(backend, cell_bytes, cells);
    debug_blocks_buffer_ = CreateMappedBuffer(backend, block_bytes, blocks);
    if (debug_cells_buffer_ == nullptr || debug_blocks_buffer_ == nullptr) {
        ReleaseDebugBuffers();
        return false;
    }

    debug_cells_mapped_ = static_cast<std::uint8_t*>(cells);
    debug_blocks_mapped_ = static_cast<std::uint32_t*>(blocks);
    debug_grid_.FlushTo(debug_cells_mapped_, debug_blocks_mapped_);
    if (cell_bytes > cell_count) {
        std::memset(debug_cells_mapped_ + cell_count, 0, cell_bytes - cell_count);
    }
    return true;
}

void MazeGeneratorApp::ReleaseDebugBuffers() noexcept {
    if (debug_cells_buffer_ != nullptr) {
        debug_cells_buffer_->Unmap();
        debug_cells_buffer_.reset();
    }
    if (debug_blocks_buffer_ != nullptr) {
        debug_blocks_buffer_->Unmap();
        debug_blocks_buffer_.reset();
    }
    debug_cells_mapped_ = nullptr;
    debug_blocks_mapped_ = nullptr;
}

void MazeGeneratorApp::FlushDebugGrid() {
    if (debug_cells_mapped_ == nullptr || debug_blocks_mapped_ == nullptr) {
        return;
    }
    debug_grid_.FlushTo(debug_cells_mapped_, debug_blocks_mapped_);
}

void MazeGeneratorApp::ReleaseGpuBuffers() noexcept {
    ReleaseDebugBuffers();
    density_buffer_.reset();
    maze_buffer_.reset();
    debug_pass_ = nullptr;
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

    debug_constants_.view = {center_x_, center_y_, pixels_per_cell_, 1.0f};
    debug_constants_.screen = {
        static_cast<float>(width), static_cast<float>(height),
        static_cast<float>(static_cast<std::uint32_t>(tier_)),
        static_cast<float>(DebugGrid::kBlockSize)};
    debug_constants_.grid = {maze_.Width(), maze_.Height(), debug_grid_.BlocksX(),
                             ui_.debug_categories};
    const std::uint32_t start_index = start_.y * maze_.Width() + start_.x;
    const std::uint32_t goal_index = goal_.y * maze_.Width() + goal_.x;
    debug_constants_.extra = {start_index, goal_index, 0u, 0u};
    if (debug_pass_ != nullptr) {
        debug_pass_->SetConstants(debug_constants_);
    }
}

} // namespace MazeGenerator::App
