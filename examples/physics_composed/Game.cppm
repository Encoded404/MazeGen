module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

export module Examples.PhysicsComposed.Game;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

export namespace Examples::PhysicsComposed::Game {

// The composed-path example (rung 4 of the engine modularization): a physics
// scene built through module registration instead of a hand-written setup
// script. The engine's default preset (device bootstrap, physics, resource
// hooks, frame-phase seeds) is running underneath — already started by Setup —
// and this game adds its own Scene-stage modules and a frame-phase system,
// ordering them by identity against the engine's preset slots. Nothing here
// edits the engine: the litmus test is that a composed game can replace any
// engine default (lighting, frame system, module) without it.
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

    // The scene module's body: ground + crate stack, spawned through the
    // prefab seam (rung 3), with physics coming from the engine's own module.
    [[nodiscard]] bool BuildScene(VulkanEngine::Modules::ModuleContext& mctx);
    // The telemetry module's body: a PostPresent frame-phase system.
    [[nodiscard]] bool RegisterTelemetry();

    VulkanEngine::Application::ApplicationHooks hooks_{};

    VulkanShared::ScopedHandle<bool(VulkanEngine::Application::ApplicationContext&)> setup_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> fixed_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_render_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> shutdown_token_{};

    std::filesystem::path exe_dir_{};
    VulkanEngine::GameEngine engine_game_{};

    // Scene state (owned by plain composition of the engine's surfaces).
    glm::vec3 camera_offset_{0.0f, 6.0f, 10.0f};
    std::uint32_t quad_mesh_id_ = 0;
    VulkanEngine::FrameId follow_system_{};
    VulkanEngine::FrameId telemetry_system_id_{};
    std::uint32_t frame_telemetry_counter_ = 0;
};

} // namespace Examples::PhysicsComposed::Game