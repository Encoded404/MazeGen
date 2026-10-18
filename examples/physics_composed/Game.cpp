module;

#include <SDL3/SDL_keycode.h>
#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <logging/logging_macros.hpp>

module Examples.PhysicsComposed.Game;

import std;

import logiface;

import VulkanEngine.GameEngine;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.ECS.Prefabs;
import VulkanEngine.Modules.EngineModules;
import VulkanEngine.ShaderManager;

namespace Examples::PhysicsComposed::Game {

Game::Game(const std::filesystem::path& executable_path)
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

Game::~Game() = default;

bool Game::OnSetup(VulkanEngine::Application::ApplicationContext& ctx) {
    // ── Configuration: where a game overrides engine defaults ──
    VulkanEngine::GameConfig config{};
    config.enable_imgui = false;
    config.renderer_config.clear_color = {0.08f, 0.09f, 0.12f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();
    // The engine preset's lighting is configuration now (rung 2); this scene
    // reads better with a steeper, cooler sun. No engine edit anywhere.
    config.lighting.sun_direction = {0.2f, -0.85f, 0.45f};
    config.lighting.sun_color = {0.85f, 0.95f, 1.0f, 2.20f};
    // The physics module's shape cache is off by default ("cook everything");
    // a scene that wants a disk cache registers its own physics module with a
    // configured PhysicsWorldConfig instead of the preset's.

    // ── The composed path ──
    // The scene is not a setup script: it is an engine module. It runs at the
    // scene stage (once renderer-side resources exist), ordered by identity
    // against the engine's own preset slots (engine.device, engine.physics,
    // engine.frame_pipeline) — never by editing the engine. Physics is not
    // wired here at all: the preset's physics module already started its world
    // and its frame contract during Setup, and the prefab's RigidBody bundles
    // attach through the registry hooks that module installed.
    // Ordering against preset slots is an identity statement (see the
    // startup-stage pattern in any composed game that runs before renderer
    // init); a Scene-stage module needs no such constraint: the stage itself
    // starts only after renderer-side resources exist.
    engine_game_.Modules().Add(VulkanEngine::Modules::EngineModuleDesc{
        .stage = VulkanEngine::Modules::Stage::Scene,
        .label = "composed.scene",
        .startup = [this](VulkanEngine::Modules::ModuleContext& mctx) -> bool {
            return BuildScene(mctx);
        },
    });

    // Registration order is startup order: the telemetry module registered
    // after the scene module also starts after it (no constraint needed —
    // AddAfter/AddBefore exist for when a real ordering is required, and the
    // scene stage itself already guarantees renderer readiness after the
    // renderer-init choreography completes).
    engine_game_.Modules().Add(VulkanEngine::Modules::EngineModuleDesc{
        .stage = VulkanEngine::Modules::Stage::Scene,
        .label = "composed.telemetry",
        .startup = [this](VulkanEngine::Modules::ModuleContext& /*mctx*/) -> bool {
            return RegisterTelemetry();
        },
    });

    // ── The engine runs the preset (and this game's modules) through the same
    //    public seam; the scene stage starts at the end of InitRenderer. ──
    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    // Renderer init in direct-mode style: the engine's standard shaders.
    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    auto& shader_ids = engine_game_.GetContext().GetShaderIds();
    if (!engine_game_.InitRenderer(ctx, shader_ids.main_indir_vert,
                                   shader_ids.standard_mesh_frag, &shader_mgr)) {
        return false;
    }

    // The camera is an entity convenience the composed scene keeps.
    engine_game_.CreateCamera(engine_game_.GetContext().GetComponentRegistry());
    if (auto* camera = engine_game_.GetCamera()) {
        camera->position = camera_offset_;
        camera->target = {0.0f, 0.0f, 0.0f};
    }

    engine_game_.MarkSceneValid();

    // Per-frame work joins the frame-phase seam (rung 1): a Bridge-phase
    // camera-follow constrained after the engine's own mesh-processing slot.
    // Ordering is a registration statement against identity — never an edit to
    // the frame pipeline. The camera is re-fetched each frame (dense storage:
    // no camera pointers are cached).
    follow_system_ = engine_game_.AddFrameSystemAfter(
        VulkanEngine::FrameSlot::MeshProcessFrame,
        [this](VulkanEngine::FramePhaseContext& fctx) {
            auto* camera = engine_game_.GetCamera();
            if (camera == nullptr) return;
            glm::vec3 average{0.0f, 0.0f, 0.0f};
            std::size_t crates = 0;
            auto& registry = fctx.Engine().component_registry;
            registry.ForEach<VulkanEngine::Components::Transform, VulkanEngine::Components::MeshReference>(
                [&](VulkanEngine::Entity& /*entity*/,
                    VulkanEngine::Components::Transform& transform,
                    VulkanEngine::Components::MeshReference& /*mesh*/) {
                    average += transform.position;
                    ++crates;
                });
            if (crates == 0) return;
            average /= static_cast<float>(crates);

            const glm::vec3 follow =
                glm::vec3{average.x, 0.0f, average.z} + camera_offset_;
            camera->position += (follow - camera->position) * 0.1f;
            camera->target = average;
        },
        "composed.camera_follow");

    // A quit action keeps the smoke run pleasant.
    ctx.quit_action_handle = ctx.input_system->BindAction(
        "quit", VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));

    return true;
}

bool Game::BuildScene(VulkanEngine::Modules::ModuleContext& mctx) {
    auto& registry = mctx.Registry();

    // One procedural quad, registered once: the crate's visual. The material is
    // the engine's fallback (technique id 0) — content, not code.
    auto quad = VulkanEngine::SceneLoader::ToMeshData(
        VulkanEngine::SceneLoader::CreateFallbackQuad());
    quad_mesh_id_ = engine_game_.GetMeshRegistry().Register(quad);

    // Invisible static ground: collision only.
    VulkanEngine::Physics::RigidBody ground_body{};
    ground_body.shape.type = VulkanEngine::Physics::ShapeType::Box;
    ground_body.shape.box_extent = {20.0f, 0.5f, 20.0f};
    ground_body.motion = VulkanEngine::Physics::BodyMotionType::Static;
    ground_body.mass = 0.0f;
    VulkanEngine::Prefabs::Prefab ground{};
    ground.name = "ground";
    ground.position = {0.0f, -0.5f, 0.0f};
    ground.rigid_body = ground_body;
    VulkanEngine::Prefabs::Spawn(registry, ground);

    // A stack of dynamic crates: each spawn composes the same bundle — the
    // factory code a game used to hand-roll per spawn, now one Prefab call.
    for (int i = 0; i < 6; ++i) {
        VulkanEngine::Physics::RigidBody body{};
        body.shape.type = VulkanEngine::Physics::ShapeType::Box;
        body.shape.box_extent = {0.5f, 0.5f, 0.5f};
        body.motion = VulkanEngine::Physics::BodyMotionType::Dynamic;
        body.mass = 1.0f;
        body.friction = 0.6f;

        VulkanEngine::Prefabs::Prefab crate{};
        crate.name = "crate";
        crate.position = {0.0f, 0.55f + static_cast<float>(i) * 1.05f, 0.0f};
        crate.mesh_id = quad_mesh_id_;
        crate.rigid_body = body;
        VulkanEngine::Prefabs::Spawn(registry, crate);
    }

    LOGIFACE_LOG(info, "composed.scene: ground + 6 crates spawned through "
                       "Prefabs::Spawn (physics lives in the engine preset's module)");
    return true;
}

bool Game::RegisterTelemetry() {
    // Post-present telemetry: after the frame is submitted. Added through the
    // public frame-phase surface; the phase sequence carries it — no engine
    // edit, and pre-record work can never run after recording.
    telemetry_system_id_ = engine_game_.AddFrameSystem(
        VulkanEngine::FramePhase::PostPresent,
        [this](VulkanEngine::FramePhaseContext& /*fctx*/) {
            if (++frame_telemetry_counter_ < 300) return;
            frame_telemetry_counter_ = 0;
            const auto* world = engine_game_.GetPhysicsWorld();
            LOGIFACE_LOG(info, std::format("composed.telemetry: {} physics steps, {} bodies",
                                           world != nullptr ? world->StepCount() : 0,
                                           world != nullptr ? world->BodyCount() : 0));
        },
        "composed.telemetry");
    return telemetry_system_id_.IsValid();
}

void Game::OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FixedUpdate(ctx);
}

void Game::OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameUpdate(ctx);
}

void Game::OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void Game::OnShutdown(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
    engine_game_.Shutdown();
}

} // namespace Examples::PhysicsComposed::Game