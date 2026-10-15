module;

export module Examples.PhysicsVehicle.Game;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.DebugOverlay;
import VulkanEngine.Physics.Vehicle;
#endif

export namespace Examples::PhysicsVehicle::Game {

/// Which axle(s) the differential routes engine torque to. The tuning tab
/// exposes it as a toggle; AWD splits torque 50/50 between the two axles.
enum class Drivetrain : int {
    AllWheel = 0,
    Front = 1,
    Rear = 2,
};

/// A named set of tuning-tab starting values. The dropdown writes one into the
/// live tunables; the Reset-fields button applies the normal-car preset.
/// Chassis body properties go through the world's live setters (mid-drive
/// safe); the `VehicleDesc` ones land on the next vehicle rebuild.
struct TuningPreset {
    const char* name;
    float chassis_mass;
    float chassis_friction;
    float engine_max_torque;
    int engine_min_rpm;
    int engine_max_rpm;
    Drivetrain drivetrain;
    float wheel_radius;
    float wheel_width;
    float wheel_spread;
    float wheel_base;
    float steer_angle;
    float suspension_frequency;
    float suspension_damping;
    float friction_slip;
    float sideways_friction_slip;
    float handbrake_torque;
};

// The named starting sets for the tuning tab: index 0 doubles as the
// Reset-fields default and seeds the `Game` member initializers below, so the
// literals appear exactly once — a fresh spawn and a reset land on the same
// car by construction.
inline constexpr TuningPreset kPresets[]{
    {"Normal car",
        1400.0f,
        0.4f,
        500.0f,
        1000,
        4500,
        Drivetrain::AllWheel,
        0.3f,
        0.25f,
        0.85f,
        1.3f,
        0.7f,
        1.8f,
        0.5f,
        1.05f,
        1.0f,
        2000.0f
    },
    {"Sports car",
        1200.0f,
        0.4f,
        2000.0f,
        1000,
        8000,
        Drivetrain::AllWheel,
        0.3f,
        0.25f,
        0.95f,
        1.4f,
        0.75f,
        1.8f,
        0.6f,
        3.0f,
        0.95f,
        4000.0f
    },
};
inline constexpr int kDefaultPreset = 0;

/// Binding the default preset's fields for the `Game` member initializers —
/// only a readability shorthand for `kPresets[kDefaultPreset].x`.
inline constexpr const TuningPreset& kDefaultTuning = kPresets[kDefaultPreset];

static_assert(std::size(kPresets) == 2);

// A drivable four-wheel vehicle over the physics facade.
//
// The chassis is an ordinary entity: `Transform` + `Physics::RigidBody`, with
// the body created by the registry's OnAdd hook and driven by the engine's
// registered physics systems (PhysicsInput -> PhysicsSimulate ->
// PhysicsWriteBack) every fixed step. The `Physics::WheeledVehicle` rides on
// that body: its suspension/tire constraint is a physics step listener, so the
// world's single `Step()` drives it and nothing here simulates by hand.
//
// The four wheels are render-only entities: a wheel is a ray cast from the
// chassis, not a body, so it has no physics presence and its pose is sampled
// from the vehicle each frame (`GetWheelPose` + accumulated spin). A chase
// camera follows the chassis, and the physics debug overlay draws the bodies
// the renderer cannot see (the wheels' contact rays included, via the tires'
// contact state).
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
#ifdef VKENGINE_PHYSICS
    void UpdateWheelEntity(std::uint32_t index);
    /// Draws the foldable tuning sections inside the demo's ImGui window.
    /// Returns true when a vehicle-`Initialize`-time value changed, so the
    /// vehicle must be rebuilt in the next fixed update.
    [[nodiscard]] bool DrawVehicleTuningWindow();
    /// The vehicle description from the current tunables; one path shared by
    /// the initial spawn and every rebuild.
    [[nodiscard]] VulkanEngine::Physics::VehicleDesc BuildVehicleDesc() const;
    /// Writes a preset into the tuning-tab tunables. Chassis mass/friction go
    /// through the world's live setters immediately; the `VehicleDesc`-fed
    /// values are picked up by the rebuild the caller queues.
    void ApplyPreset(const TuningPreset& preset);
#endif

    VulkanEngine::Application::ApplicationHooks hooks_{};

    VulkanShared::ScopedHandle<bool(VulkanEngine::Application::ApplicationContext&)> setup_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> fixed_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_render_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> shutdown_token_{};

    std::filesystem::path exe_dir_{};
    VulkanEngine::GameEngine engine_game_{};

    std::pair<float, float> ground_extend_ = std::make_pair(200.0f, 150.0f);

    // Vehicle tunables, all editable from the ImGui tab, initialized from the
    // default preset so the literals live only in the `kPresets` table above.
    // `WheelDesc`-fed values take effect on the next vehicle rebuild; chassis
    // body properties go through the world's live setters and apply while the
    // car is running. `wheel_radius_`/`wheel_width_` also drive the wheel
    // entity scales, so their render response is immediately visible even
    // before a rebuild.
    float wheel_radius_ = kDefaultTuning.wheel_radius;
    float steer_angle_ = kDefaultTuning.steer_angle;
    float wheel_spread_ = kDefaultTuning.wheel_spread;
    float wheel_width_ = kDefaultTuning.wheel_width;
    float wheel_base_ = kDefaultTuning.wheel_base;
    float suspension_frequency_ = kDefaultTuning.suspension_frequency;
    float suspension_damping_ = kDefaultTuning.suspension_damping;
    float friction_slip_ = kDefaultTuning.friction_slip;
    float sideways_friction_slip_ = kDefaultTuning.sideways_friction_slip;
    float handbrake_torque_ = kDefaultTuning.handbrake_torque;
    float engine_max_torque_ = kDefaultTuning.engine_max_torque;
    int engine_min_rpm_ = kDefaultTuning.engine_min_rpm;
    int engine_max_rpm_ = kDefaultTuning.engine_max_rpm;
    float chassis_mass_ = kDefaultTuning.chassis_mass;
    float chassis_friction_ = kDefaultTuning.chassis_friction;
    /// 0 = AWD (both axles at 0.5), 1 = front drive, 2 = rear drive.
    Drivetrain drivetrain_ = kDefaultTuning.drivetrain;
    /// Which entry of `kPresets` the dropdown shows; index 0 is the default the
    /// Reset-fields button re-applies.
    int preset_index_ = kDefaultPreset;

    float assisted_steering_min_ = 0.8f;
    float assisted_steering_max_ = 1.1f;
    float assisted_steering_scaler_ = 1.0f;

#ifdef VKENGINE_PHYSICS
    // Per-app state: the overlay keeps its toggles and last geometry between
    // frames; the vehicle owns the world-registered constraint and must be
    // shut down before the chassis body and the world.
    VulkanEngine::Physics::PhysicsDebugOverlay physics_overlay_{};
    VulkanShared::ScopedHandle<void()> imgui_draw_token_{};
    std::unique_ptr<VulkanEngine::Physics::WheeledVehicle> vehicle_{};
    VulkanEngine::Physics::PhysicsBodyId chassis_body_{};
    // Render-only wheel entities, held as `EntityId`s and re-fetched each use
    // (component storage may reallocate as entities are added, so a cached
    // `Transform*` is not stable).
    std::vector<VulkanEngine::EntityId> wheel_entities_{};

    // Scene reset: the world's solver state (poses, velocities, constraint
    // internals) snapshotted right after the scene and the vehicle finished
    // spawning. `SaveSnapshot`/`RestoreSnapshot` are all-or-nothing; bodies are
    // never added or removed after this point, so restoring the snapshot is
    // exactly "the scene the way it started". The button only sets the request
    // flag from the ImGui frame; the restore itself runs in the fixed update,
    // before the physics schedule, never inside a draw callback.
    std::vector<std::byte> scene_snapshot_{};
    bool reset_requested_ = false;
    // A tuning edit changed an `Initialize`-time vehicle value: shutdown +
    // re-`Initialize` the vehicle on the same chassis body in the next fixed
    // update (same discipline as `reset_requested_` — never inside the draw
    // callback).
    bool rebuild_requested_ = false;

    // Driver bindings. `drive` packs steering (x) and throttle/reverse (y);
    // brake and handbrake are plain 0..1 presses.
    VulkanEngine::Input::Action<2> drive_{};
    VulkanEngine::Input::Action<1> brake_{};
    VulkanEngine::Input::Action<1> handbrake_{};

    // One-shot evidence lines for headless runs (--max-frames).
    bool logged_first_contact_ = false;
    bool logged_overlay_geometry_ = false;
    bool logged_speed_ = false;
#endif
};

} // namespace Examples::PhysicsVehicle::Game