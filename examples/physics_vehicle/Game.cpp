module;
#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_keycode.h>
#include <imgui.h>
#include <logging/logging_macros.hpp>

module Examples.PhysicsVehicle.Game;

import std;
import logiface;

import VulkanEngine.GameEngine;
import VulkanEngine.GpuResources.MeshData;
import Shaders.Engine.StandardMeshFrag;
import Shaders.PhysicsVehicle.CheckerFrag;
import VulkanEngine.ShaderManager;

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.DebugOverlay;
import VulkanEngine.Physics.Vehicle;
#endif

namespace Examples::PhysicsVehicle::Game {

namespace {

// Global texture-density control. Every mesh below authors its UVs in world
// units (1 uv = 1 m); scaling them down here stretches the texture over the
// surfaces, so each texel of the checker pattern occupies 3x as much space
// (the pattern repeats every 3 m instead of every 1 m).
constexpr float kUvTexelScale = 1.0f / 8.0f;

// One axis-aligned box, authored at its final size (no entity scale), with
// per-face UVs scaled to the face's world dimensions so the checker density
// stays uniform across differently sized boxes.
[[nodiscard]] VulkanEngine::SceneLoader::LoadedMeshData BuildBoxMesh(const glm::vec3 extents) {
    const float hx = extents.x * 0.5f;
    const float hy = extents.y * 0.5f;
    const float hz = extents.z * 0.5f;

    VulkanEngine::SceneLoader::LoadedMeshData data{};
    data.positions.reserve(24 * 3);
    data.normals.reserve(24 * 3);
    data.uvs.reserve(24 * 2);
    data.tangents.reserve(24 * 3);
    data.tangent_handedness.reserve(24);
    data.indices.reserve(36);

    const auto push_face = [&](const glm::vec3& u_axis, const glm::vec3& v_axis,
                               const glm::vec3& normal, const float u_size, const float v_size) {
        // Face origin: the corner whose (u, v) is (0, 0), i.e. the face center
        // offset by -half_u along u and -half_v along v.
        const glm::vec3 center = normal * glm::vec3(hx, hy, hz);
        const glm::vec3 face_u = u_axis * u_size * 0.5f;
        const glm::vec3 face_v = v_axis * v_size * 0.5f;
        const glm::vec3 origin = center - face_u - face_v;
        const glm::vec3 tangent = u_axis;

        const auto push_vertex = [&](const glm::vec3& position, const float u, const float v) {
            data.positions.insert(data.positions.end(), {position.x, position.y, position.z});
            data.normals.insert(data.normals.end(), {normal.x, normal.y, normal.z});
            data.uvs.insert(data.uvs.end(), {u, v});
            data.tangents.insert(data.tangents.end(), {tangent.x, tangent.y, tangent.z});
            data.tangent_handedness.push_back(1.0f);
        };

        // Counter-clockwise seen from outside: origin -> +2u -> +2u +2v -> +2v.
        // `face_u`/`face_v` are half-edges (the offsets from the face center to
        // the edges), so the far corners are two of them away from the origin.
        const auto base = static_cast<std::uint32_t>(data.positions.size() / 3);
        const glm::vec3 far_u = face_u * 2.0f;
        const glm::vec3 far_v = face_v * 2.0f;
        push_vertex(origin, 0.0f, 0.0f);                       // (0, 0)
        push_vertex(origin + far_u, u_size, 0.0f);             // (1, 0)
        push_vertex(origin + far_u + far_v, u_size, v_size);   // (1, 1)
        push_vertex(origin + far_v, 0.0f, v_size);             // (0, 1)
        data.indices.insert(data.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    };

    // +z front, -z back (u along x, v along y)
    push_face({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 2.0f * hx, 2.0f * hy);
    push_face({-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, 2.0f * hx, 2.0f * hy);
    // +y top (u along z, v along x), -y bottom (u along x, v along z)
    push_face({0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 2.0f * hz, 2.0f * hx);
    push_face({1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, 2.0f * hx, 2.0f * hz);
    // +x right (u along y, v along z), -x left (u along z, v along y)
    push_face({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, 2.0f * hy, 2.0f * hz);
    push_face({0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 2.0f * hz, 2.0f * hy);

    auto& submesh = data.submeshes.emplace_back();
    submesh.index_start = 0;
    submesh.index_count = 36;
    submesh.material_id = VulkanEngine::MaterialId{0};
    submesh.sphere = VulkanEngine::BoundingSphere{
        {0.0f, 0.0f, 0.0f}, glm::length(extents) * 0.5f};
    submesh.obb = VulkanEngine::BoundingOBB{
        {0.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, hx,
        {0.0f, 1.0f, 0.0f}, hy, {0.0f, 0.0f, 1.0f}, hz};

    // Stretch the authored world-unit UVs uniformly so each texel covers 3x
    // the surface area (see `kUvTexelScale`).
    for (auto& uv : data.uvs) {
        uv *= kUvTexelScale;
    }
    return data;
}

// A wheel as a short disc (low-segment cylinder, axis +y, matching
// `ShapeType::Cylinder`'s and `WheelDesc::wheel_up`'s convention): side wall
// plus two capped faces. The cap checker pattern makes the wheel's spin, steer
// and suspension travel visible.
[[nodiscard]] VulkanEngine::SceneLoader::LoadedMeshData BuildWheelDisc(const float radius, const float width) {
    constexpr std::uint32_t kSegments = 12;
    const float hw = width * 0.5f;

    VulkanEngine::SceneLoader::LoadedMeshData data{};
    data.indices.reserve(kSegments * 4 * 3);

    glm::vec3 ring[kSegments];
    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float a = static_cast<float>(i) / static_cast<float>(kSegments) * 2.0f * 3.14159265f;
        ring[i] = {radius * std::cos(a), 0.0f, radius * std::sin(a)};
    }

    const auto tangent_h = [&](const glm::vec3& t) {
        data.tangents.insert(data.tangents.end(), {t.x, t.y, t.z});
        data.tangent_handedness.push_back(1.0f);
    };
    const auto position = [&](const glm::vec3& p) {
        data.positions.insert(data.positions.end(), {p.x, p.y, p.z});
    };
    const auto normal = [&](const glm::vec3& n) {
        data.normals.insert(data.normals.end(), {n.x, n.y, n.z});
    };

    // Side wall: bottom/top ring pairs with radial normals; u accumulates arc
    // length so the checker rolls with the wheel instead of sliding. One extra
    // iteration repeats segment 0 to close the seam.
    std::uint32_t prev_bottom = 0;
    std::uint32_t prev_top = 0;
    for (std::uint32_t i = 0; i <= kSegments; ++i) {
        const float a = static_cast<float>(i) / static_cast<float>(kSegments) * 2.0f * 3.14159265f;
        const glm::vec3 radial{std::cos(a), 0.0f, std::sin(a)};

        const bool first = i == 0;
        const auto bottom = static_cast<std::uint32_t>(data.positions.size() / 3);
        position({radial.x * radius, -hw, radial.z * radius});
        normal(radial);
        tangent_h({-radial.z, 0.0f, radial.x});
        const auto top = bottom + 1;
        position({radial.x * radius, hw, radial.z * radius});
        normal(radial);
        tangent_h({-radial.z, 0.0f, radial.x});

        const float arc = a * radius;
        data.uvs.insert(data.uvs.end(), {arc, 0.0f});
        data.uvs.insert(data.uvs.end(), {arc, 1.0f});

        if (!first) {
            // Wound to face the outward radial normal: walking the ring at
            // increasing angle, the quad front face is
            // prev_bottom <- bottom -> top (and prev_bottom -> top -> prev_top),
            // the mirror of the first version, whose normals pointed inward and
            // were back-face culled away.
            data.indices.insert(data.indices.end(),
                                {bottom, prev_bottom, top, top, prev_bottom, prev_top});
        }
        prev_bottom = bottom;
        prev_top = top;
    }

    // Caps: one center vertex per cap, fan from duplicated ring vertices.
    // The top cap walks the ring in decreasing angle; the bottom cap in
    // increasing angle, so both fans face outward.
    for (const int cap_sign : {-1, 1}) {
        const glm::vec3 cap_normal{0.0f, static_cast<float>(cap_sign), 0.0f};
        const auto center = static_cast<std::uint32_t>(data.positions.size() / 3);
        position({0.0f, cap_sign * hw, 0.0f});
        normal(cap_normal);
        tangent_h({1.0f, 0.0f, 0.0f});
        data.uvs.insert(data.uvs.end(), {0.0f, 0.0f});

        std::uint32_t prev_rim = center;
        for (std::uint32_t step = 0; step <= kSegments; ++step) {
            const std::uint32_t k = cap_sign > 0 ? (kSegments - step) % kSegments : step % kSegments;
            const auto rim = static_cast<std::uint32_t>(data.positions.size() / 3);
            position({ring[k].x, cap_sign * hw, ring[k].z});
            normal(cap_normal);
            tangent_h({1.0f, 0.0f, 0.0f});
            // Cap UVs in world units (1 uv = 1 m, matching the side wall's arc
            // length mapping) so checker density matches the rest of the wheel.
            data.uvs.insert(data.uvs.end(), {ring[k].x, ring[k].z});
            if (step > 0) {
                data.indices.insert(data.indices.end(), {center, prev_rim, rim});
            }
            prev_rim = rim;
        }
    }

    auto& submesh = data.submeshes.emplace_back();
    submesh.index_start = 0;
    submesh.index_count = static_cast<std::uint32_t>(data.indices.size());
    submesh.material_id = VulkanEngine::MaterialId{0};
    submesh.sphere = VulkanEngine::BoundingSphere{
        {0.0f, 0.0f, 0.0f}, std::sqrt(radius * radius + hw * hw)};
    submesh.obb = VulkanEngine::BoundingOBB{
        {0.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, radius,
        {0.0f, 1.0f, 0.0f}, hw, {0.0f, 0.0f, 1.0f}, radius};

    // Same global texel stretch as the boxes (see `kUvTexelScale`); applied
    // after authoring so the side wall's arc-length mapping and the caps'
    // world-unit mapping stay consistent with each other.
    for (auto& uv : data.uvs) {
        uv *= kUvTexelScale;
    }
    return data;
}

} // namespace

// The named starting sets for the tuning tab (`kPresets`, `kDefaultPreset`)
// live in `Game.cppm`, next to `TuningPreset` — the table is the single source
// of truth for the tunables' initial values there, and `Game.cpp` only
// consumes it (the field-initializer wiring happens in the module interface).

// Sample one wheel's world pose — steering and chassis orientation, spin
// excluded — from the vehicle, and re-apply the accumulated spin to the
// render-only wheel entity's Transform. The wheel model space used by the
// engine (and the disc mesh) puts the spin axle on +y, and Jolt accumulates
// the spin as a negated rotation about that axle — the convention its own
// debug renderer uses — so the spin goes in model space, post-multiplied.
void Game::UpdateWheelEntity(const std::uint32_t index) {
    auto* world = engine_game_.GetPhysicsWorld();
    if (world == nullptr || vehicle_ == nullptr) return;
    if (index >= wheel_entities_.size()) return;
    auto* entity =
        engine_game_.GetContext().GetComponentRegistry().TryGetEntity(wheel_entities_[index]);
    auto* transform = entity != nullptr
        ? entity->GetComponent<VulkanEngine::Components::Transform>()
        : nullptr;
    if (transform == nullptr) return;

    const auto pose = vehicle_->GetWheelPose(index);
    const glm::quat spin = glm::angleAxis(
        -vehicle_->GetWheelRotationAngle(index), glm::vec3{0.0f, 1.0f, 0.0f});
    transform->position = pose.position;
    transform->rotation = pose.rotation * spin;
    // The disc mesh is authored at unit radius and unit width; these scales
    // make a live radius/width tunable move the render without touching the
    // scene. (The scale is symmetric about the spin axle, so it commutes with
    // the spin rotation.)
    transform->scale = {wheel_radius_, wheel_width_, wheel_radius_};
}

// The tuning tab: every vehicle tunable, grouped under foldable headers so the
// window stays small until a section is expanded.
//
// Two apply semantics, side by side:
// - Chassis body properties (mass, friction) go through the world's live
//   setters and take effect on the next step, mid-drive — safe here because
//   they never mutate the world's structure, only body properties (the
//   body-interface locking discipline is honored; nothing structural is
//   touched).
// - Everything that lives in the `VehicleDesc` (engine, differentials, wheel
//   and tire settings) is consumed once by `WheeledVehicle::Initialize`, so an
//   edit marks the vehicle dirty; the rebuild happens in the next fixed update
//   on the same chassis body — the same discipline the Reset button follows.
bool Game::DrawVehicleTuningWindow() {
    auto* world = engine_game_.GetPhysicsWorld();
    const bool live_chassis = world != nullptr && chassis_body_.IsValid();
    bool rebuild = false;

    // Presets and field reset: both write whole preset sets into the tunables
    // through the same `ApplyPreset` path — the live Chassis setters inside it
    // cover mass/friction, and the returned `rebuild = true` queues the
    // `VehicleDesc`-level rebuild in the next fixed update.
    if (ImGui::CollapsingHeader("Preset", ImGuiTreeNodeFlags_DefaultOpen)) {
        int preset = preset_index_;
        // The label is this widget's ImGui ID; "##" keeps it hidden so it can't
        // clash with the header's own "Preset" ID.
        if (ImGui::Combo("##preset", &preset, "Normal car\0Sports car\0\0")) {
            preset_index_ = preset;
            ApplyPreset(kPresets[preset]);
            rebuild = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset fields")) {
            preset_index_ = kDefaultPreset;
            ApplyPreset(kPresets[kDefaultPreset]);
            rebuild = true;
        }
    }

    if (ImGui::CollapsingHeader("Chassis", ImGuiTreeNodeFlags_DefaultOpen)) {
        float mass = chassis_mass_;
        if (ImGui::SliderFloat("Mass (kg)", &mass, 200.0f, 3000.0f)) {
            chassis_mass_ = mass;
            if (live_chassis) world->SetBodyMass(chassis_body_, chassis_mass_);
        }
        float friction = chassis_friction_;
        if (ImGui::SliderFloat("Chassis friction", &friction, 0.0f, 1.0f)) {
            chassis_friction_ = friction;
            // Only felt when the bare body scrapes ground — tire grip is the
            // separate `friction_slip`.
            if (live_chassis) world->SetBodyFriction(chassis_body_, chassis_friction_);
        }
    }

    if (ImGui::CollapsingHeader("Engine", ImGuiTreeNodeFlags_DefaultOpen)) {
        rebuild |= ImGui::SliderFloat("Max torque", &engine_max_torque_, 100.0f, 2000.0f);
        rebuild |= ImGui::SliderInt("Min RPM", &engine_min_rpm_, 500, 4000);
        rebuild |= ImGui::SliderInt("Max RPM", &engine_max_rpm_, 1500, 9000);
        engine_min_rpm_ = std::min(engine_min_rpm_, engine_max_rpm_ - 100);
        engine_max_rpm_ = std::max(engine_max_rpm_, engine_min_rpm_ + 100);
    }

    if (ImGui::CollapsingHeader("Drivetrain")) {
        // Wheel indices follow `BuildVehicleDesc`'s push order (front 0/1,
        // rear 2/3); the split is fixed per mode here so the torque ratios
        // always sum to 1.
        int drive = static_cast<int>(drivetrain_);
        if (ImGui::Combo("Drive", &drive, "AWD (50:50)\0FWD\0RWD\0\0")) {
            drivetrain_ = static_cast<Drivetrain>(drive);
            rebuild = true;
        }
    }

    if (ImGui::CollapsingHeader("Wheels & suspension")) {
        // Radius/width also rescale the render-only wheel entities
        // (`UpdateWheelEntity`), so a drag is visible immediately even though
        // the physics picks it up on the rebuild.
        rebuild |= ImGui::SliderFloat("Radius", &wheel_radius_, 0.2f, 0.6f);
        rebuild |= ImGui::SliderFloat("Width", &wheel_width_, 0.15f, 0.4f);
        rebuild |= ImGui::SliderFloat("Track spread", &wheel_spread_, 0.5f, 1.2f);
        rebuild |= ImGui::SliderFloat("Wheelbase", &wheel_base_, 0.9f, 1.8f);
        rebuild |= ImGui::SliderFloat("Max steer angle", &steer_angle_, 0.2f, 1.0f);
        rebuild |= ImGui::SliderFloat("Suspension frequency", &suspension_frequency_, 0.8f, 3.0f);
        rebuild |= ImGui::SliderFloat("Suspension damping", &suspension_damping_, 0.2f, 1.2f);
    }

    if (ImGui::CollapsingHeader("Tire grip")) {
        rebuild |= ImGui::SliderFloat("Longitudinal grip", &friction_slip_, 0.3f, 3.0f);
        rebuild |= ImGui::SliderFloat("Lateral grip", &sideways_friction_slip_, 0.1f, 1.5f);
        rebuild |= ImGui::SliderFloat("Handbrake torque", &handbrake_torque_, 0.0f, 4000.0f);
    }

    return rebuild;
}

// Overwrites every tuning-tab tunable with the preset's values, with the two
// apply semantics the tab documents: chassis body properties go through the
// world's live setters (mid-drive safe), the `VehicleDesc`-fed ones are plain
// field writes the caller's queued rebuild picks up. The wheel entity scale
// reads `wheel_radius_`/`wheel_width_` per frame, so a preset's wheel size is
// visible immediately even before the rebuild.
void Game::ApplyPreset(const TuningPreset& preset) {
    chassis_mass_ = preset.chassis_mass;
    chassis_friction_ = preset.chassis_friction;
    engine_max_torque_ = preset.engine_max_torque;
    engine_min_rpm_ = preset.engine_min_rpm;
    engine_max_rpm_ = preset.engine_max_rpm;
    drivetrain_ = preset.drivetrain;
    wheel_radius_ = preset.wheel_radius;
    wheel_width_ = preset.wheel_width;
    wheel_spread_ = preset.wheel_spread;
    wheel_base_ = preset.wheel_base;
    steer_angle_ = preset.steer_angle;
    suspension_frequency_ = preset.suspension_frequency;
    suspension_damping_ = preset.suspension_damping;
    friction_slip_ = preset.friction_slip;
    sideways_friction_slip_ = preset.sideways_friction_slip;
    handbrake_torque_ = preset.handbrake_torque;

    if (auto* world = engine_game_.GetPhysicsWorld();
        world != nullptr && chassis_body_.IsValid()) {
        world->SetBodyMass(chassis_body_, chassis_mass_);
        world->SetBodyFriction(chassis_body_, chassis_friction_);
    }
}

// The vehicle description from the current tuning tab values — the single path
// both the initial spawn and every rebuild go through, so the tab can be read
// as the truth for the whole desc.
VulkanEngine::Physics::VehicleDesc Game::BuildVehicleDesc() const {
    const auto wheel = [this](const glm::vec3 position, const float max_steer_angle,
                              const float max_handbrake_torque) {
        VulkanEngine::Physics::WheelDesc desc{};
        desc.position = position;
        desc.suspension_direction = {0.0f, -1.0f, 0.0f};
        desc.steering_axis = {0.0f, 1.0f, 0.0f};
        desc.wheel_up = {0.0f, 1.0f, 0.0f};
        desc.wheel_forward = {0.0f, 0.0f, 1.0f};
        desc.radius = wheel_radius_;
        desc.width = wheel_width_;
        desc.suspension_min_length = 0.1f;
        desc.suspension_max_length = 0.35f;
        desc.suspension_frequency = suspension_frequency_;
        desc.suspension_damping = suspension_damping_;
        // Street-car grip: below Jolt's slick default of 8, so the
        // tires themselves give up before the body rolls.
        desc.friction_slip = friction_slip_;
        desc.sideways_friction_slip = sideways_friction_slip_;
        desc.max_steer_angle = max_steer_angle;
        desc.max_handbrake_torque = max_handbrake_torque;
        return desc;
    };

    VulkanEngine::Physics::VehicleDesc desc{};
    desc.chassis = chassis_body_;
    desc.up = {0.0f, 1.0f, 0.0f};
    desc.forward = {0.0f, 0.0f, 1.0f};
    desc.max_pitch_roll_angle = std::numbers::pi; // dont prevent roll

    // Front pair steers, rear pair carries the handbrake — a classic layout
    // either way the differential routes the torque.
    desc.wheels.push_back(wheel({-wheel_spread_, -0.1f, wheel_base_}, steer_angle_, 0.0f));
    desc.wheels.push_back(wheel({wheel_spread_, -0.1f, wheel_base_}, steer_angle_, 0.0f));
    desc.wheels.push_back(wheel({-wheel_spread_, -0.1f, -wheel_base_}, 0.0f, handbrake_torque_));
    desc.wheels.push_back(wheel({wheel_spread_, -0.1f, -wheel_base_}, 0.0f, handbrake_torque_));

    desc.engine_max_torque = engine_max_torque_;
    desc.engine_min_rpm = static_cast<float>(engine_min_rpm_);
    desc.engine_max_rpm = static_cast<float>(engine_max_rpm_);

    // Drivetrain: one differential per driven axle, torque ratios summing to 1.
    // Wheel indices follow the push order: front left/right 0/1, rear 2/3.
    const auto push_differential = [&desc](const std::int32_t left, const std::int32_t right,
                                           const float torque_ratio) {
        VulkanEngine::Physics::DifferentialDesc differential{};
        differential.left_wheel = left;
        differential.right_wheel = right;
        differential.engine_torque_ratio = torque_ratio;
        desc.differentials.push_back(differential);
    };
    switch (drivetrain_) {
    case Drivetrain::Front:
        push_differential(0, 1, 1.0f);
        break;
    case Drivetrain::Rear:
        push_differential(2, 3, 1.0f);
        break;
    default: // all-wheel drive
        push_differential(0, 1, 0.5f);
        push_differential(2, 3, 0.5f);
        break;
    }
    return desc;
}

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
    // 1. Configure and init engine subsystems
    VulkanEngine::GameConfig config{};
    // The physics debug overlay draws through ImGui's draw list, so the ImGui
    // system and its pass must be part of the frame.
    config.enable_imgui = true;
    config.renderer_config.clear_color = {0.1f, 0.12f, 0.15f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();

    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    // 2. Register the app fragment shader and init the renderer
    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    auto frag_id = Shaders::PhysicsVehicle::CheckerFrag::Register(shader_mgr, config.shader_data_dir);
    auto vert_id = engine_game_.GetContext().GetShaderIds().main_indir_vert;
    if (!engine_game_.InitRenderer(ctx, vert_id, frag_id, &shader_mgr)) {
        return false;
    }

    // 3. Meshes: authored boxes for the whole static scene and the chassis,
    // plus the wheel disc, all with the checker shader.
    auto& mesh_registry = engine_game_.GetMeshRegistry();
    const auto register_box = [&mesh_registry](const glm::vec3 extents) {
        return mesh_registry.Register(
            VulkanEngine::SceneLoader::ToMeshData(BuildBoxMesh(extents)));
    };
    const std::uint32_t ground_mesh_id = register_box({ground_extend_.first, 1.0f, ground_extend_.second});
    const std::uint32_t crate_mesh_id = register_box({1.0f, 1.0f, 1.0f});
    const std::uint32_t block_mesh_id = register_box({2.0f, 2.0f, 2.0f});
    const std::uint32_t ramp_mesh_id = register_box({4.0f, 0.4f, 10.0f});
    const std::uint32_t chassis_mesh_id = register_box({1.6f, 0.5f, 3.6f});
    // The wheel disc is authored at unit radius and unit width; each wheel
    // entity scales it by `wheel_radius_`/`wheel_width_` (see
    // `UpdateWheelEntity`), so the radius tunable moves the render live.
    const std::uint32_t wheel_mesh_id = mesh_registry.Register(
        VulkanEngine::SceneLoader::ToMeshData(BuildWheelDisc(1.0f, 1.0f)));

    engine_game_.MarkSceneValid();

    // 4. Camera (moved by the chase camera logic each frame)
    auto& component_registry = engine_game_.GetContext().GetComponentRegistry();
    engine_game_.CreateCamera(component_registry);
    if (auto* camera = engine_game_.GetCamera()) {
        camera->position = {0.0f, 4.0f, -14.0f};
        camera->target = {0.0f, 1.0f, 0.0f};
    }

    // 5. Input: quit, plus the driver bindings.
    ctx.quit_action_handle = ctx.input_system->BindAction("quit",
        VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));
#ifdef VKENGINE_PHYSICS
    drive_ = ctx.input_system->BindAction<2>("drive", VulkanEngine::Input::ActionConfig{});
    drive_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_A).Component(0, -1.0f));
    drive_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_D).Component(0, +1.0f));
    drive_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_W).Component(1, +1.0f));
    drive_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_S).Component(1, -1.0f));
    // Gamepad: left stick x steers, triggers throttle (they idle at 0, so the
    // scaled -1 backward binding contributes nothing at rest).
    drive_.Bind(VulkanEngine::Input::InputBinding::GamepadAxis(SDL_GAMEPAD_AXIS_LEFTX)
                    .Component(0, +1.0f)
                    .Deadzone(0.15f));
    drive_.Bind(VulkanEngine::Input::InputBinding::GamepadAxis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
                    .Component(1, +1.0f)
                    .Deadzone(0.05f));
    drive_.Bind(VulkanEngine::Input::InputBinding::GamepadAxis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER)
                    .Component(1, -1.0f)
                    .Deadzone(0.05f));
    brake_ = ctx.input_system->BindAction<1>("brake", VulkanEngine::Input::ActionConfig{});
    brake_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_SPACE));
    // Gamepad: X / south button brakes, triangle / north handbrakes.
    brake_.Bind(VulkanEngine::Input::InputBinding::GamepadButton(SDL_GAMEPAD_BUTTON_SOUTH));
    handbrake_ = ctx.input_system->BindAction<1>("handbrake", VulkanEngine::Input::ActionConfig{});
    handbrake_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_LSHIFT));
    handbrake_.Bind(VulkanEngine::Input::InputBinding::GamepadButton(SDL_GAMEPAD_BUTTON_NORTH));
#endif

#ifdef VKENGINE_PHYSICS
    // 6. Physics scene.
    auto* world = engine_game_.GetPhysicsWorld();
    if (world == nullptr) {
        LOGIFACE_LOG(warn, "physics: world unavailable; running without simulation");
    } else {
        // Static helpers are plain entities: `Transform` carries pose, physics
        // picks it up when the `RigidBody` component is added, and the
        // engine-registered slots (PhysicsInput -> PhysicsSimulate ->
        // PhysicsWriteBack) run every fixed step.
        const auto spawn_body = [&](const glm::vec3 position, const glm::quat rotation,
                                    const glm::vec3 box_extent,
                                    const VulkanEngine::Physics::BodyMotionType motion,
                                    const float mass, const float friction,
                                    const float angular_damping = 0.05f) {
            auto& spawned = component_registry.CreateEntity();
            auto& transform =
                component_registry.AddComponent<VulkanEngine::Components::Transform>(spawned);
            transform.position = position;
            transform.rotation = rotation;

            VulkanEngine::Physics::RigidBody body{};
            body.shape.type = VulkanEngine::Physics::ShapeType::Box;
            body.shape.box_extent = box_extent;
            body.motion = motion;
            body.mass = mass;
            body.friction = friction;
            body.angular_damping = angular_damping;
            component_registry.AddComponent<VulkanEngine::Physics::RigidBody>(spawned, body);
            return &spawned;
        };

        const auto spawn_static_with_mesh = [&](const glm::vec3 position, const glm::quat rotation,
                                                const glm::vec3 box_extent,
                                                const std::uint32_t mesh_id) {
            auto* spawned = spawn_body(position, rotation, box_extent,
                                       VulkanEngine::Physics::BodyMotionType::Static, 0.0f, 0.9f);
            auto& mesh_ref = component_registry.AddComponent<VulkanEngine::Components::MeshReference>(*spawned);
            mesh_ref.loaded_mesh_id = mesh_id;
        };

        using VulkanEngine::Physics::BodyMotionType;

        // Ground: a 60 x 1 x 60 slab whose top face sits on the y = 0 plane.
        spawn_static_with_mesh({0.0f, -0.5f, 0.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
                               {ground_extend_.first, 1.0f, ground_extend_.second}, ground_mesh_id);

        // A gentle ramp the chassis can climb at speed, low edge resting on
        // the ground plane.
        const glm::quat ramp_pitch = glm::angleAxis(glm::radians(-9.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        spawn_static_with_mesh({0.0f, 0.75f, 8.0f}, ramp_pitch, {4.0f, 0.4f, 10.0f}, ramp_mesh_id);

        // Obstacles on the far side: a heavy block and a small crate stack.
        spawn_static_with_mesh({-3.0f, 1.0f, 22.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
                               {2.0f, 2.0f, 2.0f}, block_mesh_id);
        for (std::uint32_t i = 0; i < 3; ++i) {
            auto* spawned = spawn_body({3.0f, 0.55f + static_cast<float>(i) * 1.05f, 22.0f},
                                       glm::quat{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f},
                                       BodyMotionType::Dynamic, 30.0f, 0.6f);
            auto& mesh_ref =
                component_registry.AddComponent<VulkanEngine::Components::MeshReference>(*spawned);
            mesh_ref.loaded_mesh_id = crate_mesh_id;
        }

        // The chassis: an ordinary dynamic entity with a mesh, so the renderer
        // draws it exactly like any other scene object. The angular damping
        // damps the roll/pitch rate directly — the cheap part of not tipping.
        auto* chassis = spawn_body({0.0f, 1.2f, -4.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
                                   {1.6f, 0.5f, 3.6f}, BodyMotionType::Dynamic, chassis_mass_,
                                   chassis_friction_, 0.2f);
        auto& chassis_mesh =
            component_registry.AddComponent<VulkanEngine::Components::MeshReference>(*chassis);
        chassis_mesh.loaded_mesh_id = chassis_mesh_id;
        chassis_body_ = component_registry
                            .GetComponent<VulkanEngine::Physics::RigidBody>(*chassis)
                            ->body;
        if (!chassis_body_.IsValid()) {
            LOGIFACE_LOG(error, "vehicle: chassis body was not created; no vehicle");
        } else {
            // The vehicle desc comes from the tuning-tab values shared with the
            // rebuild path (see `BuildVehicleDesc`): front pair steers, rear
            // pair carries the handbrake, differentials from the drivetrain
            // toggle.
            VulkanEngine::Physics::VehicleDesc desc = BuildVehicleDesc();

            vehicle_ = std::make_unique<VulkanEngine::Physics::WheeledVehicle>();
            if (!vehicle_->Initialize(*world, desc)) {
                LOGIFACE_LOG(error, "vehicle: initialization failed; driving is disabled");
                vehicle_.reset();
            } else {
                LOGIFACE_LOG(info, "vehicle: four wheels initialized on the chassis body");

                // Render-only wheel entities: a wheel is a ray cast from the
                // chassis, not a body, so it has no physics presence. Its pose
                // is sampled from the vehicle each frame instead.
                for (std::uint32_t i = 0; i < 4; ++i) {
                    auto& spawned = component_registry.CreateEntity();
                    component_registry.AddComponent<VulkanEngine::Components::Transform>(spawned);
                    auto& mesh_ref =
                        component_registry.AddComponent<VulkanEngine::Components::MeshReference>(spawned);
                    mesh_ref.loaded_mesh_id = wheel_mesh_id;
                    wheel_entities_.push_back(spawned.GetId());
                    UpdateWheelEntity(i);
                }
                LOGIFACE_LOG(info, std::format(
                    "vehicle: {} wheel entities attached", wheel_entities_.size()));
            }
        }

        // Log the first contact purely as headless-run evidence.
        world->SetContactCallback([this](const VulkanEngine::Physics::ContactEvent& event) {
            if (logged_first_contact_) return;
            logged_first_contact_ = true;
            LOGIFACE_LOG(info, std::format(
                "physics: first contact (phase {}) between bodies {} and {} at "
                "({:.2f}, {:.2f}, {:.2f}), penetration {:.4f}",
                VulkanEngine::Physics::ContactPhaseName(event.phase), event.body_a.index(),
                event.body_b.index(), event.position.x, event.position.y, event.position.z,
                event.penetration));
        });

        // Anchor the scene: the solver state of every body exactly as spawned
        // (the vehicle's suspension at rest length, velocities zero). The Reset
        // button restores this snapshot. The world's body set is final here, so
        // `RestoreSnapshot` will always accept it.
        scene_snapshot_ = world->SaveSnapshot();
        LOGIFACE_LOG(info, std::format(
            "vehicle: scene snapshot anchored ({} bytes, {} bodies)",
            scene_snapshot_.size(), world->BodyCount()));
    }

    // 7. Physics debug overlay (wireframe outlines over the meshed scene).
    physics_overlay_.Settings().wireframe = true;
    physics_overlay_.Settings().draw_bounding_boxes = false;
    if (auto* imgui = engine_game_.GetImGuiSystem()) {
        const auto* context = &ctx;
        imgui_draw_token_ = imgui->draw_callbacks.Register([this, context]() {
            auto* world = engine_game_.GetPhysicsWorld();
            auto* camera = engine_game_.GetCamera();
            if (world == nullptr || camera == nullptr) return;

            std::uint32_t width = 0;
            std::uint32_t height = 0;
            (void)context->bootstrap->GetBackend().GetSwapchainExtent(width, height);
            const float aspect =
                height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
            physics_overlay_.Draw(*world, *camera, aspect);

            if (!logged_overlay_geometry_ && world->StepCount() > 0) {
                const auto& geometry = physics_overlay_.Geometry();
                if (!geometry.lines.empty() || !geometry.triangles.empty()) {
                    logged_overlay_geometry_ = true;
                    LOGIFACE_LOG(info, std::format(
                        "physics overlay: {} debug lines, {} debug triangles, {} bodies, "
                        "{} fixed steps",
                        geometry.lines.size(), geometry.triangles.size(), world->BodyCount(),
                        world->StepCount()));
                }
            }

            // The demo's own window: scene reset. Only the request flag is set
            // here — the restore runs in the next fixed update, before the
            // physics schedule, so it never mutates the world mid-frame.
            ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("Vehicle", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
            if (ImGui::Button("Reset scene")) {
                reset_requested_ = true;
            }
            if (DrawVehicleTuningWindow()) {
                rebuild_requested_ = true;
            }
            ImGui::End();
        });
    }
#endif

    return true;
}

void Game::OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
#ifdef VKENGINE_PHYSICS
    // A Reset request queued by the previous frame's ImGui button: restore the
    // anchored scene snapshot before the physics schedule runs, so the world
    // steps this frame from the spawned state. The restore is all-or-nothing —
    // a rejected snapshot (version, size, hash) leaves the world untouched.
    if (reset_requested_) {
        reset_requested_ = false;
        if (auto* world = engine_game_.GetPhysicsWorld(); world != nullptr) {
            if (world->RestoreSnapshot(scene_snapshot_)) {
                LOGIFACE_LOG(info, "vehicle: scene reset to the anchored snapshot");
            } else {
                LOGIFACE_LOG(error, "vehicle: scene reset failed; snapshot was rejected");
            }
        }
    }

    // A tuning edit that touches a `VehicleDesc` value: rebuild the vehicle —
    // its controller/wheel settings are baked into the Jolt constraint at
    // `Initialize`, so there is no live path. Shutdown first (the constraint
    // references the chassis body and must be removed first, like any other
    // constraint), then re-`Initialize` on the same body: the car keeps its
    // momentum, only the tuning changes.
    if (rebuild_requested_) {
        rebuild_requested_ = false;
        auto* world = engine_game_.GetPhysicsWorld();
        if (world != nullptr && chassis_body_.IsValid()) {
            if (vehicle_ != nullptr) vehicle_->Shutdown();
            vehicle_.reset();
            auto rebuilt = std::make_unique<VulkanEngine::Physics::WheeledVehicle>();
            if (rebuilt->Initialize(*world, BuildVehicleDesc())) {
                vehicle_ = std::move(rebuilt);
                LOGIFACE_LOG(info, "vehicle: rebuilt with the current tuning");
            } else {
                LOGIFACE_LOG(error, "vehicle: tuning rebuild failed; driving is disabled");
            }
        }
    }

    // Driver input: stored on the vehicle's controller and applied by the
    // world's next Step() inside the physics simulate slot — so it must be
    // written before the engine's fixed update runs the schedule.
    if (vehicle_ != nullptr) {
        const float fspeed = std::abs(vehicle_->GetSpeed());
        const float scaled_fspeed = fspeed / assisted_steering_scaler_;
        const float temp = 1.0f / scaled_fspeed;
        const float min_addition = temp + assisted_steering_min_;
        const float cap = std::min(min_addition, assisted_steering_max_);

        const auto drive = drive_.Value();
        const auto brake = brake_.Value();
        const auto handbrake = handbrake_.Value();
        vehicle_->SetDriverInput(drive[1], drive[0] * cap, brake[0], handbrake[0]);
    }
#endif
    engine_game_.FixedUpdate(ctx);

#ifdef VKENGINE_PHYSICS
    // One speed log per run; headless runs use it as driving evidence.
    if (!logged_speed_ && vehicle_ != nullptr && vehicle_->GetSpeed() > 3.0f) {
        logged_speed_ = true;
        LOGIFACE_LOG(info, std::format("vehicle: rolling at {:.1f} m/s", vehicle_->GetSpeed()));
    }
#endif
}

void Game::OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameUpdate(ctx);
#ifdef VKENGINE_PHYSICS
    // Chase camera: behind and above the chassis, looking at its mid-height.
    auto* world = engine_game_.GetPhysicsWorld();
    auto* camera = engine_game_.GetCamera();
    if (world != nullptr && camera != nullptr && chassis_body_.IsValid()) {
        const auto pose = world->GetPose(chassis_body_);
        camera->position = pose.position + pose.rotation * glm::vec3{0.0f, 3.2f, -8.5f};
        camera->target = pose.position + glm::vec3{0.0f, 0.6f, 0.0f};
    }
    for (std::uint32_t i = 0; i < wheel_entities_.size(); ++i) {
        UpdateWheelEntity(i);
    }
#endif
}

void Game::OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void Game::OnShutdown(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
#ifdef VKENGINE_PHYSICS
    // Detach before the ImGui system (and its callback list) is destroyed, and
    // drop the vehicle before the world: the constraint references the chassis
    // body and must be removed first, exactly like any other constraint.
    imgui_draw_token_ = {};
    if (vehicle_ != nullptr) {
        vehicle_->Shutdown();
    }
    vehicle_.reset();
    wheel_entities_.clear();
#endif
    engine_game_.Shutdown();
}

} // namespace Examples::PhysicsVehicle::Game