module;
#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_keycode.h>
#include <imgui.h>

#include <logging/logging_macros.hpp>

module Examples.InfiniteRunner.Game;

import std;

import logiface;

import VulkanEngine.GameEngine;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.ShaderManager;
import VulkanEngine.Components.MaterialOverride;
import VulkanEngine.DataCipher;
import Examples.InfiniteRunner.Balance;
import Examples.InfiniteRunner.Sweep;
import Examples.InfiniteRunner.Wall;
import Examples.InfiniteRunner.Leaderboard.Account;
import Examples.InfiniteRunner.Leaderboard.Client;
import Examples.InfiniteRunner.Leaderboard.Config;
import Examples.InfiniteRunner.Leaderboard.Log;
import Examples.InfiniteRunner.Leaderboard.Protocol;
import Examples.InfiniteRunner.Account.ProfileStore;

namespace Examples::InfiniteRunner::Game {

// anonymous namespaces are a c++ stable, a way to hide functions to be hidden from other translation units
namespace {

// UTC calendar date for a score timestamp, or "-" when it predates date
// recording. Local time would be friendlier but needs the OS timezone, and the
// server is the authority on when a score was recorded.
[[nodiscard]] std::string FormatDate(std::uint64_t unix_seconds) {
    if (unix_seconds == 0) {
        return "-";
    }
    const std::time_t seconds = static_cast<std::time_t>(unix_seconds);
    const std::tm* utc = std::gmtime(&seconds);
    if (utc == nullptr) {
        return "-";
    }
    return std::format("{:04d}-{:02d}-{:02d}", utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday);
}

// Player cube box centred on the corridor at (center_x, 0, 0), for the
// discrete overlap check. Extents come from the active ruleset.
[[nodiscard]] Sweep::Aabb PlayerBox(const float center_x, const float player_size) {
    const float half = player_size * 0.5f;
    return {{center_x - half, -half, -half}, {center_x + half, half, half}};
}

// Builds a unit cube (centred on the origin, half-extent 0.5) as LoadedMeshData.
// Bounding volumes and the indexed-draw vertex window are filled in later by
// MeshRegistry::Register via EnsureSubmeshBounds.
VulkanEngine::SceneLoader::LoadedMeshData MakeUnitCube() {
    VulkanEngine::SceneLoader::LoadedMeshData cube{};

    const auto add_face = [&cube](const glm::vec3& normal, const glm::vec3& up) {
        const glm::vec3 right = glm::normalize(glm::cross(up, normal));
        const glm::vec3 center = normal * 0.5f;
        const glm::vec3 corners[4] = {
            center - right * 0.5f - up * 0.5f,
            center + right * 0.5f - up * 0.5f,
            center + right * 0.5f + up * 0.5f,
            center - right * 0.5f + up * 0.5f,
        };
        constexpr float uvs[4][2] = {{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}};

        const auto base = static_cast<std::uint32_t>(cube.positions.size() / 3U);
        for (int i = 0; i < 4; ++i) {
            cube.positions.push_back(corners[i].x);
            cube.positions.push_back(corners[i].y);
            cube.positions.push_back(corners[i].z);
            cube.normals.push_back(normal.x);
            cube.normals.push_back(normal.y);
            cube.normals.push_back(normal.z);
            cube.uvs.push_back(uvs[i][0]);
            cube.uvs.push_back(uvs[i][1]);
        }
        cube.indices.insert(cube.indices.end(),
                            {base + 0U, base + 1U, base + 2U, base + 2U, base + 3U, base + 0U});
    };

    add_face({0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f});
    add_face({0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f});
    add_face({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    add_face({-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    add_face({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f});
    add_face({0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f});

    VulkanEngine::SubMesh submesh{};
    submesh.index_start = 0;
    submesh.index_count = static_cast<std::uint32_t>(cube.indices.size());
    cube.submeshes.push_back(submesh);
    return cube;
}

struct SunSettings {
    float azimuth_degrees   = 45.0f;   // rotation about +Y
    float elevation_degrees = 45.0f;   // above horizon (default 0.5,-0.707,0.5 ≈ 45/45)
    float strength          = 2.0f;    // direct multiplier  -> Light::color.w
    glm::vec3 color         = {1.0f, 0.95f, 0.9f};
    float ambient_strength  = 1.0f;    // -> SceneHeader::ambient_color.a
    glm::vec3 ambient_tint  = {0.03f, 0.03f, 0.03f};
};

// Direction the light travels, from an azimuth/elevation pair.
glm::vec3 SunTravelDirection(float azimuth_deg, float elevation_deg) {
    const float az = glm::radians(azimuth_deg);
    const float el = glm::radians(elevation_deg);
    return glm::normalize(glm::vec3{
        std::cos(el) * std::sin(az),
        -std::sin(el),
        std::cos(el) * std::cos(az),
    });
}

void ApplySun(VulkanEngine::GameEngine& engine, const SunSettings& sun) {
    using VulkanEngine::SceneRenderer::SceneHeader;
    using VulkanEngine::SceneRenderer::Light;

    const glm::vec3 dir = SunTravelDirection(sun.azimuth_degrees, sun.elevation_degrees);

    SceneHeader header{};
    header.ambient_color[0] = sun.ambient_tint.r;
    header.ambient_color[1] = sun.ambient_tint.g;
    header.ambient_color[2] = sun.ambient_tint.b;
    header.ambient_color[3] = sun.ambient_strength;
    header.sun_direction[0] = dir.x;
    header.sun_direction[1] = dir.y;
    header.sun_direction[2] = dir.z;
    header.sun_color[0] = sun.color.r;
    header.sun_color[1] = sun.color.g;
    header.sun_color[2] = sun.color.b;
    header.sun_color[3] = sun.strength;

    Light sun_light{};
    sun_light.direction[0] = dir.x;
    sun_light.direction[1] = dir.y;
    sun_light.direction[2] = dir.z;
    sun_light.color[0] = sun.color.r;
    sun_light.color[1] = sun.color.g;
    sun_light.color[2] = sun.color.b;
    sun_light.color[3] = sun.strength;   // intensity
    sun_light.position[3] = 0.0f;        // 0 = directional

    std::array<Light, 1> lights = {sun_light};
    header.light_count = 1;

    engine.GetSceneRenderer().UploadLighting(header, lights, engine.GetStagingPool());
}

} // namespace

Game::Game(const std::filesystem::path& executable_path, EndpointOverride endpoint)
    : exe_dir_(executable_path.parent_path())
    , endpoint_override_(std::move(endpoint)) {
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

VulkanEngine::EntityId Game::CreateCubeEntity(
    const float x, const float y, const float z,
    const float sx, const float sy, const float sz,
    const VulkanEngine::MaterialManager::MaterialRef material) {
    auto& registry = engine_game_.GetContext().GetComponentRegistry();

    auto& entity = registry.CreateEntity();
    auto& transform = registry.AddComponent<VulkanEngine::Components::Transform>(entity);
    transform.position = glm::vec3{x, y, z};
    transform.scale = glm::vec3{sx, sy, sz};

    auto& mesh_ref = registry.AddComponent<VulkanEngine::Components::MeshReference>(entity);
    mesh_ref.loaded_mesh_id = cube_mesh_id_;

    // One cube mesh for everything; the colour comes from the per-object
    // material override. The override adopts this mesh on first resolve.
    auto& override_comp = registry.AddComponent<VulkanEngine::Components::MaterialOverride>(entity);
    override_comp.Set(0, material);

    return entity.GetId();
}

bool Game::OnSetup(VulkanEngine::Application::ApplicationContext& ctx) {
    // Fingerprint the active ruleset once; scores are reported under it.
    balance_hash_ = balance_.Hash();

    // Route leaderboard logging into the engine logger, and let the level be
    // raised without a rebuild (VKENGINE_LEADERBOARD_LOG=trace|debug|info|warn|error).
    if (const char* level = std::getenv("VKENGINE_LEADERBOARD_LOG"); level != nullptr) {
        Leaderboard::SetLogLevel(
            Leaderboard::ParseLogLevel(level).value_or(Leaderboard::LogLevel::Info));
    }
    Leaderboard::SetLogSink([](Leaderboard::LogLevel level, std::string_view message) {
        switch (level) {
            case Leaderboard::LogLevel::Trace: LOGIFACE_LOG(trace, std::string{message}); break;
            case Leaderboard::LogLevel::Debug: LOGIFACE_LOG(debug, std::string{message}); break;
            case Leaderboard::LogLevel::Info: LOGIFACE_LOG(info, std::string{message}); break;
            case Leaderboard::LogLevel::Warn: LOGIFACE_LOG(warn, std::string{message}); break;
            case Leaderboard::LogLevel::Error: LOGIFACE_LOG(error, std::string{message}); break;
            case Leaderboard::LogLevel::Off: break;
        }
    });

    // 1. Engine setup + renderer using the engine's standard PBR mesh shader.
    VulkanEngine::GameConfig config{};
    config.enable_imgui = true;
    config.renderer_config.clear_color = {0.8f, 0.8f, 0.80f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();

    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    auto& shader_ids = engine_game_.GetContext().GetShaderIds();
    if (!engine_game_.InitRenderer(ctx, shader_ids.main_indir_vert,
                                   shader_ids.standard_mesh_frag, &shader_mgr)) {
        return false;
    }

    SunSettings sun{};
    sun.azimuth_degrees   = 180.0f;  // light from the left
    sun.elevation_degrees = 40.0f;   // low afternoon sun
    sun.strength          = 1.5f;
    ApplySun(engine_game_, sun);

    // 2. Solid-colour materials uploaded to the bindless heap.
    auto& material_mgr = engine_game_.GetContext().GetMaterialManager();
    auto& resource_mgr = engine_game_.GetContext().GetResourceManager();

    const auto make_material = [&](const std::array<std::uint8_t, 4>& rgba,
                                   const float roughness) {
        auto texture = VulkanEngine::DefaultTextureFactory::CreateSolidColorTexture(resource_mgr, rgba);
        const std::uint32_t slot = engine_game_.UploadTextureToBindless(
            ctx, texture.get(),
            VulkanEngine::Textures::TextureSemantic::BaseColor,
            VulkanEngine::Textures::TextureNormalEncoding::Standard,
            VulkanEngine::Textures::SamplerDesc{});
        textures_.push_back(std::move(texture));

        auto handle = material_mgr.Register<VulkanEngine::TechniqueManager::DefaultMeshTechnique>(
            VulkanEngine::MaterialManager::BlendMode::Opaque,
            VulkanEngine::TechniqueManager::DefaultMeshPerMaterialData{
                .albedo_texture = slot,
                .roughness_factor = roughness,
                .metallic_factor = 0.0f,
                .ao_factor = 1.0f,
            });
        return handle.Ref();
    };

    player_material_ = make_material({70, 200, 255, 255}, 0.45f);
    wall_material_ = make_material({225, 90, 70, 255}, 0.85f);
    floor_material_ = make_material({40, 45, 58, 255}, 0.95f);

    // 3. One procedural cube registered once, reused by every entity.
    const auto cube = VulkanEngine::SceneLoader::ToMeshData(MakeUnitCube());
    cube_mesh_id_ = engine_game_.GetMeshRegistry().Register(cube);

    engine_game_.MarkSceneValid();

    // 4. Camera: fixed behind the player, looking down the corridor (-Z).
    // Orientation comes from `forward` rather than a look-at `target`, so the
    // camera stays parallel to the corridor instead of yawing toward a fixed
    // world point as the player moves sideways.
    auto& registry = engine_game_.GetContext().GetComponentRegistry();
    camera_entity_ = engine_game_.CreateCamera(registry);
    if (auto* camera = engine_game_.GetCamera()) {
        camera->position = glm::vec3{0.0f, 2.0f, 6.5f};
        camera->up = glm::vec3{0.0f, 1.0f, 0.0f};
        camera->orientation = VulkanEngine::Components::CameraOrientation::Direction;
        camera->forward = glm::normalize(glm::vec3{0.0f, -1.6f, -12.5f});
        camera->fov_degrees = 60.0f;
        camera->near_plane = 0.1f;
        camera->far_plane = 220.0f;
    }

    // 5. Floor + player + wall pool.
    const float floor_height = 0.2f;
    const float wall_spacing = balance_.WallSpacing();
    const int wall_count = balance_.wall_count;
    CreateCubeEntity(0.0f, -(balance_.player_size / 2) - (floor_height / 2), -(wall_spacing * static_cast<float>(wall_count) + 20.0f) / 2 + 20.0f, 2.0f * balance_.corridor_half + 2.0f, floor_height, wall_spacing * static_cast<float>(wall_count) + 20.0f, floor_material_);
    player_entity_ = CreateCubeEntity(0.0f, 0.0f, 0.0f, balance_.player_size, balance_.player_size, balance_.player_size, player_material_);

    walls_.reserve(static_cast<std::size_t>(wall_count));
    for (int i = 0; i < wall_count; ++i) {
        WallSlot slot{};
        slot.left = CreateCubeEntity(0.0f, 0.0f, 0.0f, 1.0f, balance_.wall_height, balance_.wall_depth, wall_material_);
        slot.right = CreateCubeEntity(0.0f, 0.0f, 0.0f, 1.0f, balance_.wall_height, balance_.wall_depth, wall_material_);
        slot.wall.z = balance_.wall_initial_spawn_z + balance_.wall_spawn_z + static_cast<float>(i) * wall_spacing;
        RandomizeWall(slot.wall);
        ApplyWallTransform(slot);
        walls_.push_back(slot);
    }

    // 6. Input. One 1D action takes both the keyboard keys and the gamepad's
    // left stick X axis, so every device feeds the same movement value.
    auto* input = ctx.input_system;
    move_ = input->BindAction<1>("move", VulkanEngine::Input::ActionConfig{
        .processing = VulkanEngine::Input::ActionProcessing::ClampLength,
    });
    move_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_A).Component(0, -1.0f));
    move_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_LEFT).Component(0, -1.0f));
    move_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_D).Component(0, +1.0f));
    move_.Bind(VulkanEngine::Input::InputBinding::Key(SDLK_RIGHT).Component(0, +1.0f));
    move_.Bind(VulkanEngine::Input::InputBinding::GamepadAxis(SDL_GAMEPAD_AXIS_LEFTX)
                   .Component(0, +1.0f)
                   .Deadzone(0.15f));

    restart_handle_ = input->BindAction("restart", VulkanEngine::Input::InputBinding::Key(SDLK_R));
    input->AddBinding(restart_handle_, VulkanEngine::Input::InputBinding::GamepadButton(SDL_GAMEPAD_BUTTON_NORTH));
    ctx.quit_action_handle = input->BindAction("quit", VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));

    // 7. Local profiles, then the leaderboard client.
    //
    // A fresh clone has neither: no profile exists, and no sealed endpoint means
    // no server. Both degrade to offline play, and the login window says so.
    if (ctx.storage != nullptr) {
        profiles_ = Account::ProfileStore::Create(*ctx.storage);
    } else {
        LOGIFACE_LOG(warn, "per-user storage unavailable; profiles disabled");
    }

    const std::optional<Leaderboard::Endpoint> sealed_endpoint = Leaderboard::LoadEndpoint();
    Leaderboard::Endpoint endpoint{};
    if (sealed_endpoint.has_value()) {
        endpoint = *sealed_endpoint;
    }
    if (!endpoint_override_.host.empty()) {
        endpoint.host = endpoint_override_.host;
    }
    if (endpoint_override_.port != 0) {
        endpoint.port = endpoint_override_.port;
    }

    if (!endpoint.host.empty() && endpoint.port != 0) {
        Leaderboard::ClientOptions options;
        options.host = endpoint.host;
        options.port = endpoint.port;
        options.config_hash = balance_hash_;
        options.server_public_key = Leaderboard::LoadServerPublicKey();
        leaderboard_ = std::make_unique<Leaderboard::Client>(std::move(options));

        // Credentials must be present before Start so the worker's first
        // connection signs in automatically.
        if (profiles_.has_value()) {
            if (const Account::StoredProfile* active = profiles_->Active();
                active != nullptr && !active->token.empty()) {
                leaderboard_->SetCredentials(active->username, active->token);
            }
        }
        leaderboard_->Start();
    } else {
        LOGIFACE_LOG(warn, "leaderboard disabled: no sealed endpoint or --leaderboard-host/"
                           "--leaderboard-port override available");
    }

    // Open the login window unless an existing profile can sign in silently.
    login_open_ = true;
    if (profiles_.has_value()) {
        if (const Account::StoredProfile* active = profiles_->Active();
            active != nullptr && !active->token.empty()) {
            login_open_ = false;
        }
    }

    // 8. HUD + login window.
    if (auto* imgui = engine_game_.GetImGuiSystem()) {
        imgui_draw_handle_ = imgui->draw_callbacks.Register([this]() {
            DrawLoginWindow();

            const ImVec2 screen = ImGui::GetMainViewport()->Size;
            ImGui::SetNextWindowPos(ImVec2(screen.x * 0.78f, screen.y * 0.05f), ImGuiCond_Always);
            ImGui::Begin("Infinite Runner", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize/* | ImGuiWindowFlags_NoMove*/);
            ImGui::Text("Score: %d", score_);
            ImGui::Separator();
            if (game_over_) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "GAME OVER");
                ImGui::TextUnformatted("Press R to restart");
            } else {
                ImGui::TextUnformatted("A/D, Left/Right, or left stick to move");
                ImGui::TextUnformatted("R to restart, Esc to quit");
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Leaderboard");
            bool leaderboard_filters_changed = false;
            if (profiles_.has_value() && profiles_->Active() != nullptr) {
                const Account::StoredProfile* profile = profiles_->Active();
                ImGui::Text("Player: %s", profile->display_name.c_str());
                ImGui::Text("Local best: %d", profiles_->LocalBest(balance_hash_));
                if (ImGui::Button("Players / Account")) {
                    login_open_ = true;
                }

                // Display filters. Local-only: they change what this machine
                // shows, not what the server records.
                Account::LocalSettings local = profile->local;
                int range = 0;
                if (local.leaderboard_days >= 30) {
                    range = 3;
                } else if (local.leaderboard_days >= 7) {
                    range = 2;
                } else if (local.leaderboard_days >= 1) {
                    range = 1;
                }
                ImGui::SetNextItemWidth(140.0f);
                if (ImGui::Combo("Range", &range, "All time\0Today\0Last 7 days\0Last 30 days\0")) {
                    local.leaderboard_days =
                        range == 0 ? 0U : range == 1 ? 1U : range == 2 ? 7U : 30U;
                    leaderboard_filters_changed = true;
                }
                if (ImGui::Checkbox("Best per player", &local.leaderboard_best_per_account)) {
                    leaderboard_filters_changed = true;
                }
                if (ImGui::Checkbox("Only my scores", &local.leaderboard_only_mine)) {
                    leaderboard_filters_changed = true;
                }
                if (leaderboard_filters_changed) {
                    profiles_->UpdateLocal(profile->id, local);
                }
            } else {
                ImGui::TextUnformatted("offline (no profile)");
            }

            if (leaderboard_ != nullptr) {
                const Leaderboard::Client::Snapshot snapshot = leaderboard_->GetSnapshot();
                if (!snapshot.connected) {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%s", snapshot.status.c_str());
                } else if (!snapshot.authenticated) {
                    ImGui::TextUnformatted("connected (not signed in)");
                } else if (snapshot.has_rank) {
                    const bool per_account = profiles_.has_value() &&
                                             profiles_->Active() != nullptr &&
                                             profiles_->Active()->local.leaderboard_best_per_account;
                    const std::int32_t rank =
                        per_account ? snapshot.last_rank_accounts : snapshot.last_rank_runs;
                    ImGui::Text("Rank %d (server best %d)", rank, snapshot.last_best);
                }
                for (const Leaderboard::TopEntry& entry : snapshot.top) {
                    const std::string who =
                        entry.display_name.empty() ? std::string{"anonymous"} : entry.display_name;
                    const std::string line = std::format("#{:<2} {:<16} {:>6}  {}", entry.rank, who,
                                                         entry.score, FormatDate(entry.recorded_at));
                    const bool mine = snapshot.authenticated && entry.user_id != 0 &&
                                      entry.user_id == snapshot.account.id;
                    if (mine) {
                        ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f), "%s", line.c_str());
                    } else {
                        ImGui::TextUnformatted(line.c_str());
                    }
                }
            }
            if (leaderboard_filters_changed && !login_open_) {
                // Apply immediately rather than waiting for the periodic refresh.
                RequestLeaderboard();
            }
            ImGui::End();
        });
    }

    return true;
}

void Game::DrawLoginWindow() {
    if (!login_open_) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(460.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Player", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (leaderboard_ == nullptr) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Offline: no leaderboard endpoint");
    } else {
        ImGui::Text("Server: %s", leaderboard_->GetSnapshot().status.c_str());
    }
    if (!login_status_.empty()) {
        ImGui::TextWrapped("%s", login_status_.c_str());
    }
    ImGui::Separator();

    if (profiles_.has_value() && !profiles_->Profiles().empty()) {
        ImGui::TextUnformatted("Local players");
        // A table keeps the name and the delete button on one row and gives
        // them the same height, instead of a same-line widget of a different
        // scale next to the selectable.
        if (ImGui::BeginTable("local_players", 2,
                              ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_NoPadOuterX)) {
            ImGui::TableSetupColumn("Player", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed);

            for (const Account::StoredProfile& profile : profiles_->Profiles()) {
                // The visible label is not unique (two profiles may share a
                // display name), so each row is scoped to the profile id.
                ImGui::PushID(profile.id.c_str());
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                const std::string label = profile.display_name + "  (" + profile.username + ")" +
                                          (profile.token.empty() ? "  [not registered]" : "");
                // SpanAllColumns makes the whole row selectable; AllowOverlap
                // lets the button in the next column receive its own clicks.
                if (ImGui::Selectable(label.c_str(), profile.id == profiles_->ActiveId(),
                                      ImGuiSelectableFlags_SpanAllColumns |
                                          ImGuiSelectableFlags_AllowOverlap,
                                      ImVec2(0.0f, ImGui::GetFrameHeight()))) {
                    SelectProfile(profile.id);
                }

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Button("Delete")) {
                    delete_candidate_id_ = profile.id;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::Separator();
    }

    // Delete confirmation. Drawn once, outside the row loop, so the popup has a
    // single stable identity.
    if (!delete_candidate_id_.empty()) {
        ImGui::OpenPopup("Delete player?");
    }
    if (ImGui::BeginPopupModal("Delete player?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        delete_popup_open_ = true;
        const Account::StoredProfile* target =
            profiles_.has_value() ? profiles_->Find(delete_candidate_id_) : nullptr;
        if (target == nullptr) {
            delete_candidate_id_.clear();
            ImGui::CloseCurrentPopup();
        } else {
            const std::string id = target->id;
            ImGui::Text("Delete '%s' (%s)?", target->display_name.c_str(), target->username.c_str());
            ImGui::TextUnformatted("This removes the local profile and its saved sign-in token.");
            ImGui::TextUnformatted("Scores already submitted to the server are kept.");
            if (target->token.empty()) {
                ImGui::TextUnformatted("This player was never registered, so the name is released too.");
            } else {
                ImGui::TextUnformatted("The username stays reserved on the server.");
            }
            ImGui::Separator();
            if (ImGui::Button("Delete")) {
                profiles_->RemoveProfile(id);
                delete_candidate_id_.clear();
                if (profiles_->Active() != nullptr) {
                    // Moves the selection and re-points the connection at the
                    // surviving profile.
                    SelectProfile(profiles_->ActiveId());
                } else {
                    RefreshActiveCredentials();
                }
                login_status_ = "Player removed.";
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                delete_candidate_id_.clear();
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    } else if (delete_popup_open_) {
        // Dismissed with Escape or the close button; drop the pending target so
        // it does not reopen next frame.
        delete_popup_open_ = false;
        delete_candidate_id_.clear();
    }

    ImGui::TextUnformatted("New player");
    ImGui::InputText("Username", username_input_.data(), username_input_.size());
    ImGui::SameLine();
    ImGui::TextDisabled("(3-20, permanent)");
    ImGui::InputText("Display name", display_name_input_.data(), display_name_input_.size());
    if (ImGui::Button("Create")) {
        const std::string username = username_input_.data();
        const std::string display = display_name_input_.data();
        const std::optional<std::string> canonical = Leaderboard::CanonicalizeUsername(username);
        const std::optional<std::string> clean = Leaderboard::SanitizeDisplayName(display);
        if (!canonical.has_value() || !clean.has_value()) {
            login_status_ = "Username: 3-20 of a-z, 0-9, _ (start with a letter). Display name: 1-32 characters.";
        } else if (profiles_.has_value()) {
            StartRegistration(*canonical, *clean);
        } else {
            login_status_ = "Per-user storage is unavailable; cannot create a profile.";
        }
    }

    if (!pending_future_.has_value()) {
        ImGui::SameLine();
        if (ImGui::Button("Play offline")) {
            login_open_ = false;
        }
    }

    if (profiles_.has_value() && profiles_->Active() != nullptr) {
        const Account::StoredProfile* profile = profiles_->Active();
        ImGui::Separator();
        ImGui::Text("Selected: %s (%s)", profile->display_name.c_str(), profile->username.c_str());

        if (profile->token.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f),
                               "Local only: scores stay on this machine");
            const bool blocked = pending_future_.has_value() || leaderboard_ == nullptr;
            if (blocked) {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Register with server")) {
                RegisterLocalProfile(profile->id);
            }
            if (blocked) {
                ImGui::EndDisabled();
            }
            if (leaderboard_ == nullptr) {
                ImGui::SameLine();
                ImGui::TextDisabled("(needs a server)");
            }
        } else {
            ImGui::TextUnformatted("Registered account");
        }

        ImGui::InputText("Display name##edit", display_edit_.data(), display_edit_.size());
        const bool can_sync = leaderboard_ != nullptr && leaderboard_->GetSnapshot().authenticated;
        if (!can_sync) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Rename")) {
            pending_op_ = PendingOp::Rename;
            pending_profile_id_ = profile->id;
            pending_future_ = leaderboard_->Rename(display_edit_.data());
        }
        if (!can_sync) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
        if (ImGui::Button("Play")) {
            login_open_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Deregister locally")) {
            profiles_->SetToken(profile->id, {});
            RefreshActiveCredentials();
            login_status_ = "Cleared the local sign-in token for this player.";
        }
    }

    ImGui::End();
}

void Game::StartRegistration(std::string username, std::string display_name) {
    if (!profiles_.has_value()) {
        return;
    }
    const Account::StoredProfile& profile = profiles_->AddProfile(username, display_name);
    pending_profile_id_ = profile.id;
    pending_username_ = username;
    std::snprintf(display_edit_.data(), display_edit_.size(), "%s", profile.display_name.c_str());

    if (leaderboard_ == nullptr) {
        login_status_ = "Created an offline profile. It is not registered with a server.";
        return;
    }
    login_status_ = "Registering...";
    pending_op_ = PendingOp::Register;
    pending_future_ = leaderboard_->Register(username, display_name);
}

void Game::RegisterLocalProfile(std::string_view profile_id) {
    if (!profiles_.has_value()) {
        return;
    }
    if (leaderboard_ == nullptr) {
        login_status_ = "Cannot register: no leaderboard server is configured.";
        return;
    }
    const Account::StoredProfile* profile = profiles_->Find(profile_id);
    if (profile == nullptr) {
        return;
    }
    if (!profile->token.empty()) {
        login_status_ = "This player is already registered.";
        return;
    }
    // The server owns username uniqueness, so a collision is possible when an
    // offline profile's name was claimed elsewhere; the reply reports it.
    pending_profile_id_ = profile->id;
    pending_username_ = profile->username;
    login_status_ = "Registering " + profile->username + " with the server...";
    pending_op_ = PendingOp::Register;
    pending_future_ = leaderboard_->Register(profile->username, profile->display_name);
}

void Game::PollPendingAccount() {
    if (!pending_future_.has_value()) {
        return;
    }
    if (pending_future_->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        return;
    }
    Leaderboard::AccountResult result = pending_future_->get();
    pending_future_.reset();
    const PendingOp op = pending_op_;
    pending_op_ = PendingOp::None;

    if (op == PendingOp::Register) {
        if (result.ok() && profiles_.has_value()) {
            profiles_->SetToken(pending_profile_id_, result.token);
            profiles_->UpdateDisplayName(pending_profile_id_, result.display_name);
            profiles_->SetActive(pending_profile_id_);
            std::snprintf(display_edit_.data(), display_edit_.size(), "%s",
                          result.display_name.c_str());
            login_status_ = "Registered. Username " + result.username + " is permanent.";
        } else if (!result.ok()) {
            login_status_ = "Registration failed: " + result.message;
        }
    } else if (op == PendingOp::Rename) {
        if (result.ok() && profiles_.has_value()) {
            profiles_->UpdateDisplayName(pending_profile_id_, result.display_name);
            std::snprintf(display_edit_.data(), display_edit_.size(), "%s",
                          result.display_name.c_str());
            login_status_ = "Renamed.";
            // Names are resolved live, so refresh the visible board.
            RequestLeaderboard();
        } else if (!result.ok()) {
            login_status_ = "Rename failed: " + result.message;
        }
    }
}

void Game::RequestLeaderboard() {
    if (leaderboard_ == nullptr) {
        return;
    }
    std::size_t count = 5;
    bool best_per_account = false;
    bool only_mine = false;
    std::uint32_t days = 0;
    if (profiles_.has_value()) {
        if (const Account::StoredProfile* active = profiles_->Active(); active != nullptr) {
            count = active->local.top_count;
            best_per_account = active->local.leaderboard_best_per_account;
            only_mine = active->local.leaderboard_only_mine;
            days = active->local.leaderboard_days;
        }
    }
    std::uint64_t since = 0;
    if (days != 0) {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        const auto window = static_cast<std::int64_t>(days) * 24 * 60 * 60;
        since = static_cast<std::uint64_t>(std::max<std::int64_t>(0, now - window));
    }
    leaderboard_->RequestTop(count, best_per_account, since, only_mine);
}

void Game::RefreshActiveCredentials() {
    if (!profiles_.has_value() || leaderboard_ == nullptr) {
        return;
    }
    if (const Account::StoredProfile* active = profiles_->Active();
        active != nullptr && !active->token.empty()) {
        leaderboard_->SetCredentials(active->username, active->token);
    } else {
        leaderboard_->ClearCredentials();
    }
    // The connection was authenticated for the previous profile, so it has to be
    // rebuilt rather than reused.
    leaderboard_->RequestReconnect();
}

void Game::SelectProfile(std::string_view profile_id) {
    if (!profiles_.has_value()) {
        return;
    }
    profiles_->SetActive(profile_id);
    const Account::StoredProfile* profile = profiles_->Active();
    if (profile == nullptr) {
        return;
    }
    std::snprintf(display_edit_.data(), display_edit_.size(), "%s", profile->display_name.c_str());
    if (leaderboard_ == nullptr) {
        return;
    }
    if (!profile->token.empty()) {
        leaderboard_->SetCredentials(profile->username, profile->token);
    } else {
        leaderboard_->ClearCredentials();
    }
    leaderboard_->RequestReconnect();
}

void Game::SubmitRun(std::int32_t score) {
    bool best_per_account = true;
    if (profiles_.has_value() && profiles_->Active() != nullptr) {
        best_per_account = profiles_->Active()->local.leaderboard_best_per_account;
    }
    std::uint64_t run_id = 0;
    if (leaderboard_ != nullptr) {
        run_id = leaderboard_->SubmitScore(score, best_per_account);
    } else {
        const std::vector<std::byte> bytes = VulkanEngine::Security::RandomBytes(8);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            run_id |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[i])) << (8 * i);
        }
    }
    if (profiles_.has_value() && profiles_->Active() != nullptr) {
        const Account::StoredProfile* profile = profiles_->Active();
        profiles_->RecordRun(balance_hash_, run_id, score, profile->display_name);
    }
}

void Game::RandomizeWall(Wall& wall) {
    const float HoleSizeDifficulty = 2 - std::pow(std::max(1, score_), balance_.wall_hole_size_pow_scaling);
    const float PlacementSizeDifficulty = 2 - std::pow(std::max(1, score_), balance_.wall_hole_placement_pow_scaling);
    std::uniform_real_distribution<float> half_dist(balance_.wall_hole_min * HoleSizeDifficulty, balance_.wall_hole_max * HoleSizeDifficulty);
    const float gap_half = half_dist(rng_);

    const float scaled_wall_hole_placement_min = balance_.wall_hole_placement_min * PlacementSizeDifficulty;
    const float scaled_wall_hole_placement_max = balance_.wall_hole_placement_max * PlacementSizeDifficulty;

    constexpr float margin = 0.25f;
    const float limit = balance_.corridor_half - gap_half - margin;
    const float distance_clamp_size_left =
        std::max(-limit, prevGapCenter_ - scaled_wall_hole_placement_min)
        - std::max(-limit, prevGapCenter_ - scaled_wall_hole_placement_max);
    const float distance_clamp_size_right =
        std::min(limit, prevGapCenter_ + scaled_wall_hole_placement_max)
        - std::min(limit, prevGapCenter_ + scaled_wall_hole_placement_min);
    const float distance_clamp_size = distance_clamp_size_left + distance_clamp_size_right;
    std::uniform_real_distribution<float> center_dist(0, distance_clamp_size);
    const float local_r_pos = center_dist(rng_);
    const float offset = local_r_pos - distance_clamp_size_left;
    const float sign = (offset < 0.0f ? -1.0f : 1.0f);
    const float gap_center = prevGapCenter_ + offset + scaled_wall_hole_placement_min * sign;

#ifndef NDEBUG
    const float gap_past_distance = std::abs(gap_center - prevGapCenter_);
    if (gap_past_distance > scaled_wall_hole_placement_max || gap_past_distance < scaled_wall_hole_placement_min)
    {
        LOGIFACE_LOG(warn, "Gap past distance is out of bounds: " + std::to_string(gap_past_distance));
    }
#endif

    prevGapCenter_ = gap_center;

    wall.gap_left_x = gap_center - gap_half;
    wall.gap_right_x = gap_center + gap_half;
    wall.passed = false;
}

void Game::ApplyWallTransform(WallSlot& slot) {
    // Render transforms are derived from the same block geometry the sweep
    // tests against, so the visuals and the hitboxes cannot drift apart.
    // Transform is a data component, so each wall re-fetches it by EntityId.
    auto& registry = engine_game_.GetContext().GetComponentRegistry();
    const auto place = [&registry](const VulkanEngine::EntityId id, const Sweep::Aabb& box) {
        if (auto* entity = registry.TryGetEntity(id)) {
            if (auto* transform = entity->GetComponent<VulkanEngine::Components::Transform>()) {
                transform->position = box.Center();
                transform->scale = box.Size();
            }
        }
    };

    place(slot.left, slot.wall.LeftBlock(balance_));
    place(slot.right, slot.wall.RightBlock(balance_));
}

void Game::ResetWalls() {
    const float difficulty = std::pow(std::max(1, score_), balance_.wall_spacing_pow_scaling);
    const float wall_spacing = balance_.WallSpacing();
    for (int i = 0; i < static_cast<int>(walls_.size()); ++i) {
        auto& slot = walls_[static_cast<std::size_t>(i)];
        slot.wall.z = balance_.wall_initial_spawn_z + balance_.wall_spawn_z + static_cast<float>(i) * wall_spacing * std::max(1.0f, difficulty / balance_.difficulty_scaling_divider);
        RandomizeWall(slot.wall);
        ApplyWallTransform(slot);
    }
}

void Game::ResetRun() {
    player_x_ = 0.0f;
    score_ = 0;
    game_over_ = false;
    auto& registry = engine_game_.GetContext().GetComponentRegistry();
    if (auto* entity = registry.TryGetEntity(player_entity_)) {
        if (auto* transform = entity->GetComponent<VulkanEngine::Components::Transform>()) {
            transform->position = glm::vec3{0.0f, 0.0f, 0.0f};
        }
    }
    ResetWalls();
}

void Game::UpdatePlayer(const VulkanEngine::Application::ApplicationContext& /*ctx*/, const float delta_time) {
    const float direction = move_.Scalar();

    const float limit = balance_.corridor_half - balance_.player_size * 0.5f;
    player_x_ = std::clamp(player_x_ + direction * balance_.player_speed * delta_time, -limit, limit);
    auto& registry = engine_game_.GetContext().GetComponentRegistry();
    if (auto* entity = registry.TryGetEntity(player_entity_)) {
        if (auto* transform = entity->GetComponent<VulkanEngine::Components::Transform>()) {
            transform->position = glm::vec3{player_x_, 0.0f, 0.0f};
        }
    }
    if (auto* camera = engine_game_.GetCamera()) {
        camera->position.x = player_x_;
    }
}

bool Game::StepWalls(const float delta_time, const float player_x_before, const float player_dx) {
    const float difficulty = static_cast<float>(std::max(1, score_));
    const float wall_dz = balance_.wall_speed * delta_time * std::max(1.0f, difficulty / balance_.difficulty_scaling_divider);

    // Player-centre sweep for this frame, expressed relative to a wall: the
    // wall's own +Z scroll enters as -wall_dz, so a wall that reaches the
    // player between frames is caught instead of tunnelling through. A
    // thin ray keeps the hitbox forgiving relative to the visible cube.
    const glm::vec3 from{player_x_before, 0.0f, 0.0f};
    const glm::vec3 to{player_x_before + player_dx, 0.0f, -wall_dz};

    bool hit = false;
    for (auto& slot : walls_) {
        Wall& wall = slot.wall;

        // Tested against the pre-scroll box, before recycle/randomize, so the
        // geometry and z used here are the ones the wall actually had.
        hit = hit || Sweep::SegmentIntersects(wall.LeftBlock(balance_), from, to)
                  || Sweep::SegmentIntersects(wall.RightBlock(balance_), from, to);

        wall.z += wall_dz;
        if (wall.z > balance_.wall_recycle_z) {
            wall.z -= (balance_.wall_recycle_z - balance_.wall_spawn_z);
            RandomizeWall(wall);
        }

        if (!wall.passed && wall.z >= 0.0f) {
            wall.passed = true;
            ++score_;
        }

        ApplyWallTransform(slot);
    }
    return hit;
}

bool Game::PlayerOverlapsAnyWall() const {
    const Sweep::Aabb player = PlayerBox(player_x_, balance_.player_size);
    for (const auto& slot : walls_) {
        if (Sweep::Intersects(player, slot.wall.LeftBlock(balance_)) ||
            Sweep::Intersects(player, slot.wall.RightBlock(balance_))) {
            return true;
        }
    }
    return false;
}

void Game::OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    // Engine simulation. Components integrate against the fixed timestep, which
    // equals the wall-clock delta while fixed stepping is disabled (the default),
    // so this runs exactly once per iteration today and 0..N times once a fixed
    // rate is configured.
    engine_game_.FixedUpdate(ctx);

    // While the login window is open the run is paused and the game's own logic is
    // skipped. The engine simulation above still runs so the scene stays
    // consistent behind the modal.
    if (login_open_) {
        return;
    }

    const float delta_time = ctx.frame.fixed_delta_time;

    if (!game_over_) {
        const float player_x_before = player_x_;
        UpdatePlayer(ctx, delta_time);
        // Continuous check catches fast walls crossing between frames; the
        // discrete box check covers overlap at the settled end-of-frame poses.
        const bool swept_hit = StepWalls(delta_time, player_x_before, player_x_ - player_x_before);
        const bool overlap_hit = PlayerOverlapsAnyWall();
        if (swept_hit || overlap_hit) {
            game_over_ = true;
            SubmitRun(score_);
        }
    }

    // Gameplay timer: accumulates simulated time, not wall-clock time, so the
    // leaderboard refresh rate does not follow the render rate.
    if (leaderboard_ != nullptr) {
        leaderboard_request_timer_ += delta_time;
        if (leaderboard_request_timer_ >= 2.0f) {
            leaderboard_request_timer_ = 0.0f;
            RequestLeaderboard();
        }
    }
}

void Game::OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    // Account work is polled every frame, modal or not, so a registration or
    // settings update completes without blocking the frame loop.
    PollPendingAccount();

    // While the login window is open the run is paused and the game's own
    // actions are ignored. Events are deliberately not filtered: Esc must still
    // reach the quit action, and ImGui consumes its own text input.
    if (login_open_) {
        engine_game_.FrameUpdate(ctx);
        return;
    }

    // Restart is edge-triggered, so it is sampled on the frame hook: the fixed
    // hook may run several times in one frame, which would fire the edge once per
    // step. It therefore takes effect from the next simulation step rather than
    // mid-iteration.
    if (ctx.input_system->WasActionStarted(restart_handle_)) {
        ResetRun();
    }

    engine_game_.FrameUpdate(ctx);
}

void Game::OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void Game::OnShutdown(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
    imgui_draw_handle_ = {};
    if (leaderboard_ != nullptr) {
        leaderboard_->Stop();
        leaderboard_.reset();
    }
    engine_game_.Shutdown();
}

} // namespace Examples::InfiniteRunner::Game
