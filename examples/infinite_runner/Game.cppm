module;

export module Examples.InfiniteRunner.Game;

import std;

export import VulkanEngine.GameEngine;
import VulkanShared.CallbackList;
import VulkanShared.Storage;
import Examples.InfiniteRunner.Balance;
import Examples.InfiniteRunner.Wall;
import Examples.InfiniteRunner.Leaderboard.Account;
import Examples.InfiniteRunner.Leaderboard.Client;
import Examples.InfiniteRunner.Account.ProfileStore;

export namespace Examples::InfiniteRunner::Game {

// Optional command-line overrides for the sealed leaderboard endpoint. An empty
// host or zero port leaves the corresponding value from the sealed entry intact,
// so either half can be redirected on its own.
struct EndpointOverride {
    std::string host;
    std::uint16_t port = 0;
};

// Infinite runner built entirely on the game layer.
//
// A fixed pool of wall entities scrolls toward a stationary player cube.
// Each wall is two blocks with a randomly placed horizontal gap.
// the player must line up with the gap to pass.
// There is no engine collision so the player is tested both as a discrete box against the block geometry and as a ray swept between frames.
class Game {
public:
    explicit Game(const std::filesystem::path& executable_path, EndpointOverride endpoint = {});
    ~Game();

    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

    [[nodiscard]] const VulkanEngine::Application::ApplicationHooks& GetHooks() const { return hooks_; }

private:
    bool OnSetup(VulkanEngine::Application::ApplicationContext& ctx);

    // Simulation: player and wall integration plus collision, and gameplay timers.
    // Runs on the fixed-step hook so the run is rate-independent of the render rate.
    void OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx);


    void OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx);
    void OnShutdown(VulkanEngine::Application::ApplicationContext& ctx);

    // A wall's simulated state plus the render handles it drives. The value
    // type owns the geometry; the game owns the ECS plumbing.
    struct WallSlot {
        Wall wall{};
        VulkanEngine::EntityId left{};
        VulkanEngine::EntityId right{};
    };

    VulkanEngine::EntityId CreateCubeEntity(float x, float y, float z,
                                            float sx, float sy, float sz,
                                            VulkanEngine::MaterialManager::MaterialRef material);

    void RandomizeWall(Wall& wall);
    void ApplyWallTransform(WallSlot& slot);
    void ResetWalls();
    void ResetRun();
    void UpdatePlayer(const VulkanEngine::Application::ApplicationContext& ctx, float delta_time);

    // Integrates every wall and sweeps the player against their pre-move boxes.
    // Owns the before/after pair explicitly, so the collision never depends on
    // where an earlier update left shared state. Returns true on a hit.
    bool StepWalls(float delta_time, float player_x_before, float player_dx);

    // Discrete box-vs-box overlap at the settled end-of-frame positions.
    bool PlayerOverlapsAnyWall() const;

    VulkanEngine::Application::ApplicationHooks hooks_{};

    VulkanShared::ScopedHandle<bool(VulkanEngine::Application::ApplicationContext&)> setup_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> fixed_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_update_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> frame_render_token_{};
    VulkanShared::ScopedHandle<void(VulkanEngine::Application::ApplicationContext&)> shutdown_token_{};
    VulkanShared::ScopedHandle<void()> imgui_draw_handle_{};

    std::filesystem::path exe_dir_{};
    EndpointOverride endpoint_override_{};
    VulkanEngine::GameEngine engine_game_{};

    // Single source of truth for every gameplay-affecting value. Rehashed once
    // at setup and submitted with the score to select the right leaderboard.
    BalanceConfig balance_{};
    std::uint64_t balance_hash_ = 0;

    std::uint32_t cube_mesh_id_ = 0;
    VulkanEngine::MaterialManager::MaterialRef player_material_{};
    VulkanEngine::MaterialManager::MaterialRef wall_material_{};
    VulkanEngine::MaterialManager::MaterialRef floor_material_{};

    // Solid-colour textures are uploaded to the bindless heap immediately
    // keep the CPU resources alive for the lifetime of the run.
    std::vector<std::shared_ptr<VulkanEngine::TextureResource>> textures_{};

    VulkanEngine::EntityId camera_entity_{};
    VulkanEngine::EntityId player_entity_{};
    std::vector<WallSlot> walls_{};

    VulkanEngine::Input::Action<1> move_{};
    VulkanEngine::Input::ActionHandle restart_handle_{};

    // Optional: present only when a sealed endpoint is available.
    std::unique_ptr<Examples::InfiniteRunner::Leaderboard::Client> leaderboard_{};
    float leaderboard_request_timer_ = 0.0f;

    // ── local profiles / account ──
    // Present only when per-user storage was resolved. Holds the profile index,
    // each profile's private token, local bests and settings.
    std::optional<Examples::InfiniteRunner::Account::ProfileStore> profiles_{};

    // The login/profile window. Gameplay input is ignored while it is open.
    bool login_open_ = false;
    std::string login_status_{};
    std::array<char, 64> username_input_{};
    std::array<char, 64> display_name_input_{};
    std::array<char, 64> display_edit_{};

    // An account request (register / rename) in flight. The UI thread must not
    // block on the network, so the future is polled once per frame.
    enum class PendingOp { None, Register, Rename };
    PendingOp pending_op_ = PendingOp::None;
    std::optional<std::future<Examples::InfiniteRunner::Leaderboard::AccountResult>> pending_future_{};
    std::string pending_profile_id_{};
    std::string pending_username_{};
    std::string pending_registration_token_{};

    // Profile awaiting delete confirmation; empty when no dialog is open.
    std::string delete_candidate_id_{};
    bool delete_popup_open_ = false;

    void DrawLoginWindow();
    void PollPendingAccount();
    void StartRegistration(std::string username, std::string display_name);

    // Turns an offline, local-only profile into a registered account.
    void RegisterLocalProfile(std::string_view profile_id);
    void SelectProfile(std::string_view profile_id);

    // Re-points the client at whichever profile is active after a removal.
    void RefreshActiveCredentials();

    // Issues a top-list request using the active profile's display filters.
    void RequestLeaderboard();
    void SubmitRun(std::int32_t score);

    float player_x_ = 0.0f;
    int score_ = 0;
    bool game_over_ = false;
    std::mt19937 rng_{std::random_device{}()};
    float prevGapCenter_ = 0.0f;
};

} // namespace Examples::InfiniteRunner::Game
