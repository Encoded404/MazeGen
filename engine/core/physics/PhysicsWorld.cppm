// The physics facade: the engine's physics world, with Jolt behind it.
//
// This interface names no Jolt type. The `PhysicsSystem`, the layer tables, the
// temp allocator and the job-system bridge all live in `Impl`/`detail`, which are
// defined in implementation files — so replacing Jolt, or compiling the engine
// without physics, never reaches game code.
//
// The world is owned by `EngineContext` and stepped exactly once per fixed step
// from the `SystemSlot::PhysicsSimulate` system. There is no variable-dt path.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.World;

import std;

export import VulkanEngine.Physics.Types;
export import VulkanEngine.Physics.Events;

export namespace VulkanEngine::Physics {

/// Caps and policy fixed at `Initialize`. All defaults are Jolt's documented
/// stable configuration: 60 Hz with one collision step.
struct PhysicsWorldConfig {
    std::uint32_t max_bodies = 65536;
    std::uint32_t max_body_pairs = 65536;
    std::uint32_t max_contact_constraints = 10240;

    glm::vec3 gravity{0.0f, -9.81f, 0.0f};

    /// The only step the engine has. Rendering interpolates; physics does not.
    float fixed_delta_time = 1.0f / 60.0f;
    std::uint32_t collision_steps = 1;

    /// Physics job concurrency ceiling. 0 means "the shared scheduler's worker
    /// count"; Jolt's own ceiling (`cMaxConcurrency = 32`) applies above it.
    std::uint32_t max_concurrency = 0;

    /// Directory for cooked shape blobs. Empty disables the cache entirely (the
    /// default): a cooked shape is a *cache*, never primary data — Jolt's shape
    /// binary format changes between versions, so staleness must be automatic.
    ///
    /// The blob is keyed by (Jolt's `JPH_VERSION_ID`, `ShapeDesc::GeometryHash()`),
    /// so a Jolt upgrade silently misses instead of restoring a shape the
    /// library no longer understands. Any error on the read path — a short
    /// file, a stale version, a corrupt payload — is a miss: the cache can
    /// never turn a bad disk into a failed world. Writes are best-effort too.
    std::filesystem::path shape_cache_directory{};
};

/// Everything needed to create a body, as data.
///
/// No Jolt type appears here, and nothing in it is a pointer into Jolt: a caller
/// describes a body and gets an opaque handle back.
struct BodyDesc {
    ShapeDesc shape{};
    CollisionLayer layer = CollisionLayer::Default;
    BodyMotionType motion = BodyMotionType::Static;
    /// Explicit mass in kg. `0` means "derive it from the shape's density", which
    /// is what a scene that only declares a size wants.
    float mass = 1.0f;
    float friction = 0.5f;
    float restitution = 0.0f;
    float linear_damping = 0.05f;
    float angular_damping = 0.05f;
    /// Continuous collision detection. Costs a linear cast per step; worth it for
    /// fast, thin bodies.
    bool ccd = false;
    /// Trigger volume: reported by contact/query paths, never resolved.
    bool sensor = false;
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    /// Free 64-bit payload. The body bridge packs the owning `EntityId` here.
    std::uint64_t user_data = 0;
};

/// A rigid transform, in engine units.
struct BodyPose {
    glm::vec3 position{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// How a gameplay character is shaped and tuned.
struct CharacterDesc {
    /// Usually a capsule (`ShapeType::Capsule`). A virtual character is a swept
    /// shape, so any shape works, but a capsule is what walks stairs well.
    ShapeDesc shape{};
    CollisionLayer layer = CollisionLayer::Character;
    float mass = 70.0f;
    /// Maximum force with which the character pushes dynamic bodies (N). `0`
    /// makes it unable to push anything.
    float max_strength = 100.0f;
    float max_slope_angle_degrees = 50.0f;
    /// How far the sweep tries to stay clear of geometry; keeps it unstuck.
    float character_padding = 0.02f;
    // `ExtendedUpdate`'s traversal tuning, in metres. These are exactly what make
    // a character climb a stair instead of stopping at it, and stay glued to a
    // slope instead of launching off the top.
    float stick_to_floor_step_down = 0.5f;
    float walk_stairs_step_up = 0.4f;
    float walk_stairs_min_step_forward = 0.02f;
    float walk_stairs_step_forward_test = 0.15f;
};

/// Opaque, generation-checked handle to a virtual character.
struct CharacterId {
    std::uint32_t slot = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t generation = 0; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] constexpr bool IsValid() const noexcept { return generation != 0; }

    friend constexpr bool operator==(const CharacterId&, const CharacterId&) = default;
};

/// A ray cast request. `from`/`to` are world space; the fraction in the result is
/// along that segment, so `fraction = 1` is `to`.
struct RayCastRequest {
    glm::vec3 from{0.0f, 0.0f, 0.0f};
    glm::vec3 to{0.0f, 0.0f, -1.0f};
    /// Which object layers the query may hit, one bit per `CollisionLayer`.
    /// `0` means "every layer".
    std::uint32_t layer_mask = 0;
    /// Trigger volumes are hit unless this is false.
    bool hit_sensors = true;
    bool hit_static = true;
    bool hit_kinematic = true;
    bool hit_dynamic = true;
};

/// A shape sweep (`CastRay`'s sibling): sweep `shape` from the request's `from`
/// pose toward `to`, stopping on the first blocking hit.
struct ShapeCastRequest {
    ShapeDesc shape{};
    glm::vec3 from{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 to{0.0f, 0.0f, 0.0f};
    std::uint32_t layer_mask = 0;
    bool hit_sensors = false;
    bool hit_static = true;
    bool hit_kinematic = true;
    bool hit_dynamic = true;
};

/// Face-culling contract carried on debug triangle vertices. Jolt hands each
/// solid draw an `ECullMode` and documents `DrawTriangle` as "a single back
/// face culled triangle": realizing that contract needs a screen-space winding
/// test, which only the renderer can do — so the decision travels with the
/// triangle instead of being lost in the `DrawTriangle` callback. Line
/// vertices ignore it (winding does not apply to segments).
enum class PhysicsDebugFaceCulling : std::uint8_t {
    None,      ///< draw both sides (Jolt `ECullMode::Off`, e.g. open cones)
    BackFace,  ///< skip back-facing triangles (Jolt `ECullMode::CullBackFace`)
    FrontFace, ///< skip front-facing triangles (Jolt `ECullMode::CullFrontFace`)
};

/// Debug visualization output. Vertices carry Jolt's packed ARGB color so the
/// renderer can draw them without knowing anything about physics.
struct PhysicsDebugVertex {
    glm::vec3 position{0.0f, 0.0f, 0.0f}; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t color = 0xff000000u; // NOLINT(misc-non-private-member-variables-in-classes)
    /// Meaningful on triangle vertices only; all three of a triangle carry the
    /// same value. The default keeps hand-built geometry double-sided.
    PhysicsDebugFaceCulling culling = PhysicsDebugFaceCulling::None; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// Collect-only geometry: a line list (pairs) and a triangle list (triples).
/// The engine has no debug line pass, so the physics overlay draws these through
/// the ImGui draw list; keeping the data in engine form means any future debug
/// renderer consumes the same buffer.
struct PhysicsDebugGeometry {
    std::vector<PhysicsDebugVertex> lines; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<PhysicsDebugVertex> triangles; // NOLINT(misc-non-private-member-variables-in-classes)

    void Clear() noexcept {
        lines.clear();
        triangles.clear();
    }
};

/// What to draw, mirroring Jolt's `BodyManager::DrawSettings` plus a layer mask.
struct PhysicsDebugSettings {
    bool draw_shapes = true;
    bool draw_bounding_boxes = false;
    bool draw_centers_of_mass = false;
    bool draw_constraints = true;
    bool draw_constraint_limits = false;
    /// Jolt's wireframe switch. Wireframe is what the line list is for.
    bool wireframe = true;
    /// Object layers to draw, one bit per `CollisionLayer`. `0` means all.
    std::uint32_t layer_mask = 0;
};

namespace detail {

/// The world's Jolt state: the physics system, layer tables, temp allocator,
/// shape cache and event capture. Defined in `PhysicsWorld.cpp` so that no Jolt
/// type appears anywhere in this interface.
struct WorldImpl;

/// Engine-internal lifecycle hook for a sibling engine module that owns state
/// attached to a world. The constraints registry (`VulkanEngine.Physics.
/// Constraints`) is the first user: its constraints reference Jolt bodies, so
/// it must release them before those bodies are destroyed.
///
/// `World::AttachExtension` stores the extension; `World::Shutdown` calls
/// `OnWorldShutdown` while the physics system is still alive, then releases the
/// extension after every body is gone. This interface names no Jolt type, so it
/// belongs in the Jolt-free facade. Game code never touches it.
struct WorldExtension {
    virtual ~WorldExtension() = default;
    /// Release anything that references Jolt bodies/constraints. Runs from
    /// `World::Shutdown`, before the physics system is torn down. Must not
    /// throw.
    virtual void OnWorldShutdown() {}
    /// A body is about to be removed from the physics system: release anything
    /// that references it. The constraints registry removes that body's joints
    /// here, so destroying a body can never leave a dangling constraint. Must
    /// not throw, and must not create or destroy other bodies.
    virtual void OnBodyDestroyed(PhysicsBodyId /*body*/) {}
};

/// Live Jolt allocation counters, proving the allocator override is installed.
struct JoltMemoryStats {
    std::uint64_t live_allocations = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint64_t total_allocations = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint64_t total_bytes = 0; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// Allocations Jolt has made through the engine's hooks.
[[nodiscard]] JoltMemoryStats JoltMemory();

/// The Jolt job-system bridge over the engine's shared scheduler.
///
/// Declared without any Jolt type so this interface stays Jolt-free; the Jolt
/// subclass lives in `PhysicsJobs.cpp`. The bridge runs Jolt's whole update burst
/// at the scheduler's Realtime tier and drains the pool with participating
/// waits, so a simulation step never leaves pool capacity idle behind a blocked
/// main thread.
class JobBridge {
public:
    virtual ~JobBridge() = default;
    /// The `JPH::JobSystem*`, as an opaque pointer for the implementation only.
    [[nodiscard]] virtual void* NativeJobSystem() = 0;
};

/// Creates the bridge over `VulkanShared::JobScheduler::Global()`.
[[nodiscard]] std::unique_ptr<JobBridge> CreateJobBridge();

/// Installs the engine's allocation hooks on Jolt. Idempotent per process.
void InstallAllocatorOverride();

/// Bootstraps Jolt (`RegisterDefaultAllocator`, `Factory`, `RegisterTypes`) and
/// installs the trace/assert routing. Idempotent per process; `RegisterTypes`
/// verifies the Jolt ABI, so a define mismatch fails loudly here rather than as
/// undefined behaviour later.
bool BootstrapJolt();

/// Reverses `BootstrapJolt`. Called when the last world shuts down.
void ShutdownJolt();

/// True once `BootstrapJolt` has succeeded.
[[nodiscard]] bool IsJoltBootstrapped();

} // namespace detail

/// The engine's physics world.
class World {
public:
    World();
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) = delete;
    World& operator=(World&&) = delete;

    /// Bootstrap Jolt if needed, then build the physics system, layer tables,
    /// temp allocator and job bridge. Returns false (with a log) on failure;
    /// a second call on an initialized world is a no-op that returns true.
    bool Initialize(const PhysicsWorldConfig& config = {});
    void Shutdown();

    /// One blocking fixed step. Jolt fans its solver out onto the shared
    /// scheduler and this thread participates in the wait. Must not be called
    /// concurrently, and no body may be mutated while it runs.
    void Step();

    [[nodiscard]] bool IsInitialized() const noexcept { return initialized_; }
    [[nodiscard]] const PhysicsWorldConfig& Config() const noexcept { return config_; }

    /// Bodies currently in the system (diagnostics and tests).
    [[nodiscard]] std::uint32_t BodyCount() const noexcept;
    /// Engine-internal escape hatch: the `JPH::PhysicsSystem*` as an opaque
    /// pointer, for engine modules that must reach Jolt directly (constraints,
    /// ragdolls, vehicles) while keeping this interface free of Jolt types. Game
    /// code must never use it; the facade exists so it does not have to.
    [[nodiscard]] void* NativePhysicsSystem() noexcept;

    /// Engine-internal: attach module-owned state to this world and be told when
    /// the world shuts down (see `detail::WorldExtension`). A sibling engine
    /// module (constraints) uses it; game code never does. The previous
    /// extension, if any, is released.
    void AttachExtension(std::shared_ptr<detail::WorldExtension> extension);

    /// Fixed steps taken since `Initialize`.
    [[nodiscard]] std::uint64_t StepCount() const noexcept { return step_count_; }
    /// `EPhysicsUpdateError` bits from the last step (0 = clean). Surfaced
    /// loudly rather than silently degrading when a cap is exceeded.
    [[nodiscard]] std::uint32_t LastUpdateErrors() const noexcept { return last_errors_; }
    /// Worker threads the job bridge reports to Jolt.
    [[nodiscard]] std::uint32_t JobConcurrency() const noexcept { return job_concurrency_; }

    /// True when the layer table says the two object layers may collide. The
    /// policy is a table; this is how tests and debug UI read it.
    [[nodiscard]] bool LayersCollide(CollisionLayer a, CollisionLayer b) const noexcept;

    // ── Body lifetime ───────────────────────────────────────────────────────
    /// Cooks (or reuses a cooked) shape and creates a body. Returns an invalid
    /// id when the description cannot be cooked. All mutation goes through
    /// Jolt's locking body interface, so this is safe from any thread that is
    /// not inside `Step()`.
    ///
    /// `children` are extra shapes placed in the body frame by their own
    /// position/rotation/scale: together with `desc.shape` they form one compound
    /// body, which is how one entity is one collider made of several pieces.
    [[nodiscard]] PhysicsBodyId CreateBody(const BodyDesc& desc,
                                           std::span<const ShapeDesc> children = {});
    /// `CreateBody` with a caller-chosen id (`BodyInterface::CreateBodyWithID`),
    /// for deterministic replay and rollback: a snapshot names bodies by id, so a
    /// world rebuilt from the same script must hand out the same ids.
    ///
    /// **The caller owns id determinism.** Jolt allocates ids in creation order,
    /// so replay determinism holds exactly when creation is driven by a
    /// deterministic iteration order (entity iteration order, per design §8) and
    /// every body is created with the id the original run assigned it. This call
    /// does not invent an id or repair a collision: an id already in use, an
    /// index past the body cap, or an id that cannot be allocated returns an
    /// invalid handle (with a log) instead of silently picking another id.
    [[nodiscard]] PhysicsBodyId CreateBodyWithID(const BodyDesc& desc, PhysicsBodyId required_id,
                                                 std::span<const ShapeDesc> children = {});
    /// Removes and destroys a body. Any constraint attached to it is released
    /// first, through the world's extension (the constraints registry), because
    /// Jolt requires constraints to be removed before their bodies.
    void DestroyBody(PhysicsBodyId body);
    [[nodiscard]] bool IsBodyValid(PhysicsBodyId body) const;

    // ── Body state ──────────────────────────────────────────────────────────
    [[nodiscard]] BodyPose GetPose(PhysicsBodyId body) const;
    void SetPose(PhysicsBodyId body, const BodyPose& pose, bool activate = true);
    /// Kinematic move: Jolt derives the velocity that reaches `pose` within
    /// `delta_time`, so a moving platform pushes whatever rests on it.
    void MoveKinematic(PhysicsBodyId body, const BodyPose& pose, float delta_time);

    [[nodiscard]] glm::vec3 GetLinearVelocity(PhysicsBodyId body) const;
    [[nodiscard]] glm::vec3 GetAngularVelocity(PhysicsBodyId body) const;
    void SetLinearVelocity(PhysicsBodyId body, glm::vec3 velocity, bool activate = true);
    void SetAngularVelocity(PhysicsBodyId body, glm::vec3 velocity, bool activate = true);
    void SetLinearAndAngularVelocity(PhysicsBodyId body, glm::vec3 linear, glm::vec3 angular,
                                     bool activate = true);

    void AddForce(PhysicsBodyId body, glm::vec3 force, bool activate = true);
    void AddTorque(PhysicsBodyId body, glm::vec3 torque, bool activate = true);
    void AddImpulse(PhysicsBodyId body, glm::vec3 impulse);
    void AddAngularImpulse(PhysicsBodyId body, glm::vec3 impulse);

    void SetMotionType(PhysicsBodyId body, BodyMotionType motion, bool activate = true);
    [[nodiscard]] BodyMotionType GetMotionType(PhysicsBodyId body) const;
    [[nodiscard]] bool IsActive(PhysicsBodyId body) const;
    void Activate(PhysicsBodyId body);

    void SetUserData(PhysicsBodyId body, std::uint64_t user_data);
    [[nodiscard]] std::uint64_t GetUserData(PhysicsBodyId body) const;

    // ── Body property tuning ────────────────────────────────────────────────
    // Live tuning of a body's contact and mass properties. Every setter reads
    // the fresh value on the next step, so these are safe to change mid-run,
    // including under an attached constraint (a vehicle, for instance). Mass
    // changes on a dynamic body scale the inertia tensor along with the mass
    // (`MotionProperties::ScaleToMass`), so the body stays kinematically
    // consistent; a static body has no motion properties, so mass is rejected
    // there with a log. Invalid (dead) ids silently no-op, like the setters
    // above.

    /// Scale a dynamic body's mass (with its inertia) to `mass` kilograms.
    void SetBodyMass(PhysicsBodyId body, float mass);
    /// The body's mass; `0` for a static body (infinite against dynamics).
    [[nodiscard]] float GetBodyMass(PhysicsBodyId body) const;

    /// Static + kinetic contact friction the body contributes to contacts
    /// (blended with the other body's value by the solver when they touch).
    void SetBodyFriction(PhysicsBodyId body, float friction);
    [[nodiscard]] float GetBodyFriction(PhysicsBodyId body) const;

    /// Restitution (bounciness) the body contributes to contacts.
    void SetBodyRestitution(PhysicsBodyId body, float restitution);
    [[nodiscard]] float GetBodyRestitution(PhysicsBodyId body) const;

    /// Cooked shapes currently cached (diagnostics and tests).
    [[nodiscard]] std::size_t CachedShapeCount() const noexcept;
    /// Cooked geometry shapes restored from `PhysicsWorldConfig::shape_cache_directory`
    /// instead of cooked (diagnostics and tests). Zero while the cache is disabled.
    [[nodiscard]] std::uint64_t ShapeCacheDiskHits() const noexcept;
    /// Geometry shapes that had to be cooked because the disk cache could not
    /// serve them — no blob, a stale one, or a corrupt one. Misses are then
    /// written back, so a corrupt blob heals on the next world.
    [[nodiscard]] std::uint64_t ShapeCacheDiskMisses() const noexcept;
    /// Every live body id, ascending — for tests and save/replay ordering.
    [[nodiscard]] std::vector<PhysicsBodyId> BodyIds() const;

    // ── Save / replay / rollback ────────────────────────────────────────────
    //
    // A snapshot is the solver-visible state of the world, so a restored world
    // continues bit-identically from where the snapshot was taken. It is *not* a
    // scene save: body configuration (shapes, friction, layers, motion types)
    // is not in it and must be rebuilt from primary data first — per design §8,
    // body creation is driven by deterministic entity iteration order and
    // `CreateBodyWithID`, then the snapshot restores the dynamic state on top.
    //
    // The bytes are ours, not raw Jolt output: a framed header (magic, format
    // and Jolt version, payload size, checksum) makes a truncated, foreign or
    // corrupt stream rejectable *before* any byte reaches Jolt. That is what
    // makes `RestoreSnapshot` all-or-nothing: a rejected snapshot leaves the
    // world exactly as it was.
    //
    // ── `JPH_CROSS_PLATFORM_DETERMINISTIC` decision (design §11) ─────────────
    // It stays OFF, which is what the vcpkg port pins. Enabling it costs ~8% of
    // the solver's throughput and buys only bit-identical results *across*
    // architectures and compilers — a requirement that exists solely for
    // networked cross-platform play, which the engine does not have. The
    // contract this rung delivers is **per-binary determinism**: identical
    // mutation order, identical fixed steps, identical bytes out, on one build
    // of one binary. That is what `SaveSnapshot`/`SnapshotHash` and the
    // rollback tests assert, and it is the contract until a networked rung
    // needs more.

    /// The state Jolt must save for rollback: `EStateRecorderState::All`
    /// (Global | Bodies | Contacts | Constraints).
    ///
    /// Bodies alone would restore positions and velocities but not the contact
    /// cache, whose warm-start impulses feed the next step's solver — the
    /// continuation after a restore would drift from the original. Constraints
    /// carry the same warm-start state for joints. `All` is therefore the only
    /// value that makes "step M more reproduces the original continuation"
    /// true, which is the rollback contract. It is also the only mode Jolt's
    /// validation path accepts, should a future rung want it.
    [[nodiscard]] std::vector<std::byte> SaveSnapshot() const;

    /// Restores a snapshot produced by `SaveSnapshot` on this world. Returns
    /// false (with a log) for a snapshot that is not ours: wrong magic, wrong
    /// format or Jolt version, a payload size that disagrees with the frame, a
    /// failed checksum, a body set that differs from the world's, or a Jolt
    /// rejection — and in every rejected case the world is left with the state
    /// it had, never half-applied.
    ///
    /// The body-set check matters because Jolt's restore matches bodies by id
    /// and applies as it reads: a snapshot from a world with a different set of
    /// bodies would otherwise misalign part-way through and corrupt the world.
    /// The configuration (which bodies exist) is the caller's to reproduce, via
    /// `CreateBodyWithID`; this call restores dynamic state on top of it.
    bool RestoreSnapshot(std::span<const std::byte> snapshot);

    /// FNV-1a over the snapshot bytes: the determinism tripwire. Two runs that
    /// took the same steps produce the same hash; a divergence changes it.
    [[nodiscard]] static std::uint64_t SnapshotHash(const std::vector<std::byte>& snapshot) noexcept;

    // ── Characters ──────────────────────────────────────────────────────────
    /// Creates a swept character. It is not a body: it is absent from the
    /// broadphase and is moved by `StepCharacter` *outside* the solver step, which
    /// is what makes it walk stairs, stick to slopes and ride moving platforms.
    [[nodiscard]] CharacterId CreateCharacter(const CharacterDesc& desc, const BodyPose& pose,
                                              std::uint64_t user_data = 0);
    void DestroyCharacter(CharacterId character);
    [[nodiscard]] bool IsCharacterValid(CharacterId character) const;

    /// One sweep of `delta_time`. Call from the frame's `PhysicsInput` slot,
    /// before the solver step: the character queries the world, so it must not run
    /// during `Step`.
    void StepCharacter(CharacterId character, float delta_time);

    [[nodiscard]] BodyPose GetCharacterPose(CharacterId character) const;
    void SetCharacterPose(CharacterId character, const BodyPose& pose);
    [[nodiscard]] glm::vec3 GetCharacterVelocity(CharacterId character) const;
    void SetCharacterVelocity(CharacterId character, glm::vec3 velocity);
    [[nodiscard]] bool IsCharacterOnGround(CharacterId character) const;
    /// Velocity of the surface the character stands on, so gameplay can ride a
    /// moving platform instead of sliding off it.
    [[nodiscard]] glm::vec3 GetCharacterGroundVelocity(CharacterId character) const;
    [[nodiscard]] std::uint32_t CharacterCount() const noexcept;

    // ── Render interpolation ─────────────────────────────────────────────────
    //
    // Physics steps at a fixed rate; frames arrive whenever they arrive. A
    // renderer that read the live fixed-step pose would show the step pattern as
    // judder, so a body that is rendered — and only such a body — has its
    // previous and current fixed-step poses kept here, and the renderer samples
    // a blend of the two with `ApplicationContext::frame.interpolation_alpha`.
    //
    // This is deliberately opt-in and sparse: the world stores state only for
    // bodies the render gather actually draws, so an untracked body costs
    // nothing per step, and the step's tracking work is proportional to the
    // tracked set, never to every body in the system. `Transform` keeps holding
    // the fixed-step gameplay truth; only the render matrix is interpolated.

    /// Registers a body whose pose is interpolated for rendering. Tracked bodies have
    /// their previous and current fixed-step poses kept by the world; untracked bodies
    /// pay nothing. Seeds both poses from the live pose, so tracking starts with no
    /// visible jump. Idempotent.
    void TrackRenderPose(PhysicsBodyId body);
    void UntrackRenderPose(PhysicsBodyId body);
    /// True when this body's pose is being interpolated. The bridge uses this to
    /// carry tracking across a body rebuild.
    [[nodiscard]] bool IsRenderPoseTracked(PhysicsBodyId body) const noexcept;
    /// The interpolated render pose for a tracked body: `alpha` in [0,1] blends the
    /// previous fixed-step pose (0) into the current one (1). An untracked body
    /// returns its live pose.
    [[nodiscard]] BodyPose GetRenderPose(PhysicsBodyId body, float alpha) const;

    /// Same for virtual characters.
    void TrackCharacterRenderPose(CharacterId character);
    void UntrackCharacterRenderPose(CharacterId character);
    [[nodiscard]] bool IsCharacterRenderPoseTracked(CharacterId character) const noexcept;
    [[nodiscard]] BodyPose GetCharacterRenderPose(CharacterId character, float alpha) const;

    // ── Events ──────────────────────────────────────────────────────────────
    /// Installs the contact replay callback (null clears it). Called once per
    /// fixed step from `Step`, after the simulation, in deterministic order.
    void SetContactCallback(ContactCallback callback);
    void SetActivationCallback(ActivationCallback callback);
    /// Contacts replayed during the last step (diagnostics and tests).
    [[nodiscard]] std::uint64_t ContactEventCount() const noexcept { return contact_event_count_; }
    /// Activation events replayed during the last step.
    [[nodiscard]] std::uint64_t ActivationEventCount() const noexcept {
        return activation_event_count_;
    }

    // ── Queries ─────────────────────────────────────────────────────────────
    /// Closest body along the segment, or nullopt. Deterministic: the closest
    /// hit wins and ties break on the body's canonical id.
    [[nodiscard]] std::optional<RayHit> CastRay(const RayCastRequest& request) const;
    /// Shape sweep to the first blocking hit, or nullopt. Non-const because a
    /// sweep cooks (and therefore mutates) the shape cache.
    [[nodiscard]] std::optional<RayHit> CastShape(const ShapeCastRequest& request);

    // ── Debug visualization ─────────────────────────────────────────────────
    /// Appends Jolt's body/constraint debug geometry for `settings` to `out`
    /// (call `Clear` first for a fresh frame). This is the only place a Jolt
    /// `DebugRenderer` is involved, and nothing Jolt-shaped leaves this call.
    void CollectDebugGeometry(const PhysicsDebugSettings& settings, PhysicsDebugGeometry& out);

private:
    /// Drains the listener queues, sorts them deterministically and replays them
    /// into the game-facing callbacks. Runs on the stepping thread after Update.
    void DrainAndReplayEvents();

    /// The two halves of the interpolation contract, both run inside `Step` so a
    /// single rotation point governs bodies and characters alike: `prev <- cur`
    /// before Jolt updates, `cur <- live` after. Only tracked entries are
    /// touched.
    void RotateRenderPoses();
    void CaptureRenderPoses();

    PhysicsWorldConfig config_{};
    bool initialized_ = false;
    std::uint64_t step_count_ = 0;
    std::uint32_t last_errors_ = 0;
    std::uint32_t job_concurrency_ = 0;
    std::uint64_t contact_event_count_ = 0;
    std::uint64_t activation_event_count_ = 0;
    ContactCallback contact_callback_;
    ActivationCallback activation_callback_;

    std::unique_ptr<detail::WorldImpl> impl_;
    std::unique_ptr<detail::JobBridge> jobs_;
    std::shared_ptr<detail::WorldExtension> extension_;
};

} // namespace VulkanEngine::Physics
