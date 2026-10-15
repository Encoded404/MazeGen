// Jolt world construction, layer policy and stepping.
//
// This is the only place the engine builds a `JPH::PhysicsSystem`, and the only
// place that decides the object/broadphase layer tables. Everything above the
// facade sees `PhysicsBodyId`, `CollisionLayer` and glm types.

module;

#include <logging/logging_macros.hpp>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#if defined(_WIN32)
#include <malloc.h>
#endif

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/IssueReporting.h>
#include <Jolt/Core/JobSystem.h>
#include <Jolt/Core/Memory.h>
#include <Jolt/Core/QuickSort.h>
#include <Jolt/Core/StreamWrapper.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MotionProperties.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/StateRecorder.h>
#include <Jolt/Physics/StateRecorderImpl.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Renderer/DebugRenderer.h>
#include <Jolt/Renderer/DebugRendererSimple.h>

module VulkanEngine.Physics.World;

import std;
import std.compat;

import logiface;

import VulkanShared.Jobs;

namespace VulkanEngine::Physics {

namespace {

// ── Allocator override ──────────────────────────────────────────────────────
// Jolt 5.6 spells its allocation entry points as function pointers, so routing
// Jolt's allocations through the engine is an assignment, not a virtual
// subclass. The counters below are the accounting seam: an engine host allocator
// plugs in here without touching anything else in the world.
std::atomic<std::uint64_t> g_live_allocations{0};
std::atomic<std::uint64_t> g_total_allocations{0};
std::atomic<std::uint64_t> g_total_bytes{0};

void NoteAllocation(std::size_t in_size) noexcept {
    g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    g_total_allocations.fetch_add(1, std::memory_order_relaxed);
    g_total_bytes.fetch_add(in_size, std::memory_order_relaxed);
}

void* EngineAllocate(std::size_t in_size) {
    NoteAllocation(in_size);
    return std::malloc(in_size);
}

void* EngineReallocate(void* in_block, std::size_t in_old_size, std::size_t in_new_size) {
    (void)in_old_size;
    if (in_block == nullptr) return EngineAllocate(in_new_size);
    g_total_bytes.fetch_add(in_new_size, std::memory_order_relaxed);
    return std::realloc(in_block, in_new_size);
}

void EngineFree(void* in_block) {
    if (in_block == nullptr) return;
    g_live_allocations.fetch_sub(1, std::memory_order_relaxed);
    std::free(in_block);
}

void* EngineAlignedAllocate(std::size_t in_size, std::size_t in_alignment) {
    NoteAllocation(in_size);
#if defined(_WIN32)
    return _aligned_malloc(in_size, in_alignment);
#else
    // aligned_alloc requires the size to be an integral multiple of the
    // alignment; rounding up keeps the block freeable with std::free.
    const std::size_t rounded = ((in_size + in_alignment - 1) / in_alignment) * in_alignment;
    return std::aligned_alloc(in_alignment, rounded);
#endif
}

void EngineAlignedFree(void* in_block) {
    if (in_block == nullptr) return;
    g_live_allocations.fetch_sub(1, std::memory_order_relaxed);
#if defined(_WIN32)
    _aligned_free(in_block);
#else
    std::free(in_block);
#endif
}

// ── Trace / assert routing ──────────────────────────────────────────────────
void EngineTrace(const char* in_format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, in_format);
    std::vsnprintf(buffer, sizeof(buffer), in_format, args);
    va_end(args);
    LOGIFACE_LOG(info, std::string("[jolt] ") + buffer);
}

#ifdef JPH_ENABLE_ASSERTS
// Jolt only declares `JPH::AssertFailed` (and only routes assertions at all)
// when it is built with asserts on, which is the case in Debug and not in
// Release/RelWithDebInfo. Guarding here keeps both the hook and its assignment out
// of the configurations where the symbol does not exist.
bool EngineAssertFailed(const char* in_expression, const char* in_message, const char* in_file,
                        JPH::uint in_line) {
    // Logged at error level, which the startup console sink tees into the crash
    // session log, so a fired assert lands in the crash report. Returning false
    // lets Jolt continue: an assert is a bug report, not a reason to take a
    // shipped build down mid-frame.
    LOGIFACE_LOG(error, std::format("[jolt] assert failed: {} ({}) at {}:{}", in_expression,
                                    in_message != nullptr ? in_message : "", in_file,
                                    static_cast<unsigned>(in_line)));
    return false;
}
#endif // JPH_ENABLE_ASSERTS

std::atomic<int> g_bootstrap_refcount{0};

// ── Layer policy ────────────────────────────────────────────────────────────
// Two broadphase layers, split by expected update frequency: world geometry,
// sensors and other rarely-moving bodies share one tree, and the moving layers
// share the other. Layering is an optimization only — the pair table below is
// what actually decides collisions.
constexpr JPH::uint kBroadPhaseStatic = 0;
constexpr JPH::uint kBroadPhaseMoving = 1;
constexpr JPH::uint kNumBroadPhaseLayers = 2;

[[nodiscard]] constexpr JPH::uint BroadPhaseFor(CollisionLayer layer) noexcept {
    switch (layer) {
        case CollisionLayer::Debris:
        case CollisionLayer::Character:
        case CollisionLayer::Projectile:
            return kBroadPhaseMoving;
        case CollisionLayer::Default:
        case CollisionLayer::Sensor:
        case CollisionLayer::Count:
            break;
    }
    return kBroadPhaseStatic;
}

} // namespace

namespace detail {

void InstallAllocatorOverride() {
    // Idempotent by construction: this only *assigns* Jolt's five allocation entry
    // points, so re-running it cannot wrap anything. It must be re-runnable because
    // `BootstrapJolt` calls `RegisterDefaultAllocator()` on every 0 -> 1 refcount
    // transition, and that resets the pointers to Jolt's own malloc-backed
    // functions: a once-per-process guard would silently leave Jolt unaccounted
    // after the first full shutdown in a process.
    JPH::Allocate = &EngineAllocate;
    JPH::Reallocate = &EngineReallocate;
    JPH::Free = &EngineFree;
    JPH::AlignedAllocate = &EngineAlignedAllocate;
    JPH::AlignedFree = &EngineAlignedFree;
}

bool BootstrapJolt() {
    if (g_bootstrap_refcount.fetch_add(1, std::memory_order_acq_rel) > 0) {
        return true;
    }

    JPH::RegisterDefaultAllocator();
    InstallAllocatorOverride();

    if (JPH::Factory::sInstance == nullptr) {
        JPH::Factory::sInstance = new JPH::Factory();
    }

    JPH::Trace = &EngineTrace;
#ifdef JPH_ENABLE_ASSERTS
    JPH::AssertFailed = &EngineAssertFailed;
#endif

    // Verifies the Jolt ABI: a `JPH_*` define mismatch between the library and
    // this translation unit reports through Trace and aborts here rather than
    // corrupting memory later.
    JPH::RegisterTypes();

    LOGIFACE_LOG(info, "[jolt] bootstrapped: types registered, engine allocator installed");
    return true;
}

void ShutdownJolt() {
    if (g_bootstrap_refcount.fetch_sub(1, std::memory_order_acq_rel) > 1) {
        return;
    }
    // Last world out reverses the bootstrap in the documented order.
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
    LOGIFACE_LOG(info, "[jolt] shut down");
}

bool IsJoltBootstrapped() {
    return g_bootstrap_refcount.load(std::memory_order_acquire) > 0;
}

JoltMemoryStats JoltMemory() {
    return JoltMemoryStats{
        .live_allocations = g_live_allocations.load(std::memory_order_relaxed),
        .total_allocations = g_total_allocations.load(std::memory_order_relaxed),
        .total_bytes = g_total_bytes.load(std::memory_order_relaxed),
    };
}

} // namespace detail

// ── World ───────────────────────────────────────────────────────────────────

namespace {

[[nodiscard]] PhysicsDebugFaceCulling FromJoltCullMode(
    JPH::DebugRenderer::ECullMode cull_mode) noexcept {
    switch (cull_mode) {
        case JPH::DebugRenderer::ECullMode::CullBackFace:  return PhysicsDebugFaceCulling::BackFace;
        case JPH::DebugRenderer::ECullMode::CullFrontFace: return PhysicsDebugFaceCulling::FrontFace;
        case JPH::DebugRenderer::ECullMode::Off:           return PhysicsDebugFaceCulling::None;
    }
    return PhysicsDebugFaceCulling::None;
}

/// Appends Jolt's debug draw calls to engine-owned vertex buffers.
///
/// Derives from `JPH::DebugRendererSimple`, which implements the batch/geometry
/// plumbing (and initializes the box geometry Jolt's wireframe helpers draw) and
/// leaves only these three callbacks. Deriving from `DebugRenderer` directly
/// would mean reimplementing that plumbing — and a nulled batch makes Jolt's own
/// `DrawWireBox` dereference a null geometry.
///
/// Jolt passes each solid draw a `DebugRenderer::ECullMode` and documents
/// `DrawTriangle` as "a single back face culled triangle": culling by winding is
/// the renderer's job. A screen-space winding test only exists after projection,
/// which happens downstream in the debug overlay, so `DrawGeometry` records the
/// cull contract on the triangles it emits instead of culling here.
class EngineDebugRenderer final : public JPH::DebugRendererSimple {
public:
    explicit EngineDebugRenderer(PhysicsDebugGeometry& out) : out_(out) {
        // Jolt's debug renderer expects this from the concrete implementation's
        // constructor: it builds the shared box/batch geometry the wireframe
        // helpers use.
        Initialize();
    }

    void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override {
        const std::uint32_t color = inColor.GetUInt32();
        out_.lines.push_back(PhysicsDebugVertex{ToGlm(inFrom), color});
        out_.lines.push_back(PhysicsDebugVertex{ToGlm(inTo), color});
    }

    void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3,
                      JPH::ColorArg inColor, ECastShadow /*inCastShadow*/) override {
        const std::uint32_t color = inColor.GetUInt32();
        out_.triangles.push_back(PhysicsDebugVertex{ToGlm(inV1), color, current_culling_});
        out_.triangles.push_back(PhysicsDebugVertex{ToGlm(inV2), color, current_culling_});
        out_.triangles.push_back(PhysicsDebugVertex{ToGlm(inV3), color, current_culling_});
    }

    void DrawText3D(JPH::RVec3Arg /*inPosition*/, const std::string_view& /*inString*/,
                    JPH::ColorArg /*inColor*/, float /*inHeight*/) override {}

    void DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox& inWorldSpaceBounds,
                      float inLODScaleSq, JPH::ColorArg inModelColor,
                      const JPH::DebugRenderer::GeometryRef& inGeometry, ECullMode inCullMode,
                      ECastShadow inCastShadow, EDrawMode inDrawMode) override {
        // Draw calls do not nest, but the save/restore keeps a cull mode from
        // leaking into triangles emitted after a callback throws or early-outs.
        const PhysicsDebugFaceCulling previous = current_culling_;
        current_culling_ = FromJoltCullMode(inCullMode);
        JPH::DebugRendererSimple::DrawGeometry(inModelMatrix, inWorldSpaceBounds, inLODScaleSq,
                                               inModelColor, inGeometry, inCullMode, inCastShadow,
                                               inDrawMode);
        current_culling_ = previous;
    }

private:
    [[nodiscard]] static glm::vec3 ToGlm(JPH::RVec3Arg value) {
        return glm::vec3(static_cast<float>(value.GetX()), static_cast<float>(value.GetY()),
                         static_cast<float>(value.GetZ()));
    }

    PhysicsDebugFaceCulling current_culling_ = PhysicsDebugFaceCulling::None;
    PhysicsDebugGeometry& out_;
};

/// Applies the debug overlay's layer mask through Jolt's own body-locked filter,
/// so nothing has to be matched back to a body after geometry is emitted.
class LayerBodyFilter final : public JPH::BodyDrawFilter {
public:
    explicit LayerBodyFilter(std::uint32_t mask) : mask_(mask) {}
    [[nodiscard]] bool ShouldDraw(const JPH::Body& body) const override {
        const auto layer = static_cast<std::uint32_t>(body.GetObjectLayer());
        return layer < 32 && (mask_ & (1u << layer)) != 0;
    }

private:
    std::uint32_t mask_ = 0;
};

} // namespace


namespace {

// ── glm <-> Jolt conversion ─────────────────────────────────────────────────
// The engine is meter-scale and render-local, so single precision is the right
// default; these would need `JPH_DOUBLE_PRECISION` handling if that ever changed.
[[nodiscard]] JPH::Vec3 ToVec3(const glm::vec3& value) { return JPH::Vec3(value.x, value.y, value.z); }
[[nodiscard]] JPH::RVec3 ToRVec3(const glm::vec3& value) { return JPH::RVec3(value.x, value.y, value.z); }
[[nodiscard]] JPH::Quat ToQuat(const glm::quat& value) { return JPH::Quat(value.x, value.y, value.z, value.w); }
[[nodiscard]] glm::vec3 FromVec3(JPH::Vec3Arg value) { return glm::vec3(value.GetX(), value.GetY(), value.GetZ()); }
[[nodiscard]] glm::vec3 FromRVec3(JPH::RVec3Arg value) {
    return glm::vec3(static_cast<float>(value.GetX()), static_cast<float>(value.GetY()),
                     static_cast<float>(value.GetZ()));
}
[[nodiscard]] glm::quat FromQuat(JPH::QuatArg value) {
    return glm::quat(value.GetW(), value.GetX(), value.GetY(), value.GetZ());
}

[[nodiscard]] JPH::EMotionType ToJoltMotion(BodyMotionType motion) noexcept {
    switch (motion) {
        case BodyMotionType::Static:    return JPH::EMotionType::Static;
        case BodyMotionType::Kinematic: return JPH::EMotionType::Kinematic;
        case BodyMotionType::Dynamic:   return JPH::EMotionType::Dynamic;
    }
    return JPH::EMotionType::Static;
}

[[nodiscard]] BodyMotionType FromJoltMotion(JPH::EMotionType motion) noexcept {
    switch (motion) {
        case JPH::EMotionType::Static:    return BodyMotionType::Static;
        case JPH::EMotionType::Kinematic: return BodyMotionType::Kinematic;
        case JPH::EMotionType::Dynamic:   return BodyMotionType::Dynamic;
    }
    return BodyMotionType::Static;
}

[[nodiscard]] JPH::ObjectLayer ToJoltLayer(CollisionLayer layer) noexcept {
    return static_cast<JPH::ObjectLayer>(layer);
}

[[nodiscard]] JPH::BodyID ToJoltBodyId(PhysicsBodyId body) noexcept {
    return JPH::BodyID(body.value);
}

[[nodiscard]] PhysicsBodyId FromJoltBodyId(const JPH::BodyID& body) noexcept {
    return PhysicsBodyId{body.GetIndexAndSequenceNumber()};
}

[[nodiscard]] JPH::EActivation ToActivation(bool activate) noexcept {
    return activate ? JPH::EActivation::Activate : JPH::EActivation::DontActivate;
}

/// Scratch Jolt may use per update without falling back to malloc. Large enough
/// that a normal step never touches the fallback allocator, small enough to be a
/// rounding error in the engine's footprint.
constexpr JPH::uint kTempAllocatorBytes = 16 * 1024 * 1024;

// ── Event capture ───────────────────────────────────────────────────────────
// A raw contact, straight from a solver thread. Kept in Jolt's own types until
// the step ends, because that is the only thing safe to touch there.
struct RawContact {
    JPH::BodyID body1;
    JPH::BodyID body2;
    std::uint64_t user_data1 = 0;
    std::uint64_t user_data2 = 0;
    ContactPhase phase = ContactPhase::Added;
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    float penetration = 0.0f;
    std::uint32_t sub_shape1 = 0;
    std::uint32_t sub_shape2 = 0;
    bool sensor = false;
};

/// Contact listener that only queues.
///
/// Jolt calls this from several threads while all bodies are locked, so nothing
/// here may allocate a body lock, mutate physics state, or read a body through a
/// locking interface. `OnContactRemoved` cannot read the bodies at all, which is
/// why the add/persist path caches each pair's user data for it.
class EngineContactListener final : public JPH::ContactListener {
public:
    void OnContactAdded(const JPH::Body& inBody1, const JPH::Body& inBody2,
                        const JPH::ContactManifold& inManifold,
                        JPH::ContactSettings& ioSettings) override {
        Queue(ContactPhase::Added, inBody1, inBody2, inManifold, ioSettings.mIsSensor);
    }

    void OnContactPersisted(const JPH::Body& inBody1, const JPH::Body& inBody2,
                            const JPH::ContactManifold& inManifold,
                            JPH::ContactSettings& ioSettings) override {
        Queue(ContactPhase::Persisted, inBody1, inBody2, inManifold, ioSettings.mIsSensor);
    }

    void OnContactRemoved(const JPH::SubShapeIDPair& inSubShapePair) override {
        RawContact raw{};
        raw.body1 = inSubShapePair.GetBody1ID();
        raw.body2 = inSubShapePair.GetBody2ID();
        raw.phase = ContactPhase::Removed;
        raw.sub_shape1 = inSubShapePair.GetSubShapeID1().GetValue();
        raw.sub_shape2 = inSubShapePair.GetSubShapeID2().GetValue();

        const std::scoped_lock lock(mutex_);
        if (const auto it = pair_user_data_.find(PairKey(raw.body1, raw.body2));
            it != pair_user_data_.end()) {
            raw.user_data1 = it->second.first;
            raw.user_data2 = it->second.second;
            pair_user_data_.erase(it);
        }
        events_.push_back(raw);
    }

    /// Move the queued events out. Called on the stepping thread, after Update.
    void Drain(std::vector<RawContact>& out) {
        const std::scoped_lock lock(mutex_);
        out.insert(out.end(), events_.begin(), events_.end());
        events_.clear();
    }

private:
    [[nodiscard]] static std::uint64_t PairKey(const JPH::BodyID& a, const JPH::BodyID& b) noexcept {
        return (static_cast<std::uint64_t>(a.GetIndexAndSequenceNumber()) << 32) |
               b.GetIndexAndSequenceNumber();
    }

    void Queue(ContactPhase phase, const JPH::Body& inBody1, const JPH::Body& inBody2,
               const JPH::ContactManifold& inManifold, bool sensor) {
        RawContact raw{};
        raw.body1 = inBody1.GetID();
        raw.body2 = inBody2.GetID();
        raw.user_data1 = inBody1.GetUserData();
        raw.user_data2 = inBody2.GetUserData();
        raw.phase = phase;
        raw.penetration = inManifold.mPenetrationDepth;
        raw.sub_shape1 = inManifold.mSubShapeID1.GetValue();
        raw.sub_shape2 = inManifold.mSubShapeID2.GetValue();
        raw.sensor = sensor;
        raw.normal = FromVec3(inManifold.mWorldSpaceNormal);
        if (inManifold.mRelativeContactPointsOn1.size() > 0) {
            const JPH::RVec3 point = inManifold.GetWorldSpaceContactPointOn1(0);
            raw.position = glm::vec3(static_cast<float>(point.GetX()),
                                     static_cast<float>(point.GetY()),
                                     static_cast<float>(point.GetZ()));
        }

        const std::scoped_lock lock(mutex_);
        pair_user_data_[PairKey(raw.body1, raw.body2)] = {raw.user_data1, raw.user_data2};
        events_.push_back(raw);
    }

    std::mutex mutex_;
    std::vector<RawContact> events_;
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> pair_user_data_;
};

class EngineActivationListener final : public JPH::BodyActivationListener {
public:
    void OnBodyActivated(const JPH::BodyID& inBodyID, JPH::uint64 inBodyUserData) override {
        Queue(inBodyID, inBodyUserData, true);
    }
    void OnBodyDeactivated(const JPH::BodyID& inBodyID, JPH::uint64 inBodyUserData) override {
        Queue(inBodyID, inBodyUserData, false);
    }

    void Drain(std::vector<ActivationEvent>& out) {
        const std::scoped_lock lock(mutex_);
        out.insert(out.end(), events_.begin(), events_.end());
        events_.clear();
    }

private:
    void Queue(const JPH::BodyID& id, JPH::uint64 user_data, bool activated) {
        const std::scoped_lock lock(mutex_);
        events_.push_back(ActivationEvent{FromJoltBodyId(id), user_data, activated});
    }

    std::mutex mutex_;
    std::vector<ActivationEvent> events_;
};

// ── Query filters ───────────────────────────────────────────────────────────
/// Object-layer filter from a bit mask. `0` means "every layer".
class LayerMaskFilter final : public JPH::ObjectLayerFilter {
public:
    explicit LayerMaskFilter(std::uint32_t mask) : mask_(mask) {}
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer layer) const override {
        if (mask_ == 0) return true;
        return layer < 32 && (mask_ & (1u << layer)) != 0;
    }

private:
    std::uint32_t mask_ = 0;
};

/// Motion-type and sensor filter. Narrow-phase queries already hold the body
/// lock, so this reads the body directly instead of re-locking it.
class MotionTypeFilter final : public JPH::BodyFilter {
public:
    MotionTypeFilter(bool hit_static, bool hit_kinematic, bool hit_dynamic, bool hit_sensors)
        : hit_static_(hit_static), hit_kinematic_(hit_kinematic), hit_dynamic_(hit_dynamic),
          hit_sensors_(hit_sensors) {}

    [[nodiscard]] bool ShouldCollideLocked(const JPH::Body& body) const override {
        if (!hit_sensors_ && body.IsSensor()) return false;
        switch (body.GetMotionType()) {
            case JPH::EMotionType::Static:    return hit_static_;
            case JPH::EMotionType::Kinematic: return hit_kinematic_;
            case JPH::EMotionType::Dynamic:   return hit_dynamic_;
        }
        return false;
    }

private:
    bool hit_static_ = true;
    bool hit_kinematic_ = true;
    bool hit_dynamic_ = true;
    bool hit_sensors_ = true;
};

class AllBroadPhaseLayers final : public JPH::BroadPhaseLayerFilter {
public:
    [[nodiscard]] bool ShouldCollide(JPH::BroadPhaseLayer) const override { return true; }
};

} // namespace

struct detail::WorldImpl {
    // Declared so the physics system is destroyed first (reverse order): the
    // layer interfaces it references stay alive until it is gone.
    std::unique_ptr<JPH::BroadPhaseLayerInterfaceTable> broad_phase_layers;
    std::unique_ptr<JPH::ObjectLayerPairFilterTable> object_pair_filter;
    std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilterTable> object_vs_broad_phase;
    std::unique_ptr<JPH::TempAllocatorImplWithMallocFallback> temp_allocator;
    std::unique_ptr<JPH::PhysicsSystem> physics;

    // Cooked shapes, keyed by `ShapeDesc::GeometryHash()`. Ref-counted by Jolt:
    // a shape stays alive while a body uses it, so eviction is unnecessary.
    std::unordered_map<std::size_t, JPH::RefConst<JPH::Shape>> shape_cache;

    // Cooked-shape disk cache state. `shape_cache_directory` is empty when the
    // cache is disabled, which is the default: nothing here touches the disk
    // unless a host asks for a directory at `Initialize`.
    std::filesystem::path shape_cache_directory;
    std::uint64_t shape_cache_disk_hits = 0;
    std::uint64_t shape_cache_disk_misses = 0;

    // Render interpolation. Two fixed-step poses per *tracked* body, keyed by
    // `PhysicsBodyId::value` (index+sequence, so a recycled slot is a different
    // key) and by `CharacterId::slot` for characters. A body that is never
    // tracked has no entry and costs the step nothing, which is the point:
    // interpolation is for what the render gather draws, not for the world.
    struct RenderPose {
        BodyPose prev{}; // pose at the end of the previous fixed step
        BodyPose cur{};  // pose at the end of the current fixed step
    };
    std::unordered_map<std::uint32_t, RenderPose> render_bodies;
    std::unordered_map<std::uint32_t, RenderPose> render_characters;

    // Event capture: listeners queue during the step, the world replays after.
    std::unique_ptr<EngineContactListener> contact_listener;
    std::unique_ptr<EngineActivationListener> activation_listener;
    std::vector<RawContact> raw_contacts;
    std::vector<ActivationEvent> raw_activations;

    // Debug visualization scratch: the Jolt debug renderer appends here, and a
    // collection call copies the frame's geometry into the caller's buffer.
    std::unique_ptr<EngineDebugRenderer> debug_renderer;
    PhysicsDebugGeometry debug_geometry;

    // Virtual characters. Slot allocation is generational, like every other
    // engine handle, so a stale CharacterId cannot name a reused slot.
    struct CharacterSlot {
        std::unique_ptr<JPH::CharacterVirtual> character;
        CharacterDesc desc{};
        std::uint32_t generation = 0;
    };
    std::vector<std::unique_ptr<CharacterSlot>> character_slots;
    std::vector<std::uint32_t> character_free_slots;
    std::uint32_t next_character_generation = 1;
};

World::World() = default;

World::~World() {
    Shutdown();
}

namespace {

// ── Cooked-shape disk cache ─────────────────────────────────────────────────
//
// A cooked shape is a cache, never primary data: `ShapeDesc` is the truth, and
// cooking it is cheap next to what it saves. The blob on disk is this header
// followed by Jolt's own `Shape::SaveWithChildren` payload:
//
//     [ magic | jolt version | geometry hash | payload bytes | checksum ]
//     [ ... Jolt shape binary ... ]
//
// The header is not decoration. `Shape::sRestoreFromBinaryState` reads a shape
// subtype straight out of the stream and indexes Jolt's shape registry with it,
// so bytes that are not a shape are an out-of-bounds read rather than a returned
// error. Everything that could make a file untrustworthy — the version in the
// filename, a truncated file, a flipped bit — is therefore rejected here,
// before the payload is parsed, and a rejected file is simply a miss.
constexpr std::uint64_t kShapeCacheMagic = 0x5348415045360001ull; // "SHAPE6\0\1"
constexpr std::uint64_t kFnv1aOffset = 0xcbf29ce484222325ull;
constexpr std::uint64_t kFnv1aPrime = 0x100000001b3ull;

struct ShapeCacheHeader {
    std::uint64_t magic = kShapeCacheMagic;
    std::uint32_t jolt_version = 0; // JPH_VERSION_ID; Jolt's format is not stable
    std::uint32_t reserved = 0;     // spelled out so the layout is not implicit
    std::uint64_t geometry_hash = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_checksum = 0; // FNV-1a over the payload
};
static_assert(sizeof(ShapeCacheHeader) == 40, "the cache header is a disk format");

/// FNV-1a, the hash `ShapeDesc::GeometryHash` already uses, so the codebase has
/// one hashing story. Not cryptographic: this detects corruption, not intent.
[[nodiscard]] std::uint64_t HashBytes(std::uint64_t seed, std::span<const std::byte> bytes) noexcept {
    for (const std::byte byte : bytes) {
        seed ^= std::to_integer<std::uint8_t>(byte);
        seed *= kFnv1aPrime;
    }
    return seed;
}

/// Order-independent checksum of a body set: Jolt's restore matches bodies by
/// id, so a stream is only applicable to a world with exactly the same ids —
/// order does not matter. The ids are sorted with Jolt's own `QuickSort` (the
/// codebase's physics-path sort) before folding, so two worlds that hold the
/// same bodies agree regardless of allocation order.
[[nodiscard]] std::uint64_t BodySetChecksum(const std::vector<PhysicsBodyId>& ids) noexcept {
    std::vector<std::uint32_t> values;
    values.reserve(ids.size());
    for (const PhysicsBodyId id : ids) values.push_back(id.value);
    JPH::QuickSort(values.begin(), values.end());
    std::uint64_t seed = kFnv1aOffset;
    for (const std::uint32_t value : values) {
        seed = HashBytes(seed, std::as_bytes(std::span{&value, 1}));
    }
    return seed;
}

/// The Jolt version a blob is valid for, which is the one thing that makes cache
/// staleness automatic: Jolt's shape binary format is explicitly not stable
/// across versions, so a blob written by another one is a miss.
///
/// `JPH_VERSION_ID` expands to a Jolt-namespace expression (`uint64(...)`), so it
/// only compiles where that name is visible; this reads it once, here, instead of
/// at each use site.
[[nodiscard]] constexpr std::uint32_t JoltShapeFormatVersion() noexcept {
    using namespace JPH; // NOLINT(google-build-using-namespace) — only for the macro's spelling
    return static_cast<std::uint32_t>(JPH_VERSION_ID);
}

/// The blob's name. The Jolt version is in the name *and* in the header: a Jolt
/// upgrade changes the name, so stale blobs are never even opened, and the
/// header catches a file that was renamed or copied by hand.
[[nodiscard]] std::filesystem::path ShapeCachePath(const std::filesystem::path& directory,
                                                   std::size_t geometry_hash) {
    return directory / std::format("shape_{:08x}_{:016x}.jshape", JoltShapeFormatVersion(),
                                   static_cast<std::uint64_t>(geometry_hash));
}

/// Creates the cache directory, or turns the cache off. Either is fine: a host
/// with a read-only data directory still gets a world, it just cooks every shape.
void PrepareShapeCacheDirectory(detail::WorldImpl& impl) {
    if (impl.shape_cache_directory.empty()) return;

    std::error_code error;
    std::filesystem::create_directories(impl.shape_cache_directory, error);
    if (error) {
        LOGIFACE_LOG(warn, std::format(
            "physics: cooked-shape cache disabled, cannot create '{}': {}",
            impl.shape_cache_directory.string(), error.message()));
        impl.shape_cache_directory.clear();
        return;
    }
    LOGIFACE_LOG(debug, std::format("physics: cooked-shape cache at '{}' (Jolt version {:08x})",
                                    impl.shape_cache_directory.string(),
                                    JoltShapeFormatVersion()));
}

/// Reads back a blob this world wrote for this geometry. Returns null for *any*
/// reason at all — no file, a short read, a version or hash mismatch, a failed
/// checksum, or Jolt refusing the payload — because a cache is never allowed to
/// fail the world.
[[nodiscard]] JPH::RefConst<JPH::Shape> LoadCookedShape(const detail::WorldImpl& impl,
                                                        std::size_t geometry_hash) {
    const std::filesystem::path path = ShapeCachePath(impl.shape_cache_directory, geometry_hash);

    std::error_code error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
    if (error) return {}; // no blob yet: the ordinary first-run miss, not worth a log
    if (file_bytes < sizeof(ShapeCacheHeader)) {
        LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache blob '{}' is too small to hold a header, re-cooking",
                                       path.string()));
        return {};
    }

    ShapeCacheHeader header{};
    std::uint64_t checksum = kFnv1aOffset;
    {
        // Pass one: validate the header and checksum the payload. No payload byte
        // is parsed until the file has been read once and found intact.
        std::ifstream file(path, std::ios::binary);
        if (!file) return {};

        JPH::StreamInWrapper reader(file);
        reader.Read(header);
        if (reader.IsEOF() || reader.IsFailed()) {
            LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache truncated header in '{}'",
                                           path.string()));
            return {};
        }
        if (header.magic != kShapeCacheMagic) {
            LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache blob '{}' is not a shape blob, re-cooking",
                                           path.string()));
            return {};
        }
        if (header.jolt_version != JoltShapeFormatVersion()) {
            LOGIFACE_LOG(debug, std::format(
                "physics: cooked-shape cache blob '{}' was written by Jolt {:08x}, re-cooking",
                path.string(), header.jolt_version));
            return {};
        }
        if (header.geometry_hash != geometry_hash ||
            header.payload_bytes != file_bytes - sizeof(ShapeCacheHeader)) {
            LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache blob '{}' does not match its name, re-cooking",
                                           path.string()));
            return {};
        }

        std::array<char, 64 * 1024> buffer{};
        std::uint64_t remaining = header.payload_bytes;
        while (remaining > 0) {
            const auto chunk = static_cast<std::streamsize>(
                std::min<std::uint64_t>(remaining, buffer.size()));
            file.read(buffer.data(), chunk);
            if (file.gcount() != chunk) {
                LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache blob '{}' is truncated, re-cooking",
                                               path.string()));
                return {};
            }
            checksum = HashBytes(checksum, std::as_bytes(std::span{buffer.data(), static_cast<std::size_t>(chunk)}));
            remaining -= static_cast<std::uint64_t>(chunk);
        }
    }
    if (checksum != header.payload_checksum) {
        LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache blob '{}' is corrupt, re-cooking",
                                       path.string()));
        return {};
    }

    // Pass two: the payload, now known to be exactly the bytes this world wrote
    // for this geometry under this Jolt version.
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    file.seekg(static_cast<std::streamoff>(sizeof(ShapeCacheHeader)));
    if (!file) return {};

    JPH::StreamInWrapper reader(file);
    JPH::Shape::IDToShapeMap shapes;
    JPH::Shape::IDToMaterialMap materials;
    const JPH::ShapeSettings::ShapeResult result =
        JPH::Shape::sRestoreWithChildren(reader, shapes, materials);
    if (result.HasError() || result.Get() == nullptr) {
        LOGIFACE_LOG(warn, std::format("physics: cannot restore cooked shape from '{}': {}, re-cooking",
                                       path.string(),
                                       result.HasError() ? result.GetError() : "empty shape"));
        return {};
    }
    if (reader.IsFailed()) {
        LOGIFACE_LOG(warn, std::format("physics: cooked-shape cache read failed for '{}', re-cooking",
                                       path.string()));
        return {};
    }
    return JPH::RefConst<JPH::Shape>{result.Get()};
}

/// Writes Jolt's shape binary to a file while measuring it, so the header can be
/// completed without a second pass over the payload. Composes the stream wrapper
/// the rest of the engine uses; the counting is the only addition.
class ShapeCacheWriter final : public JPH::StreamOut {
public:
    explicit ShapeCacheWriter(std::ostream& out) : wrapped_(out) {}

    void WriteBytes(const void* data, std::size_t bytes) override {
        wrapped_.WriteBytes(data, bytes);
        payload_bytes_ += bytes;
        payload_checksum_ = HashBytes(payload_checksum_,
                                      std::span<const std::byte>(static_cast<const std::byte*>(data), bytes));
    }
    [[nodiscard]] bool IsFailed() const override { return wrapped_.IsFailed(); }

    [[nodiscard]] std::uint64_t PayloadBytes() const noexcept { return payload_bytes_; }
    [[nodiscard]] std::uint64_t PayloadChecksum() const noexcept { return payload_checksum_; }

private:
    JPH::StreamOutWrapper wrapped_;
    std::uint64_t payload_bytes_ = 0;
    std::uint64_t payload_checksum_ = kFnv1aOffset;
};

/// Best-effort write of a freshly cooked geometry shape. The file is published
/// by its header, not by its creation: the placeholder written first carries a
/// zero magic, so a write that is interrupted at *any* point leaves a file the
/// reader rejects (and the next world re-cooks and overwrites).
void SaveCookedShape(const detail::WorldImpl& impl, const JPH::Shape& shape,
                     std::size_t geometry_hash) {
    const std::filesystem::path path = ShapeCachePath(impl.shape_cache_directory, geometry_hash);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        LOGIFACE_LOG(warn, std::format("physics: cannot write cooked-shape cache '{}'", path.string()));
        return;
    }

    ShapeCacheHeader header{};
    header.magic = 0;
    header.jolt_version = JoltShapeFormatVersion();
    header.geometry_hash = geometry_hash;
    {
        JPH::StreamOutWrapper writer(file);
        writer.Write(header); // placeholder: replaced below
        ShapeCacheWriter payload(file);
        JPH::Shape::ShapeToIDMap shapes;
        JPH::Shape::MaterialToIDMap materials;
        shape.SaveWithChildren(payload, shapes, materials);
        header.payload_bytes = payload.PayloadBytes();
        header.payload_checksum = payload.PayloadChecksum();
        if (payload.IsFailed()) {
            LOGIFACE_LOG(warn, std::format("physics: cannot write cooked-shape cache '{}'", path.string()));
            // Leave nothing behind: the reader would reject a half-written blob
            // anyway, but a failed write should not litter the cache directory.
            file.close();
            std::error_code error;
            std::filesystem::remove(path, error);
            return;
        }
    }

    header.magic = kShapeCacheMagic;
    file.seekp(0);
    JPH::StreamOutWrapper writer(file);
    writer.Write(header);
    file.flush();
    if (!file) {
        LOGIFACE_LOG(warn, std::format("physics: cannot finalize cooked-shape cache '{}'", path.string()));
        return;
    }
    LOGIFACE_LOG(debug, std::format("physics: cooked shape cached at '{}' ({} bytes)", path.string(),
                                    header.payload_bytes));
}

// ── Render interpolation ────────────────────────────────────────────────────

/// `alpha` 0 is `from`, 1 is `to`. Rotation goes by the shortest arc — both
/// `glm::slerp` and `glm::mix` negate one end when the quaternions are more than
/// 90 degrees apart — so a body spinning past half a turn never unwinds
/// backwards through the blend. The endpoints are returned exactly rather than
/// computed, so `alpha == 1` is bit-for-bit the step's own pose.
[[nodiscard]] BodyPose BlendPoses(const BodyPose& from, const BodyPose& to, float alpha) {
    if (alpha <= 0.0f) return from;
    if (alpha >= 1.0f) return to;
    return BodyPose{
        glm::mix(from.position, to.position, alpha),
        glm::slerp(from.rotation, to.rotation, alpha),
    };
}

} // namespace

bool World::Initialize(const PhysicsWorldConfig& config) {
    if (initialized_) return true;
    if (config.fixed_delta_time <= 0.0f) {
        LOGIFACE_LOG(error, "physics: fixed_delta_time must be positive");
        return false;
    }

    if (!detail::BootstrapJolt()) {
        LOGIFACE_LOG(error, "physics: Jolt bootstrap failed");
        return false;
    }

    config_ = config;
    impl_ = std::make_unique<detail::WorldImpl>();
    impl_->shape_cache_directory = config.shape_cache_directory;
    PrepareShapeCacheDirectory(*impl_);

    const auto object_layers = static_cast<JPH::uint>(CollisionLayer::Count);

    impl_->broad_phase_layers =
        std::make_unique<JPH::BroadPhaseLayerInterfaceTable>(object_layers, kNumBroadPhaseLayers);
    for (JPH::uint layer = 0; layer < object_layers; ++layer) {
        impl_->broad_phase_layers->MapObjectToBroadPhaseLayer(
            layer, JPH::BroadPhaseLayer(BroadPhaseFor(static_cast<CollisionLayer>(layer))));
    }

    // Collision policy as a table: everything collides with everything except
    // sensor-vs-sensor, which cannot mean anything. Sensors still collide with
    // ordinary bodies — that is how a trigger volume detects one — they simply
    // never resolve a contact.
    impl_->object_pair_filter = std::make_unique<JPH::ObjectLayerPairFilterTable>(object_layers);
    for (JPH::uint a = 0; a < object_layers; ++a) {
        for (JPH::uint b = a; b < object_layers; ++b) {
            const auto layer_a = static_cast<CollisionLayer>(a);
            const auto layer_b = static_cast<CollisionLayer>(b);
            if (layer_a == CollisionLayer::Sensor && layer_b == CollisionLayer::Sensor) continue;
            impl_->object_pair_filter->EnableCollision(a, b);
        }
    }

    impl_->object_vs_broad_phase = std::make_unique<JPH::ObjectVsBroadPhaseLayerFilterTable>(
        *impl_->broad_phase_layers, kNumBroadPhaseLayers, *impl_->object_pair_filter, object_layers);

    impl_->physics = std::make_unique<JPH::PhysicsSystem>();
    // inNumBodyMutexes = 0 lets Jolt size the body locks from the body count.
    impl_->physics->Init(config_.max_bodies, /*inNumBodyMutexes=*/0, config_.max_body_pairs,
                         config_.max_contact_constraints, *impl_->broad_phase_layers,
                         *impl_->object_vs_broad_phase, *impl_->object_pair_filter);

    impl_->physics->SetGravity(
        JPH::Vec3(config_.gravity.x, config_.gravity.y, config_.gravity.z));

    impl_->temp_allocator =
        std::make_unique<JPH::TempAllocatorImplWithMallocFallback>(kTempAllocatorBytes);

    // Event listeners. They only queue; `Step` sorts and replays afterwards, so
    // the observable order is deterministic even though the calls arrive from
    // several solver threads.
    impl_->contact_listener = std::make_unique<EngineContactListener>();
    impl_->activation_listener = std::make_unique<EngineActivationListener>();
    impl_->physics->SetContactListener(impl_->contact_listener.get());
    impl_->physics->SetBodyActivationListener(impl_->activation_listener.get());

    jobs_ = detail::CreateJobBridge();
    if (config_.max_concurrency != 0) {
        job_concurrency_ = config_.max_concurrency;
    } else {
        auto* job_system = static_cast<JPH::JobSystem*>(jobs_->NativeJobSystem());
        job_concurrency_ = static_cast<std::uint32_t>(job_system->GetMaxConcurrency());
    }

    initialized_ = true;
    LOGIFACE_LOG(info, std::format(
        "physics: world initialized ({} bodies, {} pairs, {} contacts, {} job workers, {} Hz)",
        config_.max_bodies, config_.max_body_pairs, config_.max_contact_constraints,
        job_concurrency_, 1.0f / config_.fixed_delta_time));
    return true;
}

void World::Shutdown() {
    if (!initialized_) return;
    initialized_ = false;

    // Extensions first: an attached module (constraints) must release the Jolt
    // objects it owns while the physics system is still alive. The extension
    // itself is released only *after* the bodies are gone, because a Jolt body's
    // `CollisionGroup` can hold a reference to a filter the extension owns.
    if (extension_) extension_->OnWorldShutdown();

    // The bridge next: Jolt's job system must not outlive the world's use of it,
    // and the physics system holds no reference to it after an Update returns.
    jobs_.reset();
    impl_.reset();

    extension_.reset();

    detail::ShutdownJolt();

    job_concurrency_ = 0;
    step_count_ = 0;
    last_errors_ = 0;
}

void World::Step() {
    if (!initialized_) return;

    // The interpolation contract's one rotation point. Everything that happened
    // since the last step — the solver's bodies and the characters the bridge
    // swept during the `PhysicsInput` slot — becomes the *previous* pose, so the
    // blend the renderer samples after this step runs from here to there.
    RotateRenderPoses();

    auto* job_system = static_cast<JPH::JobSystem*>(jobs_->NativeJobSystem());
    const JPH::EPhysicsUpdateError errors = impl_->physics->Update(
        config_.fixed_delta_time, static_cast<int>(config_.collision_steps),
        impl_->temp_allocator.get(), job_system);

    last_errors_ = static_cast<std::uint32_t>(errors);
    ++step_count_;

    CaptureRenderPoses();

    DrainAndReplayEvents();

    // Caps return an error bitmask rather than silently dropping bodies; surface
    // it loudly so an overloaded world is visible in the log.
    if (last_errors_ != 0) {
        LOGIFACE_LOG(warn, std::format(
            "physics: update reported errors 0x{:x} ({} bodies, {} body pairs, {} contacts)",
            last_errors_, impl_->physics->GetNumBodies(), config_.max_body_pairs,
            config_.max_contact_constraints));
    }
}

void World::DrainAndReplayEvents() {
    impl_->raw_contacts.clear();
    impl_->contact_listener->Drain(impl_->raw_contacts);
    impl_->raw_activations.clear();
    impl_->activation_listener->Drain(impl_->raw_activations);

    // Deterministic replay order: the owning entity pair first, then the phase,
    // then the sub-shape pair. Jolt's `QuickSort` is used rather than `std::sort`
    // because it is the sort the rest of the physics path is built on and it does
    // not allocate.
    std::vector<RawContact>& contacts = impl_->raw_contacts;
    JPH::QuickSort(contacts.begin(), contacts.end(), [](const RawContact& a, const RawContact& b) {
        if (a.user_data1 != b.user_data1) return a.user_data1 < b.user_data1;
        if (a.user_data2 != b.user_data2) return a.user_data2 < b.user_data2;
        if (a.phase != b.phase) return a.phase < b.phase;
        if (a.sub_shape1 != b.sub_shape1) return a.sub_shape1 < b.sub_shape1;
        return a.sub_shape2 < b.sub_shape2;
    });

    contact_event_count_ = contacts.size();
    if (contact_callback_) {
        for (const RawContact& raw : contacts) {
            contact_callback_(ContactEvent{
                .body_a = FromJoltBodyId(raw.body1),
                .body_b = FromJoltBodyId(raw.body2),
                .user_data_a = raw.user_data1,
                .user_data_b = raw.user_data2,
                .phase = raw.phase,
                .position = raw.position,
                .normal = raw.normal,
                .penetration = raw.penetration,
                .sub_shape_a = raw.sub_shape1,
                .sub_shape_b = raw.sub_shape2,
                .sensor = raw.sensor,
            });
        }
    }

    std::vector<ActivationEvent>& activations = impl_->raw_activations;
    JPH::QuickSort(activations.begin(), activations.end(),
                   [](const ActivationEvent& a, const ActivationEvent& b) {
                       if (a.user_data != b.user_data) return a.user_data < b.user_data;
                       if (a.activated != b.activated) return a.activated < b.activated;
                       return a.body.value < b.body.value;
                   });

    activation_event_count_ = activations.size();
    if (activation_callback_) {
        for (const ActivationEvent& event : activations) activation_callback_(event);
    }
}

void World::SetContactCallback(ContactCallback callback) {
    contact_callback_ = std::move(callback);
}

void World::SetActivationCallback(ActivationCallback callback) {
    activation_callback_ = std::move(callback);
}

void* World::NativePhysicsSystem() noexcept {
    return impl_ ? impl_->physics.get() : nullptr;
}

void World::AttachExtension(std::shared_ptr<detail::WorldExtension> extension) {
    if (extension_) extension_->OnWorldShutdown();
    extension_ = std::move(extension);
}

std::uint32_t World::BodyCount() const noexcept {
    return impl_ && impl_->physics ? impl_->physics->GetNumBodies() : 0;
}

bool World::LayersCollide(CollisionLayer a, CollisionLayer b) const noexcept {
    if (!impl_ || !impl_->object_pair_filter) return false;
    if (a == CollisionLayer::Count || b == CollisionLayer::Count) return false;
    return impl_->object_pair_filter->ShouldCollide(static_cast<JPH::ObjectLayer>(a),
                                                    static_cast<JPH::ObjectLayer>(b));
}

// ── Shape cooking ───────────────────────────────────────────────────────────

namespace {

/// Cooks the geometry half of a descriptor. Placement is not baked in: it is
/// applied per body by `ApplyPlacement`, so one cooked box serves every instance
/// of that box regardless of where it sits or how it is rotated.
JPH::RefConst<JPH::Shape> CookGeometry(const ShapeDesc& desc) {
    const auto fail = [](const char* what, std::string_view error) {
        LOGIFACE_LOG(error, std::format("physics: cannot cook {} shape: {}", what, error));
        return JPH::RefConst<JPH::Shape>{};
    };
    const auto finish = [&fail](const char* what, const JPH::ShapeSettings& settings) {
        const JPH::ShapeSettings::ShapeResult result = settings.Create();
        if (!result.IsValid()) return fail(what, result.GetError());
        return JPH::RefConst<JPH::Shape>{result.Get()};
    };

    switch (desc.type) {
        case ShapeType::Box: {
            // The descriptor carries full extents; Jolt wants half extents.
            const glm::vec3 half = desc.box_extent * 0.5f;
            if (half.x <= 0.0f || half.y <= 0.0f || half.z <= 0.0f) {
                return fail("box", "non-positive extent");
            }
            return finish("box", JPH::BoxShapeSettings(ToVec3(half)));
        }
        case ShapeType::Sphere: {
            if (desc.radius <= 0.0f) return fail("sphere", "non-positive radius");
            return finish("sphere", JPH::SphereShapeSettings(desc.radius));
        }
        case ShapeType::Capsule: {
            // Height is the full capsule height, so the cylindrical half is
            // height/2 - radius.
            const float half_cylinder = std::max(0.0f, desc.height * 0.5f - desc.radius);
            if (desc.radius <= 0.0f) return fail("capsule", "non-positive radius");
            return finish("capsule", JPH::CapsuleShapeSettings(half_cylinder, desc.radius));
        }
        case ShapeType::Cylinder: {
            if (desc.radius <= 0.0f || desc.height <= 0.0f) {
                return fail("cylinder", "non-positive radius or height");
            }
            return finish("cylinder", JPH::CylinderShapeSettings(desc.height * 0.5f, desc.radius));
        }
        case ShapeType::ConvexHull: {
            if (!desc.points || desc.points->size() < 4) {
                return fail("convex-hull", "fewer than 4 points");
            }
            JPH::Array<JPH::Vec3> points;
            points.reserve(desc.points->size());
            for (const glm::vec3& point : *desc.points) points.push_back(ToVec3(point));
            return finish("convex-hull", JPH::ConvexHullShapeSettings(points, desc.hull_shrink));
        }
        case ShapeType::Mesh: {
            if (!desc.points || !desc.indices || desc.indices->size() < 3) {
                return fail("mesh", "missing vertices or triangles");
            }
            JPH::VertexList vertices;
            vertices.reserve(desc.points->size());
            for (const glm::vec3& point : *desc.points) vertices.push_back(JPH::Float3(point.x, point.y, point.z));
            JPH::IndexedTriangleList triangles;
            triangles.reserve(desc.indices->size() / 3);
            for (std::size_t i = 0; i + 2 < desc.indices->size(); i += 3) {
                triangles.push_back(JPH::IndexedTriangle((*desc.indices)[i], (*desc.indices)[i + 1],
                                                         (*desc.indices)[i + 2], 0));
            }
            return finish("mesh", JPH::MeshShapeSettings(std::move(vertices), std::move(triangles)));
        }
        case ShapeType::HeightField: {
            if (!desc.heights || desc.heightfield_samples < 2 ||
                desc.heights->size() !=
                    static_cast<std::size_t>(desc.heightfield_samples) * desc.heightfield_samples) {
                return fail("height-field", "sample count does not match the height array");
            }
            return finish("height-field",
                          JPH::HeightFieldShapeSettings(desc.heights->data(), JPH::Vec3::sZero(),
                                                       ToVec3(desc.heightfield_scale),
                                                       desc.heightfield_samples));
        }
    }
    return fail("unknown", "unrecognized shape type");
}

/// Wraps a cooked shape with the descriptor's scale and placement. Both
/// decorators are shapes in their own right, so the body's shape stays one
/// object.
JPH::RefConst<JPH::Shape> ApplyPlacement(const JPH::RefConst<JPH::Shape>& geometry,
                                        const ShapeDesc& desc) {
    JPH::RefConst<JPH::Shape> result = geometry;

    if (desc.scale != glm::vec3(1.0f)) {
        const JPH::ScaledShapeSettings settings(result, ToVec3(desc.scale));
        const auto scaled = settings.Create();
        if (!scaled.IsValid()) {
            LOGIFACE_LOG(error, std::format("physics: cannot scale shape: {}", scaled.GetError()));
            return {};
        }
        result = scaled.Get();
    }

    if (desc.position != glm::vec3(0.0f) || desc.rotation != glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
        const JPH::RotatedTranslatedShapeSettings settings(ToVec3(desc.position), ToQuat(desc.rotation),
                                                          result);
        const auto placed = settings.Create();
        if (!placed.IsValid()) {
            LOGIFACE_LOG(error, std::format("physics: cannot place shape: {}", placed.GetError()));
            return {};
        }
        result = placed.Get();
    }

    return result;
}

/// Cook-or-reuse the geometry shape for `desc` from the world's cache.
///
/// With `shape_cache_directory` set the cache is two-level: the in-memory map
/// serves a repeat of the same geometry within one world, and the disk blob —
/// keyed by (Jolt version, geometry hash) — serves the *next* world, which is
/// what turns a scene load from "cook every collider" into "restore every
/// collider". Everything about the disk half is best-effort: a hit that fails to
/// restore is a miss, and a miss that fails to save is still a cooked shape.
JPH::RefConst<JPH::Shape> AcquireGeometry(detail::WorldImpl& impl, const ShapeDesc& desc) {
    const std::size_t key = desc.GeometryHash();
    if (const auto it = impl.shape_cache.find(key); it != impl.shape_cache.end()) return it->second;

    const bool disk_cache_enabled = !impl.shape_cache_directory.empty();
    if (disk_cache_enabled) {
        if (JPH::RefConst<JPH::Shape> restored = LoadCookedShape(impl, key); restored != nullptr) {
            ++impl.shape_cache_disk_hits;
            LOGIFACE_LOG(debug, std::format("physics: restored cooked shape {} from disk", key));
            impl.shape_cache.emplace(key, restored);
            return restored;
        }
        ++impl.shape_cache_disk_misses;
    }

    JPH::RefConst<JPH::Shape> geometry = CookGeometry(desc);
    if (geometry == nullptr) return {};
    impl.shape_cache.emplace(key, geometry);
    if (disk_cache_enabled) SaveCookedShape(impl, *geometry, key);
    return geometry;
}

/// The one body-creation path: cook the shape set, build the settings, then
/// create the body with Jolt's allocator or with the caller's id. `CreateBody`
/// and `CreateBodyWithID` differ only in which Jolt entry point is used, so
/// deterministic replay cannot drift from ordinary creation.
[[nodiscard]] PhysicsBodyId CreateBodyChecked(detail::WorldImpl& impl,
                                              const PhysicsWorldConfig& config,
                                              const BodyDesc& desc,
                                              std::span<const ShapeDesc> children,
                                              PhysicsBodyId required_id) {
    // Cook-once, reuse-by-hash: a scene with a thousand identical crates cooks
    // one shape. Jolt ref-counts shapes, so a cached shape stays alive while any
    // body references it and eviction is unnecessary.
    JPH::RefConst<JPH::Shape> shape = ApplyPlacement(AcquireGeometry(impl, desc.shape), desc.shape);
    if (shape == nullptr) return {};

    if (!children.empty()) {
        // One body, several pieces: a static compound. Each child carries its own
        // placement, which is how a compound is assembled from the same cooked
        // child geometries as single-shape bodies.
        JPH::StaticCompoundShapeSettings compound;
        compound.AddShape(ToVec3(desc.shape.position), ToQuat(desc.shape.rotation), shape);
        for (const ShapeDesc& child : children) {
            JPH::RefConst<JPH::Shape> child_shape = ApplyPlacement(AcquireGeometry(impl, child), child);
            if (child_shape == nullptr) return {};
            compound.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), child_shape);
        }
        const JPH::ShapeSettings::ShapeResult result = compound.Create();
        if (!result.IsValid()) {
            LOGIFACE_LOG(error, std::format("physics: cannot cook compound shape: {}", result.GetError()));
            return {};
        }
        shape = result.Get();
    }

    JPH::BodyInterface& bodies = impl.physics->GetBodyInterface();
    JPH::BodyCreationSettings settings(shape.GetPtr(), ToRVec3(desc.position), ToQuat(desc.rotation),
                                       ToJoltMotion(desc.motion), ToJoltLayer(desc.layer));
    if (desc.mass > 0.0f) {
        // An explicit mass with inertia computed from the shape: a scene that
        // declares "10 kg" gets sane rotational behaviour without also declaring
        // an inertia tensor.
        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        settings.mMassPropertiesOverride.mMass = desc.mass;
    }
    settings.mFriction = desc.friction;
    settings.mRestitution = desc.restitution;
    settings.mLinearDamping = desc.linear_damping;
    settings.mAngularDamping = desc.angular_damping;
    settings.mIsSensor = desc.sensor;
    settings.mUserData = desc.user_data;
    if (desc.ccd) settings.mMotionQuality = JPH::EMotionQuality::LinearCast;

    JPH::Body* body = nullptr;
    if (required_id.IsValid()) {
        // `CreateBodyWithID` returns null when the id is already taken or its
        // index is past the cap — it never picks a different id, which is what
        // makes replay id-exact and a collision loud rather than silent.
        body = bodies.CreateBodyWithID(ToJoltBodyId(required_id), settings);
        if (body == nullptr) {
            LOGIFACE_LOG(warn, std::format(
                "physics: CreateBodyWithID rejected id {} (index {}, sequence {}): already in use "
                "or past the {} body cap",
                required_id.value, required_id.index(), required_id.sequence(), config.max_bodies));
            return {};
        }
    } else {
        body = bodies.CreateBody(settings);
        if (body == nullptr) {
            LOGIFACE_LOG(warn, std::format("physics: body limit reached ({}), body not created",
                                           config.max_bodies));
            return {};
        }
    }

    const JPH::BodyID id = body->GetID();
    bodies.AddBody(id, desc.motion == BodyMotionType::Static ? JPH::EActivation::DontActivate
                                                            : JPH::EActivation::Activate);
    return FromJoltBodyId(id);
}

// ── Snapshot framing ────────────────────────────────────────────────────────
//
// A `SaveState` payload is opaque to us, and Jolt's restore path reads it
// sequentially and applies as it goes: handed a truncated or foreign stream it
// can half-apply before returning false. The frame below is therefore the
// rejection gate — magic, a format version, the exact Jolt version that wrote
// the payload, the payload's length and an FNV-1a checksum over it. A stream
// that does not carry all of that, exactly, never reaches Jolt.
constexpr std::uint64_t kSnapshotMagic = 0x564B50534E415030ull; // "VKPSNAP0"
constexpr std::uint32_t kSnapshotFormatVersion = 1;
// `JPH_VERSION_ID` expands to an unqualified `uint64(...)` that only parses where
// Jolt's namespace is in scope, so the tag is composed from the plain version
// macros. It exists to reject a stream written by a different Jolt, whose binary
// format is explicitly unstable (design §4/§8); `RegisterTypes` covers ABI drift
// within one version.
constexpr std::uint32_t kJoltVersionTag =
    (JPH_VERSION_MAJOR * 10000u) + (JPH_VERSION_MINOR * 100u) + JPH_VERSION_PATCH;

struct SnapshotHeader {
    std::uint64_t magic = kSnapshotMagic;
    std::uint32_t format_version = kSnapshotFormatVersion;
    std::uint32_t jolt_version = kJoltVersionTag; // a foreign library's stream is rejected
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_checksum = 0; // FNV-1a over the payload
    std::uint64_t body_count = 0;       // diagnostics and structural sanity
    std::uint64_t body_set_checksum = 0; // order-independent; a different world is rejected
};
static_assert(sizeof(SnapshotHeader) == 48, "the snapshot header is a wire format");

} // namespace

// ── Body lifetime ───────────────────────────────────────────────────────────

PhysicsBodyId World::CreateBody(const BodyDesc& desc, std::span<const ShapeDesc> children) {
    if (!initialized_) return {};
    return CreateBodyChecked(*impl_, config_, desc, children, PhysicsBodyId{});
}

PhysicsBodyId World::CreateBodyWithID(const BodyDesc& desc, PhysicsBodyId required_id,
                                      std::span<const ShapeDesc> children) {
    if (!initialized_) return {};
    if (!required_id.IsValid()) {
        LOGIFACE_LOG(error, "physics: CreateBodyWithID requires a valid id");
        return {};
    }
    // A body created on replay must not keep a stale interpolation entry alive
    // under the same key: the id is the interpolation key, so a reused id starts
    // its tracking clean rather than blending from the previous occupant.
    impl_->render_bodies.erase(required_id.value);
    return CreateBodyChecked(*impl_, config_, desc, children, required_id);
}

void World::DestroyBody(PhysicsBodyId body) {
    if (!initialized_ || !body.IsValid()) return;
    JPH::BodyInterface& bodies = impl_->physics->GetBodyInterface();
    const JPH::BodyID id = ToJoltBodyId(body);
    if (!bodies.IsAdded(id)) return;
    // Constraints reference bodies; an attached extension (the constraints
    // registry) releases this body's joints before it leaves the system, so a
    // body can never be destroyed out from under a live constraint.
    if (extension_) extension_->OnBodyDestroyed(body);
    bodies.RemoveBody(id);
    bodies.DestroyBody(id);
    // A dead handle must not keep interpolating: the tracking maps are keyed by
    // the body's index+sequence, so nothing else can ever name this entry again.
    impl_->render_bodies.erase(body.value);
}

bool World::IsBodyValid(PhysicsBodyId body) const {
    if (!impl_ || !body.IsValid()) return false;
    return impl_->physics->GetBodyInterface().IsAdded(ToJoltBodyId(body));
}

// ── Body state ──────────────────────────────────────────────────────────────

BodyPose World::GetPose(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return {};
    JPH::RVec3 position;
    JPH::Quat rotation;
    impl_->physics->GetBodyInterface().GetPositionAndRotation(ToJoltBodyId(body), position, rotation);
    return BodyPose{FromRVec3(position), FromQuat(rotation)};
}

void World::SetPose(PhysicsBodyId body, const BodyPose& pose, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetPositionAndRotation(ToJoltBodyId(body), ToRVec3(pose.position),
                                                             ToQuat(pose.rotation), ToActivation(activate));
}

void World::MoveKinematic(PhysicsBodyId body, const BodyPose& pose, float delta_time) {
    if (!IsBodyValid(body) || delta_time <= 0.0f) return;
    impl_->physics->GetBodyInterface().MoveKinematic(ToJoltBodyId(body), ToRVec3(pose.position),
                                                    ToQuat(pose.rotation), delta_time);
}

glm::vec3 World::GetLinearVelocity(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return {};
    return FromVec3(impl_->physics->GetBodyInterface().GetLinearVelocity(ToJoltBodyId(body)));
}

glm::vec3 World::GetAngularVelocity(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return {};
    return FromVec3(impl_->physics->GetBodyInterface().GetAngularVelocity(ToJoltBodyId(body)));
}

void World::SetLinearVelocity(PhysicsBodyId body, glm::vec3 velocity, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetLinearVelocity(ToJoltBodyId(body), ToVec3(velocity));
    if (activate) impl_->physics->GetBodyInterface().ActivateBody(ToJoltBodyId(body));
}

void World::SetAngularVelocity(PhysicsBodyId body, glm::vec3 velocity, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetAngularVelocity(ToJoltBodyId(body), ToVec3(velocity));
    if (activate) impl_->physics->GetBodyInterface().ActivateBody(ToJoltBodyId(body));
}

void World::SetLinearAndAngularVelocity(PhysicsBodyId body, glm::vec3 linear, glm::vec3 angular,
                                       bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetLinearAndAngularVelocity(ToJoltBodyId(body), ToVec3(linear),
                                                                  ToVec3(angular));
    if (activate) impl_->physics->GetBodyInterface().ActivateBody(ToJoltBodyId(body));
}

void World::AddForce(PhysicsBodyId body, glm::vec3 force, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().AddForce(ToJoltBodyId(body), ToVec3(force),
                                               ToActivation(activate));
}

void World::AddTorque(PhysicsBodyId body, glm::vec3 torque, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().AddTorque(ToJoltBodyId(body), ToVec3(torque),
                                                ToActivation(activate));
}

void World::AddImpulse(PhysicsBodyId body, glm::vec3 impulse) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().AddImpulse(ToJoltBodyId(body), ToVec3(impulse));
}

void World::AddAngularImpulse(PhysicsBodyId body, glm::vec3 impulse) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().AddAngularImpulse(ToJoltBodyId(body), ToVec3(impulse));
}

void World::SetMotionType(PhysicsBodyId body, BodyMotionType motion, bool activate) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetMotionType(ToJoltBodyId(body), ToJoltMotion(motion),
                                                    ToActivation(activate));
}

BodyMotionType World::GetMotionType(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return BodyMotionType::Static;
    return FromJoltMotion(impl_->physics->GetBodyInterface().GetMotionType(ToJoltBodyId(body)));
}

bool World::IsActive(PhysicsBodyId body) const {
    return IsBodyValid(body) && impl_->physics->GetBodyInterface().IsActive(ToJoltBodyId(body));
}

void World::Activate(PhysicsBodyId body) {
    if (IsBodyValid(body)) impl_->physics->GetBodyInterface().ActivateBody(ToJoltBodyId(body));
}

void World::SetUserData(PhysicsBodyId body, std::uint64_t user_data) {
    if (IsBodyValid(body)) impl_->physics->GetBodyInterface().SetUserData(ToJoltBodyId(body), user_data);
}

std::uint64_t World::GetUserData(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return 0;
    return impl_->physics->GetBodyInterface().GetUserData(ToJoltBodyId(body));
}

// ── Body property tuning ────────────────────────────────────────────────────

void World::SetBodyMass(PhysicsBodyId body, float mass) {
    if (!IsBodyValid(body)) return;
    if (!(mass > 0.0f)) {
        LOGIFACE_LOG(error, "physics: SetBodyMass with a non-positive mass; ignored");
        return;
    }
    // Mass lives on the body's motion properties, which the body interface does
    // not expose — take the body lock (the same locking discipline `Step`
    // honors for everything else) and scale mass and inertia together, so the
    // body keeps a consistent response to forces.
    JPH::BodyLockWrite lock(impl_->physics->GetBodyLockInterface(), ToJoltBodyId(body));
    if (!lock.Succeeded()) return; // removed concurrently; nothing to tune
    JPH::MotionProperties* motion = lock.GetBody().GetMotionPropertiesUnchecked();
    if (motion == nullptr) {
        LOGIFACE_LOG(warn, "physics: SetBodyMass on a static body has no motion properties; ignored");
        return;
    }
    motion->ScaleToMass(mass);
}

float World::GetBodyMass(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return 0.0f;
    const JPH::BodyLockRead lock(impl_->physics->GetBodyLockInterface(), ToJoltBodyId(body));
    if (!lock.Succeeded()) return 0.0f;
    const JPH::MotionProperties* motion = lock.GetBody().GetMotionPropertiesUnchecked();
    if (motion == nullptr) return 0.0f;
    const float inverse = motion->GetInverseMassUnchecked();
    return inverse > 0.0f ? 1.0f / inverse : 0.0f;
}

void World::SetBodyFriction(PhysicsBodyId body, float friction) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetFriction(ToJoltBodyId(body), friction);
}

float World::GetBodyFriction(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return 0.0f;
    return impl_->physics->GetBodyInterface().GetFriction(ToJoltBodyId(body));
}

void World::SetBodyRestitution(PhysicsBodyId body, float restitution) {
    if (!IsBodyValid(body)) return;
    impl_->physics->GetBodyInterface().SetRestitution(ToJoltBodyId(body), restitution);
}

float World::GetBodyRestitution(PhysicsBodyId body) const {
    if (!IsBodyValid(body)) return 0.0f;
    return impl_->physics->GetBodyInterface().GetRestitution(ToJoltBodyId(body));
}

std::size_t World::CachedShapeCount() const noexcept {
    return impl_ ? impl_->shape_cache.size() : 0;
}

std::uint64_t World::ShapeCacheDiskHits() const noexcept {
    return impl_ ? impl_->shape_cache_disk_hits : 0;
}

std::uint64_t World::ShapeCacheDiskMisses() const noexcept {
    return impl_ ? impl_->shape_cache_disk_misses : 0;
}

std::vector<PhysicsBodyId> World::BodyIds() const {
    std::vector<PhysicsBodyId> ids;
    if (!impl_ || !impl_->physics) return ids;
    JPH::BodyIDVector jolt_ids;
    impl_->physics->GetBodies(jolt_ids);
    ids.reserve(jolt_ids.size());
    for (const JPH::BodyID& id : jolt_ids) ids.push_back(FromJoltBodyId(id));
    return ids;
}

// ── Save / replay / rollback ────────────────────────────────────────────────

std::vector<std::byte> World::SaveSnapshot() const {
    std::vector<std::byte> snapshot;
    if (!impl_ || !impl_->physics) return snapshot;

    // `EStateRecorderState::All` — see the interface comment: contacts and
    // constraints carry warm-start impulses the next step's solver consumes, so
    // bodies alone would not reproduce a continuation. No filter: a partial or
    // filtered record could not be validated by Jolt either.
    JPH::StateRecorderImpl recorder;
    impl_->physics->SaveState(recorder, JPH::EStateRecorderState::All);

    const std::string payload = recorder.GetData();
    SnapshotHeader header;
    header.jolt_version = kJoltVersionTag;
    header.payload_bytes = payload.size();
    header.body_count = impl_->physics->GetNumBodies();
    header.body_set_checksum = BodySetChecksum(BodyIds());
    header.payload_checksum = HashBytes(kFnv1aOffset,
                                        std::as_bytes(std::span{payload.data(), payload.size()}));

    snapshot.resize(sizeof(SnapshotHeader) + payload.size());
    std::memcpy(snapshot.data(), &header, sizeof(header));
    std::memcpy(snapshot.data() + sizeof(header), payload.data(), payload.size());
    return snapshot;
}

bool World::RestoreSnapshot(std::span<const std::byte> snapshot) {
    if (!impl_ || !impl_->physics) {
        LOGIFACE_LOG(error, "physics: RestoreSnapshot on an uninitialized world");
        return false;
    }

    // Reject before a single byte reaches Jolt. Jolt's restore reads the stream
    // sequentially and applies as it goes, so this frame check is what makes the
    // operation all-or-nothing for every stream that is not exactly ours.
    if (snapshot.size() < sizeof(SnapshotHeader)) {
        LOGIFACE_LOG(error, std::format("physics: snapshot rejected: {} bytes is smaller than the "
                                        "{} byte header",
                                        snapshot.size(), sizeof(SnapshotHeader)));
        return false;
    }
    SnapshotHeader header;
    std::memcpy(&header, snapshot.data(), sizeof(header));
    if (header.magic != kSnapshotMagic) {
        LOGIFACE_LOG(error, "physics: snapshot rejected: not an engine snapshot (bad magic)");
        return false;
    }
    if (header.format_version != kSnapshotFormatVersion) {
        LOGIFACE_LOG(error, std::format("physics: snapshot rejected: format version {} (this build "
                                        "writes {})",
                                        header.format_version, kSnapshotFormatVersion));
        return false;
    }
    if (header.jolt_version != kJoltVersionTag) {
        LOGIFACE_LOG(error, std::format("physics: snapshot rejected: written by Jolt {} (this build "
                                        "is {})",
                                        header.jolt_version, kJoltVersionTag));
        return false;
    }
    if (snapshot.size() != sizeof(SnapshotHeader) + header.payload_bytes) {
        LOGIFACE_LOG(error, std::format("physics: snapshot rejected: frame says {} payload bytes, "
                                        "stream has {}",
                                        header.payload_bytes,
                                        snapshot.size() - sizeof(SnapshotHeader)));
        return false;
    }
    const auto payload = snapshot.subspan(sizeof(SnapshotHeader));
    if (HashBytes(kFnv1aOffset, payload) != header.payload_checksum) {
        LOGIFACE_LOG(error, "physics: snapshot rejected: payload checksum mismatch");
        return false;
    }

    // Jolt's restore matches bodies by id and applies state as it reads, so a
    // stream recorded from a different body set would misalign *inside* the
    // read and corrupt the world. Checking the set first is what keeps the
    // restore all-or-nothing for that case too.
    if (BodySetChecksum(BodyIds()) != header.body_set_checksum) {
        LOGIFACE_LOG(error, std::format("physics: snapshot rejected: the world's body set differs "
                                        "from the one the snapshot was taken from ({} vs {} bodies)",
                                        BodyCount(), header.body_count));
        return false;
    }

    // The frame is intact, so Jolt rejecting the payload is a bug (version
    // skew inside the frame, or a body set that changed under us). Keep the
    // pre-restore state so a rejection can never leave a half-applied world:
    // the stream we save here is one we just produced, so restoring it is the
    // same operation in reverse.
    std::vector<std::byte> rollback = SaveSnapshot();
    JPH::StateRecorderImpl recorder;
    recorder.WriteBytes(payload.data(), payload.size());
    recorder.Rewind();
    if (!impl_->physics->RestoreState(recorder)) {
        LOGIFACE_LOG(error, "physics: snapshot rejected by Jolt; rolling the world back");
        // Re-run the frame checks are unnecessary (we produced it), and a failure
        // here would mean the world is beyond saving; report it loudly.
        if (!RestoreSnapshot(rollback)) {
            LOGIFACE_LOG(error, "physics: rollback after a rejected snapshot also failed");
        }
        return false;
    }

    // Interpolation state is not part of the solver state and is keyed by id, so
    // a restored body resumes from its restored pose on the next step rather than
    // blending from a pose that no longer exists.
    impl_->render_bodies.clear();
    impl_->render_characters.clear();
    return true;
}

std::uint64_t World::SnapshotHash(const std::vector<std::byte>& snapshot) noexcept {
    return HashBytes(kFnv1aOffset, snapshot);
}

// ── Queries ─────────────────────────────────────────────────────────────────

std::optional<RayHit> World::CastRay(const RayCastRequest& request) const {
    if (!impl_ || !impl_->physics) return std::nullopt;

    const glm::vec3 delta = request.to - request.from;
    if (delta == glm::vec3(0.0f)) return std::nullopt;

    // Jolt takes the direction with its length, so the segment *is* the ray.
    const JPH::RRayCast ray{ToRVec3(request.from), ToVec3(delta)};
    JPH::RayCastResult result;
    const LayerMaskFilter layer_filter(request.layer_mask);
    const MotionTypeFilter body_filter(request.hit_static, request.hit_kinematic, request.hit_dynamic,
                                       request.hit_sensors);
    const AllBroadPhaseLayers broad_phase_filter;

    if (!impl_->physics->GetNarrowPhaseQuery().CastRay(ray, result, broad_phase_filter, layer_filter,
                                                      body_filter)) {
        return std::nullopt;
    }

    RayHit hit{};
    hit.body = FromJoltBodyId(result.mBodyID);
    hit.fraction = result.mFraction;
    hit.sub_shape = result.mSubShapeID2.GetValue();
    hit.position = request.from + delta * result.mFraction;

    // The narrow-phase query has released the body lock by now, so the surface
    // normal takes a fresh read lock; that is also why it is not part of the
    // collector.
    const JPH::BodyLockRead lock(impl_->physics->GetBodyLockInterface(), result.mBodyID);
    if (lock.Succeeded()) {
        hit.normal = FromVec3(
            lock.GetBody().GetWorldSpaceSurfaceNormal(result.mSubShapeID2, ToRVec3(hit.position)));
    }
    return hit;
}

std::optional<RayHit> World::CastShape(const ShapeCastRequest& request) {
    if (!impl_ || !impl_->physics) return std::nullopt;

    JPH::RefConst<JPH::Shape> shape = AcquireGeometry(*impl_, request.shape);
    if (shape == nullptr) return std::nullopt;
    if (request.shape.scale != glm::vec3(1.0f)) {
        // A swept scaled shape is a scaled shape swept along the direction.
        const JPH::ScaledShapeSettings scaled(shape, ToVec3(request.shape.scale));
        const auto result = scaled.Create();
        if (!result.IsValid()) {
            LOGIFACE_LOG(error, std::format("physics: cannot scale swept shape: {}", result.GetError()));
            return std::nullopt;
        }
        shape = result.Get();
    }

    const JPH::RMat44 start = JPH::RMat44::sRotationTranslation(ToQuat(request.rotation),
                                                               ToRVec3(request.from));
    const JPH::RShapeCast cast = JPH::RShapeCast::sFromWorldTransform(
        shape.GetPtr(), JPH::Vec3::sReplicate(1.0f), start, ToVec3(request.to - request.from));
    JPH::ShapeCastSettings settings;
    JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;

    const LayerMaskFilter layer_filter(request.layer_mask);
    const MotionTypeFilter body_filter(request.hit_static, request.hit_kinematic, request.hit_dynamic,
                                       request.hit_sensors);
    const AllBroadPhaseLayers broad_phase_filter;
    impl_->physics->GetNarrowPhaseQuery().CastShape(cast, settings, JPH::RVec3::sZero(), collector,
                                                    broad_phase_filter, layer_filter, body_filter);
    if (!collector.HadHit()) return std::nullopt;

    const JPH::ShapeCastResult& result = collector.mHit;
    RayHit hit{};
    hit.body = FromJoltBodyId(result.mBodyID2);
    hit.fraction = result.mFraction;
    hit.sub_shape = result.mSubShapeID2.GetValue();
    hit.position = request.from + (request.to - request.from) * result.mFraction;
    // The penetration axis points out of body 2, so the surface normal faces back
    // along the sweep.
    hit.normal = -FromVec3(result.mPenetrationAxis.Normalized());
    return hit;
}

// ── Debug visualization ─────────────────────────────────────────────────────

void World::CollectDebugGeometry(const PhysicsDebugSettings& settings, PhysicsDebugGeometry& out) {
    if (!impl_ || !impl_->physics) return;

    if (impl_->debug_renderer == nullptr) {
        impl_->debug_renderer = std::make_unique<EngineDebugRenderer>(impl_->debug_geometry);
    }
    impl_->debug_geometry.Clear();

    if (settings.draw_shapes || settings.draw_bounding_boxes || settings.draw_centers_of_mass) {
        JPH::BodyManager::DrawSettings draw{};
        draw.mDrawShape = settings.draw_shapes;
        draw.mDrawShapeWireframe = settings.wireframe;
        draw.mDrawBoundingBox = settings.draw_bounding_boxes;
        draw.mDrawCenterOfMassTransform = settings.draw_centers_of_mass;

        // `layer_mask == 0` means "every layer", so no filter is installed.
        std::optional<LayerBodyFilter> filter;
        if (settings.layer_mask != 0) filter.emplace(settings.layer_mask);
        impl_->physics->DrawBodies(draw, impl_->debug_renderer.get(),
                                   filter.has_value() ? &*filter : nullptr);
    }
    if (settings.draw_constraints) {
        impl_->physics->DrawConstraints(impl_->debug_renderer.get());
    }
    if (settings.draw_constraint_limits) {
        impl_->physics->DrawConstraintLimits(impl_->debug_renderer.get());
    }

    out.lines.insert(out.lines.end(), impl_->debug_geometry.lines.begin(),
                     impl_->debug_geometry.lines.end());
    out.triangles.insert(out.triangles.end(), impl_->debug_geometry.triangles.begin(),
                         impl_->debug_geometry.triangles.end());
}

// ── Characters ──────────────────────────────────────────────────────────────

CharacterId World::CreateCharacter(const CharacterDesc& desc, const BodyPose& pose,
                                   std::uint64_t user_data) {
    if (!initialized_) return {};

    JPH::RefConst<JPH::Shape> shape = ApplyPlacement(AcquireGeometry(*impl_, desc.shape), desc.shape);
    if (shape == nullptr) return {};

    JPH::CharacterVirtualSettings settings;
    settings.mShape = shape;
    settings.mUp = JPH::Vec3::sAxisY();
    settings.mMaxSlopeAngle = glm::radians(desc.max_slope_angle_degrees);
    settings.mMass = desc.mass;
    settings.mMaxStrength = desc.max_strength;
    settings.mCharacterPadding = desc.character_padding;
    settings.mInnerBodyLayer = ToJoltLayer(desc.layer);
    if (desc.shape.radius > 0.0f) {
        // The supporting volume is the plane below which a contact counts as
        // ground: a disc at the bottom of the shape. Without it the character
        // treats every side contact as supportable.
        settings.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -desc.shape.radius);
    }

    auto slot = std::make_unique<detail::WorldImpl::CharacterSlot>();
    slot->desc = desc;
    slot->generation = impl_->next_character_generation++;
    slot->character = std::make_unique<JPH::CharacterVirtual>(
        &settings, ToRVec3(pose.position), ToQuat(pose.rotation), user_data, impl_->physics.get());

    std::uint32_t index = 0;
    if (!impl_->character_free_slots.empty()) {
        index = impl_->character_free_slots.back();
        impl_->character_free_slots.pop_back();
        impl_->character_slots[index] = std::move(slot);
    } else {
        index = static_cast<std::uint32_t>(impl_->character_slots.size());
        impl_->character_slots.push_back(std::move(slot));
    }
    return CharacterId{index, impl_->character_slots[index]->generation};
}

void World::DestroyCharacter(CharacterId character) {
    if (!IsCharacterValid(character)) return;
    impl_->character_slots[character.slot]->character.reset();
    impl_->character_slots[character.slot]->generation = 0;
    impl_->character_free_slots.push_back(character.slot);
    impl_->render_characters.erase(character.slot);
}

bool World::IsCharacterValid(CharacterId character) const {
    if (!impl_ || !character.IsValid() || character.slot >= impl_->character_slots.size()) {
        return false;
    }
    const detail::WorldImpl::CharacterSlot* slot = impl_->character_slots[character.slot].get();
    return slot != nullptr && slot->character != nullptr && slot->generation == character.generation;
}

void World::StepCharacter(CharacterId character, float delta_time) {
    if (!IsCharacterValid(character) || delta_time <= 0.0f) return;
    detail::WorldImpl::CharacterSlot& slot = *impl_->character_slots[character.slot];

    JPH::CharacterVirtual::ExtendedUpdateSettings update{};
    update.mStickToFloorStepDown = JPH::Vec3(0.0f, -slot.desc.stick_to_floor_step_down, 0.0f);
    update.mWalkStairsStepUp = JPH::Vec3(0.0f, slot.desc.walk_stairs_step_up, 0.0f);
    update.mWalkStairsMinStepForward = slot.desc.walk_stairs_min_step_forward;
    update.mWalkStairsStepForwardTest = slot.desc.walk_stairs_step_forward_test;

    // The character's sweeps honour the same layer table the solver does, so
    // "walk through a wall but not through a character-only barrier" is one policy.
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(CollisionLayer::Count); ++i) {
        if (LayersCollide(slot.desc.layer, static_cast<CollisionLayer>(i))) mask |= 1u << i;
    }
    const LayerMaskFilter layer_filter(mask);
    const AllBroadPhaseLayers broad_phase_filter;
    // Sensors are triggers, not walls: a character never stands on one.
    const MotionTypeFilter body_filter(/*static=*/true, /*kinematic=*/true, /*dynamic=*/true,
                                       /*sensors=*/false);
    const JPH::ShapeFilter shape_filter;

    slot.character->ExtendedUpdate(delta_time, ToVec3(config_.gravity), update, broad_phase_filter,
                                   layer_filter, body_filter, shape_filter,
                                   *impl_->temp_allocator);
}

BodyPose World::GetCharacterPose(CharacterId character) const {
    if (!IsCharacterValid(character)) return {};
    const JPH::CharacterVirtual& value = *impl_->character_slots[character.slot]->character;
    return BodyPose{FromRVec3(value.GetPosition()), FromQuat(value.GetRotation())};
}

void World::SetCharacterPose(CharacterId character, const BodyPose& pose) {
    if (!IsCharacterValid(character)) return;
    impl_->character_slots[character.slot]->character->SetPosition(ToRVec3(pose.position));
    impl_->character_slots[character.slot]->character->SetRotation(ToQuat(pose.rotation));
}

glm::vec3 World::GetCharacterVelocity(CharacterId character) const {
    if (!IsCharacterValid(character)) return {};
    return FromVec3(impl_->character_slots[character.slot]->character->GetLinearVelocity());
}

void World::SetCharacterVelocity(CharacterId character, glm::vec3 velocity) {
    if (!IsCharacterValid(character)) return;
    impl_->character_slots[character.slot]->character->SetLinearVelocity(ToVec3(velocity));
}

bool World::IsCharacterOnGround(CharacterId character) const {
    if (!IsCharacterValid(character)) return false;
    return impl_->character_slots[character.slot]->character->GetGroundState() ==
           JPH::CharacterVirtual::EGroundState::OnGround;
}

glm::vec3 World::GetCharacterGroundVelocity(CharacterId character) const {
    if (!IsCharacterValid(character)) return {};
    return FromVec3(impl_->character_slots[character.slot]->character->GetGroundVelocity());
}

std::uint32_t World::CharacterCount() const noexcept {
    if (!impl_) return 0;
    std::uint32_t count = 0;
    for (const auto& slot : impl_->character_slots) {
        if (slot != nullptr && slot->character != nullptr) ++count;
    }
    return count;
}

// ── Render interpolation ─────────────────────────────────────────────────────

namespace {

/// The character's live pose, read straight from its slot. `CaptureRenderPoses`
/// goes through this rather than through a handle: the tracking map is keyed by
/// slot, and a slot that is present in it is known to be live.
[[nodiscard]] std::optional<BodyPose> CharacterSlotPose(const detail::WorldImpl& impl,
                                                        std::uint32_t slot) {
    if (slot >= impl.character_slots.size()) return std::nullopt;
    const detail::WorldImpl::CharacterSlot* entry = impl.character_slots[slot].get();
    if (entry == nullptr || entry->character == nullptr) return std::nullopt;
    return BodyPose{FromRVec3(entry->character->GetPosition()),
                    FromQuat(entry->character->GetRotation())};
}

} // namespace

void World::TrackRenderPose(PhysicsBodyId body) {
    if (!impl_ || !IsBodyValid(body)) return;
    // Both poses start at the live pose: a body begins being interpolated *at*
    // where it is, so tracking never introduces a jump of its own. The next
    // `Step` rotates that into `prev` and captures the first new pose.
    const BodyPose pose = GetPose(body);
    impl_->render_bodies[body.value] = detail::WorldImpl::RenderPose{pose, pose};
}

void World::UntrackRenderPose(PhysicsBodyId body) {
    if (impl_) impl_->render_bodies.erase(body.value);
}

bool World::IsRenderPoseTracked(PhysicsBodyId body) const noexcept {
    return impl_ != nullptr && impl_->render_bodies.contains(body.value);
}

BodyPose World::GetRenderPose(PhysicsBodyId body, float alpha) const {
    if (!impl_) return {};
    const auto it = impl_->render_bodies.find(body.value);
    // Untracked (or dead): there is no previous pose to blend from, so the live
    // pose is the answer. `GetPose` already resolves a dead handle to the
    // identity, which is what the caller can draw.
    if (it == impl_->render_bodies.end()) return GetPose(body);
    return BlendPoses(it->second.prev, it->second.cur, std::clamp(alpha, 0.0f, 1.0f));
}

void World::TrackCharacterRenderPose(CharacterId character) {
    if (!impl_ || !IsCharacterValid(character)) return;
    const BodyPose pose = GetCharacterPose(character);
    impl_->render_characters[character.slot] = detail::WorldImpl::RenderPose{pose, pose};
}

void World::UntrackCharacterRenderPose(CharacterId character) {
    // Gated on validity, unlike the body version: a character's key is its slot
    // alone (there is no sequence in it), so an invalid handle would otherwise
    // reach into whichever character now occupies that slot. A body's key carries
    // its index *and* sequence, so its erase is safe unconditionally.
    if (!impl_ || !IsCharacterValid(character)) return;
    impl_->render_characters.erase(character.slot);
}

bool World::IsCharacterRenderPoseTracked(CharacterId character) const noexcept {
    // The validity check matters: the tracking map is keyed by slot, and a stale
    // or default handle must not read the state of whichever character now lives
    // in "its" slot.
    return impl_ != nullptr && IsCharacterValid(character) &&
           impl_->render_characters.contains(character.slot);
}

BodyPose World::GetCharacterRenderPose(CharacterId character, float alpha) const {
    if (!impl_ || !IsCharacterValid(character)) return GetCharacterPose(character);
    const auto it = impl_->render_characters.find(character.slot);
    if (it == impl_->render_characters.end()) return GetCharacterPose(character);
    return BlendPoses(it->second.prev, it->second.cur, std::clamp(alpha, 0.0f, 1.0f));
}

void World::RotateRenderPoses() {
    // `prev <- cur` for the tracked set only. A body that nothing renders has no
    // entry here, so this loop is proportional to what is drawn, not to the
    // world's body count.
    for (auto& [key, pose] : impl_->render_bodies) {
        (void)key;
        pose.prev = pose.cur;
    }
    for (auto& [slot, pose] : impl_->render_characters) {
        (void)slot;
        pose.prev = pose.cur;
    }
}

void World::CaptureRenderPoses() {
    // `cur <- live` after the update, for bodies and characters alike. Characters
    // were swept earlier in the frame (`PhysicsInput` runs before `Step`), but
    // they are captured here anyway so one rotation point governs both and the
    // blend after a step always runs this step's poses.
    for (auto& [key, pose] : impl_->render_bodies) {
        pose.cur = GetPose(PhysicsBodyId{key});
    }
    for (auto& [slot, pose] : impl_->render_characters) {
        if (const std::optional<BodyPose> live = CharacterSlotPose(*impl_, slot)) pose.cur = *live;
    }
}

} // namespace VulkanEngine::Physics
