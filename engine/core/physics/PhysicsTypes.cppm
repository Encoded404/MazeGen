// Engine-owned physics vocabulary.
//
// Units and conventions are the engine's already: meters, seconds, kilograms,
// right-handed, Y-up, radians. Jolt's defaults agree, so no conversion layer is
// needed — but this header states the contract so a future change has one place
// to be wrong.
//
// Nothing here names a Jolt type. `PhysicsBodyId` is the one boundary value
// Jolt owns (its `BodyID` is a packed index+sequence pair); it is wrapped as a
// POD of the same shape so the mapping is free while Jolt stays confined to
// implementation files.

module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Physics.Types;

import std;

export namespace VulkanEngine::Physics {

/// Object-layer count. Jolt's `ObjectLayer` is 16 bits by default; the collision
/// policy in `PhysicsWorld` decides which pairs may collide, as a table rather
/// than as code.
inline constexpr std::uint16_t kObjectLayerBits = 16;

/// Opaque, generation-checked handle to a physics body.
///
/// Carries Jolt's `BodyID` encoding verbatim — a 23-bit body index with an 8-bit
/// sequence number above it — so the mapping is exact and lossless, while no Jolt
/// type crosses the facade. `value == 0` is the "no body" handle: Jolt's sequence
/// numbers pre-increment from zero, so the first body ever created at index 0
/// already encodes as non-zero, and no live body can collide with the sentinel.
///
/// Note that body index 0 is a *valid* body in Jolt — only `0xffffffff` is
/// invalid — so validity is not "index != 0". `sequence()` is what distinguishes a
/// reused index from the body that previously occupied it.
struct PhysicsBodyId {
    std::uint32_t value = 0; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] constexpr bool IsValid() const noexcept { return value != 0; }
    /// Body array index (23 bits).
    [[nodiscard]] constexpr std::uint32_t index() const noexcept { return value & 0x7fffffu; }
    /// Reuse counter (8 bits), incremented every time this index is reallocated.
    [[nodiscard]] constexpr std::uint32_t sequence() const noexcept { return value >> 23; }

    friend constexpr bool operator==(const PhysicsBodyId&, const PhysicsBodyId&) = default;
};

/// How a body is allowed to move. Static bodies never move and are cheapest to
/// keep asleep; kinematic bodies are moved by gameplay but push dynamics;
/// dynamic bodies are moved by the solver.
enum class BodyMotionType : std::uint8_t {
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
};

/// Collision policy identity. The table in `PhysicsWorld` decides which of these
/// may collide; the enumerator is what components and queries carry.
enum class CollisionLayer : std::uint16_t {
    Default = 0,    // ordinary geometry
    Sensor = 1,     // trigger volumes: reported by queries, never resolved
    Debris = 2,
    Character = 3,
    Projectile = 4,
    Count = 5,
};

/// Diagnostic spelling of a layer (failure messages and debug UI).
[[nodiscard]] constexpr std::string_view CollisionLayerName(CollisionLayer layer) noexcept {
    switch (layer) {
        case CollisionLayer::Default:    return "default";
        case CollisionLayer::Sensor:     return "sensor";
        case CollisionLayer::Debris:     return "debris";
        case CollisionLayer::Character:  return "character";
        case CollisionLayer::Projectile: return "projectile";
        case CollisionLayer::Count:      break;
    }
    return "<unknown>";
}

/// Shape kinds a `ShapeDesc` can describe.
enum class ShapeType : std::uint8_t {
    Box = 0,
    Sphere = 1,
    Capsule = 2,     // along local Y
    Cylinder = 3,    // along local Y
    ConvexHull = 4,  // from `points`
    Mesh = 5,        // triangle mesh from `points` + `indices`; static only
    HeightField = 6, // regular grid from `heights`
};

[[nodiscard]] constexpr std::string_view ShapeTypeName(ShapeType type) noexcept {
    switch (type) {
        case ShapeType::Box:         return "box";
        case ShapeType::Sphere:      return "sphere";
        case ShapeType::Capsule:     return "capsule";
        case ShapeType::Cylinder:    return "cylinder";
        case ShapeType::ConvexHull:  return "convex-hull";
        case ShapeType::Mesh:        return "mesh";
        case ShapeType::HeightField: return "height-field";
    }
    return "<unknown>";
}

/// A shape as *data*, not as a cooked object.
///
/// This POD is the primary data (it is what a scene file or an asset carries);
/// the cooked Jolt shape is a cache keyed by this descriptor's hash plus the Jolt
/// version. Jolt's shape binary format is explicitly unstable across versions, so
/// a cooked blob may never be the primary representation.
struct ShapeDesc {
    ShapeType type = ShapeType::Box;

    // Box: full extents along each axis.
    glm::vec3 box_extent{1.0f, 1.0f, 1.0f};
    // Sphere/cylinder/capsule: radius; capsule/cylinder height is the distance
    // between the hemisphere centers (capsule) or the full cylinder height.
    float radius = 0.5f;
    float height = 1.0f;
    // Convex hull: the shrink applied when building the hull from its points.
    float hull_shrink = 0.02f;

    // ConvexHull / Mesh geometry. Shared, because several entities typically
    // describe the same mesh, and the shape cache deduplicates on content.
    std::shared_ptr<const std::vector<glm::vec3>> points;
    std::shared_ptr<const std::vector<std::uint32_t>> indices;

    // HeightField: an `heightfield_samples` x `heightfield_samples` grid of
    // heights, spaced by scale. Jolt's height field is square.
    std::uint32_t heightfield_samples = 0;
    std::shared_ptr<const std::vector<float>> heights;
    glm::vec3 heightfield_scale{1.0f, 1.0f, 1.0f};

    // Local placement of this shape inside the body frame. A `RigidBody`'s own
    // shape uses the identity; compound children and `Collider` shapes offset.
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    friend bool operator==(const ShapeDesc&, const ShapeDesc&) = default;

    /// Geometry-only hash: the shape-cache key.
    ///
    /// Placement (`position`/`rotation`/`scale`) is deliberately excluded: it is
    /// applied by the body as a decorator shape, so a hundred boxes of one size
    /// at a hundred positions cook one shape, not a hundred. Hashes the geometry
    /// *contents*, not the pointers, so two descriptors that describe the same
    /// mesh agree.
    [[nodiscard]] std::size_t GeometryHash() const noexcept {
        std::size_t seed = 0xcbf29ce484222325ULL; // FNV-1a offset basis
        const auto mix = [&seed](const auto& value) {
            const auto bytes = std::as_bytes(std::span{&value, 1});
            for (const std::byte byte : bytes) {
                seed ^= static_cast<std::size_t>(std::to_integer<std::uint8_t>(byte));
                seed *= 0x100000001b3ULL;
            }
        };
        mix(type);
        mix(box_extent);
        mix(radius);
        mix(height);
        mix(hull_shrink);
        mix(heightfield_samples);
        mix(heightfield_scale);
        if (points) {
            mix(points->size());
            for (const glm::vec3& point : *points) mix(point);
        }
        if (indices) {
            mix(indices->size());
            for (const std::uint32_t index : *indices) mix(index);
        }
        if (heights) {
            mix(heights->size());
            for (const float sample : *heights) mix(sample);
        }
        return seed;
    }

    /// Full content hash: geometry plus placement. This is the identity of the
    /// *body shape*, and (combined with the Jolt version) the cooked-blob cache
    /// key for a scene's primary shape data.
    [[nodiscard]] std::size_t Hash() const noexcept {
        std::size_t seed = GeometryHash();
        const auto mix = [&seed](const auto& value) {
            const auto bytes = std::as_bytes(std::span{&value, 1});
            for (const std::byte byte : bytes) {
                seed ^= static_cast<std::size_t>(std::to_integer<std::uint8_t>(byte));
                seed *= 0x100000001b3ULL;
            }
        };
        mix(position);
        mix(rotation);
        mix(scale);
        return seed;
    }

    /// True when this shape is placed somewhere other than the body origin.
    [[nodiscard]] bool HasPlacement() const noexcept {
        return position != glm::vec3(0.0f) || rotation != glm::quat(1.0f, 0.0f, 0.0f, 0.0f) ||
               scale != glm::vec3(1.0f);
    }
};

/// Surface properties. A table entry rather than per-body code: Jolt combines the
/// two bodies' values, so a "slippery ice" material is a row, not a special case.
struct PhysicsMaterial {
    float friction = 0.5f; // NOLINT(misc-non-private-member-variables-in-classes)
    float restitution = 0.0f; // NOLINT(misc-non-private-member-variables-in-classes)
};

/// Result of a ray cast through the facade.
struct RayHit {
    PhysicsBodyId body{};
    float fraction = 1.0f; // distance along the ray, in [0, 1]
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    /// Which child shape was hit, for compound shapes (a sub-entity mapping).
    std::uint32_t sub_shape = 0;
};

} // namespace VulkanEngine::Physics
