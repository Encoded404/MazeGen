module;

#include <vulkan/vulkan_core.h>

export module VulkanEngine.TextureTypes;

import std;

import vulkan;

export namespace VulkanEngine::Textures {

// ── Requirement-level enums ───────────────────────────────────────────

enum class TextureSemantic : std::uint8_t {
    BaseColor,
    Emissive,
    Normal,
    ORM,
    Mask,
    UI,
    HDR,
    Unknown,
};

// Declared at import/authoring time; NOT derived from the GPU format.
// Determines both the shader decode path and which upload formats are legal.
enum class TextureNormalEncoding : std::uint8_t {
    Standard = 0, // unit tangent-space normal; reconstruct z = sqrt(1 - x^2 - y^2) from RG.
    Full     = 1, // authored RGB normal, possibly non-unit or z < 0. Use RGB as-is.
    Bent     = 2, // stylized/bent direction; use RGB as-is and apply normal_scale to xy.
};

// Per-texture-slot KHR_texture_transform; stored here so
// the type exists once (a consumer reads it once UV authoring carries it).
struct TextureTransform {
    float offset_u{0.0f}, offset_v{0.0f};
    float scale_u{1.0f}, scale_v{1.0f};
    float rotation{0.0f};
};

// CPU-side sampler description, resolved to a VkSampler by the owning layer.
// No swizzle here: a component mapping is a view property (TextureData.swizzle
// applied through the image view), never a sampler property. Clamp to device
// limits before hashing/creating.
struct SamplerDesc {
    vk::Filter min_filter{vk::Filter::eLinear}, mag_filter{vk::Filter::eLinear};
    vk::SamplerMipmapMode mip_mode{vk::SamplerMipmapMode::eLinear};
    vk::SamplerAddressMode address_u{vk::SamplerAddressMode::eRepeat};
    vk::SamplerAddressMode address_v{vk::SamplerAddressMode::eRepeat};
    vk::SamplerAddressMode address_w{vk::SamplerAddressMode::eRepeat};
    float mip_lod_bias{0.0f};
    float min_lod{0.0f};
    float max_lod{1000.0f}; // VK_LOD_CLAMP_NONE
    bool anisotropy{false};
    std::uint32_t max_anisotropy{1};
    vk::CompareOp compare{vk::CompareOp::eNever};
    vk::BorderColor border{vk::BorderColor::eFloatOpaqueBlack};

    bool operator==(const SamplerDesc&) const = default;
};

// ── Sampler cache policy (pure; device-free testable) ─────────────────
//
// A SamplerDesc is clamped to the device's limits before it is hashed or
// turned into a VkSampler, so equal requests collapse to one cache entry and
// two devices with different limits never collide on the same key. The
// clamped value is the cache key.
[[nodiscard]] inline SamplerDesc ClampSamplerDesc(const SamplerDesc& desired,
                                                  bool anisotropy_supported,
                                                  std::uint32_t device_max_anisotropy,
                                                  std::uint32_t mip_levels) {
    SamplerDesc clamped = desired;
    if (!anisotropy_supported || !clamped.anisotropy) {
        clamped.anisotropy = false;
        clamped.max_anisotropy = 1;
    } else {
        const std::uint32_t device_max = std::max(1U, device_max_anisotropy);
        clamped.max_anisotropy = std::max(1U, std::min(clamped.max_anisotropy, device_max));
    }
    // Never advertise lod levels the chain does not have.
    const float chain_max_lod = mip_levels > 1U ? static_cast<float>(mip_levels - 1U) : 0.0f;
    clamped.max_lod = mip_levels > 1U ? std::min(clamped.max_lod, chain_max_lod) : 0.0f;
    return clamped;
}

[[nodiscard]] inline std::size_t HashSamplerDesc(const SamplerDesc& desc) noexcept {
    std::size_t seed = 0;
    const auto mix = [&seed](std::size_t value) {
        // Boost-style combine: stable and dependency-free.
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    };
    mix(static_cast<std::size_t>(desc.min_filter));
    mix(static_cast<std::size_t>(desc.mag_filter));
    mix(static_cast<std::size_t>(desc.mip_mode));
    mix(static_cast<std::size_t>(desc.address_u));
    mix(static_cast<std::size_t>(desc.address_v));
    mix(static_cast<std::size_t>(desc.address_w));
    mix(static_cast<std::size_t>(desc.compare));
    mix(static_cast<std::size_t>(desc.border));
    mix(desc.anisotropy ? 1U : 0U);
    mix(static_cast<std::size_t>(desc.max_anisotropy));
    mix(static_cast<std::size_t>(std::bit_cast<std::uint32_t>(desc.mip_lod_bias)));
    mix(static_cast<std::size_t>(std::bit_cast<std::uint32_t>(desc.min_lod)));
    mix(static_cast<std::size_t>(std::bit_cast<std::uint32_t>(desc.max_lod)));
    return seed;
}

struct SamplerDescHash {
    [[nodiscard]] std::size_t operator()(const SamplerDesc& desc) const noexcept {
        return HashSamplerDesc(desc);
    }
};

// Distance between two clamped sampler descriptions. Used only when the cache
// is at capacity: the closest existing entry is substituted so a draw never
// fails for want of a sampler. Discrete state dominates; lod bias is a weak
// tiebreak so two samplers differing only in bias still rank apart.
[[nodiscard]] inline std::uint64_t SamplerDescDistance(const SamplerDesc& a, const SamplerDesc& b) noexcept {
    std::uint64_t distance = 0;
    const auto weigh = [&distance](bool differs, std::uint64_t weight) {
        if (differs) {
            distance += weight;
        }
    };
    weigh(a.min_filter != b.min_filter, 8);
    weigh(a.mag_filter != b.mag_filter, 4);
    weigh(a.mip_mode != b.mip_mode, 2);
    weigh(a.address_u != b.address_u, 2);
    weigh(a.address_v != b.address_v, 2);
    weigh(a.address_w != b.address_w, 2);
    weigh(a.compare != b.compare, 4);
    weigh(a.border != b.border, 1);
    weigh(a.anisotropy != b.anisotropy, 16);
    if (a.anisotropy && b.anisotropy) {
        weigh(a.max_anisotropy != b.max_anisotropy, 1);
    }
    weigh(a.mip_lod_bias != b.mip_lod_bias, 1);
    weigh(a.min_lod != b.min_lod, 1);
    weigh(a.max_lod != b.max_lod, 1);
    return distance;
}

// Translates an already-clamped description into a VkSamplerCreateInfo. Single
// source shared by the sampler cache and the uncached bootstrap path so both
// build identical samplers.
[[nodiscard]] inline vk::SamplerCreateInfo MakeSamplerCreateInfo(const SamplerDesc& clamped) {
    vk::SamplerCreateInfo info{};
    info.minFilter = clamped.min_filter;
    info.magFilter = clamped.mag_filter;
    info.mipmapMode = clamped.mip_mode;
    info.addressModeU = clamped.address_u;
    info.addressModeV = clamped.address_v;
    info.addressModeW = clamped.address_w;
    info.mipLodBias = clamped.mip_lod_bias;
    info.minLod = clamped.min_lod;
    info.maxLod = clamped.max_lod;
    info.anisotropyEnable = clamped.anisotropy ? vk::True : vk::False;
    info.maxAnisotropy = static_cast<float>(clamped.max_anisotropy);
    info.compareEnable = clamped.compare != vk::CompareOp::eNever ? vk::True : vk::False;
    info.compareOp = clamped.compare;
    info.borderColor = clamped.border;
    return info;
}

// Index of the entry nearest to `desired`; 0 when `candidates` is empty.
[[nodiscard]] inline std::size_t FindNearestSamplerIndex(std::span<const SamplerDesc> candidates,
                                                         const SamplerDesc& desired) noexcept {
    std::size_t best = 0;
    std::uint64_t best_distance = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const std::uint64_t distance = SamplerDescDistance(candidates[i], desired);
        if (distance < best_distance) {
            best_distance = distance;
            best = i;
        }
    }
    return best;
}

// ── Alpha analysis (moved from the loader-local type; used by material
//    validation and by callers deciding fallbacks) ──────────────────────

struct AlphaAnalysis {
    bool hasAlphaChannel = false; //NOLINT(misc-non-private-member-variables-in-classes)
    bool hasFractionalAlpha = false; //NOLINT(misc-non-private-member-variables-in-classes)
    bool hasZeroAlpha = false; //NOLINT(misc-non-private-member-variables-in-classes)
    float opaqueCoverage = 1.0f; //NOLINT(misc-non-private-member-variables-in-classes)
};

// ── Per-slot GPU metadata buffer ──────────────────────────────────────

// One element per bindless slot, exposed to shaders as a
// `StructuredBuffer<GpuTextureInfo>` (set 0, binding 1). A shader that needs a
// texture's extent, chain length, layer count or device format reads it here
// instead of inferring it from the sampled value; that is how mip-aware
// sampling, procedural fallbacks and debug views stay possible with no
// per-texture push constant.
//
// CDataLayout-compatible: six tightly packed 32-bit words, no padding, so the
// C++ and shader views are byte-identical. `format` carries the numeric value
// of the resolved vk::Format and tracks the descriptor, including while a
// reserved slot still points at the fallback.
struct GpuTextureInfo {
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t mip_levels{1};
    std::uint32_t array_layers{1};
    std::uint32_t format{0};  // vk::Format numeric value; 0 == eUndefined
    std::uint32_t flags{0};   // reserved for future packed metadata
};

static_assert(sizeof(GpuTextureInfo) == 24,
              "GpuTextureInfo must stay six packed 32-bit words for the shader mirror");

// ── Subresource-complete texture payload ──────────────────────────────

// One entry per (mip, layer, face). gpu_layer = layer * face_count + face.
struct TextureSubresource {
    std::uint32_t mip{0};
    std::uint32_t array_layer{0};
    std::uint64_t offset{0}, size{0};       // into TextureData::blob, block-tight
    std::uint32_t width{0}, height{0}, depth{1};
};

// Canonical loader output: the loader-local struct of the same name is deleted;
// this is the single type produced by every texture loader. Loaders keep the
// container payload as the file provides it (supercompression included) and
// never inflate or transcode; the uploader/worker resolves the device format
// and performs inflate + transcode.
struct TextureData {
    std::uint32_t width{0}, height{0}, depth{1};
    std::uint32_t mip_levels{1}, array_layers{1}, face_count{1};
    vk::Format source_format{vk::Format::eUndefined};
    bool needs_transcode{false};       // KTX2 Basis (UASTC/ETC1S)
    bool is_hdr_source{false};
    std::uint8_t source_channels{0};   // from the DFD channel model, not the raw count
    std::uint8_t source_alpha{0};      // 0/1
    bool supercompressed{false};       // Zstd/ZLIB payload present in blob
    bool cube_complete{true};          // false when KTXcubemapIncomplete is set
    TextureSemantic semantic{TextureSemantic::Unknown};
    vk::ComponentMapping swizzle{};    // from KTXswizzle; applied to the view
    AlphaAnalysis alpha_analysis{};
    std::vector<TextureSubresource> subresources;
    // Container payload as the file provides it. For PNG/JPG loaders this is
    // decoded RGBA8 pixels (those containers have no supercompression).
    std::vector<std::byte> blob;

    void Reset() noexcept {
        width = 0;
        height = 0;
        depth = 1;
        mip_levels = 1;
        array_layers = 1;
        face_count = 1;
        source_format = vk::Format::eUndefined;
        needs_transcode = false;
        is_hdr_source = false;
        source_channels = 0;
        source_alpha = 0;
        supercompressed = false;
        cube_complete = true;
        semantic = TextureSemantic::Unknown;
        swizzle = vk::ComponentMapping{};
        alpha_analysis = AlphaAnalysis{};
        subresources.clear();
        blob.clear();
    }
};

} // namespace VulkanEngine::Textures
