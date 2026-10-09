module;

export module VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;

import std;

import VulkanEngine.TextureTypes;
import VulkanEngine.MaterialManager.MaterialId;

// glTF material/texture/sampler import (Phase 4B). This interface exposes only
// CPU types and standard-library values: fastgltf is confined to the module
// implementation unit, so a consumer (and a device-free test) imports this
// module without linking the parser.
//
// The importer reports what the asset declares; it never names a technique and
// never touches the device. The application maps a glTF material index through
// its `material_bindings` table to a MaterialId it registered (K15), and builds
// the technique's per-material payload from these values.

export namespace VulkanEngine::FileLoaders::Mesh {

using VulkanEngine::MaterialManager::MaterialId;
using VulkanEngine::Textures::SamplerDesc;
using VulkanEngine::Textures::TextureNormalEncoding;

// ── Neutral value types (no Vulkan handles, no fastgltf types) ─────────

enum class GltfAlphaMode : std::uint8_t { Opaque, Mask, Blend };

enum class GltfWrap : std::uint8_t { ClampToEdge, MirroredRepeat, Repeat };

enum class GltfFilter : std::uint8_t {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
};

// glTF sampler. `specified == false` means the texture used no sampler object,
// so the glTF default applies (linear/linear, mip linear, repeat wrap).
struct GltfSamplerInfo {
    bool specified{false};
    GltfFilter min_filter{GltfFilter::Linear};
    GltfFilter mag_filter{GltfFilter::Linear};
    GltfWrap wrap_s{GltfWrap::Repeat};
    GltfWrap wrap_t{GltfWrap::Repeat};
};

// KHR_texture_transform for one texture slot. `present` is false when the
// extension is absent, so a consumer can pay nothing for the untransformed case.
struct GltfTextureTransform {
    bool present{false};
    float offset_u{0.0f};
    float offset_v{0.0f};
    float scale_u{1.0f};
    float scale_v{1.0f};
    float rotation{0.0f};
};

// A glTF texture node: which image it uses and which sampler, if any.
// `valid == false` marks a texture reference whose image could not be resolved
// (missing index), so the caller can substitute the fallback.
struct GltfTextureRef {
    bool valid{false};
    std::size_t image_index{0};
    // Index into GltfImportResult::samplers, or kNoSampler when the texture
    // declared none (the glTF default sampler applies).
    std::size_t sampler_index{0};
    bool basisu{false}; // resolved through KHR_texture_basisu (a KTX2 image)
};

inline constexpr std::size_t kNoGltfSampler = std::numeric_limits<std::size_t>::max();

// A glTF image source. `Uri` carries the raw external URI (a relative path the
// caller resolves against the asset directory); `Embedded` carries the payload
// bytes (a GLB-embedded image, a `data:` URI, which fastgltf always decodes, or
// an image stored in a buffer view), so no second read is needed.
struct GltfImageRef {
    enum class Source : std::uint8_t { None, Uri, Embedded };
    Source source{Source::None};
    std::string uri;                       // set for Source::Uri
    std::string mime_type;                 // best-effort; may be empty
    std::vector<std::byte> embedded_bytes; // set for Source::Embedded
};

// One material texture slot. Slot indices are 0=baseColor, 1=normal, 2=ORM
// (metallicRoughness/occlusion), 3=emissive.
struct GltfTextureSlot {
    bool present{false};
    std::size_t texture_index{0}; // into GltfImportResult::textures
    std::uint32_t tex_coord{0};   // glTF TEXCOORD_n
    GltfTextureTransform transform{};
};

struct GltfMaterialInfo {
    std::size_t index{0};
    std::string name;
    bool unlit{false};
    GltfAlphaMode alpha_mode{GltfAlphaMode::Opaque};
    float alpha_cutoff{0.5f};
    bool double_sided{false};

    float base_color_factor[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic_factor{1.0f};
    float roughness_factor{1.0f};
    float ao_factor{1.0f}; // always 1.0f; glTF has no occlusion factor
    float emissive_factor[3]{0.0f, 0.0f, 0.0f};
    float emissive_strength{1.0f};
    float normal_scale{1.0f};
    float occlusion_strength{1.0f};

    // Declared content, never derived from a resolved upload format. glTF
    // normal maps are authored RGB, so a material with a normal texture is
    // Full unless the content pipeline packed RG (then the app overrides this).
    TextureNormalEncoding normal_encoding{TextureNormalEncoding::Standard};

    GltfTextureSlot base_color_texture{};
    GltfTextureSlot normal_texture{};
    GltfTextureSlot metallic_roughness_texture{};
    GltfTextureSlot occlusion_texture{};
    GltfTextureSlot emissive_texture{};
};

struct GltfImportResult {
    std::vector<GltfMaterialInfo> materials;
    std::vector<GltfTextureRef> textures;
    std::vector<GltfImageRef> images;
    std::vector<GltfSamplerInfo> samplers;
};

// ── Pure mapping helpers (device-free) ────────────────────────────────

// Maps a glTF material index to a MaterialId through the app-supplied binding
// table, matching the OBJ/BIN path. Returns MaterialId{0} (the default/fallback
// material) when there is no table or the index is out of range.
[[nodiscard]] MaterialId MapGltfMaterialIndex(std::size_t material_index,
                                              const std::vector<MaterialId>* bindings) noexcept;

// Packs a material's per-slot texCoord selection into the technique's
// `DefaultMeshPerMaterialData::uv_sets` (2 bits per slot, slot order:
// 0=baseColor, 1=normal, 2=ORM, 3=emissive). Values above 3 are clamped. The
// ORM slot follows the metallicRoughness texture's texCoord, falling back to
// the occlusion texture's when only occlusion is present.
[[nodiscard]] std::uint32_t ComputeGltfUvSets(const GltfMaterialInfo& material) noexcept;

// True when any texture slot carries a KHR_texture_transform, i.e. the
// material's `has_transform` flag should be set.
[[nodiscard]] bool GltfMaterialHasTransform(const GltfMaterialInfo& material) noexcept;

// Maps a glTF sampler to the engine's CPU sampler description. The glTF default
// (no sampler object) is linear filtering with linear mip interpolation and
// repeat wrap. Mip-aware min filters set the mip mode; the rest of the fields
// keep the engine defaults (no anisotropy, no lod clamp) for the uploader to
// resolve against device limits.
[[nodiscard]] SamplerDesc ToSamplerDesc(const GltfSamplerInfo& sampler) noexcept;

// ── Entry point ───────────────────────────────────────────────────────

// Parses materials, textures, images, and samplers from a .gltf or .glb file.
// External buffers and images are resolved relative to the file's directory
// when present. Throws std::runtime_error on an unreadable or invalid asset.
[[nodiscard]] GltfImportResult ImportGltfMaterials(const std::filesystem::path& path);

} // namespace VulkanEngine::FileLoaders::Mesh
