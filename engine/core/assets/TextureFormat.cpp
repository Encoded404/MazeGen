module;

module VulkanEngine.TextureFormat;

import std;

import vulkan;

import VulkanEngine.TextureTypes;

namespace VulkanEngine::Textures {

namespace {

// Same-bit sRGB/UNORM variant pairs the resolver may switch between; formats
// outside the table have no variant and never reinterpret.
struct FormatVariantPair {
    vk::Format format;
    vk::Format srgb;
    vk::Format unorm;
};

constexpr std::array<FormatVariantPair, 14> kVariantPairs{{
    {vk::Format::eR8Unorm,           vk::Format::eR8Srgb,           vk::Format::eR8Unorm},
    {vk::Format::eR8G8Unorm,         vk::Format::eR8G8Srgb,         vk::Format::eR8G8Unorm},
    {vk::Format::eR8G8B8A8Unorm,     vk::Format::eR8G8B8A8Srgb,     vk::Format::eR8G8B8A8Unorm},
    {vk::Format::eB8G8R8A8Unorm,     vk::Format::eB8G8R8A8Srgb,     vk::Format::eB8G8R8A8Unorm},
    {vk::Format::eBc1RgbUnormBlock,  vk::Format::eBc1RgbSrgbBlock,  vk::Format::eBc1RgbUnormBlock},
    {vk::Format::eBc1RgbaUnormBlock, vk::Format::eBc1RgbaSrgbBlock, vk::Format::eBc1RgbaUnormBlock},
    {vk::Format::eBc3UnormBlock,     vk::Format::eBc3SrgbBlock,     vk::Format::eBc3UnormBlock},
    {vk::Format::eBc4UnormBlock,     vk::Format::eUndefined,       vk::Format::eBc4UnormBlock},
    {vk::Format::eBc5UnormBlock,     vk::Format::eUndefined,       vk::Format::eBc5UnormBlock},
    {vk::Format::eBc7UnormBlock,     vk::Format::eBc7SrgbBlock,     vk::Format::eBc7UnormBlock},
    {vk::Format::eAstc4x4SrgbBlock,  vk::Format::eAstc4x4SrgbBlock, vk::Format::eUndefined},
}};

[[nodiscard]] const FormatVariantPair* FindVariantPair(vk::Format format) {
    const auto it = std::ranges::find(kVariantPairs, format, &FormatVariantPair::format);
    return it != kVariantPairs.end() ? &*it : nullptr;
}

[[nodiscard]] bool IsBcCompressed(vk::Format format) {
    return format >= vk::Format::eBc1RgbUnormBlock && format <= vk::Format::eBc7SrgbBlock;
}

[[nodiscard]] bool IsUncompressed8Bit(vk::Format format) {
    switch (format) {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Srgb:
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Srgb:
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Srgb:
            return true;
        default:
            return false;
    }
}

} // namespace

bool IsColorSemantic(TextureSemantic semantic) noexcept {
    switch (semantic) {
        case TextureSemantic::BaseColor:
        case TextureSemantic::Emissive:
        case TextureSemantic::UI:
        case TextureSemantic::HDR:
            return true;
        case TextureSemantic::Normal:
        case TextureSemantic::ORM:
        case TextureSemantic::Mask:
        case TextureSemantic::Unknown:
            return false;
    }
    return false;
}

ResolvedFormat ResolveUploadFormat(const TextureSemantic semantic,
                                   const TextureNormalEncoding normal_encoding,
                                   const vk::Format source,
                                   const bool needs_transcode,
                                   const std::uint8_t source_channels,
                                   const bool source_alpha,
                                   const bool hdr,
                                   const FormatSupportQuery& caps) {
    // ── Basis transcode path ──
    if (needs_transcode) {
        // HDR Basis never downconverts to 8-bit; libktx 4.x has no BC6H
        // transcode target, so the HDR path lands in float16.
        if (hdr) {
            return {.format = vk::Format::eR16G16B16A16Sfloat, .reinterpreted = false, .transcoded = true};
        }

        // 2-channel content declared Full/Bent must never land in BC5;
        // force it through the 3/4-channel target list below.
        const bool two_channel_forced_rgb =
            source_channels == 2 && normal_encoding != TextureNormalEncoding::Standard;

        // 2 independent channels (RG / RRRG) and a Standard normal -> BC5.
        if (source_channels == 2 && !two_channel_forced_rgb) {
            if (caps.sampled(vk::Format::eBc5UnormBlock) && caps.transfer_dst(vk::Format::eBc5UnormBlock)) {
                return {.format = vk::Format::eBc5UnormBlock, .reinterpreted = false, .transcoded = true};
            }
        }

        // 1 channel used as data -> BC4 (no sRGB; a data-only target).
        if (source_channels == 1 && !IsColorSemantic(semantic)) {
            if (caps.sampled(vk::Format::eBc4UnormBlock) && caps.transfer_dst(vk::Format::eBc4UnormBlock)) {
                return {.format = vk::Format::eBc4UnormBlock, .reinterpreted = false, .transcoded = true};
            }
        }

        // 3/4 channels (and 1-channel color, and forced-RGB 2-channel):
        // BC7 > BC3 > BC1, with BC1 only for opaque content. BC4 has no sRGB,
        // so a color 1-channel goes through BC7/RGBA8.
        const bool wants_srgb = IsColorSemantic(semantic);
        struct Target { vk::Format unorm; vk::Format srgb; bool alpha_ok; };
        constexpr std::array<Target, 3> rgb_targets{{
            {vk::Format::eBc7UnormBlock, vk::Format::eBc7SrgbBlock, true},
            {vk::Format::eBc3UnormBlock, vk::Format::eBc3SrgbBlock, true},
            {vk::Format::eBc1RgbaUnormBlock, vk::Format::eBc1RgbaSrgbBlock, false},
        }};
        for (const auto& target : rgb_targets) {
            if (source_alpha && !target.alpha_ok) {
                continue; // BC1 cannot represent alpha
            }
            const vk::Format candidate = wants_srgb ? target.srgb : target.unorm;
            if (caps.sampled(candidate) && caps.transfer_dst(candidate)) {
                return {.format = candidate, .reinterpreted = false, .transcoded = true};
            }
        }

        // No BCn at all: uncompressed fallback (or float16 for HDR sources,
        // which never downconvert to 8-bit).
        if (hdr) {
            return {.format = vk::Format::eR16G16B16A16Sfloat, .reinterpreted = false, .transcoded = true};
        }
        return {.format = wants_srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
                .reinterpreted = false, .transcoded = true};
    }

    // ── Native BCn source ──
    if (IsBcCompressed(source)) {
        if (!caps.sampled(source) || !caps.transfer_dst(source)) {
            // No runtime decode exists: fail honestly rather than expand.
            return {.format = vk::Format::eUndefined, .reinterpreted = false, .transcoded = false};
        }
        // A native 2-channel BC5 is a Standard normal; Full/Bent on it cannot
        // be honored without a decode -> fail.
        if (source == vk::Format::eBc5UnormBlock && normal_encoding != TextureNormalEncoding::Standard) {
            return {.format = vk::Format::eUndefined, .reinterpreted = false, .transcoded = false};
        }
        // Reinterpret to the semantic's variant when one exists.
        if (const FormatVariantPair* pair = FindVariantPair(source); pair != nullptr) {
            const vk::Format wanted = IsColorSemantic(semantic) ? pair->srgb : pair->unorm;
            if (wanted != source && wanted != vk::Format::eUndefined && caps.sampled(wanted)) {
                return {.format = wanted, .reinterpreted = true, .transcoded = false};
            }
        }
        return {.format = source, .reinterpreted = false, .transcoded = false};
    }

    // ── Uncompressed source ──
    if (hdr || source == vk::Format::eR16G16B16A16Sfloat) {
        // HDR sources never downconvert to 8-bit.
        return {.format = vk::Format::eR16G16B16A16Sfloat, .reinterpreted = false, .transcoded = false};
    }

    if (source == vk::Format::eUndefined || IsUncompressed8Bit(source)) {
        const bool color = IsColorSemantic(semantic);
        // Single/dual channel data maps to R8/R8G8; everything else to RGBA8.
        if (!color && source_channels == 1 && !source_alpha) {
            return {.format = vk::Format::eR8Unorm, .reinterpreted = false, .transcoded = false};
        }
        if (!color && source_channels == 2 && normal_encoding == TextureNormalEncoding::Standard) {
            return {.format = vk::Format::eR8G8Unorm, .reinterpreted = false, .transcoded = false};
        }
        // Rule 2: an explicit _SRGB source on a data semantic reinterprets to
        // _UNORM so it is not hardware-linearized. Rule 1 is symmetric: an
        // explicit _UNORM source on a color semantic reinterprets to _SRGB.
        if (!color && source == vk::Format::eR8G8B8A8Srgb) {
            return {.format = vk::Format::eR8G8B8A8Unorm, .reinterpreted = true, .transcoded = false};
        }
        if (color && source == vk::Format::eR8G8B8A8Unorm) {
            return {.format = vk::Format::eR8G8B8A8Srgb, .reinterpreted = true, .transcoded = false};
        }
        return {.format = color ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
                .reinterpreted = false, .transcoded = false};
    }

    // Any other native format (e.g. A2B10G10R10_UNORM_PACK32): a
    // color semantic without a same-bit _SRGB variant is a content error that
    // fails the load; data semantics keep the native format.
    if (IsColorSemantic(semantic)) {
        return {.format = vk::Format::eUndefined, .reinterpreted = false, .transcoded = false};
    }
    return {.format = source, .reinterpreted = false, .transcoded = false};
}

} // namespace VulkanEngine::Textures
