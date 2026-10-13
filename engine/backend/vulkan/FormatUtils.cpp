module;

module VulkanBackend.Vulkan.FormatUtils;

import std;

import vulkan;

namespace VulkanBackend::Vulkan {

namespace {

FormatTraits MakeUncompressed(vk::Format format, std::uint32_t texel_bytes) {
    FormatTraits traits{};
    traits.block = vk::Extent3D{1, 1, 1};
    traits.block_bytes = texel_bytes;
    // Same-bit sRGB pairs only; formats without one stay Undefined.
    switch (format) {
        case vk::Format::eR8Unorm:
            traits.srgb_variant = vk::Format::eR8Srgb;
            traits.unorm_variant = format;
            break;
        case vk::Format::eR8Srgb:
            traits.srgb_variant = format;
            traits.unorm_variant = vk::Format::eR8Unorm;
            traits.srgb = true;
            break;
        case vk::Format::eR8G8Unorm:
            traits.srgb_variant = vk::Format::eR8G8Srgb;
            traits.unorm_variant = format;
            break;
        case vk::Format::eR8G8Srgb:
            traits.srgb_variant = format;
            traits.unorm_variant = vk::Format::eR8G8Unorm;
            traits.srgb = true;
            break;
        case vk::Format::eR8G8B8A8Unorm:
            traits.srgb_variant = vk::Format::eR8G8B8A8Srgb;
            traits.unorm_variant = format;
            break;
        case vk::Format::eR8G8B8A8Srgb:
            traits.srgb_variant = format;
            traits.unorm_variant = vk::Format::eR8G8B8A8Unorm;
            traits.srgb = true;
            break;
        case vk::Format::eB8G8R8A8Unorm:
            traits.srgb_variant = vk::Format::eB8G8R8A8Srgb;
            traits.unorm_variant = format;
            break;
        case vk::Format::eB8G8R8A8Srgb:
            traits.srgb_variant = format;
            traits.unorm_variant = vk::Format::eB8G8R8A8Unorm;
            traits.srgb = true;
            break;
        default:
            break;
    }
    return traits;
}

FormatTraits MakeCompressed(vk::Format format, vk::Format counterpart, std::uint32_t block_bytes, bool srgb) {
    FormatTraits traits{};
    traits.compressed = true;
    traits.srgb = srgb;
    traits.block = vk::Extent3D{4, 4, 1};
    traits.block_bytes = block_bytes;
    traits.srgb_variant = srgb ? format : counterpart;
    traits.unorm_variant = srgb ? counterpart : format;
    return traits;
}

} // namespace

FormatTraits GetFormatTraits(vk::Format format) {
    switch (format) {
        case vk::Format::eR8Unorm: return MakeUncompressed(format, 1);
        case vk::Format::eR8Srgb: return MakeUncompressed(format, 1);
        case vk::Format::eR8G8Unorm: return MakeUncompressed(format, 2);
        case vk::Format::eR8G8Srgb: return MakeUncompressed(format, 2);
        case vk::Format::eR8G8B8A8Unorm: return MakeUncompressed(format, 4);
        case vk::Format::eR8G8B8A8Srgb: return MakeUncompressed(format, 4);
        case vk::Format::eB8G8R8A8Unorm: return MakeUncompressed(format, 4);
        case vk::Format::eB8G8R8A8Srgb: return MakeUncompressed(format, 4);
        case vk::Format::eR16G16B16A16Sfloat: {
            FormatTraits traits = MakeUncompressed(format, 8);
            return traits;
        }
        case vk::Format::eBc1RgbUnormBlock: return MakeCompressed(format, vk::Format::eBc1RgbSrgbBlock, 8, false);
        case vk::Format::eBc1RgbSrgbBlock: return MakeCompressed(format, vk::Format::eBc1RgbUnormBlock, 8, true);
        case vk::Format::eBc1RgbaUnormBlock: return MakeCompressed(format, vk::Format::eBc1RgbaSrgbBlock, 8, false);
        case vk::Format::eBc1RgbaSrgbBlock: return MakeCompressed(format, vk::Format::eBc1RgbaUnormBlock, 8, true);
        case vk::Format::eBc2UnormBlock: return MakeCompressed(format, vk::Format::eBc2SrgbBlock, 16, false);
        case vk::Format::eBc2SrgbBlock: return MakeCompressed(format, vk::Format::eBc2UnormBlock, 16, true);
        case vk::Format::eBc3UnormBlock: return MakeCompressed(format, vk::Format::eBc3SrgbBlock, 16, false);
        case vk::Format::eBc3SrgbBlock: return MakeCompressed(format, vk::Format::eBc3UnormBlock, 16, true);
        // BC4/BC5 have no sRGB variants in Vulkan (data-only targets).
        case vk::Format::eBc4UnormBlock:
            return MakeCompressed(format, vk::Format::eUndefined, 8, false);
        case vk::Format::eBc5UnormBlock:
            return MakeCompressed(format, vk::Format::eUndefined, 16, false);
        case vk::Format::eBc6HSfloatBlock:
        case vk::Format::eBc6HUfloatBlock: {
            FormatTraits traits = MakeCompressed(format, format, 16, false);
            return traits;
        }
        case vk::Format::eBc7UnormBlock: return MakeCompressed(format, vk::Format::eBc7SrgbBlock, 16, false);
        case vk::Format::eBc7SrgbBlock: return MakeCompressed(format, vk::Format::eBc7UnormBlock, 16, true);
        case vk::Format::eAstc4x4UnormBlock: return MakeCompressed(format, vk::Format::eAstc4x4SrgbBlock, 16, false);
        case vk::Format::eAstc4x4SrgbBlock: return MakeCompressed(format, vk::Format::eAstc4x4UnormBlock, 16, true);
        case vk::Format::eD32Sfloat:
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32SfloatS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD16UnormS8Uint: {
            FormatTraits traits{};
            traits.depth_stencil = true;
            traits.block = vk::Extent3D{1, 1, 1};
            traits.block_bytes = 4;
            return traits;
        }
        default:
            return FormatTraits{}; // block_bytes == 0 marks the format unknown
    }
}

vk::Extent3D MipExtent(std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t mip) {
    const auto shrink = [mip](std::uint32_t dim) {
        return mip == 0 ? dim : std::max<std::uint32_t>(1u, dim >> mip);
    };
    return vk::Extent3D{shrink(width), shrink(height), shrink(depth)};
}

vk::DeviceSize SubresourceByteSize(vk::Extent3D mip_extent, vk::Format format) {
    const FormatTraits traits = GetFormatTraits(format);
    if (traits.block_bytes == 0) {
        return 0;
    }
    const auto blocks = [](std::uint32_t dim, std::uint32_t block_dim) {
        return (dim + block_dim - 1) / block_dim;
    };
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(blocks(mip_extent.width, traits.block.width)) *
        static_cast<std::uint64_t>(blocks(mip_extent.height, traits.block.height)) *
        static_cast<std::uint64_t>(blocks(mip_extent.depth, traits.block.depth));
    return static_cast<vk::DeviceSize>(block_count * traits.block_bytes);
}

std::uint32_t MipLevelCount(std::uint32_t width, std::uint32_t height, std::uint32_t depth) {
    const std::uint32_t max_dim = std::max({width, height, depth});
    return max_dim == 0 ? 0u : static_cast<std::uint32_t>(std::bit_width(max_dim));
}

} // namespace VulkanBackend::Vulkan
