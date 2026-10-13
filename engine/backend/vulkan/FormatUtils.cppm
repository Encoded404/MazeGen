module;

export module VulkanBackend.Vulkan.FormatUtils;

import std;

import vulkan;

export namespace VulkanBackend::Vulkan {

// Static per-format traits for the closed candidate set the texture system
// may choose from (BC1..BC7, BC4/BC5, ASTC 4x4, R8/R8G8/RGBA8, BGRA8,
// RGBA16F). Pure data: no device queries at worker time. An unknown format
// reports block_bytes == 0 and callers must treat it as unsupported.
struct FormatTraits {
    bool compressed{false};
    bool srgb{false};
    bool depth_stencil{false};
    vk::Format srgb_variant{vk::Format::eUndefined};
    vk::Format unorm_variant{vk::Format::eUndefined};
    vk::Extent3D block{1, 1, 1};
    std::uint32_t block_bytes{0};
};

[[nodiscard]] FormatTraits GetFormatTraits(vk::Format format);

// Texel dimensions of `mip`: each dimension is halved per level, never below 1.
[[nodiscard]] vk::Extent3D MipExtent(std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t mip);

// Block-tight byte size of one subresource of `mip_extent` in `format`.
// Compressed formats round each dimension up to whole blocks (including the
// sub-4x4 tail mips); uncompressed formats are byte-exact.
[[nodiscard]] vk::DeviceSize SubresourceByteSize(vk::Extent3D mip_extent, vk::Format format);

// Number of mip levels for a base extent (1 for an empty extent).
[[nodiscard]] std::uint32_t MipLevelCount(std::uint32_t width, std::uint32_t height, std::uint32_t depth);

} // namespace VulkanBackend::Vulkan
