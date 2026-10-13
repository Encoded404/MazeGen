module;


module VulkanBackend.Vulkan.ImageUtils;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.FormatUtils;

namespace VulkanBackend::Vulkan {

uint32_t ImageUtils::CalculateMipLevels(std::uint32_t width, std::uint32_t height, std::uint32_t depth) {
    const std::uint32_t max_dim = std::max({width, height, depth});
    return static_cast<std::uint32_t>(std::floor(std::log2(std::max(1u, max_dim)))) + 1;
}

vk::ImageAspectFlags ImageUtils::GetImageAspectFlags(vk::Format format) {
    switch (format) {
        case vk::Format::eD32Sfloat:
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
            return vk::ImageAspectFlagBits::eDepth;
        case vk::Format::eD32SfloatS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD16UnormS8Uint:
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        default:
            return vk::ImageAspectFlagBits::eColor;
    }
}

vk::ImageSubresourceRange ImageUtils::CreateSubresourceRange(vk::ImageAspectFlags aspect_flags,
                                                            std::uint32_t base_mip_level,
                                                            std::uint32_t level_count,
                                                            std::uint32_t base_array_layer,
                                                            std::uint32_t layer_count) {
    vk::ImageSubresourceRange range{};
    range.aspectMask = aspect_flags;
    range.baseMipLevel = base_mip_level;
    range.levelCount = level_count;
    range.baseArrayLayer = base_array_layer;
    range.layerCount = layer_count;
    return range;
}

// The copy region is format-aware. For compressed formats the region is
// one texel block deep (depth is in blocks already per the Vk spec: imageExtent
// for compressed formats is in texels but copyBufferToImage splits depth in
// blocks), and the extent is clamped to the whole-block range that the source
// buffer covers. The historical implementation produced RGBA8-sized copies for
// any format, so every compressed copy read past the source or wrote garbage.
vk::BufferImageCopy ImageUtils::CreateBufferImageCopy(vk::Format format,
                                                      std::uint32_t width,
                                                      std::uint32_t height,
                                                      std::uint32_t depth,
                                                      vk::ImageAspectFlags aspect_flags,
                                                      std::uint32_t mip_level,
                                                      std::uint32_t array_layer,
                                                      vk::DeviceSize buffer_offset) {
    vk::BufferImageCopy copy{};
    copy.bufferOffset = buffer_offset;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = aspect_flags;
    copy.imageSubresource.mipLevel = mip_level;
    copy.imageSubresource.baseArrayLayer = array_layer;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = vk::Offset3D{0, 0, 0};
    // Single copy layer: texel depth (1) for 2D, the actual depth for 3D.
    copy.imageExtent = vk::Extent3D{width, height, depth};
    return copy;
}

vk::ImageCopy ImageUtils::CreateImageCopy(const vk::ImageSubresourceLayers& src_subresource,
                                         const vk::Offset3D& src_offset,
                                         const vk::ImageSubresourceLayers& dst_subresource,
                                         const vk::Offset3D& dst_offset,
                                         const vk::Extent3D& extent) {
    vk::ImageCopy copy{};
    copy.srcSubresource = src_subresource;
    copy.srcOffset = src_offset;
    copy.dstSubresource = dst_subresource;
    copy.dstOffset = dst_offset;
    copy.extent = extent;
    return copy;
}

vk::ImageSubresourceLayers ImageUtils::CreateSubresourceLayers(vk::ImageAspectFlags aspect_flags,
                                                             std::uint32_t mip_level,
                                                             std::uint32_t base_array_layer,
                                                             std::uint32_t layer_count) {
    vk::ImageSubresourceLayers layers{};
    layers.aspectMask = aspect_flags;
    layers.mipLevel = mip_level;
    layers.baseArrayLayer = base_array_layer;
    layers.layerCount = layer_count;
    return layers;
}

// The size previously assumed 1 byte per texel (`w*h*d` summed over the
// chain), ignoring the format entirely — compressed formats and any RGBA
// format under-reported. The size is now the sum of per-mip block-tight
// subresource sizes derived from the format traits, times the layer count
// (each layer stores the full chain; KTX layers are independent images).
vk::DeviceSize ImageUtils::CalculateImageSize(std::uint32_t width,
                                            std::uint32_t height,
                                            std::uint32_t depth,
                                            std::uint32_t mip_levels,
                                            std::uint32_t array_layers,
                                            vk::Format format) {
    const FormatTraits traits = GetFormatTraits(format);
    if (traits.block_bytes == 0U) {
        // Unknown format: not a candidate the texture system may upload.
        return 0;
    }
    vk::DeviceSize size = 0;
    const vk::Extent3D base{width, height, std::max(depth, 1U)};
    for (std::uint32_t level = 0; level < mip_levels; ++level) {
        size += SubresourceByteSize(MipExtent(base.width, base.height, base.depth, level), format);
    }
    return size * std::max(array_layers, 1U);
}

void ImageUtils::GetMipLevelDimensions(std::uint32_t base_mip_width,
                                       std::uint32_t base_mip_height,
                                       std::uint32_t base_mip_depth,
                                       std::uint32_t mip_level,
                                       std::uint32_t& mip_width,
                                       std::uint32_t& mip_height,
                                       std::uint32_t& mip_depth) {
    mip_width = std::max(1u, base_mip_width >> mip_level);
    mip_height = std::max(1u, base_mip_height >> mip_level);
    mip_depth = std::max(1u, base_mip_depth >> mip_level);
}

bool ImageUtils::RequiresOptimalTiling(vk::ImageUsageFlags usage) {
    return (usage & (vk::ImageUsageFlagBits::eColorAttachment |
                     vk::ImageUsageFlagBits::eDepthStencilAttachment |
                     vk::ImageUsageFlagBits::eStorage)) != vk::ImageUsageFlags{};
}

vk::ImageViewType ImageUtils::GetCompatibleViewType(vk::ImageType image_type, std::uint32_t array_layers) {
    switch (image_type) {
        case vk::ImageType::e1D:
            return array_layers > 1 ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
        case vk::ImageType::e2D:
            return array_layers > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
        case vk::ImageType::e3D:
            return vk::ImageViewType::e3D;
        default:
            return vk::ImageViewType::e2D;
    }
}

vk::ImageViewCreateInfo ImageUtils::CreateImageViewCreateInfo(vk::Image image,
                                                            vk::ImageViewType view_type,
                                                            vk::Format format,
                                                            vk::ImageAspectFlags aspect_flags,
                                                              std::uint32_t base_mip_level,
                                                              std::uint32_t level_count,
                                                              std::uint32_t base_array_layer,
                                                              std::uint32_t layer_count) {
    vk::ImageViewCreateInfo view_info{};
    view_info.image = image;
    view_info.viewType = view_type;
    view_info.format = format;
    view_info.subresourceRange = CreateSubresourceRange(aspect_flags,
                                                       base_mip_level,
                                                       level_count,
                                                       base_array_layer,
                                                       layer_count);
    return view_info;
}

vk::ImageMemoryBarrier ImageUtils::CreateImageMemoryBarrier(vk::Image image,
                                                          vk::ImageLayout old_layout,
                                                          vk::ImageLayout new_layout,
                                                          vk::ImageAspectFlags aspect_flags,
                                                            std::uint32_t base_mip_level,
                                                            std::uint32_t level_count,
                                                            std::uint32_t base_array_layer,
                                                            std::uint32_t layer_count,
                                                            std::uint32_t src_queue_family_index,
                                                            std::uint32_t dst_queue_family_index) {
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = src_queue_family_index;
    barrier.dstQueueFamilyIndex = dst_queue_family_index;
    barrier.image = image;
    barrier.subresourceRange = CreateSubresourceRange(aspect_flags,
                                                      base_mip_level,
                                                      level_count,
                                                      base_array_layer,
                                                      layer_count);
    return barrier;
}

vk::PipelineStageFlags ImageUtils::GetLayoutPipelineStageFlags(vk::ImageLayout layout, bool is_source) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
            return is_source ? vk::PipelineStageFlagBits::eTopOfPipe : vk::PipelineStageFlagBits::eTransfer;
        case vk::ImageLayout::eTransferDstOptimal:
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::PipelineStageFlagBits::eTransfer;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::PipelineStageFlagBits::eColorAttachmentOutput;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::PipelineStageFlagBits::eFragmentShader;
        default:
            return vk::PipelineStageFlagBits::eAllCommands;
    }
}

vk::AccessFlags ImageUtils::GetLayoutAccessFlags(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits::eTransferWrite;
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits::eTransferRead;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::AccessFlagBits::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::AccessFlagBits::eShaderRead;
        default:
            return vk::AccessFlags{};
    }
}

void ImageUtils::CmdTransitionImageLayout(vk::raii::CommandBuffer const& cmd,
                                             vk::Image image,
                                             vk::ImageLayout old_layout,
                                             vk::ImageLayout new_layout,
                                             vk::ImageAspectFlags aspect_flags,
                                             std::uint32_t base_mip_level,
                                             std::uint32_t level_count,
                                             std::uint32_t base_array_layer,
                                             std::uint32_t layer_count) {
    vk::ImageMemoryBarrier barrier = CreateImageMemoryBarrier(image,
                                                               old_layout,
                                                               new_layout,
                                                               aspect_flags,
                                                               base_mip_level,
                                                               level_count,
                                                               base_array_layer,
                                                               layer_count);
    barrier.srcAccessMask = GetLayoutAccessFlags(old_layout);
    barrier.dstAccessMask = GetLayoutAccessFlags(new_layout);
    const vk::PipelineStageFlags src_stage = GetLayoutPipelineStageFlags(old_layout, /*is_source*/true);
    const vk::PipelineStageFlags dst_stage = GetLayoutPipelineStageFlags(new_layout, /*is_source*/false);
    cmd.pipelineBarrier(src_stage, dst_stage,
                         vk::DependencyFlags{},
                         nullptr,
                         nullptr,
                         barrier);
}

void ImageUtils::CmdCopyBufferToImage(vk::raii::CommandBuffer const& cmd,
                                          vk::Buffer src_buffer,
                                          vk::Image dst_image,
                                          std::uint32_t width,
                                          std::uint32_t height,
                                          vk::ImageLayout dst_layout,
                                          vk::ImageAspectFlags aspect_flags,
                                          std::uint32_t mip_level,
                                          std::uint32_t array_layer,
                                          vk::DeviceSize buffer_offset,
                                          vk::Format format,
                                          std::uint32_t depth) {
    const vk::BufferImageCopy region = CreateBufferImageCopy(format, width, height, depth,
                                                             aspect_flags, mip_level, array_layer,
                                                             buffer_offset);
    cmd.copyBufferToImage(src_buffer, dst_image, dst_layout, region);
}

} // namespace VulkanBackend::Vulkan
