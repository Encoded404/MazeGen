module;

export module VulkanBackend.Vulkan.ImageUtils;

import std;
import std.compat;

import vulkan;

export namespace VulkanBackend::Vulkan {

class ImageUtils {
public:
    static std::uint32_t CalculateMipLevels(std::uint32_t width, std::uint32_t height, std::uint32_t depth = 1);
    static vk::ImageAspectFlags GetImageAspectFlags(vk::Format format);
    static vk::ImageSubresourceRange CreateSubresourceRange(vk::ImageAspectFlags aspect_flags,
                                                          std::uint32_t base_mip_level = 0,
                                                          std::uint32_t level_count = vk::RemainingMipLevels,
                                                          std::uint32_t base_array_layer = 0,
                                                          std::uint32_t layer_count = vk::RemainingArrayLayers);
    static vk::BufferImageCopy CreateBufferImageCopy(vk::Format format,
                                                   std::uint32_t width,
                                                   std::uint32_t height,
                                                   std::uint32_t depth = 1,
                                                   vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor,
                                                   std::uint32_t mip_level = 0,
                                                   std::uint32_t array_layer = 0,
                                                   vk::DeviceSize buffer_offset = 0);

    static vk::ImageCopy CreateImageCopy(const vk::ImageSubresourceLayers& src_subresource,
                                      const vk::Offset3D& src_offset,
                                      const vk::ImageSubresourceLayers& dst_subresource,
                                      const vk::Offset3D& dst_offset,
                                      const vk::Extent3D& extent);

    static vk::ImageSubresourceLayers CreateSubresourceLayers(vk::ImageAspectFlags aspect_flags,
                                                           std::uint32_t mip_level = 0,
                                                           std::uint32_t base_array_layer = 0,
                                                           std::uint32_t layer_count = 1);

    static vk::DeviceSize CalculateImageSize(std::uint32_t width, std::uint32_t height, std::uint32_t depth,
                                          std::uint32_t mip_levels, std::uint32_t array_layers,
                                          vk::Format format);

    static void GetMipLevelDimensions(std::uint32_t base_mip_width, std::uint32_t base_mip_height, std::uint32_t base_mip_depth,
                                     std::uint32_t mip_level,
                                     std::uint32_t& mip_width, std::uint32_t& mip_height, std::uint32_t& mip_depth);

    static bool RequiresOptimalTiling(vk::ImageUsageFlags usage);
    static vk::ImageViewType GetCompatibleViewType(vk::ImageType image_type, std::uint32_t array_layers);

    static vk::ImageViewCreateInfo CreateImageViewCreateInfo(vk::Image image,
                                                          vk::ImageViewType view_type,
                                                          vk::Format format,
                                                          vk::ImageAspectFlags aspect_flags,
                                                          std::uint32_t base_mip_level = 0,
                                                          std::uint32_t level_count = vk::RemainingMipLevels,
                                                          std::uint32_t base_array_layer = 0,
                                                          std::uint32_t layer_count = vk::RemainingArrayLayers);

    static vk::ImageMemoryBarrier CreateImageMemoryBarrier(vk::Image image,
                                                         vk::ImageLayout old_layout,
                                                         vk::ImageLayout new_layout,
                                                         vk::ImageAspectFlags aspect_flags,
                                                         std::uint32_t base_mip_level = 0,
                                                         std::uint32_t level_count = vk::RemainingMipLevels,
                                                         std::uint32_t base_array_layer = 0,
                                                         std::uint32_t layer_count = vk::RemainingArrayLayers,
                                                         std::uint32_t src_queue_family_index = vk::QueueFamilyIgnored,
                                                         std::uint32_t dst_queue_family_index = vk::QueueFamilyIgnored);

    static vk::PipelineStageFlags GetLayoutPipelineStageFlags(vk::ImageLayout layout, bool is_source);
    static vk::AccessFlags GetLayoutAccessFlags(vk::ImageLayout layout);

    static void CmdTransitionImageLayout(vk::raii::CommandBuffer const& cmd,
                                         vk::Image image,
                                         vk::ImageLayout old_layout,
                                         vk::ImageLayout new_layout,
                                         vk::ImageAspectFlags aspect_flags,
                                         std::uint32_t base_mip_level = 0,
                                         std::uint32_t level_count = vk::RemainingMipLevels,
                                         std::uint32_t base_array_layer = 0,
                                         std::uint32_t layer_count = vk::RemainingArrayLayers);

    static void CmdCopyBufferToImage(vk::raii::CommandBuffer const& cmd,
                                     vk::Buffer src_buffer,
                                     vk::Image dst_image,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     vk::ImageLayout dst_layout,
                                     vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor,
                                     std::uint32_t mip_level = 0,
                                     std::uint32_t array_layer = 0,
                                     vk::DeviceSize buffer_offset = 0,
                                     vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                     std::uint32_t depth = 1);
};

} // namespace VulkanBackend::Vulkan
