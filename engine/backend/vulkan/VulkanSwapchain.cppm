module;

export module VulkanBackend.Vulkan.VulkanSwapchain;

import std;

import vulkan;

import VulkanBackend.Vulkan.CommonTypes;
import VulkanBackend.Vulkan.VulkanDevice;
import VulkanBackend.Vulkan.VulkanInstance;

export namespace VulkanBackend::Vulkan {

class VulkanSwapchain {
public:
    [[nodiscard]] bool Initialize(const VulkanInstance& instance, const VulkanDevice& device, std::uint32_t preferred_image_count, PresentMode present_mode);
    void Shutdown();
    [[nodiscard]] bool Recreate(const VulkanInstance& instance, const VulkanDevice& device, std::uint32_t preferred_image_count, PresentMode present_mode);

    [[nodiscard]] std::uint32_t GetImageCount() const { return swapchain_image_count_; }
    [[nodiscard]] std::uint32_t GetWidth() const { return swapchain_extent_.width; }
    [[nodiscard]] std::uint32_t GetHeight() const { return swapchain_extent_.height; }
    [[nodiscard]] const vk::SurfaceFormatKHR& GetSurfaceFormat() const { return surface_format_; }
    [[nodiscard]] const vk::raii::SwapchainKHR& GetSwapchain() const { return *swapchain_; }
    [[nodiscard]] const std::vector<vk::Image>& GetImages() const { return swapchain_images_; }
    [[nodiscard]] const std::vector<vk::raii::ImageView>& GetImageViews() const { return swapchain_image_views_; }
    [[nodiscard]] std::vector<bool>& GetImageInitializedFlags() { return swapchain_image_initialized_; }

    // Depth buffer support (per swapchain image)
    [[nodiscard]] vk::Format GetDepthFormat() const { return depth_format_; }
    [[nodiscard]] const vk::raii::ImageView& GetDepthImageView(std::uint32_t image_index) const { return depth_image_views_[image_index]; }
    [[nodiscard]] const vk::raii::Image& GetDepthImage(std::uint32_t image_index) const { return depth_images_[image_index]; }
    [[nodiscard]] std::uint32_t GetDepthImageCount() const { return static_cast<std::uint32_t>(depth_images_.size()); }

private:
    std::unique_ptr<vk::raii::SwapchainKHR> swapchain_{};
    std::vector<vk::Image> swapchain_images_{};
    std::vector<vk::raii::ImageView> swapchain_image_views_{};
    vk::SurfaceFormatKHR surface_format_{};
    vk::Extent2D swapchain_extent_{};
    std::uint32_t swapchain_image_count_ = 0;
    std::vector<bool> swapchain_image_initialized_{};

    // Depth buffer resources (per swapchain image)
    vk::Format depth_format_ = vk::Format::eD32Sfloat;
    std::vector<vk::raii::Image> depth_images_{};
    std::vector<vk::raii::DeviceMemory> depth_image_memories_{};
    std::vector<vk::raii::ImageView> depth_image_views_{};

    [[nodiscard]] bool RebuildSwapchainViews(const vk::raii::Device& device);
    [[nodiscard]] bool CreateDepthResources(const vk::raii::PhysicalDevice& physical_device, const vk::raii::Device& device);
};

} // namespace VulkanBackend::Vulkan
