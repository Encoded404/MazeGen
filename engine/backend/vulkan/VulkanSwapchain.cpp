module;

#include <SDL3/SDL_video.h>

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.VulkanSwapchain;

import std;
import std.compat;
import logiface;

import vulkan;

import VulkanBackend.Vulkan.CommonTypes;
import VulkanBackend.Vulkan.VulkanDevice;
import VulkanBackend.Vulkan.VulkanInstance;
import VulkanBackend.Vulkan.VulkanDebugUtils;

namespace VulkanBackend::Vulkan {

namespace {

vk::PresentModeKHR ToVkPresentMode(PresentMode mode) {
    switch (mode) {
        case PresentMode::Mailbox:     return vk::PresentModeKHR::eMailbox;
        case PresentMode::Fifo:        return vk::PresentModeKHR::eFifo;
        case PresentMode::Immediate:   return vk::PresentModeKHR::eImmediate;
        case PresentMode::FifoRelaxed: return vk::PresentModeKHR::eFifoRelaxed;
        default:                       return vk::PresentModeKHR::eFifo;
    }
}

} // anonymous namespace

bool VulkanSwapchain::Initialize(const VulkanInstance& instance, const VulkanDevice& device, std::uint32_t preferred_image_count, PresentMode present_mode) {
    try {
        const auto& vk_surface = instance.GetSurface();
        const auto& vk_physical_device = device.GetPhysicalDevice();
        const auto& vk_device = device.GetDevice();
        SDL_Window* window = instance.GetWindow();

        const auto capabilities = vk_physical_device.getSurfaceCapabilitiesKHR(*vk_surface);
        const auto formats = vk_physical_device.getSurfaceFormatsKHR(*vk_surface);
        surface_format_ = formats.empty() ? vk::SurfaceFormatKHR{} : formats.front();

        const auto present_modes = vk_physical_device.getSurfacePresentModesKHR(*vk_surface);
        vk::PresentModeKHR vk_present_mode = ToVkPresentMode(present_mode);
        if (std::ranges::find(present_modes, vk_present_mode) == present_modes.end()) {
            vk_present_mode = vk::PresentModeKHR::eFifo;
        }

        if (capabilities.currentExtent.width == UINT32_MAX || capabilities.currentExtent.height == UINT32_MAX) {
            int drawable_w = 0, drawable_h = 0;
            SDL_GetWindowSizeInPixels(window, &drawable_w, &drawable_h);
            swapchain_extent_.width = std::clamp(static_cast<std::uint32_t>(drawable_w), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            swapchain_extent_.height = std::clamp(static_cast<std::uint32_t>(drawable_h), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        } else {
            swapchain_extent_ = capabilities.currentExtent;
        }

        std::uint32_t image_count = std::max(preferred_image_count, capabilities.minImageCount);
        if (capabilities.maxImageCount > 0) {
            image_count = std::min(image_count, capabilities.maxImageCount);
        }

        vk::SwapchainCreateInfoKHR swap_info{};
        swap_info.surface = *vk_surface;
        swap_info.minImageCount = image_count;
        swap_info.imageFormat = surface_format_.format;
        swap_info.imageColorSpace = surface_format_.colorSpace;
        swap_info.imageExtent = swapchain_extent_;
        swap_info.imageArrayLayers = 1;
        swap_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                               vk::ImageUsageFlagBits::eSampled;
        swap_info.imageSharingMode = vk::SharingMode::eExclusive;
        swap_info.preTransform = capabilities.currentTransform;
        swap_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        swap_info.presentMode = vk_present_mode;
        swap_info.clipped = vk::True;

        swapchain_ = std::make_unique<vk::raii::SwapchainKHR>(vk_device, swap_info);
        VulkanBackend::Vulkan::SetVulkanObjectName(vk_device, *swapchain_, "swapchain");
        swapchain_images_ = swapchain_->getImages();
        swapchain_image_initialized_.assign(swapchain_images_.size(), false);
        swapchain_image_count_ = static_cast<std::uint32_t>(swapchain_images_.size());

        if (!RebuildSwapchainViews(vk_device)) return false;
        return CreateDepthResources(vk_physical_device, vk_device);
    } catch (const std::exception& ex) {
        LOGIFACE_LOG(error, std::string("initialization failed: ") + ex.what());
        return false;
    }
}

bool VulkanSwapchain::RebuildSwapchainViews(const vk::raii::Device& device) {
    try {
        swapchain_image_views_.clear();
        std::uint32_t view_idx = 0;
        for (const auto& image : swapchain_images_) {
            vk::ImageViewCreateInfo view_info{};
            view_info.image = image;
            view_info.viewType = vk::ImageViewType::e2D;
            view_info.format = surface_format_.format;
            view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            swapchain_image_views_.emplace_back(device, view_info);
            VulkanBackend::Vulkan::SetVulkanObjectName(device, swapchain_image_views_.back(), "swapchain-image-view-" + std::to_string(view_idx++));
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool VulkanSwapchain::CreateDepthResources(const vk::raii::PhysicalDevice& physical_device, const vk::raii::Device& device) {
    try {
        const std::uint32_t count = swapchain_image_count_;

        depth_images_.clear();
        depth_image_memories_.clear();
        depth_image_views_.clear();
        depth_images_.reserve(count);
        depth_image_memories_.reserve(count);
        depth_image_views_.reserve(count);

        vk::ImageCreateInfo image_info{};
        image_info.imageType = vk::ImageType::e2D;
        image_info.extent = vk::Extent3D{swapchain_extent_.width, swapchain_extent_.height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = depth_format_;
        image_info.tiling = vk::ImageTiling::eOptimal;
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
        image_info.samples = vk::SampleCountFlagBits::e1;
        image_info.sharingMode = vk::SharingMode::eExclusive;

        for (std::uint32_t i = 0; i < count; ++i) {
            depth_images_.emplace_back(device, image_info);
            VulkanBackend::Vulkan::SetVulkanObjectName(device, depth_images_.back(), "depth-image-" + std::to_string(i));

            const auto requirements = depth_images_.back().getMemoryRequirements();
            vk::MemoryAllocateInfo alloc_info{};
            alloc_info.allocationSize = requirements.size;

            const auto mem_properties = physical_device.getMemoryProperties();
            std::uint32_t memory_type_index = UINT32_MAX;
            for (std::uint32_t j = 0; j < mem_properties.memoryTypeCount; ++j) {
                if ((requirements.memoryTypeBits & (1u << j)) && (mem_properties.memoryTypes[j].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                    memory_type_index = j;
                    break;
                }
            }
            if (memory_type_index == UINT32_MAX) return false;

            alloc_info.memoryTypeIndex = memory_type_index;
            depth_image_memories_.emplace_back(device, alloc_info);
            VulkanBackend::Vulkan::SetVulkanObjectName(device, depth_image_memories_.back(), "depth-memory-" + std::to_string(i));
            depth_images_.back().bindMemory(depth_image_memories_.back(), 0);

            vk::ImageViewCreateInfo view_info{};
            view_info.image = *depth_images_.back();
            view_info.viewType = vk::ImageViewType::e2D;
            view_info.format = depth_format_;
            view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;

            depth_image_views_.emplace_back(device, view_info);
            VulkanBackend::Vulkan::SetVulkanObjectName(device, depth_image_views_.back(), "depth-image-view-" + std::to_string(i));
        }
        return true;
    } catch (...) {
        return false;
    }
}

void VulkanSwapchain::Shutdown() {
    depth_image_views_.clear();
    depth_image_memories_.clear();
    depth_images_.clear();
    swapchain_image_views_.clear();
    swapchain_images_.clear();
    swapchain_.reset();
}

bool VulkanSwapchain::Recreate(const VulkanInstance& instance, const VulkanDevice& device, std::uint32_t preferred_image_count, PresentMode present_mode) {
    Shutdown();
    return Initialize(instance, device, preferred_image_count, present_mode);
}

} // namespace VulkanBackend::Vulkan
