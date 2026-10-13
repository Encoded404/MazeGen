module;

#include <vulkan/vulkan.h>
// VULKAN_HPP_DEFAULT_DISPATCHER is a macro, which an `import vulkan;` cannot
// bring across the module boundary, so it must come from the macros header.
#include <vulkan/vulkan_hpp_macros.hpp>

module TestSupport.HeadlessVulkanBackend;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.CommonTypes;

namespace TestSupport {

namespace {

constexpr std::uint32_t kTestFramesInFlight = 1;

} // namespace

bool HeadlessVulkanBackend::Initialize() {
    if (device_ != nullptr) {
        return true;
    }
    return CreateInstance({}) && SelectPhysicalDevice() && CreateLogicalDevice(kTestFramesInFlight);
}

bool HeadlessVulkanBackend::CreateInstance(const VulkanBackend::Vulkan::VulkanBootstrapConfig&) {
    if (instance_ != nullptr) {
        return true;
    }
    try {
        loader_ = std::make_unique<vk::detail::DynamicLoader>();
        VULKAN_HPP_DEFAULT_DISPATCHER.init(loader_->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
        context_ = std::make_unique<vk::raii::Context>();
        vk::ApplicationInfo app_info("VulkanEngineV5-headless-tests", 1,
                                     "VulkanEngineV5-headless-tests", 1,
                                     vk::ApiVersion13);
        vk::InstanceCreateInfo info({}, &app_info);
        instance_ = std::make_unique<vk::raii::Instance>(*context_, info);
        // Instance-level and device-level entry points resolve only now.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(**instance_);
        return true;
    } catch (const std::exception& err) {
        error_message_ = err.what();
        return false;
    }
}

bool HeadlessVulkanBackend::SelectPhysicalDevice() {
    if (physical_device_ != nullptr) {
        return true;
    }
    if (instance_ == nullptr) {
        return false;
    }
    try {
        auto devices = instance_->enumeratePhysicalDevices();
        for (auto& candidate : devices) {
            const auto families = candidate.getQueueFamilyProperties();
            for (std::uint32_t i = 0; i < families.size(); ++i) {
                if ((families[i].queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{}) {
                    physical_device_ = std::make_unique<vk::raii::PhysicalDevice>(std::move(candidate));
                    graphics_family_ = i;
                    break;
                }
            }
            if (physical_device_ != nullptr) {
                break;
            }
        }
        if (physical_device_ == nullptr) {
            return false;
        }

        // Capability snapshot through the production builder path.
        const auto props2 = physical_device_->getProperties2<vk::PhysicalDeviceProperties2,
                                                              vk::PhysicalDeviceDescriptorIndexingProperties>();
        VulkanBackend::Vulkan::SupportedDeviceState supported{};
        supported.properties = props2.get<vk::PhysicalDeviceProperties2>().properties;
        supported.descriptor_indexing = props2.get<vk::PhysicalDeviceDescriptorIndexingProperties>();
        supported.format_supports.reserve(VulkanBackend::Vulkan::kCandidateFormats.size());
        for (const vk::Format format : VulkanBackend::Vulkan::kCandidateFormats) {
            VkFormatProperties3 props3{};
            props3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
            VkFormatProperties2 props2_raw{};
            props2_raw.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
            props2_raw.pNext = &props3;
            physical_device_->getDispatcher()->vkGetPhysicalDeviceFormatProperties2(
                static_cast<VkPhysicalDevice>(**physical_device_),
                static_cast<VkFormat>(format),
                &props2_raw);
            supported.format_supports.push_back(VulkanBackend::Vulkan::FormatSupport{
                .format = format,
                .linear_features = vk::FormatFeatureFlags2{props3.linearTilingFeatures},
                .optimal_features = vk::FormatFeatureFlags2{props3.optimalTilingFeatures},
            });
        }

        VulkanBackend::Vulkan::VulkanCapabilitiesBuilder builder{capabilities_, supported};

        // Query and chain the full features2 chain, mirroring
        // VulkanDevice::SelectPhysicalDevice, so SetFeatureRequest +
        // FinalizeFeatures produce the same create-time feature chain the
        // engine device carries (descriptor indexing + update-after-bind for
        // the bindless tests).
        supported.features2.pNext = &supported.vulkan11;
        supported.vulkan11.pNext = &supported.vulkan12;
        supported.vulkan12.pNext = &supported.vulkan13;
        supported.vulkan13.pNext = nullptr;
        physical_device_->getDispatcher()->vkGetPhysicalDeviceFeatures2(
            static_cast<VkPhysicalDevice>(**physical_device_),
            reinterpret_cast<VkPhysicalDeviceFeatures2*>(&supported.features2));

        for (std::size_t i = 0;
             i < static_cast<std::size_t>(VulkanBackend::Vulkan::Feature::Count); ++i) {
            const auto feature = static_cast<VulkanBackend::Vulkan::Feature>(i);
            builder.SetFeatureRequest(feature, /*required=*/false);
        }

        builder.SetProperties(supported.properties);
        builder.SetDescriptorIndexingProperties(supported.descriptor_indexing);
        builder.SetFormatSupports(supported.format_supports);
        builder.FinalizeDescriptorCapabilities(VulkanBackend::Vulkan::kBindlessAppSampledReserve);
        builder.FinalizeFeatures();
        return true;
    } catch (const std::exception& err) {
        error_message_ = err.what();
        return false;
    }
}

bool HeadlessVulkanBackend::CreateLogicalDevice(std::uint32_t frames_in_flight) {
    if (device_ != nullptr) {
        return true;
    }
    if (physical_device_ == nullptr) {
        return false;
    }
    frames_in_flight_ = std::max<std::uint32_t>(frames_in_flight, 1U);
    try {
        const float priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_info({}, graphics_family_, 1, &priority);
        // Chain the production feature snapshot so GPU tests exercise the same
        // device features the engine runs with (descriptor indexing +
        // update-after-bind for the bindless tests, compression catalogs, ...).
        // VulkanDevice::SelectPhysicalDevice filled capabilities_ through the
        // production builder, whose FinalizeFeatures chained the create-time
        // feature structs inside the snapshot.
        vk::DeviceCreateInfo device_info({}, queue_info);
        device_info.pNext = &capabilities_.GetFeatureChain();
        device_ = std::make_unique<vk::raii::Device>(*physical_device_, device_info);

        graphics_queue_ = std::make_unique<vk::raii::Queue>(device_->getQueue(graphics_family_, 0));
        command_pool_ = std::make_unique<vk::raii::CommandPool>(
            *device_, vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphics_family_});

        command_buffers_.reserve(frames_in_flight_);
        for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
            command_buffers_.emplace_back(std::move(
                device_->allocateCommandBuffers(
                    vk::CommandBufferAllocateInfo{**command_pool_, vk::CommandBufferLevel::ePrimary, 1})[0]));
        }
        semaphores_.emplace_back(*device_, vk::SemaphoreCreateInfo{});
        fence_ = std::make_unique<vk::raii::Fence>(*device_, vk::FenceCreateInfo{});

        queue_families_.clear();
        queue_families_.push_back(graphics_family_);
        return true;
    } catch (const std::exception& err) {
        error_message_ = err.what();
        return false;
    }
}

bool HeadlessVulkanBackend::CreateSwapchain(std::uint32_t, VulkanBackend::Vulkan::PresentMode, std::uint32_t&) {
    return false;
}

bool HeadlessVulkanBackend::GetSwapchainExtent(std::uint32_t&, std::uint32_t&) const { return false; }

void HeadlessVulkanBackend::Shutdown() {
    command_buffers_.clear();
    semaphores_.clear();
    fence_.reset();
    command_pool_.reset();
    graphics_queue_.reset();
    device_.reset();
    physical_device_.reset();
    instance_.reset();
    context_.reset();
    queue_families_.clear();
}

const vk::raii::SwapchainKHR& HeadlessVulkanBackend::GetSwapchain() const {
    throw std::logic_error("HeadlessVulkanBackend has no swapchain");
}
const std::vector<vk::Image>& HeadlessVulkanBackend::GetSwapchainImages() const {
    throw std::logic_error("HeadlessVulkanBackend has no swapchain");
}
const std::vector<vk::raii::ImageView>& HeadlessVulkanBackend::GetSwapchainImageViews() const {
    throw std::logic_error("HeadlessVulkanBackend has no swapchain");
}
std::vector<bool>& HeadlessVulkanBackend::GetSwapchainImageInitializedFlags() {
    throw std::logic_error("HeadlessVulkanBackend has no swapchain");
}
const vk::SurfaceFormatKHR& HeadlessVulkanBackend::GetSurfaceFormat() const {
    throw std::logic_error("HeadlessVulkanBackend has no surface");
}
vk::Format HeadlessVulkanBackend::GetDepthFormat() const { return vk::Format::eUndefined; }
const vk::raii::ImageView& HeadlessVulkanBackend::GetDepthImageView(std::uint32_t) const {
    throw std::logic_error("HeadlessVulkanBackend has no depth image");
}
const vk::raii::Image& HeadlessVulkanBackend::GetDepthImage(std::uint32_t) const {
    throw std::logic_error("HeadlessVulkanBackend has no depth image");
}
bool HeadlessVulkanBackend::AcquireNextImage(std::uint32_t, std::uint32_t&) { return false; }
bool HeadlessVulkanBackend::SubmitFrame(std::uint32_t, std::uint32_t, bool) { return false; }
bool HeadlessVulkanBackend::Present(std::uint32_t) { return false; }
bool HeadlessVulkanBackend::IsFrameComplete(std::uint32_t) { return false; }

void HeadlessVulkanBackend::WaitDeviceIdle() {
    if (device_) {
        device_->waitIdle();
    }
}

} // namespace TestSupport
