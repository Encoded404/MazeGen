module;

#include <SDL3/SDL_video.h>

export module VulkanBackend.Vulkan.VulkanInstance;

import std;

import vulkan;

import VulkanBackend.Vulkan.CommonTypes;
import VulkanBackend.Vulkan.VulkanCapabilities;

export namespace VulkanBackend::Vulkan {

class VulkanInstance {
public:
    [[nodiscard]] bool Initialize(const VulkanBootstrapConfig& config);
    void Shutdown();

    [[nodiscard]] const vk::raii::Instance& GetInstance() const { return *instance_; }
    [[nodiscard]] const vk::raii::SurfaceKHR& GetSurface() const { return *surface_; }
    [[nodiscard]] SDL_Window* GetWindow() const { return window_; }
    [[nodiscard]] const VulkanInstanceCapabilities& GetCapabilities() const { return capabilities_; }

private:
    SDL_Window* window_ = nullptr;
    std::unique_ptr<vk::detail::DynamicLoader> loader_{};
    std::unique_ptr<vk::raii::Instance> instance_{};
    // Validation/debug messages sink. Created only when VK_EXT_debug_utils is
    // enabled; without it the validation layer has nowhere to report, which is
    // why a Debug build can be silent about invalid create calls.
    std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debug_messenger_{};
    std::unique_ptr<vk::raii::SurfaceKHR> surface_{};
    VulkanInstanceCapabilities capabilities_{};
};

} // namespace VulkanBackend::Vulkan
