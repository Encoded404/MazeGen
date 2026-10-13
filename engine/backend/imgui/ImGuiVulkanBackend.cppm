module;

struct SDL_Window;

export module VulkanBackend.ImGui;

import std;

import vulkan;


export namespace VulkanBackend::ImGui {

struct ImGuiBackendConfig {
    std::uint32_t image_count = 3;
    vk::Format swapchain_format = vk::Format::eUndefined;
    std::uint32_t min_allocation_size = 1024 * 1024;
};

class IImGuiBackend {
public:
    virtual ~IImGuiBackend() = default;

    [[nodiscard]] virtual bool Initialize(SDL_Window* window, const ImGuiBackendConfig& config,
                                          vk::Instance instance, vk::PhysicalDevice physical_device,
                                          vk::Device device, std::uint32_t queue_family, vk::Queue queue,
                                          std::uint32_t api_version) = 0;
    virtual void Shutdown() = 0;
    virtual void NewFrame() = 0;
    virtual void ProcessSDLEvent(void* sdl_event) = 0;
    virtual void RenderDrawData(vk::CommandBuffer command_buffer, vk::ImageView color_attachment,
                                vk::Format render_target_format, std::uint32_t width, std::uint32_t height) = 0;
    virtual void OnResize() = 0;
    virtual void OnSwapchainRecreated(std::uint32_t new_image_count, vk::Format new_format) = 0;
};

[[nodiscard]] std::shared_ptr<IImGuiBackend> CreateImGuiBackend();

}
