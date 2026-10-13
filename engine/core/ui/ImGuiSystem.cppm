module;

export module VulkanEngine.ImGui;

import std;

import vulkan;

export import VulkanBackend.ImGui;
import VulkanShared.CallbackList;

#ifndef UINT32_MAX
constexpr std::uint32_t UINT32_MAX =
    std::numeric_limits<std::uint32_t>::max();
#endif

export namespace VulkanEngine::ImGui {

struct ImGuiConfig {
    bool enabled = true;
    bool show_metrics = false;
    bool show_style_editor = false;
    float font_size = 16.0f;
    std::string ini_file_path = "";
};

struct ImGuiSystemInitInfo {
    void* sdl_window = nullptr;
    ImGuiConfig config{};
    VulkanBackend::ImGui::ImGuiBackendConfig backend_config{};
    vk::Instance instance{};
    vk::PhysicalDevice physical_device{};
    vk::Device device{};
    std::uint32_t queue_family = UINT32_MAX;
    vk::Queue queue{};
    std::uint32_t api_version = vk::ApiVersion13;
};

class ImGuiSystem {
public:
    explicit ImGuiSystem(std::shared_ptr<VulkanBackend::ImGui::IImGuiBackend> backend);

    [[nodiscard]] bool Initialize(const ImGuiSystemInitInfo& init_info);
    void Shutdown();

    void NewFrame();
    void ProcessSDLEvent(void* sdl_event);

    VulkanShared::CallbackList<void()> draw_callbacks; // NOLINT(misc-non-private-member-variables-in-classes)

    void RenderDrawData(vk::CommandBuffer command_buffer, vk::ImageView color_attachment, std::uint32_t width, std::uint32_t height);
    void OnSwapchainRecreated(std::uint32_t new_image_count, vk::Format new_format);

    [[nodiscard]] const ImGuiConfig& GetConfig() const { return config_; }
    void SetConfig(const ImGuiConfig& config) { config_ = config; }

    [[nodiscard]] bool IsInitialized() const { return initialized_; }

private:
    std::shared_ptr<VulkanBackend::ImGui::IImGuiBackend> backend_;
    ImGuiConfig config_{};
    VulkanBackend::ImGui::ImGuiBackendConfig backend_config_{};
    bool initialized_ = false;
};

}
