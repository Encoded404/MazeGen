module;

#include <SDL3/SDL_video.h>
#include <imgui.h>

module VulkanEngine.ImGui;

import std;
import std.compat;

import vulkan;

namespace VulkanEngine::ImGui {

ImGuiSystem::ImGuiSystem(std::shared_ptr<VulkanBackend::ImGui::IImGuiBackend> backend)
    : backend_(std::move(backend)) {}

[[nodiscard]] bool ImGuiSystem::Initialize(const ImGuiSystemInitInfo& init_info) {
    config_ = init_info.config;
    backend_config_ = init_info.backend_config;

    if (!backend_->Initialize(static_cast<SDL_Window*>(init_info.sdl_window),
                              backend_config_,
                              init_info.instance,
                              init_info.physical_device,
                              init_info.device,
                              init_info.queue_family,
                              init_info.queue,
                              init_info.api_version)) {
        return false;
    }

    initialized_ = true;
    return true;
}

void ImGuiSystem::Shutdown() {
    if (initialized_) {
        backend_->Shutdown();
        initialized_ = false;
    }
}

void ImGuiSystem::NewFrame() {
    if (!initialized_) {
        return;
    }
    backend_->NewFrame();
}

void ImGuiSystem::ProcessSDLEvent(void* sdl_event) {
    if (!initialized_) {
        return;
    }
    backend_->ProcessSDLEvent(sdl_event);
}

void ImGuiSystem::OnSwapchainRecreated(std::uint32_t new_image_count, vk::Format new_format) {
    if (!initialized_) return;
    backend_config_.image_count = new_image_count;
    backend_config_.swapchain_format = new_format;
    backend_->OnSwapchainRecreated(new_image_count, new_format);
}

void ImGuiSystem::RenderDrawData(vk::CommandBuffer command_buffer, vk::ImageView color_attachment, std::uint32_t width, std::uint32_t height) {
    if (!initialized_) {
        return;
    }

    draw_callbacks.Call();

    if (config_.show_metrics) {
        ::ImGui::ShowMetricsWindow(&config_.show_metrics);
    }
    if (config_.show_style_editor) {
        ::ImGui::ShowStyleEditor();
    }

    backend_->RenderDrawData(command_buffer, color_attachment, backend_config_.swapchain_format, width, height);
}

}
