module;

#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#include <SDL3/SDL_events.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <logging/logging_macros.hpp>

module VulkanBackend.ImGui;

import std;
import std.compat;

import logiface;

import vulkan;

namespace VulkanBackend::ImGui {

class VulkanImGuiBackend : public IImGuiBackend {
public:
    ~VulkanImGuiBackend() override = default;

    [[nodiscard]] bool Initialize(SDL_Window* window, const ImGuiBackendConfig& config,
                                  vk::Instance instance, vk::PhysicalDevice physical_device,
                                  vk::Device device, std::uint32_t queue_family, vk::Queue queue,
                                  std::uint32_t api_version) override {
        config_ = config;
        device_ = device;

        ::ImGui::DebugCheckVersionAndDataLayout(IMGUI_VERSION, sizeof(ImGuiIO), sizeof(ImGuiStyle), sizeof(ImVec2), sizeof(ImVec4), sizeof(ImDrawVert), sizeof(ImDrawIdx));
        ::ImGui::CreateContext();
        ImGuiIO& io = ::ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.IniFilename = nullptr;

        ::ImGui::StyleColorsDark();

        if (!ImGui_ImplSDL3_InitForVulkan(window)) {
            return false;
        }

        // imgui_impl_vulkan 1.92.8+ allocates VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
        // sets (>= IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE per atlas)
        // plus two internal sampler sets. Let the backend create its own pool,
        // sized by the minimums it enforces, instead of hand-rolling one here.
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physical_device;
        init_info.Device = device;
        init_info.QueueFamily = queue_family;
        init_info.Queue = queue;
        init_info.DescriptorPool = VK_NULL_HANDLE;
        init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
        init_info.MinImageCount = config.image_count;
        init_info.ImageCount = config.image_count;
        init_info.UseDynamicRendering = true;
        init_info.ApiVersion = api_version;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        const vk::Format format = config.swapchain_format;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = reinterpret_cast<const VkFormat*>(&format);
        init_info.CheckVkResultFn = [](VkResult err) {
            if (err < 0) {
                LOGIFACE_LOG(warn, "An error occurred when initializing ImGui. Error code: " + std::to_string(static_cast<std::uint32_t>(err)));
            }
        };

        if (!ImGui_ImplVulkan_Init(&init_info)) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    void Shutdown() override {
        if (initialized_) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ::ImGui::DestroyContext();
            initialized_ = false;
        }
    }

    void NewFrame() override {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ::ImGui::NewFrame();
    }

    void ProcessSDLEvent(void* sdl_event) override {
        ImGui_ImplSDL3_ProcessEvent(static_cast<SDL_Event*>(sdl_event));
    }

    void RenderDrawData(vk::CommandBuffer command_buffer, vk::ImageView color_attachment,
                        vk::Format /*render_target_format*/, std::uint32_t width, std::uint32_t height) override {
        ::ImGui::Render();

        ImDrawData* draw_data = ::ImGui::GetDrawData();
        if (draw_data == nullptr || draw_data->TotalVtxCount == 0) {
            return;
        }

        vk::RenderingAttachmentInfo color_attach{};
        color_attach.imageView = color_attachment;
        color_attach.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color_attach.loadOp = vk::AttachmentLoadOp::eLoad;
        color_attach.storeOp = vk::AttachmentStoreOp::eStore;

        vk::RenderingInfo render_info{};
        render_info.renderArea = vk::Rect2D({0, 0}, {width, height});
        render_info.layerCount = 1;
        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &color_attach;

        command_buffer.beginRendering(render_info);
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
        command_buffer.endRendering();
    }

    void OnResize() override {
        ImGui_ImplVulkan_SetMinImageCount(config_.image_count);
    }

    void OnSwapchainRecreated(std::uint32_t new_image_count, vk::Format new_format) override {
        config_.image_count = new_image_count;
        config_.swapchain_format = new_format;
        ImGui_ImplVulkan_SetMinImageCount(config_.image_count);

        // imgui 1.92.1 (#8110/#8111): recreate the main pipeline against the
        // new swapchain format without a full backend shutdown/reinit. The
        // backend created the original pipeline from the same dynamic-rendering
        // info during Init; re-supply it with the new format.
        ImGui_ImplVulkan_PipelineInfo pipeline_info{};
        pipeline_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        pipeline_info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipeline_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        const vk::Format format = new_format;
        pipeline_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = reinterpret_cast<const VkFormat*>(&format);
        ImGui_ImplVulkan_CreateMainPipeline(&pipeline_info);
    }

private:
    ImGuiBackendConfig config_{};
    vk::Device device_{};
    bool initialized_ = false;
};

[[nodiscard]] std::shared_ptr<IImGuiBackend> CreateImGuiBackend() {
    return std::make_shared<VulkanImGuiBackend>();
}

}
