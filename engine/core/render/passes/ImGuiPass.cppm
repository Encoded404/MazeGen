module;

export module VulkanEngine.Render.Passes.ImGuiPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanBackend.Vulkan.VulkanBootstrap;

export namespace VulkanEngine::SceneRenderer {

class ImGuiPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    explicit ImGuiPass(VulkanBackend::Vulkan::VulkanBootstrap* bootstrap);
    ~ImGuiPass() override;

    ImGuiPass(const ImGuiPass&) = delete;
    ImGuiPass& operator=(const ImGuiPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "imgui-overlay"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    VulkanBackend::Vulkan::VulkanBootstrap* bootstrap_ = nullptr;
};

} // namespace VulkanEngine::SceneRenderer
