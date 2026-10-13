module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.Render.Passes.MainPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class MainPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    MainPass(SceneRenderer& sr, glm::vec4 clear_color);
    ~MainPass() override;

    MainPass(const MainPass&) = delete;
    MainPass& operator=(const MainPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "main-pass"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
    glm::vec4 clear_color_{0.0f};
};

} // namespace VulkanEngine::SceneRenderer
