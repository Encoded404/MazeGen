module;

export module VulkanEngine.Render.Passes.ExpandPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class ExpandPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    ExpandPass(SceneRenderer& sr);
    ~ExpandPass() override;

    ExpandPass(const ExpandPass&) = delete;
    ExpandPass& operator=(const ExpandPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "expand"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
};

} // namespace VulkanEngine::SceneRenderer
