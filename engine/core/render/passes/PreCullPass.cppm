module;

export module VulkanEngine.Render.Passes.PreCullPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class PreCullPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    PreCullPass(SceneRenderer& sr);
    ~PreCullPass() override;

    PreCullPass(const PreCullPass&) = delete;
    PreCullPass& operator=(const PreCullPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "pre-cull"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
};

} // namespace VulkanEngine::SceneRenderer
