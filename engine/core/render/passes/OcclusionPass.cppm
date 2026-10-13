module;

export module VulkanEngine.Render.Passes.OcclusionPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class OcclusionPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    OcclusionPass(SceneRenderer& sr);
    ~OcclusionPass() override;

    OcclusionPass(const OcclusionPass&) = delete;
    OcclusionPass& operator=(const OcclusionPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "occlusion"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
};

} // namespace VulkanEngine::SceneRenderer
