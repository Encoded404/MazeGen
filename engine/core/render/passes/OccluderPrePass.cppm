module;

export module VulkanEngine.Render.Passes.OccluderPrePass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class OccluderPrePass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    OccluderPrePass(SceneRenderer& sr, vk::ClearDepthStencilValue clear_depth);
    ~OccluderPrePass() override;

    OccluderPrePass(const OccluderPrePass&) = delete;
    OccluderPrePass& operator=(const OccluderPrePass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "occluder-prepass"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
    vk::ClearDepthStencilValue clear_depth_{1.0f, 0};
};

} // namespace VulkanEngine::SceneRenderer
