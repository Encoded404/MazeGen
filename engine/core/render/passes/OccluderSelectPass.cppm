module;

export module VulkanEngine.Render.Passes.OccluderSelectPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class OccluderSelectPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    OccluderSelectPass(SceneRenderer& sr);
    ~OccluderSelectPass() override;

    OccluderSelectPass(const OccluderSelectPass&) = delete;
    OccluderSelectPass& operator=(const OccluderSelectPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "occluder-select"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
};

} // namespace VulkanEngine::SceneRenderer
