module;

export module VulkanEngine.Render.Passes.HiZPass;

import std;
import vulkan;
import VulkanEngine.PipelinePass;
import VulkanEngine.SceneRenderer;

export namespace VulkanEngine::SceneRenderer {

class HiZPass : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    HiZPass(SceneRenderer& sr, std::string name);
    ~HiZPass() override;

    HiZPass(const HiZPass&) = delete;
    HiZPass& operator=(const HiZPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return name_; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

private:
    SceneRenderer& scene_renderer_;
    std::string name_;
};

} // namespace VulkanEngine::SceneRenderer
