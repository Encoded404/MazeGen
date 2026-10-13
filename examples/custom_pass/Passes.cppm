module;

export module Examples.CustomPass.Passes;

import std;
import vulkan;
import VulkanEngine.PipelinePass;

export namespace Examples::CustomPass {

using VulkanEngine::PipelinePass::FrameContext;
using VulkanEngine::PipelinePass::IPipelinePass;
using VulkanEngine::PipelinePass::PassSetupContext;
using VulkanEngine::PipelinePass::TransientBufferDesc;
using VulkanEngine::PipelinePass::TransientImageDesc;
using VulkanEngine::PipelinePass::BuiltinPass;
using VulkanEngine::Render::DescriptorDecl;
using VulkanEngine::Render::DescriptorKind;
using VulkanEngine::RenderGraph::AccessIntent;
using VulkanEngine::RenderGraph::PipelineStageIntent;

inline constexpr std::uint32_t kAppSet = 5;

// Scene colour the capture pass writes and the exposure/tonemap passes read.
// Relative to the render extent so it follows resizes.
[[nodiscard]] inline TransientImageDesc SceneColorDesc() {
    return TransientImageDesc{
        .name = "custom-scene-color",
        .format = vk::Format::eR8G8B8A8Unorm,
        .width_scale = 1.0f,
        .height_scale = 1.0f,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
    };
}

[[nodiscard]] inline TransientBufferDesc ExposureBufferDesc() {
    return TransientBufferDesc{
        .name = "custom-exposure",
        .size = sizeof(float),
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .memory_properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
    };
}

// Samples the backbuffer into the transient scene-colour image.
class CapturePass final : public IPipelinePass {
public:
    CapturePass(std::uint64_t vertex_shader, std::uint64_t fragment_shader);
    ~CapturePass() override;

    CapturePass(const CapturePass&) = delete;
    CapturePass& operator=(const CapturePass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "custom-capture"; }

    void Setup(PassSetupContext& ctx) override;
    void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) override;

private:
    std::uint64_t vertex_shader_;
    std::uint64_t fragment_shader_;
};

// Estimates exposure into a transient storage buffer.
class ExposurePass final : public IPipelinePass {
public:
    explicit ExposurePass(std::uint64_t compute_shader, bool use_async_queue);
    ~ExposurePass() override;

    ExposurePass(const ExposurePass&) = delete;
    ExposurePass& operator=(const ExposurePass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "custom-exposure-pass"; }

    void Setup(PassSetupContext& ctx) override;
    void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) override;

private:
    std::uint64_t compute_shader_;
    bool use_async_queue_ = false;
};

// Tonemaps scene colour with the computed exposure into the backbuffer.
class ToneMapPass final : public IPipelinePass {
public:
    ToneMapPass(std::uint64_t vertex_shader, std::uint64_t fragment_shader);
    ~ToneMapPass() override;

    ToneMapPass(const ToneMapPass&) = delete;
    ToneMapPass& operator=(const ToneMapPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "custom-tonemap"; }

    void Setup(PassSetupContext& ctx) override;
    void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) override;

private:
    std::uint64_t vertex_shader_;
    std::uint64_t fragment_shader_;
};

} // namespace Examples::CustomPass
