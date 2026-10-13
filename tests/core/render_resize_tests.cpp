#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.RenderPipeline;

namespace {

using namespace VulkanEngine::PipelinePass;
using VulkanEngine::RenderPipeline::RenderPipeline;

TEST(TransientResizeTest, ResolveTransientExtentScalesOrPassesThrough) {
    TransientImageDesc absolute{};
    absolute.width = 320;
    absolute.height = 200;
    auto [width, height] = ResolveTransientExtent(absolute, 1280, 720);
    EXPECT_EQ(width, 320u);
    EXPECT_EQ(height, 200u);

    TransientImageDesc half{};
    half.width_scale = 0.5f;
    half.height_scale = 0.5f;
    std::tie(width, height) = ResolveTransientExtent(half, 1280, 720);
    EXPECT_EQ(width, 640u);
    EXPECT_EQ(height, 360u);

    // Never collapses to zero, even for a sub-pixel scale.
    TransientImageDesc tiny{};
    tiny.width_scale = 0.001f;
    tiny.height_scale = 0.001f;
    std::tie(width, height) = ResolveTransientExtent(tiny, 4, 4);
    EXPECT_EQ(width, 1u);
    EXPECT_EQ(height, 1u);

    // Mixed: absolute width, relative height.
    TransientImageDesc mixed{};
    mixed.width = 64;
    mixed.height_scale = 0.25f;
    std::tie(width, height) = ResolveTransientExtent(mixed, 800, 600);
    EXPECT_EQ(width, 64u);
    EXPECT_EQ(height, 150u);
}

class ResizeRecordingPass final : public IPipelinePass {
public:
    explicit ResizeRecordingPass(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] std::string_view GetName() const override { return name_; }
    void Setup(PassSetupContext&) override {}
    void Execute(const FrameContext&, vk::CommandBuffer) override {}
    void OnRenderResize(std::uint32_t width, std::uint32_t height) override {
        ++calls;
        last_width = width;
        last_height = height;
    }

    int calls = 0;
    std::uint32_t last_width = 0;
    std::uint32_t last_height = 0;

private:
    std::string name_;
};

// A resize reallocates graph-owned transients and notifies app passes, but must
// not recompile the plan.
TEST(TransientResizeTest, ResizeNotifiesOnceAndDoesNotRecompile) {
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(800, 600);

    auto pass = std::make_unique<ResizeRecordingPass>("resize-recorder");
    auto* raw = pass.get();
    ASSERT_TRUE(pipeline.RegisterPass(std::move(pass)).has_value());
    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());
    raw->calls = 0; // ignore the notification queued by the initial extent

    const std::uint32_t revision = pipeline.GetRevision();

    // An unchanged extent is a no-op.
    pipeline.SetRenderExtent(800, 600);
    pipeline.ApplyChanges();
    EXPECT_EQ(raw->calls, 0);
    EXPECT_EQ(pipeline.GetRevision(), revision);

    // A changed extent queues exactly one notification, delivered at the frame
    // boundary; the compiled plan is untouched.
    pipeline.SetRenderExtent(1024, 768);
    EXPECT_EQ(raw->calls, 0);
    pipeline.ApplyChanges();
    EXPECT_EQ(raw->calls, 1);
    EXPECT_EQ(raw->last_width, 1024u);
    EXPECT_EQ(raw->last_height, 768u);
    EXPECT_EQ(pipeline.GetRevision(), revision);

    pipeline.ApplyChanges();
    EXPECT_EQ(raw->calls, 1);
}

TEST(TransientResizeTest, SwapchainRecreationKeepsPlanAndClearsState) {
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(640, 480);
    auto pass = std::make_unique<ResizeRecordingPass>("recreate-recorder");
    ASSERT_TRUE(pipeline.RegisterPass(std::move(pass)).has_value());
    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());
    const std::uint32_t revision = pipeline.GetRevision();

    // Recreating with a different image count must not recompile, and must not
    // disturb the resize notification queue.
    pipeline.OnSwapchainRecreated(4);
    EXPECT_TRUE(pipeline.IsCompiled());
    EXPECT_EQ(pipeline.GetRevision(), revision);

    pipeline.SetRenderExtent(1280, 720);
    pipeline.ApplyChanges();
    EXPECT_EQ(pipeline.GetRevision(), revision);
}

} // namespace
