#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.RenderPipeline;

namespace {

using namespace VulkanEngine::PipelinePass;
using VulkanEngine::RenderPipeline::PassErrorCode;
using VulkanEngine::RenderPipeline::RenderPipeline;

bool GraphHasPass(const RenderPipeline& pipeline, std::string_view name) {
    for (const auto& pass : pipeline.GetCompiledGraph().passes) {
        if (pass.name == name) {
            return true;
        }
    }
    return false;
}

// A pass with a configurable name that writes a transient image.
class NamedPass final : public IPipelinePass {
public:
    explicit NamedPass(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] std::string_view GetName() const override { return name_; }

    void Setup(PassSetupContext& ctx) override {
        const auto target = ctx.CreateTransientImage(TransientImageDesc{
            .name = name_ + "-target",
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = 64,
            .height = 64,
        });
        ctx.AddWrite(target);
    }

    void Execute(const FrameContext&, vk::CommandBuffer) override {}

private:
    std::string name_;
};

TEST(RenderPipelineRegistrationTest, RegisterAppliesAtBoundaryAndRemoves) {
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(256, 256);

    auto pass = std::make_unique<NamedPass>("app-blur");
    auto handle = pipeline.RegisterPass(std::move(pass));
    ASSERT_TRUE(handle.has_value());

    // Registered but not yet applied: absent from the compiled graph.
    pipeline.ApplyChanges();
    EXPECT_TRUE(GraphHasPass(pipeline, "app-blur"));

    EXPECT_TRUE(pipeline.RemovePass(*handle));
    pipeline.ApplyChanges();
    EXPECT_FALSE(GraphHasPass(pipeline, "app-blur"));
}

TEST(RenderPipelineRegistrationTest, DisablePrunesUntilReenabled) {
    RenderPipeline pipeline;
    auto handle = pipeline.RegisterPass(std::make_unique<NamedPass>("app-fx"));
    ASSERT_TRUE(handle.has_value());
    pipeline.ApplyChanges();
    EXPECT_TRUE(GraphHasPass(pipeline, "app-fx"));

    EXPECT_TRUE(pipeline.SetPassEnabled(*handle, false));
    pipeline.ApplyChanges();
    EXPECT_FALSE(GraphHasPass(pipeline, "app-fx"));

    EXPECT_TRUE(pipeline.SetPassEnabled(*handle, true));
    pipeline.ApplyChanges();
    EXPECT_TRUE(GraphHasPass(pipeline, "app-fx"));
}

TEST(RenderPipelineRegistrationTest, DuplicateNameReturnsStructuredError) {
    RenderPipeline pipeline;
    ASSERT_TRUE(pipeline.RegisterPass(std::make_unique<NamedPass>("dup")).has_value());

    const auto second = pipeline.RegisterPass(std::make_unique<NamedPass>("dup"));
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, PassErrorCode::DuplicateName);
    EXPECT_EQ(second.error().pass, "dup");
}

TEST(RenderPipelineRegistrationTest, MissingResolverReturnsStructuredError) {
    RenderPipeline pipeline;

    class ImportsUnknownPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "imports-unknown"; }
        void Setup(PassSetupContext& ctx) override {
            const auto imported = ctx.ImportImage("app-owned-image");
            ctx.AddRead(imported, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                        VulkanEngine::RenderGraph::AccessIntent::Read);
        }
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    const auto result = pipeline.RegisterPass(std::make_unique<ImportsUnknownPass>());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PassErrorCode::MissingResolver);

    // Providing the resolver lets it register.
    pipeline.RegisterResourceResolver("app-owned-image",
        [](std::uint32_t) { return vk::Image(reinterpret_cast<vk::Image::CType>(0x1)); },
        [](std::uint32_t) { return vk::ImageView(reinterpret_cast<vk::ImageView::CType>(0x2)); },
        vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(pipeline.RegisterPass(std::make_unique<ImportsUnknownPass>()).has_value());
}

TEST(RenderPipelineRegistrationTest, StableHandlesAcrossInterleavedMutation) {
    RenderPipeline pipeline;
    auto a = pipeline.RegisterPass(std::make_unique<NamedPass>("a"));
    auto b = pipeline.RegisterPass(std::make_unique<NamedPass>("b"));
    auto c = pipeline.RegisterPass(std::make_unique<NamedPass>("c"));
    ASSERT_TRUE(a && b && c);
    pipeline.ApplyChanges();

    // Remove the middle pass; the other handles must stay valid.
    EXPECT_TRUE(pipeline.RemovePass(*b));
    pipeline.ApplyChanges();
    EXPECT_FALSE(GraphHasPass(pipeline, "b"));
    EXPECT_TRUE(GraphHasPass(pipeline, "a"));
    EXPECT_TRUE(GraphHasPass(pipeline, "c"));

    // A new pass registered after removal gets a fresh stable slot and applies.
    auto d = pipeline.RegisterPass(std::make_unique<NamedPass>("d"));
    ASSERT_TRUE(d.has_value());
    pipeline.ApplyChanges();
    EXPECT_TRUE(GraphHasPass(pipeline, "d"));

    // The old handles still resolve for enable/disable.
    EXPECT_TRUE(pipeline.SetPassEnabled(*a, false));
    EXPECT_FALSE(pipeline.SetPassEnabled(*b, true)); // b is a tombstone
    pipeline.ApplyChanges();
    EXPECT_FALSE(GraphHasPass(pipeline, "a"));
}

TEST(RenderPipelineRegistrationTest, IdempotentApplyIsANoOp) {
    RenderPipeline pipeline;
    ASSERT_TRUE(pipeline.RegisterPass(std::make_unique<NamedPass>("once")).has_value());
    pipeline.ApplyChanges();

    const std::uint32_t revision = pipeline.GetRevision();
    pipeline.ApplyChanges();
    EXPECT_EQ(pipeline.GetRevision(), revision);
}

TEST(RenderPipelineRegistrationTest, OrdersAroundBuiltinAnchors) {
    RenderPipeline pipeline;

    // Simulate the renderer registering a built-in "main" pass and exposing it
    // as the MainPass anchor.
    const auto main_result = pipeline.RegisterPass(std::make_unique<NamedPass>("main"));
    ASSERT_TRUE(main_result.has_value());
    std::array<VulkanEngine::RenderGraph::PassHandle, kBuiltinPassCount> anchors{};
    anchors[static_cast<std::size_t>(BuiltinPass::MainPass)] = *main_result;
    pipeline.SetBuiltinHandles(anchors);

    class BeforeMainPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "before-main"; }
        void Setup(PassSetupContext& ctx) override {
            ctx.RunBefore(BuiltinPass::MainPass);
            const auto target = ctx.CreateTransientImage(TransientImageDesc{
                .name = "before-main-target", .format = vk::Format::eR8G8B8A8Unorm,
                .width = 16, .height = 16});
            ctx.AddWrite(target);
        }
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    ASSERT_TRUE(pipeline.RegisterPass(std::make_unique<BeforeMainPass>()).has_value());
    pipeline.ApplyChanges();

    const auto& passes = pipeline.GetCompiledGraph().passes;
    ASSERT_EQ(passes.size(), 2u);
    EXPECT_EQ(passes[0].name, "before-main");
    EXPECT_EQ(passes[1].name, "main");
}

// A transient image's explicit initial/final layout must reach the compiled graph
// unchanged. The final-layout path used to map eShaderReadOnlyOptimal to
// Undefined (its ternary chain handled only ColorAttachment/DepthAttachment/
// Present while the initial-layout chain also handled ShaderReadOnly), so the
// requested layout degraded to a content-discarding Undefined transition.
class LayoutRequestingPass final : public IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return "layout-requesting"; }

    void Setup(PassSetupContext& ctx) override {
        target_ = ctx.CreateTransientImage(TransientImageDesc{
            .name = "shader-read-target",
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = 32,
            .height = 32,
            .initial_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .final_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        });
        ctx.AddWrite(target_);
    }

    void Execute(const FrameContext&, vk::CommandBuffer) override {}

    VulkanEngine::RenderGraph::ResourceHandle target_{};
};

TEST(RenderPipelineRegistrationTest, ShaderReadOnlyInitialAndFinalLayoutsSurviveCompile) {
    RenderPipeline pipeline;
    auto pass = std::make_unique<LayoutRequestingPass>();
    auto* raw_pass = pass.get();
    ASSERT_TRUE(pipeline.RegisterPass(std::move(pass)).has_value());
    pipeline.ApplyChanges();
    ASSERT_TRUE(pipeline.IsCompiled());

    const auto& graph = pipeline.GetCompiledGraph();
    const std::uint32_t index = raw_pass->target_.index;

    // The requested initial layout must survive compile as the exact
    // eShaderReadOnlyOptimal: a degrade to Undefined would insert a
    // content-discarding acquire barrier.
    const auto initial = graph.InitialImageLayoutFor(index);
    ASSERT_TRUE(initial.has_value());
    EXPECT_EQ(*initial, vk::ImageLayout::eShaderReadOnlyOptimal);

    // And the compiled final transition must post the same non-Undefined
    // layout the pass requested (the original ShaderReadOnly regression).
    const auto final_layout = graph.FinalImageLayoutFor(index);
    ASSERT_TRUE(final_layout.has_value());
    EXPECT_EQ(*final_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

}  // namespace
