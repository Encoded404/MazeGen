#include <gtest/gtest.h>

#include <glm/glm.hpp>

import std;

import vulkan;
import VulkanEngine.RenderPipeline;
import VulkanEngine.SceneRenderer;
import Examples.CustomPass.Passes;

import VulkanEngine.Render.Passes.ExpandPass;
import VulkanEngine.Render.Passes.OccluderSelectPass;
import VulkanEngine.Render.Passes.OccluderPrePass;
import VulkanEngine.Render.Passes.PreCullPass;
import VulkanEngine.Render.Passes.HiZPass;
import VulkanEngine.Render.Passes.DepthPrePass;
import VulkanEngine.Render.Passes.OcclusionPass;
import VulkanEngine.Render.Passes.CollectPass;
import VulkanEngine.Render.Passes.MainPass;
import VulkanEngine.Render.Passes.ImGuiPass;

namespace {

using Examples::CustomPass::CapturePass;
using Examples::CustomPass::ExposurePass;
using Examples::CustomPass::ToneMapPass;
using VulkanEngine::RenderPipeline::RenderPipeline;

// Acts as the built-in ImGui anchor without drawing anything.
class AnchorPass final : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return "imgui-overlay"; }
    void Setup(VulkanEngine::PipelinePass::PassSetupContext&) override {}
    void Execute(const VulkanEngine::PipelinePass::FrameContext&, vk::CommandBuffer) override {}
};

TEST(CustomPassPlanTest, PassesAndTransientsCompileInOrder) {
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(1280, 720);

    // The capture pass reads the imported backbuffer, so it must resolve.
    pipeline.RegisterResourceResolver("swapchain-backbuffer",
        [](std::uint32_t) { return vk::Image{}; },
        [](std::uint32_t) { return vk::ImageView{}; },
        vk::Format::eR8G8B8A8Unorm);

    const auto capture = pipeline.RegisterPass(std::make_unique<CapturePass>(1, 2));
    const auto exposure = pipeline.RegisterPass(std::make_unique<ExposurePass>(3, false));
    const auto tonemap = pipeline.RegisterPass(std::make_unique<ToneMapPass>(1, 4));
    ASSERT_TRUE(capture && exposure && tonemap);

    (void)pipeline.AddDependency(*capture, *exposure);
    (void)pipeline.AddDependency(*exposure, *tonemap);

    // The tonemap orders itself before the ImGui anchor; expose one.
    const auto anchor = pipeline.RegisterPass(std::make_unique<AnchorPass>());
    ASSERT_TRUE(anchor.has_value());
    std::array<VulkanEngine::RenderGraph::PassHandle,
               VulkanEngine::PipelinePass::kBuiltinPassCount> anchors{};
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::ImGui)] = *anchor;
    pipeline.SetBuiltinHandles(anchors);

    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());

    const auto& passes = pipeline.GetCompiledGraph().passes;
    const auto index_of = [&](std::string_view name) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < passes.size(); ++i) {
            if (passes[i].name == name) {
                return i;
            }
        }
        return std::nullopt;
    };

    const auto capture_index = index_of("custom-capture");
    const auto exposure_index = index_of("custom-exposure-pass");
    const auto tonemap_index = index_of("custom-tonemap");
    const auto imgui_index = index_of("imgui-overlay");
    ASSERT_TRUE(capture_index && exposure_index && tonemap_index && imgui_index);
    EXPECT_LT(*capture_index, *exposure_index);
    EXPECT_LT(*exposure_index, *tonemap_index);
    EXPECT_LT(*tonemap_index, *imgui_index);

    // The compute pass consumes a transient buffer written in the same frame,
    // and the capture pass owns a transient image.
    const auto has_transient = [&](std::string_view name) {
        for (const auto& lifetime : pipeline.GetCompiledGraph().resource_lifetimes) {
            if (lifetime.name == name && lifetime.transient) {
                return true;
            }
        }
        return false;
    };
    EXPECT_TRUE(has_transient("custom-scene-color"));
    EXPECT_TRUE(has_transient("custom-exposure"));
}

// Mirrors the real renderer: every built-in pass plus the three custom passes
// in one graph. This is the device-free reproduction of the on-device graph, so
// ordering/hazard compile failures surface without a GPU.
TEST(CustomPassPlanTest, FullBuiltinAndCustomGraphCompiles) {
    VulkanEngine::SceneRenderer::SceneRenderer scene_renderer;
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(1280, 720);

    const auto register_image = [&pipeline](std::string name, vk::Format format) {
        pipeline.RegisterResourceResolver(std::move(name),
            [](std::uint32_t) { return vk::Image{}; },
            [](std::uint32_t) { return vk::ImageView{}; },
            format);
    };
    register_image("swapchain-backbuffer", vk::Format::eR8G8B8A8Unorm);
    register_image("depth-buffer", vk::Format::eD32Sfloat);
    register_image("hiz-image", vk::Format::eR32Sfloat);

    const auto reg = [&pipeline](std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass> pass) {
        auto result = pipeline.RegisterPass(std::move(pass));
        EXPECT_TRUE(result.has_value()) << (result ? std::string{} : result.error().message);
        return result.has_value() ? *result : VulkanEngine::RenderGraph::PassHandle{};
    };

    namespace SR = VulkanEngine::SceneRenderer;
    const auto expand = reg(std::make_unique<SR::ExpandPass>(scene_renderer));
    const auto occluder_select = reg(std::make_unique<SR::OccluderSelectPass>(scene_renderer));
    const auto occluder_prepass =
        reg(std::make_unique<SR::OccluderPrePass>(scene_renderer, vk::ClearDepthStencilValue{1.0f, 0}));
    const auto hiz_pre = reg(std::make_unique<SR::HiZPass>(scene_renderer, "hiz-gen-pre"));
    const auto pre_cull = reg(std::make_unique<SR::PreCullPass>(scene_renderer));
    const auto depth = reg(std::make_unique<SR::DepthPrePass>(scene_renderer));
    const auto hiz = reg(std::make_unique<SR::HiZPass>(scene_renderer, "hiz-gen"));
    const auto occlusion = reg(std::make_unique<SR::OcclusionPass>(scene_renderer));
    const auto collect = reg(std::make_unique<SR::CollectPass>(scene_renderer));
    const auto main_pass =
        reg(std::make_unique<SR::MainPass>(scene_renderer, glm::vec4{0.0f}));
    const auto imgui = reg(std::make_unique<SR::ImGuiPass>(nullptr));

    pipeline.AddDependency(expand, occluder_select);
    pipeline.AddDependency(occluder_select, occluder_prepass);
    pipeline.AddDependency(occluder_prepass, hiz_pre);
    pipeline.AddDependency(hiz_pre, pre_cull);
    pipeline.AddDependency(pre_cull, depth);
    pipeline.AddDependency(depth, hiz);
    pipeline.AddDependency(hiz, occlusion);
    pipeline.AddDependency(occlusion, collect);
    pipeline.AddDependency(collect, main_pass);

    std::array<VulkanEngine::RenderGraph::PassHandle,
               VulkanEngine::PipelinePass::kBuiltinPassCount> anchors{};
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::Expand)] = expand;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::OccluderSelect)] = occluder_select;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::OccluderPrepass)] = occluder_prepass;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::HiZGenPre)] = hiz_pre;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::PreCull)] = pre_cull;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::DepthPrepass)] = depth;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::HiZGen)] = hiz;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::Occlusion)] = occlusion;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::Collect)] = collect;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::MainPass)] = main_pass;
    anchors[static_cast<std::size_t>(VulkanEngine::PipelinePass::BuiltinPass::ImGui)] = imgui;
    pipeline.SetBuiltinHandles(anchors);

    const auto capture = reg(std::make_unique<CapturePass>(1, 2));
    const auto exposure = reg(std::make_unique<ExposurePass>(3, false));
    const auto tonemap = reg(std::make_unique<ToneMapPass>(1, 4));
    pipeline.AddDependency(capture, exposure);
    pipeline.AddDependency(exposure, tonemap);

    pipeline.Compile();
    for (const auto& diagnostic : pipeline.GetDiagnostics()) {
        if (diagnostic.severity == VulkanEngine::RenderGraph::DiagnosticSeverity::Error) {
            ADD_FAILURE() << "compile diagnostic: " << diagnostic.message;
        }
    }
    EXPECT_TRUE(pipeline.IsCompiled());
}

// Contiguous same-queue passes form one run; a queue switch starts a new run.
TEST(CustomPassPlanTest, BuildQueueRunsPartitionsByQueue) {
    using VulkanEngine::RenderGraph::BuildQueueRuns;
    using VulkanEngine::RenderGraph::CompiledRenderGraph;
    using VulkanEngine::RenderGraph::QueueType;

    CompiledRenderGraph graph{};
    graph.passes.resize(5);
    graph.passes[0].queue = QueueType::Graphics;
    graph.passes[1].queue = QueueType::Graphics;
    graph.passes[2].queue = QueueType::Compute;
    graph.passes[3].queue = QueueType::Graphics;
    graph.passes[4].queue = QueueType::Graphics;

    const auto plan = BuildQueueRuns(graph);
    ASSERT_EQ(plan.runs.size(), 3u);
    EXPECT_EQ(plan.runs[0].queue, QueueType::Graphics);
    EXPECT_EQ(plan.runs[0].first_pass, 0u);
    EXPECT_EQ(plan.runs[0].last_pass, 2u);
    EXPECT_EQ(plan.runs[1].queue, QueueType::Compute);
    EXPECT_EQ(plan.runs[1].first_pass, 2u);
    EXPECT_EQ(plan.runs[1].last_pass, 3u);
    EXPECT_EQ(plan.runs[2].queue, QueueType::Graphics);
    EXPECT_EQ(plan.runs[2].first_pass, 3u);
    EXPECT_EQ(plan.runs[2].last_pass, 5u);
    EXPECT_TRUE(plan.HasComputeRun());
}

// A device-free pass that only selects a queue, so run partitioning can be
// driven purely by registration order.
class QueueOnlyPass final : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    QueueOnlyPass(std::string name, VulkanEngine::RenderGraph::QueueType queue)
        : name_(std::move(name)), queue_(queue) {}

    [[nodiscard]] std::string_view GetName() const override { return name_; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override {
        ctx.SetQueueType(queue_);
    }

    void Execute(const VulkanEngine::PipelinePass::FrameContext&, vk::CommandBuffer) override {}

private:
    std::string name_;
    VulkanEngine::RenderGraph::QueueType queue_;
};

// The plan reports the limit and whether a graphics preamble is required.
TEST(CustomPassPlanTest, QueueRunLimitAndGraphicsStart) {
    using VulkanEngine::RenderGraph::BuildQueueRuns;
    using VulkanEngine::RenderGraph::CompiledRenderGraph;
    using VulkanEngine::RenderGraph::QueueType;

    CompiledRenderGraph alternating{};
    alternating.passes.resize(9);
    for (std::size_t i = 0; i < alternating.passes.size(); ++i) {
        alternating.passes[i].queue = (i % 2 == 0) ? QueueType::Graphics : QueueType::Compute;
    }
    const auto over = BuildQueueRuns(alternating);
    EXPECT_EQ(over.runs.size(), 9u);
    EXPECT_TRUE(over.ExceedsLimit());
    EXPECT_TRUE(over.StartsWithGraphics());

    CompiledRenderGraph compute_first{};
    compute_first.passes.resize(2);
    compute_first.passes[0].queue = QueueType::Compute;
    compute_first.passes[1].queue = QueueType::Graphics;
    const auto preamble = BuildQueueRuns(compute_first);
    EXPECT_FALSE(preamble.StartsWithGraphics());
    EXPECT_FALSE(preamble.ExceedsLimit());
}

// A graph with more queue runs than the device budgets is rejected at compile
// time rather than indexing past the per-run command buffers/semaphores.
TEST(CustomPassPlanTest, RejectsGraphExceedingQueueRunLimit) {
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(1280, 720);
    pipeline.SetAsyncComputeAvailable(true); // allow compute-queue passes

    for (int i = 0; i < 9; ++i) {
        const auto queue = (i % 2 == 0) ? VulkanEngine::RenderGraph::QueueType::Graphics
                                        : VulkanEngine::RenderGraph::QueueType::Compute;
        auto handle = pipeline.RegisterPass(
            std::make_unique<QueueOnlyPass>("queue-only-" + std::to_string(i), queue));
        ASSERT_TRUE(handle.has_value()) << (handle ? std::string{} : handle.error().message);
    }

    pipeline.Compile();
    EXPECT_FALSE(pipeline.IsCompiled());

    bool explained = false;
    for (const auto& error : pipeline.GetValidationErrors()) {
        if (error.message.find("queue runs") != std::string::npos) {
            explained = true;
        }
    }
    EXPECT_TRUE(explained);
}

// A compute-queue pass is rejected when no async compute queue is available.
TEST(CustomPassPlanTest, RejectsComputeQueueWithoutAsyncComputeQueue) {
    class ComputeQueuePass final : public VulkanEngine::PipelinePass::IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "compute-queue-pass"; }
        void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override {
            ctx.RequestComputePipeline(1);
            ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
        }
        void Execute(const VulkanEngine::PipelinePass::FrameContext&, vk::CommandBuffer) override {}
    };

    RenderPipeline pipeline;
    const auto handle = pipeline.RegisterPass(std::make_unique<ComputeQueuePass>());
    ASSERT_FALSE(handle.has_value());
    EXPECT_EQ(handle.error().code, VulkanEngine::RenderPipeline::PassErrorCode::ValidationFailed);
}

} // namespace
