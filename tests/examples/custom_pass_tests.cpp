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

// Passes.cppm returns the pass names from GetName() and exports no name
// constants, so the test keeps one constexpr table as the single point of edit:
// construction (AnchorPass) and every lookup read the same values, so a renamed
// pass is one edit away. The transient names, in contrast, are exported as
// descriptor functions (SceneColorDesc/ExposureBufferDesc) and are used through
// those instead of literals.
struct CustomPassNames {
    std::string_view capture;
    std::string_view exposure;
    std::string_view tonemap;
    std::string_view anchor;
};
constexpr CustomPassNames kCustomPassNames{
    .capture = "custom-capture",
    .exposure = "custom-exposure-pass",
    .tonemap = "custom-tonemap",
    .anchor = "imgui-overlay",
};

// Acts as the built-in ImGui anchor without drawing anything.
class AnchorPass final : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return kCustomPassNames.anchor; }
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

    const auto capture_index = index_of(kCustomPassNames.capture);
    const auto exposure_index = index_of(kCustomPassNames.exposure);
    const auto tonemap_index = index_of(kCustomPassNames.tonemap);
    const auto imgui_index = index_of(kCustomPassNames.anchor);
    ASSERT_TRUE(capture_index && exposure_index && tonemap_index && imgui_index);
    EXPECT_LT(*capture_index, *exposure_index);
    EXPECT_LT(*exposure_index, *tonemap_index);
    EXPECT_LT(*tonemap_index, *imgui_index);

    // The compute pass consumes a transient buffer written in the same frame,
    // and the capture pass owns a transient image. The names come from the
    // example's own exported descriptors, so a rename cannot desync the test.
    const auto has_transient = [&](std::string_view name) {
        for (const auto& lifetime : pipeline.GetCompiledGraph().resource_lifetimes) {
            if (lifetime.name == name && lifetime.transient) {
                return true;
            }
        }
        return false;
    };
    EXPECT_TRUE(has_transient(Examples::CustomPass::SceneColorDesc().name));
    EXPECT_TRUE(has_transient(Examples::CustomPass::ExposureBufferDesc().name));
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

// The plan reports the limit and whether a graphics preamble is required. The
// budget is the engine's named constant (RenderGraph::kMaxQueueRuns); the sweep
// only locates the smallest candidate run count that crosses it, so the test
// states the property (the budget holds below the limit, fails above it)
// without pinning the numeric boundary here.
TEST(CustomPassPlanTest, QueueRunLimitAndGraphicsStart) {
    using VulkanEngine::RenderGraph::BuildQueueRuns;
    using VulkanEngine::RenderGraph::CompiledRenderGraph;
    using VulkanEngine::RenderGraph::QueueType;

    // Candidates seeded from the named budget: straddle it without writing its
    // value. The loop visits each candidate at most once — a hard iteration cap
    // (the list's size), so the sweep cannot run past it.
    constexpr std::array<std::uint32_t, 6> kCandidates = {
        2, 4, VulkanEngine::RenderGraph::kMaxQueueRuns,
        VulkanEngine::RenderGraph::kMaxQueueRuns + 4,
        VulkanEngine::RenderGraph::kMaxQueueRuns * 2,
        VulkanEngine::RenderGraph::kMaxQueueRuns * 3};

    // One alternating Graphics/Compute chain: every pass is its own run, so the
    // run count equals the pass count.
    const std::vector<std::pair<std::uint32_t, bool>> sweep_results = [&] {
        std::vector<std::pair<std::uint32_t, bool>> results;
        results.reserve(kCandidates.size());
        for (const std::uint32_t count : kCandidates) {
            CompiledRenderGraph graph{};
            graph.passes.resize(count);
            for (std::size_t i = 0; i < graph.passes.size(); ++i) {
                graph.passes[i].queue = (i % 2 == 0) ? QueueType::Graphics : QueueType::Compute;
            }
            const auto plan = BuildQueueRuns(graph);
            EXPECT_EQ(plan.runs.size(), count);
            EXPECT_TRUE(plan.StartsWithGraphics());
            results.emplace_back(count, plan.ExceedsLimit());
        }
        return results;
    }();

    // The smallest candidate in the sweep that crosses the budget.
    const auto first_exceeding =
        std::find_if(sweep_results.begin(), sweep_results.end(),
                     [](const auto& result) { return result.second; });
    ASSERT_TRUE(first_exceeding != sweep_results.end())
        << "no candidate in the sweep exceeded the queue-run budget; extend the sweep";

    // The limit holds below the boundary and fails from it on: every candidate
    // at or above the first crossing also exceeds.
    for (const auto& [count, exceeds] : sweep_results) {
        EXPECT_EQ(exceeds, count >= first_exceeding->first)
            << count << " runs, budget " << VulkanEngine::RenderGraph::kMaxQueueRuns;
    }

    CompiledRenderGraph compute_first{};
    compute_first.passes.resize(2);
    compute_first.passes[0].queue = QueueType::Compute;
    compute_first.passes[1].queue = QueueType::Graphics;
    const auto preamble = BuildQueueRuns(compute_first);
    EXPECT_FALSE(preamble.StartsWithGraphics());
    EXPECT_FALSE(preamble.ExceedsLimit());
}

// A graph with more queue runs than the device budgets is rejected at compile
// time rather than indexing past the per-run command buffers/semaphores. The
// boundary is located with a bounded sweep seeded from the named budget
// constant: the smallest candidate that fails validation is found, while every
// smaller candidate still compiles. The loop visits each candidate at most once
// (a hard iteration cap), so it can neither loop forever nor pin the boundary.
TEST(CustomPassPlanTest, RejectsGraphExceedingQueueRunLimit) {
    using VulkanEngine::RenderGraph::QueueType;

    constexpr std::array<std::uint32_t, 6> kCandidates = {
        2, 4, VulkanEngine::RenderGraph::kMaxQueueRuns,
        VulkanEngine::RenderGraph::kMaxQueueRuns + 4,
        VulkanEngine::RenderGraph::kMaxQueueRuns * 2,
        VulkanEngine::RenderGraph::kMaxQueueRuns * 3};

    bool queue_run_rejection_reported = false;
    std::uint32_t first_rejected = 0;
    for (const std::uint32_t count : kCandidates) {
        RenderPipeline pipeline;
        pipeline.SetRenderExtent(1280, 720);
        pipeline.SetAsyncComputeAvailable(true); // allow compute-queue passes

        for (std::uint32_t i = 0; i < count; ++i) {
            const auto queue = (i % 2 == 0) ? QueueType::Graphics : QueueType::Compute;
            auto handle = pipeline.RegisterPass(
                std::make_unique<QueueOnlyPass>("queue-only-" + std::to_string(i), queue));
            ASSERT_TRUE(handle.has_value()) << (handle ? std::string{} : handle.error().message);
        }

        pipeline.Compile();
        if (!pipeline.IsCompiled()) {
            first_rejected = count;
            // The failure must be reported as the structured queue-run compile
            // error, not left as an unexplained rejection.
            for (const auto& error : pipeline.GetValidationErrors()) {
                if (error.code == VulkanEngine::RenderPipeline::PassErrorCode::CompileFailed) {
                    queue_run_rejection_reported = true;
                }
            }
            break;
        }
        EXPECT_TRUE(pipeline.GetValidationErrors().empty())
            << "a " << count << "-run graph compiled with validation errors";
    }

    ASSERT_TRUE(queue_run_rejection_reported)
        << "no candidate in the sweep was rejected; extend the sweep";

    // Every candidate at or above the first rejected count fails as well.
    for (const std::uint32_t count : kCandidates) {
        if (count < first_rejected) {
            continue;
        }
        RenderPipeline pipeline;
        pipeline.SetRenderExtent(1280, 720);
        pipeline.SetAsyncComputeAvailable(true);
        for (std::uint32_t i = 0; i < count; ++i) {
            auto handle = pipeline.RegisterPass(
                std::make_unique<QueueOnlyPass>(
                    "queue-only-" + std::to_string(i),
                    (i % 2 == 0) ? QueueType::Graphics : QueueType::Compute));
            ASSERT_TRUE(handle.has_value()) << (handle ? std::string{} : handle.error().message);
        }
        pipeline.Compile();
        EXPECT_FALSE(pipeline.IsCompiled())
            << count << " runs compiled although " << first_rejected << " did not";
        EXPECT_FALSE(pipeline.GetValidationErrors().empty());
    }
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
