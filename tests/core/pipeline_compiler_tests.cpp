#include <gtest/gtest.h>

import std;
import std.compat;

import vulkan;

import VulkanEngine.PipelineCompiler;
import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderManager;

namespace {

using namespace VulkanEngine::ShaderSystem;

// Device-free, like the whole core suite: the fake executor stands in for the
// factory's batched driver calls, so window semantics (partitioning, delivery,
// error isolation) are exercised without a VkDevice.

constexpr ShaderId kComp = 21;
constexpr ShaderId kVert = 31;
constexpr ShaderId kFrag = 32;

ComputePipelineDesc CompDesc(ShaderId id) {
    ComputePipelineDesc d{};
    d.shader = id;
    d.entry_point = "main";
    return d;
}

GraphicsPipelineDesc GfxDesc(ShaderId vert, ShaderId frag) {
    GraphicsPipelineDesc d{};
    d.vertex_shader = vert;
    d.fragment_shader = frag;
    d.color_blend_attachments = {
        vk::PipelineColorBlendAttachmentState(
            true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA)
    };
    return d;
}

struct FakeExecutor : IPipelineBatchExecutor {
    // Counters + captured descs.
    int compute_calls = 0;
    int graphics_calls = 0;
    std::vector<std::vector<ComputePipelineDesc>> compute_batches;
    std::vector<std::vector<GraphicsPipelineDesc>> graphics_batches;

    // Per-call behavior: which shader ids in the submitted batch fail.
    std::set<ShaderId> fail_shader_ids;
    // Defensive-path triggers.
    bool wrong_outcome_count = false;
    bool throw_from_executor = false;

    std::vector<PipelineOutcome> ExecuteCompute(
            ShaderManager&, std::span<const ComputePipelineDesc> descs) override {
        ++compute_calls;
        compute_batches.emplace_back(descs.begin(), descs.end());
        if (throw_from_executor) throw std::runtime_error("fake executor boom");
        std::vector<PipelineOutcome> out;
        for (const auto& d : descs) {
            if (fail_shader_ids.contains(d.shader)) {
                out.push_back(std::unexpected(PipelineError{
                    PipelineErrorKind::ShaderModuleUnavailable,
                    vk::Result::eErrorInitializationFailed, "fake: shader gone"}));
            } else {
                out.push_back(PipelineProduct{});
            }
        }
        if (wrong_outcome_count) out.pop_back();
        return out;
    }

    std::vector<PipelineOutcome> ExecuteGraphics(
            ShaderManager&, std::span<const GraphicsPipelineDesc> descs) override {
        ++graphics_calls;
        graphics_batches.emplace_back(descs.begin(), descs.end());
        if (throw_from_executor) throw std::runtime_error("fake executor boom");
        std::vector<PipelineOutcome> out;
        for (const auto& d : descs) {
            if (fail_shader_ids.contains(d.vertex_shader) ||
                fail_shader_ids.contains(d.fragment_shader)) {
                out.push_back(std::unexpected(PipelineError{
                    PipelineErrorKind::ShaderModuleUnavailable,
                    vk::Result::eErrorInitializationFailed, "fake: shader gone"}));
            } else {
                out.push_back(PipelineProduct{});
            }
        }
        if (wrong_outcome_count) out.pop_back();
        return out;
    }
};

struct Delivery {
    enum class Kind { NotDelivered, Success, Error };
    Kind kind = Kind::NotDelivered;
    std::string message;   // error message when kind == Error
    std::size_t deliveries = 0;

    std::function<void(PipelineOutcome&&)> Callback() {
        return [this](PipelineOutcome&& outcome) {
            ++deliveries;
            if (outcome.has_value()) {
                kind = Kind::Success;
            } else {
                kind = Kind::Error;
                message = outcome.error().message;
            }
        };
    }
};

// The executor seam keeps the manager behind a reference; the fake never
// dereferences it, so a laundered byte buffer stands in. The real factory
// executor dereferences it on the main thread (device-bound tests cover that
// path through the gpu label).
[[nodiscard]] ShaderManager& TestShaderManager() noexcept {
    static std::byte storage[sizeof(ShaderManager)];
    return *std::launder(reinterpret_cast<ShaderManager*>(storage));
}

// ── Partitioning ─────────────────────────────────────────────────────────────

TEST(PipelineCompiler, PartitionsWindowIntoOneExecutorCallPerKind) {
    FakeExecutor ex;
    PipelineCompiler compiler(ex);
    std::vector<Delivery> deliveries(5);

    compiler.SubmitCompute(CompDesc(kComp), "c1", deliveries[0].Callback());
    compiler.SubmitGraphics(GfxDesc(kVert, kFrag), "g1", deliveries[1].Callback());
    compiler.SubmitCompute(CompDesc(kComp + 1), "c2", deliveries[2].Callback());
    compiler.SubmitGraphics(GfxDesc(kVert + 1, kFrag), "g2", deliveries[3].Callback());
    compiler.SubmitCompute(CompDesc(kComp + 2), "c3", deliveries[4].Callback());
    ASSERT_EQ(compiler.Pending(), 5u);

    compiler.Flush(TestShaderManager());

    EXPECT_EQ(ex.compute_calls, 1);
    EXPECT_EQ(ex.graphics_calls, 1);
    ASSERT_EQ(ex.compute_batches.size(), 1u);
    ASSERT_EQ(ex.compute_batches[0].size(), 3u);
    EXPECT_EQ(ex.compute_batches[0][0].shader, kComp);
    EXPECT_EQ(ex.compute_batches[0][1].shader, kComp + 1);
    EXPECT_EQ(ex.compute_batches[0][2].shader, kComp + 2);
    ASSERT_EQ(ex.graphics_batches.size(), 1u);
    ASSERT_EQ(ex.graphics_batches[0].size(), 2u);
    EXPECT_EQ(ex.graphics_batches[0][0].vertex_shader, kVert);
    for (const auto& d : deliveries) {
        EXPECT_EQ(d.deliveries, 1u);
        EXPECT_EQ(d.kind, Delivery::Kind::Success);
    }
    EXPECT_EQ(compiler.Pending(), 0u);
}

// ── Delivery semantics ───────────────────────────────────────────────────────

TEST(PipelineCompiler, DeliversExactlyOnceInSubmissionOrderPerGroup) {
    FakeExecutor ex;
    PipelineCompiler compiler(ex);
    std::vector<Delivery> deliveries(4); // 2 compute interleaved with 2 graphics

    compiler.SubmitCompute(CompDesc(kComp), "c1", deliveries[0].Callback());
    compiler.SubmitGraphics(GfxDesc(kVert, kFrag), "g1", deliveries[1].Callback());
    compiler.SubmitCompute(CompDesc(kComp + 1), "c2", deliveries[2].Callback());
    compiler.SubmitGraphics(GfxDesc(kVert, kFrag + 1), "g2", deliveries[3].Callback());

    compiler.Flush(TestShaderManager());

    // Compute group delivered first (window partition order), then graphics.
    ASSERT_EQ(ex.compute_calls, 1);
    ASSERT_EQ(ex.graphics_calls, 1);
    for (const auto& d : deliveries) {
        EXPECT_EQ(d.deliveries, 1u);
    }
}

TEST(PipelineCompiler, EmptyWindowCallsNoExecutor) {
    FakeExecutor ex;
    PipelineCompiler compiler(ex);
    compiler.Flush(TestShaderManager());
    EXPECT_EQ(ex.compute_calls, 0);
    EXPECT_EQ(ex.graphics_calls, 0);
}

TEST(PipelineCompiler, ClearDropsUnflushedSubmissionsWithoutDelivery) {
    FakeExecutor ex;
    PipelineCompiler compiler(ex);
    Delivery d;
    compiler.SubmitCompute(CompDesc(kComp), "c", d.Callback());
    compiler.Clear();
    EXPECT_EQ(compiler.Pending(), 0u);
    compiler.Flush(TestShaderManager());
    EXPECT_EQ(d.deliveries, 0u);
    EXPECT_EQ(ex.compute_calls, 0);
}

// ── Error isolation ──────────────────────────────────────────────────────────

TEST(PipelineCompiler, FailedDescDoesNotStarveTheRestOfItsBatch) {
    FakeExecutor ex;
    ex.fail_shader_ids = {kComp}; // middle shader of three fails
    PipelineCompiler compiler(ex);
    Delivery d0, d1, d2;
    compiler.SubmitCompute(CompDesc(kComp - 1), "c0", d0.Callback());
    compiler.SubmitCompute(CompDesc(kComp), "c1", d1.Callback());
    compiler.SubmitCompute(CompDesc(kComp + 1), "c2", d2.Callback());
    compiler.Flush(TestShaderManager());

    EXPECT_EQ(d0.kind, Delivery::Kind::Success); // did not starve
    EXPECT_EQ(d2.kind, Delivery::Kind::Success); // did not starve
    EXPECT_EQ(d1.kind, Delivery::Kind::Error);
    EXPECT_NE(d1.message.find("fake: shader gone"), std::string::npos);
}

TEST(PipelineCompiler, SizeContractViolationFailsTheWholeGroupWithErrors) {
    FakeExecutor ex;
    ex.wrong_outcome_count = true; // executor returns N-1 outcomes
    PipelineCompiler compiler(ex);
    std::vector<Delivery> deliveries(3);
    for (auto& d : deliveries) {
        compiler.SubmitGraphics(GfxDesc(kVert, kFrag), "g", d.Callback());
    }
    compiler.Flush(TestShaderManager());

    for (const auto& d : deliveries) {
        ASSERT_EQ(d.deliveries, 1u);
        EXPECT_EQ(d.kind, Delivery::Kind::Error);
        // The defensive error must be self-describing.
        EXPECT_NE(d.message.find("outcomes for"), std::string::npos);
    }
}

TEST(PipelineCompiler, ThrowingExecutorFailsEveryPendingRequest) {
    FakeExecutor ex;
    ex.throw_from_executor = true;
    PipelineCompiler compiler(ex);
    Delivery d0, d1;
    compiler.SubmitCompute(CompDesc(kComp), "c", d0.Callback());
    compiler.SubmitGraphics(GfxDesc(kVert, kFrag), "g", d1.Callback());
    compiler.Flush(TestShaderManager());

    EXPECT_EQ(compiler.Pending(), 0u);
    EXPECT_EQ(d0.deliveries, 1u);
    EXPECT_EQ(d1.deliveries, 1u);
    EXPECT_EQ(d0.kind, Delivery::Kind::Error);
    EXPECT_NE(d0.message.find("executor threw"), std::string::npos);
    // Same failure mode as a driver rejection kind, message identifies windows.
    EXPECT_NE(d1.message.find("[g]"), std::string::npos);
    // A second, healthy window works again afterwards.
    ex.throw_from_executor = false;
    Delivery d2;
    compiler.SubmitCompute(CompDesc(kComp), "c", d2.Callback());
    compiler.Flush(TestShaderManager());
    EXPECT_EQ(d2.kind, Delivery::Kind::Success);
}

} // namespace