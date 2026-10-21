// GPU-labelled tests for the per-pass timestamp timer.
//
// These use the real headless device: the query pool, timestamp writes, raw
// readback, ring-slot reuse and the RenderGraphExecutor bracket are all
// exercised on hardware. The default ctest presets exclude the `gpu` label.

#include <gtest/gtest.h>

import std;

import vulkan;

import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.GpuPassTimer;
import VulkanBackend.Vulkan.RenderGraphExecutor;
import VulkanShared.RenderGraphTypes;
import test_gpu;

namespace {

using TestSupport::HeadlessVulkanBackend;
using VulkanBackend::Vulkan::GpuPassSample;
using VulkanBackend::Vulkan::GpuPassTimer;
using VulkanEngine::RenderGraph::BarrierPlan;
using VulkanEngine::RenderGraph::CompiledRenderGraph;
using VulkanEngine::RenderGraph::QueueType;

void SubmitAndWait(HeadlessVulkanBackend& backend, vk::raii::CommandBuffer& cmd) {
    cmd.end();
    const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
    backend.GetGraphicsQueue().submit(submit, {});
    backend.GetGraphicsQueue().waitIdle();
}

class GpuPassTimerGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        if (!timer_.Initialize(backend_.GetPhysicalDevice(), backend_.GetDevice(),
                               backend_.GetGraphicsQueueFamily(),
                               backend_.GetFramesInFlight())) {
            GTEST_SKIP() << "graphics queue family reports no timestamp support";
        }
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        timer_.Shutdown();
        backend_.Shutdown();
    }

    HeadlessVulkanBackend backend_{};
    GpuPassTimer timer_{};
};

} // namespace

TEST_F(GpuPassTimerGpuTest, InitializesOnADeviceWithTimestampSupport) {
    EXPECT_TRUE(timer_.Enabled());
}

TEST_F(GpuPassTimerGpuTest, CollectsOneLabelledPassAndSkipsUnwrittenIndices) {
    auto& cmd = backend_.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    timer_.BeginFrame(0);
    timer_.WriteBegin(cmd, 0, "test-pass");
    timer_.WriteEnd(cmd, 0);
    SubmitAndWait(backend_, cmd);

    timer_.BeginFrame(0);
    std::vector<GpuPassSample> samples;
    timer_.Collect(samples);

    ASSERT_EQ(samples.size(), 1u);
    EXPECT_EQ(samples[0].name, "test-pass");
    EXPECT_GE(samples[0].ms, 0.0);
    // The reserved prep index was never written, so it is not a phantom 0 ms row.
    EXPECT_EQ(samples[0].name.find("frame-prep"), std::string::npos);
}

TEST_F(GpuPassTimerGpuTest, SecondCollectWithoutNewWritesReturnsEmpty) {
    auto& cmd = backend_.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    timer_.BeginFrame(0);
    timer_.WriteBegin(cmd, 0, "test-pass");
    timer_.WriteEnd(cmd, 0);
    SubmitAndWait(backend_, cmd);

    timer_.BeginFrame(0);
    std::vector<GpuPassSample> first;
    timer_.Collect(first);
    ASSERT_EQ(first.size(), 1u);

    std::vector<GpuPassSample> second;
    timer_.Collect(second);
    EXPECT_TRUE(second.empty());
}

TEST_F(GpuPassTimerGpuTest, PrepBracketIsReported) {
    auto& cmd = backend_.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    timer_.BeginFrame(0);
    timer_.WritePrepBegin(cmd);
    timer_.WritePrepEnd(cmd);
    SubmitAndWait(backend_, cmd);

    timer_.BeginFrame(0);
    std::vector<GpuPassSample> samples;
    timer_.Collect(samples);

    ASSERT_EQ(samples.size(), 1u);
    EXPECT_NE(samples[0].name.find("frame-prep"), std::string::npos);
}

TEST_F(GpuPassTimerGpuTest, ShutdownThenCollectIsSafeAndEmpty) {
    timer_.Shutdown();
    EXPECT_FALSE(timer_.Enabled());

    std::vector<GpuPassSample> samples;
    timer_.Collect(samples);
    EXPECT_TRUE(samples.empty());
}

TEST_F(GpuPassTimerGpuTest, ExecutorBracketsEveryExecutingPass) {
    CompiledRenderGraph graph{};
    graph.success = true;
    graph.passes.resize(2);
    graph.passes[0].name = "pass-a";
    graph.passes[0].queue = QueueType::Graphics;
    graph.passes[0].execute.callback = [](const void*, vk::CommandBuffer) {};
    graph.passes[1].name = "pass-b";
    graph.passes[1].queue = QueueType::Graphics;
    graph.passes[1].execute.callback = [](const void*, vk::CommandBuffer) {};

    BarrierPlan plan{};
    plan.valid = true;

    auto& cmd = backend_.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    timer_.BeginFrame(0);
    VulkanBackend::Vulkan::ExecuteRenderGraphRange(plan, graph, 0, 2, nullptr, *cmd, &timer_);
    SubmitAndWait(backend_, cmd);

    timer_.BeginFrame(0);
    std::vector<GpuPassSample> samples;
    timer_.Collect(samples);

    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(samples[0].name, "pass-a");
    EXPECT_EQ(samples[1].name, "pass-b");
    EXPECT_GE(samples[0].ms, 0.0);
    EXPECT_GE(samples[1].ms, 0.0);
}
