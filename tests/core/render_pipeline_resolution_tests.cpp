#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.RenderGraph;

namespace {

using namespace VulkanEngine::RenderGraph;

// Builds a graph whose resources have mixed origins so the identity table can be
// checked for positional-indexing regressions.
CompiledRenderGraph BuildMixedGraph(ResourceHandle& imported_out, ResourceHandle& transient_out) {
    RenderGraphBuilder builder;

    builder.ImportResource("engine-render-target", ResourceKind::Image);
    builder.ImportResource("depth-buffer", ResourceKind::Image);
    builder.ImportResource("missing-resolver", ResourceKind::Image);
    const auto imported_buf = builder.ImportResource("scene-buffers", ResourceKind::Buffer);

    // A high-index transient: with positional/`size()`-as-count resolution this
    // is where the old code mis-resolved (defect #14).
    const auto high_transient = builder.CreateTransientResource("late-transient", ResourceKind::Image);
    builder.SetTransientImageInfo(high_transient, TransientImageInfo{
                                                      .format = vk::Format::eR8G8B8A8Unorm,
                                                      .width = 64,
                                                      .height = 64,
                                                      .aliasable = true});

    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    builder.AddWrite(pass, high_transient);
    builder.AddRead(pass, imported_buf, PipelineStageIntent::ComputeShader, AccessIntent::Read);

    imported_out = imported_buf;
    transient_out = high_transient;
    return builder.Compile();
}

TEST(RenderPipelineResolutionTest, TableKeysByResourceIndexNotPosition) {
    ResourceHandle imported{};
    ResourceHandle transient_handle{};
    const auto graph = BuildMixedGraph(imported, transient_handle);
    ASSERT_TRUE(graph.success);

    const std::unordered_set<std::string> image_resolvers{"engine-render-target", "depth-buffer"};
    const std::unordered_set<std::string> buffer_resolvers{"scene-buffers"};

    const auto table = BuildResourceResolutionTable(graph, image_resolvers, buffer_resolvers);
    ASSERT_EQ(table.size(), graph.resource_lifetimes.size());

    // Every entry is self-consistent: its resource_index matches its position
    // (lifetimes are emitted in resource-index order) and never exceeds bounds.
    for (std::uint32_t i = 0; i < table.size(); ++i) {
        EXPECT_EQ(table[i].resource_index, i);
        EXPECT_LT(table[i].resource_index, graph.resource_info.size());
    }

    EXPECT_EQ(table[imported.index].resolution, ResourceResolutionKind::ImportedBuffer);
    EXPECT_EQ(table[transient_handle.index].resolution, ResourceResolutionKind::TransientImage);
    EXPECT_EQ(table[2].resolution, ResourceResolutionKind::ImportedImageMissingResolver);
    EXPECT_EQ(table[0].resolution, ResourceResolutionKind::ImportedImage);
}

TEST(RenderPipelineResolutionTest, RuntimeOldStateSeedsFirstBarrier) {
    RenderGraphBuilder builder;
    const auto backbuffer = builder.ImportResource("swapchain-backbuffer", ResourceKind::Image);
    builder.SetFinalState(backbuffer, ResourceState::ImageState(
                                           PipelineStageIntent::BottomOfPipe, AccessIntent::None,
                                           QueueType::Graphics, ImageLayoutIntent::Present));

    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    ASSERT_TRUE(builder.AddWrite(pass, backbuffer));
    const auto graph = builder.Compile();
    ASSERT_TRUE(graph.success);

    // The swapchain image is still in Present layout from the previous frame, so
    // the first barrier is Present -> ColorAttachment, not Undefined -> ...
    RuntimeResourceStates runtime{};
    runtime.states.resize(graph.resource_info.size());
    runtime.has_state.assign(graph.resource_info.size(), false);
    runtime.states[backbuffer.index] = ResourceState::ImageState(
        PipelineStageIntent::Present, AccessIntent::None, QueueType::Graphics, ImageLayoutIntent::Present);
    runtime.has_state[backbuffer.index] = true;

    const auto plan = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{}, &runtime);
    ASSERT_EQ(plan.passes.size(), 1u);
    ASSERT_FALSE(plan.passes[0].pre_image.empty());
    EXPECT_EQ(plan.passes[0].pre_image.front().old_layout, vk::ImageLayout::ePresentSrcKHR);
    // A plain write (no attachment setup) targets General.
    EXPECT_EQ(plan.passes[0].pre_image.front().new_layout, vk::ImageLayout::eGeneral);
}

TEST(RenderPipelineResolutionTest, RuntimeStateAlreadyAtTargetDropsBarrier) {
    RenderGraphBuilder builder;
    const auto scratch = builder.CreateTransientResource("scratch", ResourceKind::Image);
    builder.SetTransientImageInfo(scratch, TransientImageInfo{.format = vk::Format::eR8G8B8A8Unorm,
                                                              .width = 64, .height = 64});
    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    ASSERT_TRUE(builder.AddWrite(pass, scratch));
    const auto graph = builder.Compile();
    ASSERT_TRUE(graph.success);

    // Force the runtime to already be in the write target (General/compute).
    RuntimeResourceStates runtime{};
    runtime.states.resize(graph.resource_info.size());
    runtime.has_state.assign(graph.resource_info.size(), false);
    runtime.states[scratch.index] = ResourceState::ImageState(
        PipelineStageIntent::ComputeShader, AccessIntent::Write, QueueType::Graphics,
        ImageLayoutIntent::General);
    runtime.has_state[scratch.index] = true;

    const auto plan = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{}, &runtime);
    EXPECT_TRUE(plan.passes[0].pre_image.empty());
    // The state is still tracked so the next frame seeds correctly.
    EXPECT_TRUE(plan.has_end_state[scratch.index]);
    EXPECT_EQ(plan.end_states[scratch.index].layout, ImageLayoutIntent::General);
}

TEST(RenderPipelineResolutionTest, EndStateRecordsActualLastLayout) {
    RenderGraphBuilder builder;
    const auto backbuffer = builder.ImportResource("swapchain-backbuffer", ResourceKind::Image);
    builder.SetFinalState(backbuffer, ResourceState::ImageState(
                                           PipelineStageIntent::BottomOfPipe, AccessIntent::None,
                                           QueueType::Graphics, ImageLayoutIntent::Present));
    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    ASSERT_TRUE(builder.AddWrite(pass, backbuffer));
    const auto graph = builder.Compile();
    ASSERT_TRUE(graph.success);

    // First use (image was never presented): Undefined -> ColorAttachment, then
    // the final transition back to Present must be recorded as the end state.
    const auto plan = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{});
    ASSERT_TRUE(plan.has_end_state[backbuffer.index]);
    EXPECT_EQ(plan.end_states[backbuffer.index].layout, ImageLayoutIntent::Present);
}

TEST(RenderPipelineResolutionTest, IndependentlyTrackedResourcesKeepSeparateOldStates) {
    RenderGraphBuilder builder;
    const auto first = builder.CreateTransientResource("first", ResourceKind::Image);
    const auto second = builder.CreateTransientResource("second", ResourceKind::Image);
    const TransientImageInfo info{.format = vk::Format::eR8G8B8A8Unorm, .width = 64, .height = 64};
    builder.SetTransientImageInfo(first, info);
    builder.SetTransientImageInfo(second, info);

    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    ASSERT_TRUE(builder.AddWrite(pass, first));
    ASSERT_TRUE(builder.AddWrite(pass, second));
    const auto graph = builder.Compile();
    ASSERT_TRUE(graph.success);

    RuntimeResourceStates runtime{};
    runtime.states.resize(graph.resource_info.size());
    runtime.has_state.assign(graph.resource_info.size(), false);
    // `first` is already General (barrier dropped); `second` is ShaderReadOnly
    // (barrier needed) so the two do not share a state.
    runtime.states[first.index] = ResourceState::ImageState(
        PipelineStageIntent::ComputeShader, AccessIntent::Write, QueueType::Graphics,
        ImageLayoutIntent::General);
    runtime.has_state[first.index] = true;
    runtime.states[second.index] = ResourceState::ImageState(
        PipelineStageIntent::ComputeShader, AccessIntent::Read, QueueType::Graphics,
        ImageLayoutIntent::ShaderReadOnly);
    runtime.has_state[second.index] = true;

    const auto plan = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{}, &runtime);
    ASSERT_EQ(plan.passes[0].pre_image.size(), 1u);
    EXPECT_EQ(plan.passes[0].pre_image.front().resource_index, second.index);
    EXPECT_EQ(plan.passes[0].pre_image.front().old_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(RenderPipelineResolutionTest, RecordedEndStateSeedsTheNextFrame) {
    RenderGraphBuilder builder;
    const auto backbuffer = builder.ImportResource("swapchain-backbuffer", ResourceKind::Image);
    builder.SetFinalState(backbuffer, ResourceState::ImageState(
                                           PipelineStageIntent::BottomOfPipe, AccessIntent::None,
                                           QueueType::Graphics, ImageLayoutIntent::Present));
    const auto pass = builder.AddPass("main", QueueType::Graphics, true, {});
    ASSERT_TRUE(builder.AddWrite(pass, backbuffer));
    const auto graph = builder.Compile();
    ASSERT_TRUE(graph.success);

    // Frame 1: no recorded state yet -> first use is Undefined.
    const auto frame1 = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{});
    ASSERT_FALSE(frame1.passes[0].pre_image.empty());
    EXPECT_EQ(frame1.passes[0].pre_image.front().old_layout, vk::ImageLayout::eUndefined);

    // Frame 2: feed frame 1's recorded end state (Present) back in.
    RuntimeResourceStates runtime{};
    runtime.states = frame1.end_states;
    runtime.has_state = frame1.has_end_state;
    const auto frame2 = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{}, &runtime);
    ASSERT_FALSE(frame2.passes[0].pre_image.empty());
    EXPECT_EQ(frame2.passes[0].pre_image.front().old_layout, vk::ImageLayout::ePresentSrcKHR);

    // Swapchain recreation: the caller clears the flag, so no state is supplied
    // and the first barrier discards contents again.
    const auto recreated = PlanBarriers(graph, ResolvedResourceHandles{}, AliasIntervals{});
    EXPECT_EQ(recreated.passes[0].pre_image.front().old_layout, vk::ImageLayout::eUndefined);
}

}  // namespace
