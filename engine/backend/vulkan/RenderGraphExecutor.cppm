module;

export module VulkanBackend.Vulkan.RenderGraphExecutor;

import std;

import vulkan;

import VulkanShared.RenderGraphTypes;

export namespace VulkanBackend::Vulkan {

// Executes a compiled graph using a pre-computed BarrierPlan. The plan carries
// all synchronization; the executor only translates it into synchronization2
// dependency info. No barrier logic lives here.
void ExecuteRenderGraph(const VulkanEngine::RenderGraph::BarrierPlan& plan,
                        const VulkanEngine::RenderGraph::CompiledRenderGraph& graph,
                        const void* user_data,
                        vk::CommandBuffer command_buffer);

// Records only passes [first_pass, last_pass) into command_buffer. Used by the
// multi-queue submission path to record one queue run per command buffer. The
// plan's barrier scopes are already clamped to each pass's queue, so this does
// not need to know which queue the command buffer belongs to.
void ExecuteRenderGraphRange(const VulkanEngine::RenderGraph::BarrierPlan& plan,
                             const VulkanEngine::RenderGraph::CompiledRenderGraph& graph,
                             std::uint32_t first_pass,
                             std::uint32_t last_pass,
                             const void* user_data,
                             vk::CommandBuffer command_buffer);

} // namespace VulkanBackend::Vulkan
