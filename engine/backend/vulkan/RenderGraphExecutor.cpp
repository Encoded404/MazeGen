module;

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.RenderGraphExecutor;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanShared.RenderGraphTypes;
import VulkanBackend.Vulkan.GpuPassTimer;
import VulkanBackend.Vulkan.VulkanDebugUtils;

using VulkanEngine::RenderGraph::BarrierPlan;
using VulkanEngine::RenderGraph::CompiledPass;
using VulkanEngine::RenderGraph::CompiledRenderGraph;
using VulkanEngine::RenderGraph::PlannedBufferBarrier;
using VulkanEngine::RenderGraph::PlannedImageBarrier;
using VulkanEngine::RenderGraph::PlannedPassBarriers;
using VulkanEngine::RenderGraph::ResourceKind;

namespace VulkanBackend::Vulkan {

namespace {

// sync2 only: a single dependency info carries the image/buffer/global
// barriers. Unresolved buffers degrade to a conservative global memory barrier
// with the same stage/access scopes, so correctness does not depend on a
// resolver having been registered.
// The planned scopes are already clamped to the pass's queue by PlanBarriers
// (see ClampStagesToQueue/ClampAccessToQueue), so this is a direct sync2
// translation with no queue-specific rewriting.
void EmitPlannedBarriers(vk::CommandBuffer command_buffer,
                         const std::vector<PlannedImageBarrier>& planned_images,
                         const std::vector<PlannedBufferBarrier>& planned_buffers) {
    std::vector<vk::ImageMemoryBarrier2> image_barriers;
    std::vector<vk::BufferMemoryBarrier2> buffer_barriers;
    std::vector<vk::MemoryBarrier2> global_barriers;

    image_barriers.reserve(planned_images.size());
    const auto add_image = [&](const PlannedImageBarrier& barrier) {
        if (!barrier.image) {
            return;
        }
        image_barriers.emplace_back(barrier.src_stage,
                                    barrier.src_access,
                                    barrier.dst_stage,
                                    barrier.dst_access,
                                    barrier.old_layout,
                                    barrier.new_layout,
                                    vk::QueueFamilyIgnored,
                                    vk::QueueFamilyIgnored,
                                    barrier.image,
                                    barrier.range);
    };

    const auto add_buffer = [&](const PlannedBufferBarrier& barrier) {
        if (!barrier.buffer) {
            global_barriers.emplace_back(barrier.src_stage,
                                         barrier.src_access,
                                         barrier.dst_stage,
                                         barrier.dst_access);
            return;
        }
        buffer_barriers.emplace_back(barrier.src_stage,
                                     barrier.src_access,
                                     barrier.dst_stage,
                                     barrier.dst_access,
                                     vk::QueueFamilyIgnored,
                                     vk::QueueFamilyIgnored,
                                     barrier.buffer,
                                     barrier.offset,
                                     barrier.size);
    };

    for (const auto& barrier : planned_images) add_image(barrier);
    for (const auto& barrier : planned_buffers) add_buffer(barrier);

    if (image_barriers.empty() && buffer_barriers.empty() && global_barriers.empty()) {
        return;
    }

    vk::DependencyInfo dependency{};
    dependency.dependencyFlags = vk::DependencyFlags{};
    dependency.memoryBarrierCount = static_cast<std::uint32_t>(global_barriers.size());
    dependency.pMemoryBarriers = global_barriers.empty() ? nullptr : global_barriers.data();
    dependency.bufferMemoryBarrierCount = static_cast<std::uint32_t>(buffer_barriers.size());
    dependency.pBufferMemoryBarriers = buffer_barriers.empty() ? nullptr : buffer_barriers.data();
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(image_barriers.size());
    dependency.pImageMemoryBarriers = image_barriers.empty() ? nullptr : image_barriers.data();

    command_buffer.pipelineBarrier2(dependency);
}

}  // namespace

void ExecuteRenderGraphRange(const BarrierPlan& plan,
                             const CompiledRenderGraph& graph,
                             std::uint32_t first_pass,
                             std::uint32_t last_pass,
                             const void* user_data,
                             vk::CommandBuffer command_buffer,
                             GpuPassTimer* gpu_pass_timer) {
    if (!graph.success || !plan.valid) {
        return;
    }

    last_pass = std::min<std::uint32_t>(last_pass, static_cast<std::uint32_t>(graph.passes.size()));
    for (std::uint32_t pass_index = first_pass; pass_index < last_pass; ++pass_index) {
        const auto& pass = graph.passes[pass_index];
        const PlannedPassBarriers* planned =
            pass_index < plan.passes.size() ? &plan.passes[pass_index] : nullptr;

        BeginDebugUtilsLabel(command_buffer, pass.name);

        if (planned != nullptr) {
            EmitPlannedBarriers(command_buffer, planned->pre_image, planned->pre_buffer);
        }

        if (pass.attachment_setup.has_value() && pass.attachment_setup->auto_begin_rendering) {
            const auto& setup = *pass.attachment_setup;

            std::vector<vk::RenderingAttachmentInfo> color_attachments;
            color_attachments.reserve(setup.color_attachments.size());
            for (const auto& attach : setup.color_attachments) {
                vk::RenderingAttachmentInfo info{};
                info.imageView = attach.image_view;
                info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
                info.loadOp = attach.load_op;
                info.storeOp = attach.store_op;
                if (attach.load_op == vk::AttachmentLoadOp::eClear) {
                    info.clearValue = vk::ClearValue(attach.clear_color);
                }
                color_attachments.push_back(info);
            }

            std::optional<vk::RenderingAttachmentInfo> depth_attachment;
            if (setup.depth_attachment.has_value()) {
                depth_attachment = vk::RenderingAttachmentInfo{};
                depth_attachment->imageView = setup.depth_attachment->image_view;
                depth_attachment->imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depth_attachment->loadOp = setup.depth_attachment->load_op;
                depth_attachment->storeOp = setup.depth_attachment->store_op;
                if (setup.depth_attachment->load_op == vk::AttachmentLoadOp::eClear) {
                    depth_attachment->clearValue = vk::ClearValue(setup.depth_attachment->clear_depth);
                }
            }

            vk::RenderingInfo render_info{};
            render_info.renderArea = setup.render_area;
            render_info.layerCount = 1;
            render_info.colorAttachmentCount = static_cast<std::uint32_t>(color_attachments.size());
            render_info.pColorAttachments = color_attachments.data();
            if (depth_attachment) {
                render_info.pDepthAttachment = &*depth_attachment;
            }

            command_buffer.beginRendering(render_info);
        }

        if (pass.execute.callback) {
            // Only passes with work are timed; the reserved prep index stays
            // untimed here because the Renderer writes that pair around its
            // pre-graph work.
            const bool timed = gpu_pass_timer != nullptr && gpu_pass_timer->Enabled() &&
                               pass_index < GpuPassTimer::kPrepPassIndex;
            if (timed) {
                gpu_pass_timer->WriteBegin(command_buffer, pass_index, pass.name);
            }
            LOGIFACE_LOG(trace, std::format("RenderGraphExecutor: executing pass '{}'", pass.name));
            pass.execute.callback(user_data, command_buffer);
            if (timed) {
                gpu_pass_timer->WriteEnd(command_buffer, pass_index);
            }
        }

        if (pass.attachment_setup.has_value() && pass.attachment_setup->auto_begin_rendering) {
            command_buffer.endRendering();
        }

        if (planned != nullptr) {
            EmitPlannedBarriers(command_buffer, planned->post_image, planned->post_buffer);
        }

        EndDebugUtilsLabel(command_buffer);
    }
}

void ExecuteRenderGraph(const BarrierPlan& plan,
                        const CompiledRenderGraph& graph,
                        const void* user_data,
                        vk::CommandBuffer command_buffer) {
    ExecuteRenderGraphRange(plan, graph, 0, static_cast<std::uint32_t>(graph.passes.size()),
                            user_data, command_buffer);
}

} // namespace VulkanBackend::Vulkan
