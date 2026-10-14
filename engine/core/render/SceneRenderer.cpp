module;

#include <cassert>

#include <logging/logging_macros.hpp>

module VulkanEngine.SceneRenderer;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.GpuResources;
import VulkanEngine.StandardMeshPipeline;
import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderRegistration;
import VulkanBackend.Vulkan.VulkanCapabilities;

namespace VulkanEngine::SceneRenderer {

SceneRenderer::~SceneRenderer() {
    Shutdown();
}

bool SceneRenderer::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& be,
                                VulkanEngine::GpuResources::DeviceBufferHeap& vh,
                                SceneCapacity initial_capacity,
                                ShaderSystem::ShaderManager& shader_mgr,
                                ShaderSystem::PipelineFactory& pipeline_factory,
                                const EngineShaderIds& shader_ids,
                                const std::uint32_t frames_in_flight,
                                DrawMode draw_mode) {
    backend_ = &be;
    shader_mgr_ = &shader_mgr;
    pipeline_factory_ = &pipeline_factory;
    shader_ids_ = shader_ids;
    const auto& dev = be.GetDevice();
    draw_indirect_count_supported_ =
        be.GetCapabilities().IsFeatureEnabled(
            VulkanBackend::Vulkan::Feature::DrawIndirectCount);
    draw_indirect_first_instance_supported_ =
        be.GetCapabilities().IsFeatureEnabled(
            VulkanBackend::Vulkan::Feature::DrawIndirectFirstInstance);
    // MID needs both: the count-driven draw and a non-zero firstInstance
    // (the submesh id). Store the combined rule once so selection and
    // Reinitialize cannot disagree.
    mid_supported_ =
        draw_indirect_count_supported_ && draw_indirect_first_instance_supported_;
    draw_mode_ = ResolveDrawMode(draw_mode);
    if (draw_mode_ != draw_mode) {
        LOGIFACE_LOG(warn, "SceneRenderer: drawIndirectCount and drawIndirectFirstInstance "
                           "not both supported; falling back to CID draw mode");
    }

    // Resolve the submesh-block budget from the device before any descriptor
    // layout is created: binding array sizes and descriptor pool sizes both
    // derive from it. Descriptor binding arrays are static, so the compile-time
    // ceiling bounds them and the device clamps below it.
    {
        const auto& limits = be.GetCapabilities().GetProperties().limits;
        const SceneLimits::DeviceLimits device_limits{
            .max_descriptor_set_storage_buffers =
                limits.maxDescriptorSetStorageBuffers,
            .max_per_stage_descriptor_storage_buffers =
                limits.maxPerStageDescriptorStorageBuffers,
        };
        max_blocks_ = std::clamp(SceneLimits::MaxBlocksForDevice(device_limits),
                                 1u, SceneLimits::kMaxBlocksCeiling);
        LOGIFACE_LOG(info, "SceneRenderer: submesh block budget max_blocks=" +
                     std::to_string(max_blocks_) + " (" + std::to_string(MaxSubmeshes()) +
                     " submeshes; ceiling " +
                     std::to_string(SceneLimits::kMaxBlocksCeiling) + ")");
        if (max_blocks_ < SceneLimits::kMaxBlocksCeiling) {
            LOGIFACE_LOG(info, "SceneRenderer: block budget clamped by device descriptor limits "
                         "(maxDescriptorSetStorageBuffers=" +
                         std::to_string(limits.maxDescriptorSetStorageBuffers) +
                         ", maxPerStageDescriptorStorageBuffers=" +
                         std::to_string(limits.maxPerStageDescriptorStorageBuffers) + ")");
        }
    }
    scene_capacity_ = SceneCapacity{
        std::max(initial_capacity.index_count, 1u),
        std::max(initial_capacity.vertex_span, 1u),
        std::max(initial_capacity.submesh_count, 1u),
    };

    // Size the per-frame resource ring to the configured pipeline depth. All
    // descriptor pools and per-frame GPU resources scale with this count.
    frames_in_flight_ = std::max(frames_in_flight, 1u);
    frames_.resize(frames_in_flight_);
    empty_sets_.clear();
    empty_sets_.reserve(frames_in_flight_);

    // Size every pipeline retire ring consistently with the frame depth.
    expand_slot_.SetFramesInFlight(frames_in_flight_);
    occlusion_slot_.SetFramesInFlight(frames_in_flight_);
    hiz_slot_.SetFramesInFlight(frames_in_flight_);
    collect_count_slot_.SetFramesInFlight(frames_in_flight_);
    collect_write_slot_.SetFramesInFlight(frames_in_flight_);
    depth_slot_.SetFramesInFlight(frames_in_flight_);
    occluder_select_slot_.SetFramesInFlight(frames_in_flight_);
    pre_cull_slot_.SetFramesInFlight(frames_in_flight_);

    // Set 1: SubmeshVertexData (block array, simple layout)
    {
        std::array<vk::DescriptorSetLayoutBinding, 1> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[0].descriptorCount = max_blocks_;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eVertex |
                           vk::ShaderStageFlagBits::eCompute;
        submesh_vertex_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{{}, static_cast<std::uint32_t>(bs.size()), bs.data()});
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *submesh_vertex_layout_, "submesh-vertex-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_buffers = frames_in_flight_ *
            SceneLimits::StorageBuffersForPlan(SceneLimits::kSubmeshVertexPlan, max_blocks_);
        submesh_vertex_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        submesh_vertex_pool_->SetDebugName(dev, "submesh-vertex-pool");
    }

    // Set 2: Vertex buffer table (binding 0) and out-of-line UV table
    // (binding 1). Both are fixed-count: the spec allows the variable-count
    // flag only on the highest binding, and neither array's runtime length is
    // ever queried, so the flag is dropped and both bindings are allocated at
    // their full fixed count (partially bound, update-after-bind).
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[0].descriptorCount = MAX_VERTEX_BUFFERS;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
        bs[1].binding = 1;
        bs[1].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[1].descriptorCount = MAX_UV_BUFFERS;
        bs[1].stageFlags = vk::ShaderStageFlagBits::eVertex;
        const std::array<vk::DescriptorBindingFlags, 2> flags = {
            vk::DescriptorBindingFlagBits::ePartiallyBound |
                vk::DescriptorBindingFlagBits::eUpdateAfterBind,
            vk::DescriptorBindingFlagBits::ePartiallyBound |
                vk::DescriptorBindingFlagBits::eUpdateAfterBind,
        };
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bind_flags{};
        bind_flags.bindingCount = static_cast<std::uint32_t>(flags.size());
        bind_flags.pBindingFlags = flags.data();
        vk::DescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
        layout_ci.pNext = &bind_flags;
        layout_ci.bindingCount = static_cast<std::uint32_t>(bs.size());
        layout_ci.pBindings = bs.data();
        vertex_buffers_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(dev, layout_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *vertex_buffers_layout_, "vertex-buffers-layout");

        const vk::DescriptorPoolSize ps{
            vk::DescriptorType::eStorageBuffer,
            frames_in_flight_ * (MAX_VERTEX_BUFFERS + MAX_UV_BUFFERS)
        };
        vk::DescriptorPoolCreateInfo pool_ci{};
        pool_ci.flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
                        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        pool_ci.maxSets = frames_in_flight_;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &ps;
        vertex_buffers_pool_ = std::make_unique<vk::raii::DescriptorPool>(dev, pool_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *vertex_buffers_pool_, "vertex-buffers-pool");

        for (auto& fr : frames_) {
            vk::DescriptorSetAllocateInfo alloc_ci{};
            alloc_ci.descriptorPool = **vertex_buffers_pool_;
            alloc_ci.descriptorSetCount = 1;
            alloc_ci.pSetLayouts = &**vertex_buffers_layout_;
            auto sets = dev.allocateDescriptorSets(alloc_ci);
            fr.vertex_buffers_set = std::move(sets[0]);
        }

        for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
            VulkanBackend::Vulkan::SetVulkanObjectName(
                dev, frames_[i].vertex_buffers_set, "vertex-buffers-frame-" + std::to_string(i));
        }

        // Write initial static blocks into ALL frame vertex sets
        for (auto& fr : frames_) {
            for (std::uint32_t bi = 0; bi < vh.GetBufferCount(); ++bi) {
                const vk::DescriptorBufferInfo bii(vh.GetBuffer(bi), 0, vk::WholeSize);
                vk::WriteDescriptorSet w{};
                w.dstSet = *fr.vertex_buffers_set;
                w.dstBinding = 0;
                w.dstArrayElement = bi;
                w.descriptorCount = 1;
                w.descriptorType = vk::DescriptorType::eStorageBuffer;
                w.pBufferInfo = &bii;
                dev.updateDescriptorSets(w, nullptr);
            }
            // UV table (binding 1) blocks are written as they are allocated, by
            // the caller that uploads the mesh, exactly like vertex blocks.
        }
    }

    // Set 3: Indirection buffer (single, update-after-bind for per-frame swapping)
    {
        std::array<vk::DescriptorSetLayoutBinding, 1> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[0].descriptorCount = 1;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
        auto flags = vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::eUpdateAfterBind);
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bind_flags{};
        bind_flags.bindingCount = 1;
        bind_flags.pBindingFlags = &flags;
        vk::DescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.pNext = &bind_flags;
        layout_ci.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
        layout_ci.bindingCount = 1;
        layout_ci.pBindings = bs.data();
        indirection_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(dev, layout_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *indirection_layout_, "indirection-layout");
        const vk::DescriptorPoolSize indir_ps{
            vk::DescriptorType::eStorageBuffer, frames_in_flight_
        };
        vk::DescriptorPoolCreateInfo indir_pool_ci{};
        indir_pool_ci.flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
                              vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        indir_pool_ci.maxSets = frames_in_flight_;
        indir_pool_ci.poolSizeCount = 1;
        indir_pool_ci.pPoolSizes = &indir_ps;
        indirection_raw_pool_ = std::make_unique<vk::raii::DescriptorPool>(dev, indir_pool_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *indirection_raw_pool_, "indirection-raw-pool");
    }

    // Index buffer table (bindless, update-after-bind) for expand - per-frame
    {
        std::array<vk::DescriptorSetLayoutBinding, 1> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[0].descriptorCount = MAX_INDEX_BUFFERS;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        auto flags = vk::DescriptorBindingFlagBits::ePartiallyBound |
                     vk::DescriptorBindingFlagBits::eUpdateAfterBind |
                     vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bind_flags{};
        bind_flags.bindingCount = 1;
        bind_flags.pBindingFlags = &flags;
        vk::DescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
        layout_ci.pNext = &bind_flags;
        layout_ci.bindingCount = 1;
        layout_ci.pBindings = bs.data();
        index_buffers_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(dev, layout_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *index_buffers_layout_, "index-buffers-layout");

        const vk::DescriptorPoolSize ps{
            vk::DescriptorType::eStorageBuffer, frames_in_flight_ * MAX_INDEX_BUFFERS
        };
        vk::DescriptorPoolCreateInfo pool_ci{};
        pool_ci.flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
                        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        pool_ci.maxSets = frames_in_flight_;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &ps;
        index_buffers_pool_ = std::make_unique<vk::raii::DescriptorPool>(dev, pool_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *index_buffers_pool_, "index-buffers-pool");

        for (auto& fr : frames_) {
            const std::uint32_t var_desc_count = MAX_INDEX_BUFFERS;
            vk::DescriptorSetVariableDescriptorCountAllocateInfo var_desc{};
            var_desc.descriptorSetCount = 1;
            var_desc.pDescriptorCounts = &var_desc_count;
            vk::DescriptorSetAllocateInfo alloc_ci{};
            alloc_ci.pNext = &var_desc;
            alloc_ci.descriptorPool = **index_buffers_pool_;
            alloc_ci.descriptorSetCount = 1;
            alloc_ci.pSetLayouts = &**index_buffers_layout_;
            auto sets = dev.allocateDescriptorSets(alloc_ci);
            fr.index_buffers_set = std::move(sets[0]);
        }

        for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
            VulkanBackend::Vulkan::SetVulkanObjectName(
                dev, frames_[i].index_buffers_set, "index-buffers-frame-" + std::to_string(i));
        }
    }

    // Set 4: Expand layout (7 bindings: 4 block arrays + vertex_indirection +
    // draw_indices + expand counter)
    {
        std::array<vk::DescriptorSetLayoutBinding, 7> bs{};
        for (std::uint32_t i = 0; i < 4; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = max_blocks_;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        for (std::uint32_t i = 4; i < 7; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = 1;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        expand_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{
                {}, static_cast<std::uint32_t>(bs.size()), bs.data() });
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *expand_layout_, "expand-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_buffers = frames_in_flight_ *
            SceneLimits::StorageBuffersForPlan(SceneLimits::kExpandPlan, max_blocks_);
        expand_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        expand_pool_->SetDebugName(dev, "expand-pool");
    }

    // Set 5: Occlusion layout (10 bindings: vertex info, cull, spheres, Hi-Z,
    // OBBs, flag table — shared by the occlusion pass and the pre-cull pass,
    // which adds the survivor compaction bindings 6-8)
    {
        std::array<vk::DescriptorSetLayoutBinding, 9> bs{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = max_blocks_;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        bs[3].binding = 3;
        bs[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bs[3].descriptorCount = 1;
        bs[3].stageFlags = vk::ShaderStageFlagBits::eCompute;
        bs[4].binding = 4;
        bs[4].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[4].descriptorCount = max_blocks_;
        bs[4].stageFlags = vk::ShaderStageFlagBits::eCompute;
        for (std::uint32_t i = 5; i < 9; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = 1;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        occlusion_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{
                {}, static_cast<std::uint32_t>(bs.size()), bs.data() });
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *occlusion_layout_, "occlusion-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_buffers = frames_in_flight_ *
            SceneLimits::StorageBuffersForPlan(SceneLimits::kOcclusionPlan, max_blocks_);
        pc.max_sampled_images = frames_in_flight_;
        pc.max_combined_image_samplers = frames_in_flight_;
        occlusion_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        occlusion_pool_->SetDebugName(dev, "occlusion-pool");
    }

    // Set 8: Occluder-select layout (8 bindings: cull blocks, transform blocks,
    // OBB blocks, flag table, full indirection, occluder indirection, draw
    // command, selection counter)
    {
        std::array<vk::DescriptorSetLayoutBinding, 8> bs{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = max_blocks_;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        for (std::uint32_t i = 3; i < 8; ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = 1;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        occluder_select_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{
                {}, static_cast<std::uint32_t>(bs.size()), bs.data() });
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *occluder_select_layout_, "occluder-select-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_buffers = frames_in_flight_ *
            SceneLimits::StorageBuffersForPlan(SceneLimits::kOccluderSelectPlan, max_blocks_);
        occluder_select_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        occluder_select_pool_->SetDebugName(dev, "occluder-select-pool");
    }

    // Set 6: Collect count/compact layout (6 bindings: cull blocks,
    // draw_indices, main output, technique results, technique region bases,
    // technique counts)
    {
        std::array<vk::DescriptorSetLayoutBinding, 6> bs{};
        bs[0].binding = 0;
        bs[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bs[0].descriptorCount = max_blocks_;
        bs[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        for (std::uint32_t i = 1; i < bs.size(); ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = 1;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        collect_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{
                {}, static_cast<std::uint32_t>(bs.size()), bs.data() });
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *collect_layout_, "collect-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        pc.max_storage_buffers = frames_in_flight_ *
            SceneLimits::StorageBuffersForPlan(SceneLimits::kCollectPlan, max_blocks_);
        collect_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        collect_pool_->SetDebugName(dev, "collect-pool");
    }

    // Set 7: Collect write shader layout (CID-only: emits one indexed
    // indirect command per technique from the technique results).
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bs{};
        for (std::uint32_t i = 0; i < bs.size(); ++i) {
            bs[i].binding = i;
            bs[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bs[i].descriptorCount = 1;
            bs[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        collect_write_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            dev, vk::DescriptorSetLayoutCreateInfo{
                {}, static_cast<std::uint32_t>(bs.size()), bs.data() });
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *collect_write_layout_, "collect-write-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_ + 1;
        pc.max_storage_buffers = frames_in_flight_ * 2;
        collect_write_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        collect_write_pool_->SetDebugName(dev, "collect-write-pool");
    }

    // Empty set (placeholder for set 0 in depth pass)
    {
        constexpr vk::DescriptorSetLayoutCreateInfo empty_ci{};
        empty_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(dev, empty_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *empty_layout_, "empty-layout");
        GpuResources::DescriptorPoolConfig pc{};
        pc.max_sets = frames_in_flight_;
        empty_pool_ = GpuResources::DescriptorPool::Create(be, pc);
        empty_pool_->SetDebugName(dev, "empty-pool");
        empty_sets_.reserve(frames_in_flight_);
        for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
            auto set = empty_pool_->Allocate(*empty_layout_);
            set.SetDebugName(dev, "empty-set-" + std::to_string(i));
            empty_sets_.push_back(std::move(set));
        }
    }

    // Create all compute/graphics pipelines in one batched submission
    if (!CreateScenePipelines(be, shader_mgr, pipeline_factory, shader_ids)) return false;

    // Per-frame ring resources
    {
        constexpr std::uint64_t technique_flags_size =
            static_cast<uint64_t>(MAX_DRAW_GROUPS) * sizeof(std::uint32_t);

        // Every per-submesh block array is bounded to the runtime block budget:
        // its descriptors are a max_blocks_-sized binding array, so growing past
        // that would write descriptors out of range.
        auto make_block_config = [max_blocks = max_blocks_](std::uint32_t entry_size,
                                                            std::uint32_t entries_per_block,
                                                            vk::BufferUsageFlags extra_usage,
                                                            vk::MemoryPropertyFlags memory) {
            GpuResources::BlockArray::Config c{};
            c.entry_size = entry_size;
            c.entries_per_block = entries_per_block;
            c.max_blocks = max_blocks;
            c.extra_usage = extra_usage;
            c.memory = memory;
            return c;
        };

        // Shared technique flag table (single buffer, not per-frame — see
        // SceneRenderer.cppm). Created before the ring loop because the
        // per-frame depth-filter descriptor sets bind it.
        technique_flags = GpuResources::GpuBuffer::Create(be,
            technique_flags_size,
            vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, technique_flags.GetBuffer(), vk::ObjectType::eBuffer, "technique-flags");

        for (auto& fr : frames_) {
            fr.dynamic_entries.Initialize(be,
                make_block_config(48, BLOCK_ENTRIES, {},
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent));
            fr.static_entries.Initialize(be,
                make_block_config(36, BLOCK_ENTRIES, {},   // StaticEntry: 9 u32 total (incl. uvSlot/uvBias)
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent));
            fr.bounding_spheres.Initialize(be,
                make_block_config(16, BLOCK_ENTRIES, {},
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent));
            fr.obb_entries.Initialize(be,
                make_block_config(64, BLOCK_ENTRIES, {},
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent));
            fr.submesh_vertex_entries.Initialize(be,
                make_block_config(188, BLOCK_ENTRIES,   // sizeof(VertexEntry) = 188, Slang CDataLayout
                                                        // (scalar block layout): 64 mvp + 12 (maxScale/mat/orm)
                                                        // + 64 modelMatrix + 36 normalMatrix + 4 slot + 4 uvSlot
                                                        // + 4 uvBias. No padding — matrices are 4B-aligned.
                                                        // Byte-identical to the VertexEntry mirror +
                                                        // static_assert in MeshGatherSystem.cpp. MUST match the
                                                        // VertexEntry definition in scene_entries.slang.
                    vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eDeviceLocal));
            fr.cull_entries.Initialize(be,
                make_block_config(16, BLOCK_ENTRIES,
                    vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eDeviceLocal));

            // Capacity- and mode-dependent GPU buffers are created by
            // CreateFrameBuffers() below (and re-created on capacity growth).
            fr.expand_set = expand_pool_->Allocate(*expand_layout_);
            fr.occlusion_set = occlusion_pool_->Allocate(*occlusion_layout_);
            fr.collect_set = collect_pool_->Allocate(*collect_layout_);
            fr.collect_write_set = collect_write_pool_->Allocate(*collect_write_layout_);
            fr.occluder_select_set = occluder_select_pool_->Allocate(*occluder_select_layout_);
            fr.submesh_vertex_set =
                submesh_vertex_pool_->Allocate(*submesh_vertex_layout_);
            {
                vk::DescriptorSetAllocateInfo alloc_ci{};
                alloc_ci.descriptorPool = **indirection_raw_pool_;
                alloc_ci.descriptorSetCount = 1;
                alloc_ci.pSetLayouts = &**indirection_layout_;
                auto sets = dev.allocateDescriptorSets(alloc_ci);
                fr.indirection_raw_set = std::move(sets[0]);
                VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.indirection_raw_set, "indirection-raw-set");
            }
            fr.hiz_set = hiz_pool_->Allocate(*hiz_layout_);

            fr.expand_set.SetDebugName(dev, "expand-set");
            fr.occlusion_set.SetDebugName(dev, "occlusion-set");
            fr.collect_set.SetDebugName(dev, "collect-set");
            fr.collect_write_set.SetDebugName(dev, "collect-write-set");
            fr.occluder_select_set.SetDebugName(dev, "occluder-select-set");
            fr.submesh_vertex_set.SetDebugName(dev, "submesh-vertex-set");
            fr.hiz_set.SetDebugName(dev, "hiz-set");

        }

        if (!CreateFrameBuffers()) return false;
    }

    (void)be.GetSwapchainExtent(depth_width_, depth_height_);

    if (!CreateHiZResources()) return false;

    // ── Lighting system (descriptor set 4) ──
    {
        const vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex |
                                             vk::ShaderStageFlagBits::eFragment;

        // Binding 0: SceneHeader (single storage buffer)
        const vk::DescriptorSetLayoutBinding header_binding(0, vk::DescriptorType::eStorageBuffer,
                                                             1, stages);
        // Binding 1: Light[] BlockArray (array of storage buffers, partially bound)
        const vk::DescriptorSetLayoutBinding lights_binding(1, vk::DescriptorType::eStorageBuffer,
                                                             MAX_LIGHT_BLOCKS, stages);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {header_binding, lights_binding};
        std::array<vk::DescriptorBindingFlags, 2> flags_arr{};
        flags_arr[1] = vk::DescriptorBindingFlagBits::ePartiallyBound;
        const vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info(
            static_cast<std::uint32_t>(flags_arr.size()), flags_arr.data());

        const vk::DescriptorSetLayoutCreateInfo layout_info({}, bindings, &flags_info);
        scene_uniform_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(dev, layout_info);

        // Pool: 1 header buffer + MAX_LIGHT_BLOCKS light block buffers
        std::vector<vk::DescriptorPoolSize> pool_sizes = {
            {vk::DescriptorType::eStorageBuffer, 1 + MAX_LIGHT_BLOCKS}};
        scene_uniform_pool_ = std::make_unique<vk::raii::DescriptorPool>(
            dev, vk::DescriptorPoolCreateInfo{
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, pool_sizes});

        vk::DescriptorSetAllocateInfo alloc_ci{};
        alloc_ci.descriptorPool = **scene_uniform_pool_;
        alloc_ci.descriptorSetCount = 1;
        alloc_ci.pSetLayouts = &**scene_uniform_layout_;
        auto sets = dev.allocateDescriptorSets(alloc_ci);
        scene_uniform_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));

        // Create device-local header buffer
        scene_header = GpuResources::GpuBuffer::Create(
            *backend_, sizeof(SceneHeader),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal, nullptr);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, scene_header.GetBuffer(),
                                                     "scene-header");

        // Write header buffer to descriptor set binding 0
        const vk::DescriptorBufferInfo header_buf_info(
            static_cast<vk::Buffer>(*scene_header.GetBuffer()), 0,
            sizeof(SceneHeader));
        vk::WriteDescriptorSet header_write{};
        header_write.dstSet = **scene_uniform_set_;
        header_write.dstBinding = 0;
        header_write.descriptorCount = 1;
        header_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        header_write.pBufferInfo = &header_buf_info;
        dev.updateDescriptorSets(header_write, nullptr);

        // Initialize BlockArray for lights
        GpuResources::BlockArray::Config light_cfg{};
        light_cfg.entry_size = sizeof(Light);
        light_cfg.entries_per_block = LIGHTS_PER_BLOCK;
        // The light descriptor binding is a fixed MAX_LIGHT_BLOCKS array; the
        // BlockArray must not grow past it.
        light_cfg.max_blocks = MAX_LIGHT_BLOCKS;
        light_cfg.memory_mode = GpuResources::MemoryMode::DeviceLocal;
        light_cfg.memory = vk::MemoryPropertyFlagBits::eDeviceLocal;
        light_cfg.extra_usage = vk::BufferUsageFlagBits::eTransferDst; // NOLINT
        scene_lights.Initialize(*backend_, light_cfg);
    }

    LOGIFACE_LOG(info, "SceneRenderer initialized");
    return true;
}

bool SceneRenderer::IsDrawModeSupported(DrawMode mode) const {
    switch (mode) {
        case DrawMode::CID: return true;
        case DrawMode::MID: return mid_supported_;
    }
    return false;
}

DrawMode SceneRenderer::ResolveDrawMode(DrawMode requested) const {
    if (requested == DrawMode::MID && !mid_supported_) {
        return DrawMode::CID;
    }
    return requested;
}

bool SceneRenderer::CreateFrameBuffers() {
    if (!backend_) return false;
    auto& be = *backend_;
    const auto& dev = be.GetDevice();

    // Any descriptor written against the previous buffers is stale; force a
    // full block-descriptor rewrite on the next PrepareCompute.
    ++binding_epoch_;

    const std::uint64_t vertex_indirection_size =
        static_cast<std::uint64_t>(scene_capacity_.vertex_span) * 8u;
    const std::uint64_t draw_indices_size =
        static_cast<std::uint64_t>(scene_capacity_.index_count) * 4u;
    const std::uint64_t compact_size = draw_indices_size;
    const std::uint64_t command_size =
        static_cast<std::uint64_t>(scene_capacity_.submesh_count) *
            sizeof(vk::DrawIndexedIndirectCommand);
    constexpr std::uint64_t tech_counts_size =
        static_cast<std::uint64_t>(MAX_DRAW_GROUPS) * sizeof(std::uint32_t);
    constexpr std::uint64_t technique_results_size =
        static_cast<std::uint64_t>(MAX_DRAW_GROUPS) * sizeof(TechniqueResult);
    constexpr std::uint64_t occluder_count_size = sizeof(std::uint32_t);

    const bool mid = draw_mode_ == DrawMode::MID;
    const vk::BufferUsageFlags index_dest_usage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eTransferDst;
    const vk::BufferUsageFlags indirect_usage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eTransferDst;
    const vk::BufferUsageFlags storage_usage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst;

    for (auto& fr : frames_) {
        // MID neither writes nor reads vertex_indirection: it carries the
        // vertex buffer slot in VertexEntry and the vertex index in
        // draw_indices. A fixed 8 B dummy keeps both descriptor writes (the
        // shared indirection set and the expand set) valid without allocating
        // 8 B per vertex window slot that MID would never touch.
        const std::uint64_t indirection_bytes =
            mid ? 8u : std::max<std::uint64_t>(vertex_indirection_size, 8u);
        fr.vertex_indirection = GpuResources::GpuBuffer::Create(
            be, indirection_bytes,
            vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.vertex_indirection.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "vertex-indirection");

        // draw_indices is bound as an index buffer only in MID mode.
        const vk::BufferUsageFlags draw_indices_usage =
            mid ? index_dest_usage : storage_usage;
        fr.draw_indices = GpuResources::GpuBuffer::Create(
            be, std::max<std::uint64_t>(draw_indices_size, 4u),
            draw_indices_usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.draw_indices.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "draw-indices");

        fr.expand_counter = GpuResources::GpuBuffer::Create(
            be, 2u * sizeof(std::uint32_t), storage_usage,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.expand_counter.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "expand-counter");

        if (!mid) {
            fr.main_indices = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(compact_size, 4u), index_dest_usage,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            fr.depth_indices = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(compact_size, 4u), index_dest_usage,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            fr.occluder_indices = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(compact_size, 4u), index_dest_usage,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.main_indices.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "main-indices");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.depth_indices.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "depth-indices");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.occluder_indices.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "occluder-indices");

            fr.depth_out_draw_command = GpuResources::GpuBuffer::Create(
                be, sizeof(vk::DrawIndexedIndirectCommand), indirect_usage,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
            fr.occluder_out_draw_command = GpuResources::GpuBuffer::Create(
                be, sizeof(vk::DrawIndexedIndirectCommand), indirect_usage,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.depth_out_draw_command.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "depth-out-draw-command");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.occluder_out_draw_command.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "occluder-out-draw-command");
        } else {
            fr.main_commands = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(command_size, sizeof(vk::DrawIndexedIndirectCommand)),
                indirect_usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            fr.depth_commands = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(command_size, sizeof(vk::DrawIndexedIndirectCommand)),
                indirect_usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            fr.occluder_commands = GpuResources::GpuBuffer::Create(
                be, std::max<std::uint64_t>(command_size, sizeof(vk::DrawIndexedIndirectCommand)),
                indirect_usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.main_commands.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "main-commands");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.depth_commands.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "depth-commands");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.occluder_commands.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "occluder-commands");

            fr.depth_out_command_count = GpuResources::GpuBuffer::Create(
                be, sizeof(std::uint32_t), indirect_usage,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
            fr.occluder_out_command_count = GpuResources::GpuBuffer::Create(
                be, sizeof(std::uint32_t), indirect_usage,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.depth_out_command_count.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "depth-out-command-count");
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.occluder_out_command_count.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "occluder-out-command-count");
        }

        // Allocated in both modes so the collect layout bindings are always valid.
        fr.technique_region_bases = GpuResources::GpuBuffer::Create(
            be, tech_counts_size, storage_usage,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.technique_region_bases.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "technique-region-bases");

        // technique_draw_commands: CID main pass only. In MID the
        // per-technique commands live in main_commands, so nothing is bound to
        // the CID-only collect-write set.
        if (!mid) {
            fr.technique_draw_commands = GpuResources::GpuBuffer::Create(
                be, static_cast<std::uint64_t>(MAX_DRAW_GROUPS) *
                        sizeof(vk::DrawIndexedIndirectCommand),
                indirect_usage,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.technique_draw_commands.GetBuffer(),
                                                       vk::ObjectType::eBuffer, "technique-draw-cmds");
        }

        fr.occluder_candidate_count = GpuResources::GpuBuffer::Create(
            be, occluder_count_size, storage_usage,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.occluder_candidate_count.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "occluder-candidate-count");

        fr.technique_counts = GpuResources::GpuBuffer::Create(
            be, tech_counts_size,
            mid ? (storage_usage | vk::BufferUsageFlagBits::eIndirectBuffer) : storage_usage,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.technique_counts.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "technique-counts");

        fr.technique_results = GpuResources::GpuBuffer::Create(
            be, technique_results_size, storage_usage,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.technique_results.GetBuffer(),
                                                   vk::ObjectType::eBuffer, "technique-results");
    }

    // CPU-prefix region bases for MID (recomputed when topology changes).
    region_base_.assign(MAX_DRAW_GROUPS, 0);
    region_count_.assign(MAX_DRAW_GROUPS, 0);
    region_total_ = 0;
    technique_count_ = 0;
    return true;
}

void SceneRenderer::DestroyFrameBuffers() {
    for (auto& fr : frames_) {
        fr.vertex_indirection = GpuResources::GpuBuffer{};
        fr.draw_indices = GpuResources::GpuBuffer{};
        fr.expand_counter = GpuResources::GpuBuffer{};
        fr.main_indices = GpuResources::GpuBuffer{};
        fr.depth_indices = GpuResources::GpuBuffer{};
        fr.occluder_indices = GpuResources::GpuBuffer{};
        fr.depth_out_draw_command = GpuResources::GpuBuffer{};
        fr.occluder_out_draw_command = GpuResources::GpuBuffer{};
        fr.main_commands = GpuResources::GpuBuffer{};
        fr.depth_commands = GpuResources::GpuBuffer{};
        fr.occluder_commands = GpuResources::GpuBuffer{};
        fr.depth_out_command_count = GpuResources::GpuBuffer{};
        fr.occluder_out_command_count = GpuResources::GpuBuffer{};
        fr.technique_region_bases = GpuResources::GpuBuffer{};
        fr.occluder_candidate_count = GpuResources::GpuBuffer{};
        fr.technique_draw_commands = GpuResources::GpuBuffer{};
        fr.technique_counts = GpuResources::GpuBuffer{};
        fr.technique_results = GpuResources::GpuBuffer{};
    }
}

void SceneRenderer::DestroyHiZResources() {
    for (auto& fr : frames_) {
        // Views must outlive the image; the image must be destroyed before the
        // device memory it is bound to is freed.
        fr.hiz_mip_views.clear();
        fr.hiz_full_view = vk::raii::ImageView(nullptr);
        fr.hiz_image = vk::raii::Image(nullptr);
        fr.hiz_memory = vk::raii::DeviceMemory(nullptr);
    }
    hiz_sampler_.reset();
    hiz_mip_count_ = 0;
}

bool SceneRenderer::CreateHiZResources() {
    if (!backend_) return false;
    auto& be = *backend_;
    const auto& dev = be.GetDevice();

    const std::uint32_t hiz_w = (depth_width_ + 1) / 2;
    const std::uint32_t hiz_h = (depth_height_ + 1) / 2;
    std::uint32_t max_dim = std::max(hiz_w, hiz_h);
    std::uint32_t mip_levels = 1;
    while (max_dim > 1) { max_dim >>= 1; ++mip_levels; }
    mip_levels = std::min(mip_levels, MAX_HIZ_MIPS);
    const vk::Format hiz_format = vk::Format::eR32Sfloat;

    vk::SamplerCreateInfo sampler_ci{};
    sampler_ci.magFilter = vk::Filter::eNearest;
    sampler_ci.minFilter = vk::Filter::eNearest;
    sampler_ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
    // Clamp instead of the default Repeat: out-of-range UVs (e.g. a sphere center
    // off-screen) must sample the edge texel, whose max depth is a conservative
    // proxy, never a wrapped texel from the opposite edge.
    sampler_ci.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    sampler_ci.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    sampler_ci.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    sampler_ci.minLod = 0.0f;
    sampler_ci.maxLod = static_cast<float>(mip_levels);
    hiz_sampler_ = std::make_unique<vk::raii::Sampler>(dev, sampler_ci);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *hiz_sampler_, "hiz-sampler");

    for (auto& fr : frames_) {
        vk::ImageCreateInfo img_ci{};
        img_ci.imageType = vk::ImageType::e2D;
        img_ci.format = hiz_format;
        img_ci.extent = vk::Extent3D(hiz_w, hiz_h, 1);
        img_ci.mipLevels = mip_levels;
        img_ci.arrayLayers = 1;
        img_ci.samples = vk::SampleCountFlagBits::e1;
        img_ci.tiling = vk::ImageTiling::eOptimal;
        img_ci.usage = vk::ImageUsageFlagBits::eStorage |
                       vk::ImageUsageFlagBits::eSampled |
                       vk::ImageUsageFlagBits::eTransferDst;
        img_ci.initialLayout = vk::ImageLayout::eUndefined;
        fr.hiz_image = vk::raii::Image(dev, img_ci);

        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.hiz_image, "hiz-image");

        const vk::MemoryRequirements mem_req = fr.hiz_image.getMemoryRequirements();
        vk::MemoryAllocateInfo alloc_ci{};
        alloc_ci.allocationSize = mem_req.size;
        alloc_ci.memoryTypeIndex = VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
            be.GetPhysicalDevice(), mem_req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        fr.hiz_memory = vk::raii::DeviceMemory(dev, alloc_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.hiz_memory, "hiz-memory");
        fr.hiz_image.bindMemory(*fr.hiz_memory, 0);

        fr.hiz_mip_views.clear();
        fr.hiz_mip_views.reserve(mip_levels);
        for (std::uint32_t mip = 0; mip < mip_levels; ++mip) {
            vk::ImageViewCreateInfo view_ci{};
            view_ci.image = *fr.hiz_image;
            view_ci.viewType = vk::ImageViewType::e2D;
            view_ci.format = hiz_format;
            view_ci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            view_ci.subresourceRange.baseMipLevel = mip;
            view_ci.subresourceRange.levelCount = 1;
            view_ci.subresourceRange.baseArrayLayer = 0;
            view_ci.subresourceRange.layerCount = 1;
            fr.hiz_mip_views.emplace_back(dev, view_ci);
            VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.hiz_mip_views.back(), "hiz-mip-view-" + std::to_string(mip));
        }

        vk::ImageViewCreateInfo full_view_ci{};
        full_view_ci.image = *fr.hiz_image;
        full_view_ci.viewType = vk::ImageViewType::e2D;
        full_view_ci.format = hiz_format;
        full_view_ci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        full_view_ci.subresourceRange.baseMipLevel = 0;
        full_view_ci.subresourceRange.levelCount = mip_levels;
        full_view_ci.subresourceRange.baseArrayLayer = 0;
        full_view_ci.subresourceRange.layerCount = 1;
        fr.hiz_full_view = vk::raii::ImageView(dev, full_view_ci);
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, fr.hiz_full_view, "hiz-full-view");

        // Bind the whole storage-image array declared by the layout
        // (descriptorCount = MAX_HIZ_MIPS). Re-pointing the tail levels (beyond
        // mip_levels) at the coarsest mip keeps every descriptor valid after a
        // shrink; leaving them would dangle at the previous resize's views.
        std::vector<vk::DescriptorImageInfo> storage_infos;
        storage_infos.reserve(MAX_HIZ_MIPS);
        for (std::uint32_t mip = 0; mip < MAX_HIZ_MIPS; ++mip) {
            const std::uint32_t view_index = std::min(mip, mip_levels - 1);
            storage_infos.emplace_back(
                nullptr, *fr.hiz_mip_views[view_index],
                vk::ImageLayout::eGeneral);
        }
        {
            vk::WriteDescriptorSet w{};
            w.dstSet = fr.hiz_set.GetHandle();
            w.dstBinding = 2;
            w.dstArrayElement = 0;
            w.descriptorCount = MAX_HIZ_MIPS;
            w.descriptorType = vk::DescriptorType::eStorageImage;
            w.pImageInfo = storage_infos.data();
            dev.updateDescriptorSets(w, nullptr);
        }
        // Bind hiz mip 0 as placeholder depth input (binding 0). Uses eGeneral
        // layout to match the storage image binding at binding 2 on same subresource.
        {
            const vk::DescriptorImageInfo depth_info(
                nullptr, *fr.hiz_mip_views[0],
                vk::ImageLayout::eGeneral);
            vk::WriteDescriptorSet w{};
            w.dstSet = fr.hiz_set.GetHandle();
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &depth_info;
            dev.updateDescriptorSets(w, nullptr);
        }
        {
            const vk::DescriptorImageInfo sampler_info(
                **hiz_sampler_, nullptr,
                vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet w{};
            w.dstSet = fr.hiz_set.GetHandle();
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampler;
            w.pImageInfo = &sampler_info;
            dev.updateDescriptorSets(w, nullptr);
        }
    }
    hiz_mip_count_ = mip_levels;

#ifndef NDEBUG
    // Invariant: DispatchHiZGen must produce every level the cull shaders can
    // sample. The generator's coverage and the shaders' MaxHizMip() clamp are
    // independent derivations of the same range, so assert they agree; a
    // regression here leaves a sampled mip unwritten, which reads as undefined
    // memory and false-culls large screen footprints on some launches.
    {
        [[maybe_unused]] std::uint32_t cull_max_mip = 0; // == floor(log2(max(hiz_w, hiz_h)))
        for (std::uint32_t m = std::max(hiz_w, hiz_h); m > 1; m >>= 1) ++cull_max_mip;
        assert(LastHiZLevelWritten(hiz_mip_count_) == hiz_mip_count_ - 1 &&
               "Hi-Z generator does not cover every level; cull shaders may sample an unwritten mip");
        assert(cull_max_mip <= LastHiZLevelWritten(hiz_mip_count_) &&
               "cull shaders can sample a Hi-Z mip the generator never writes");
    }
#endif

    return true;
}

void SceneRenderer::EnsureRenderExtent(std::uint32_t width, std::uint32_t height) {
    if (!backend_ || width == 0 || height == 0) return;
    if (width == depth_width_ && height == depth_height_) return;

    // Swapchain recreation already idled the device, but match the
    // EnsureSceneCapacity/Reinitialize pattern so this is safe however it is
    // reached. Hi-Z is derived per-frame (non-temporal), so the old pyramid is
    // discarded rather than rescaled; the next frame's hiz-gen refills it.
    backend_->GetDevice().waitIdle();

    DestroyHiZResources();
    depth_width_ = width;
    depth_height_ = height;
    if (!CreateHiZResources()) {
        LOGIFACE_LOG(error, "SceneRenderer::EnsureRenderExtent: failed to create Hi-Z resources");
        return;
    }
    hiz_initialized_ = false;

    LOGIFACE_LOG(info, "SceneRenderer: render extent changed to " +
        std::to_string(width) + "x" + std::to_string(height) +
        " (Hi-Z mips=" + std::to_string(hiz_mip_count_) + ")");
}

void SceneRenderer::EnsureSceneCapacity(const SceneCapacity& required) {
    if (!backend_) return;
    if (required.index_count <= scene_capacity_.index_count &&
        required.vertex_span <= scene_capacity_.vertex_span &&
        required.submesh_count <= scene_capacity_.submesh_count) {
        return;
    }
    backend_->GetDevice().waitIdle();
    scene_capacity_.index_count =
        std::max(required.index_count, scene_capacity_.index_count * 2);
    scene_capacity_.vertex_span =
        std::max(required.vertex_span, scene_capacity_.vertex_span * 2);
    scene_capacity_.submesh_count =
        std::max(required.submesh_count, scene_capacity_.submesh_count * 2);
    LOGIFACE_LOG(info, "SceneRenderer: capacity grew to index_count=" +
        std::to_string(scene_capacity_.index_count) + " vertex_span=" +
        std::to_string(scene_capacity_.vertex_span) + " submeshes=" +
        std::to_string(scene_capacity_.submesh_count));
    DestroyFrameBuffers();
    CreateFrameBuffers();
    // Buffer descriptors are rewritten each frame in PrepareCompute.
}

void SceneRenderer::Reinitialize(DrawMode mode) {
    if (mode == draw_mode_) return;
    if (!IsDrawModeSupported(mode)) {
        LOGIFACE_LOG(warn, "SceneRenderer::Reinitialize: requested draw mode unsupported; ignoring");
        return;
    }
    backend_->GetDevice().waitIdle();
    DestroyFrameBuffers();
    draw_mode_ = mode;
    CreateFrameBuffers();
    if (!RebuildCompactionPipelines()) {
        LOGIFACE_LOG(error, "SceneRenderer::Reinitialize: failed to rebuild compaction pipelines");
    }
    LOGIFACE_LOG(info, std::string("SceneRenderer: draw mode set to ") +
        (mode == DrawMode::MID ? "MID" : "CID"));
}

namespace {
    template<typename Desc>
    std::optional<ShaderSystem::PipelineProduct> RebuildForDesc(
        ShaderSystem::PipelineFactory& factory, const Desc& desc,
        ShaderSystem::ShaderManager& shaders) {
        if constexpr (std::same_as<Desc, ShaderSystem::ComputePipelineDesc>) {
            auto result = factory.CreateCompute(desc, shaders);
            return result
                ? std::optional<ShaderSystem::PipelineProduct>(std::move(*result))
                : std::nullopt;
        } else {
            auto result = factory.CreateGraphics(desc, shaders);
            return result
                ? std::optional<ShaderSystem::PipelineProduct>(std::move(*result))
                : std::nullopt;
        }
    }
} // anonymous namespace

void SceneRenderer::PollShaders(std::uint32_t frame_counter) {
    const auto poll = [this, frame_counter](
            ShaderSystem::PipelineSlot& slot, ShaderSystem::ShaderId id,
            auto& desc) {
        slot.RetireFrame(frame_counter);
        if (!desc) return;
        slot.PollAndRebuild(*shader_mgr_, id,
            [this, &desc](ShaderSystem::ShaderManager& s)
                -> std::optional<ShaderSystem::PipelineProduct> {
                return RebuildForDesc(*pipeline_factory_, *desc, s);
            }, frame_counter);
    };
    const auto poll_graphics = [this, frame_counter](
            ShaderSystem::PipelineSlot& slot,
            ShaderSystem::ShaderId vert_id, ShaderSystem::ShaderId frag_id,
            auto& desc) {
        slot.RetireFrame(frame_counter);
        if (!desc) return;
        slot.PollAndRebuild(*shader_mgr_, vert_id, frag_id,
            [this, &desc](ShaderSystem::ShaderManager& s)
                -> std::optional<ShaderSystem::PipelineProduct> {
                return RebuildForDesc(*pipeline_factory_, *desc, s);
            }, frame_counter);
    };
    poll(expand_slot_, shader_ids_.expand_comp, expand_desc_);
    poll(occlusion_slot_, shader_ids_.occlusion_cull_comp, occlusion_desc_);
    poll(occluder_select_slot_, shader_ids_.occluder_select_comp, occluder_select_desc_);
    poll(pre_cull_slot_, shader_ids_.pre_cull_comp, pre_cull_desc_);
    poll(hiz_slot_, shader_ids_.hiz_gen_comp, hiz_desc_);
    poll(collect_count_slot_, shader_ids_.collect_count_compact_comp, collect_count_desc_);
    poll(collect_write_slot_, shader_ids_.collect_write_comp, collect_write_desc_);
    poll_graphics(depth_slot_, shader_ids_.depth_indir_vert, shader_ids_.depth_prepass_frag, depth_desc_);
}

void SceneRenderer::Shutdown() {
    for (auto& fr : frames_) {
        fr.dynamic_entries.Shutdown();
        fr.static_entries.Shutdown();
        fr.bounding_spheres.Shutdown();
        fr.obb_entries.Shutdown();
        fr.submesh_vertex_entries.Shutdown();
        fr.cull_entries.Shutdown();
    }
    scene_lights.Shutdown();
    scene_header = GpuResources::GpuBuffer{};
    technique_flags = GpuResources::GpuBuffer{};
    technique_flags_cache_.clear();
    scene_uniform_set_.reset();
    scene_uniform_pool_.reset();
    scene_uniform_layout_.reset();
    backend_ = nullptr;
}

vk::DescriptorSetLayout* SceneRenderer::GetSubmeshVertexEntriesLayout() const {
    return submesh_vertex_layout_
        ? const_cast<vk::DescriptorSetLayout*>(&**submesh_vertex_layout_)
        : nullptr;
}

vk::DescriptorSetLayout* SceneRenderer::GetVertexBuffersLayout() const {
    return vertex_buffers_layout_
        ? const_cast<vk::DescriptorSetLayout*>(&**vertex_buffers_layout_)
        : nullptr;
}

vk::DescriptorSetLayout* SceneRenderer::GetIndirectionLayout() const {
    return indirection_layout_
        ? const_cast<vk::DescriptorSetLayout*>(&**indirection_layout_)
        : nullptr;
}

void SceneRenderer::UpdateVertexBufferArrayElement(std::uint32_t frame_index,
                                                     std::uint32_t buffer_index,
                                                     vk::Buffer buffer,
                                                     std::uint64_t size) {
    const auto& fr = frames_[frame_index % frames_in_flight_];
    if (!buffer) {
        LOGIFACE_LOG(warn, "UpdateVertexBufferArrayElement: null buffer for slot " +
                     std::to_string(buffer_index));
        return;
    }
    const vk::DescriptorBufferInfo bii(buffer, 0, size);
    vk::WriteDescriptorSet w{};
    w.dstSet = *fr.vertex_buffers_set;
    w.dstBinding = 0;
    w.dstArrayElement = buffer_index;
    w.descriptorCount = 1;
    w.descriptorType = vk::DescriptorType::eStorageBuffer;
    w.pBufferInfo = &bii;
    backend_->GetDevice().updateDescriptorSets(w, nullptr);
}

void SceneRenderer::UpdateIndexBufferArrayElement(std::uint32_t frame_index,
                                                    std::uint32_t buffer_index,
                                                    vk::Buffer buffer,
                                                    std::uint64_t size) {
    const auto& fr = frames_[frame_index % frames_in_flight_];
    if (!buffer) {
        LOGIFACE_LOG(warn, "UpdateIndexBufferArrayElement: null buffer for slot " +
                     std::to_string(buffer_index));
        return;
    }
    const vk::DescriptorBufferInfo bii(buffer, 0, size);
    vk::WriteDescriptorSet w{};
    w.dstSet = *fr.index_buffers_set;
    w.dstBinding = 0;
    w.dstArrayElement = buffer_index;
    w.descriptorCount = 1;
    w.descriptorType = vk::DescriptorType::eStorageBuffer;
    w.pBufferInfo = &bii;
    backend_->GetDevice().updateDescriptorSets(w, nullptr);
}

void SceneRenderer::UpdateUvBufferArrayElement(std::uint32_t frame_index,
                                                std::uint32_t buffer_index,
                                                vk::Buffer buffer,
                                                std::uint64_t size) {
    const auto& fr = frames_[frame_index % frames_in_flight_];
    if (!buffer) {
        LOGIFACE_LOG(warn, "UpdateUvBufferArrayElement: null buffer for slot " +
                     std::to_string(buffer_index));
        return;
    }
    const vk::DescriptorBufferInfo bii(buffer, 0, size);
    vk::WriteDescriptorSet w{};
    w.dstSet = *fr.vertex_buffers_set;
    w.dstBinding = 1;
    w.dstArrayElement = buffer_index;
    w.descriptorCount = 1;
    w.descriptorType = vk::DescriptorType::eStorageBuffer;
    w.pBufferInfo = &bii;
    backend_->GetDevice().updateDescriptorSets(w, nullptr);
}

void SceneRenderer::UpdateAllFrameUvBufferArrayElements(std::uint32_t buffer_index,
                                                         vk::Buffer buffer,
                                                         std::uint64_t size) {
    for (std::uint32_t fi = 0; fi < frames_in_flight_; ++fi) {
        UpdateUvBufferArrayElement(fi, buffer_index, buffer, size);
    }
}

void SceneRenderer::UpdateAllFrameVertexBufferArrayElements(std::uint32_t buffer_index,
                                                              vk::Buffer buffer,
                                                              std::uint64_t size) {
    for (std::uint32_t fi = 0; fi < frames_in_flight_; ++fi) {
        UpdateVertexBufferArrayElement(fi, buffer_index, buffer, size);
    }
}

void SceneRenderer::UpdateAllFrameIndexBufferArrayElements(std::uint32_t buffer_index,
                                                             vk::Buffer buffer,
                                                             std::uint64_t size) {
    for (std::uint32_t fi = 0; fi < frames_in_flight_; ++fi) {
        UpdateIndexBufferArrayElement(fi, buffer_index, buffer, size);
    }
}

void SceneRenderer::UpdateBlockArrayDescriptor(vk::DescriptorSet desc_set,
                                                 std::uint32_t binding,
                                                 GpuResources::BlockArray& buf,
                                                 vk::DescriptorType desc_type) {
    if (!backend_) return;
    auto& dev = backend_->GetDevice();
    for (std::uint32_t bi = 0; bi < buf.BlockCount(); ++bi) {
        const vk::DescriptorBufferInfo bii(buf.GetBlockArray(bi), 0, buf.BlockSize());
        vk::WriteDescriptorSet w{};
        w.dstSet = desc_set;
        w.dstBinding = binding;
        w.dstArrayElement = bi;
        w.descriptorCount = 1;
        w.descriptorType = desc_type;
        w.pBufferInfo = &bii;
        dev.updateDescriptorSets(w, nullptr);
    }
}

void SceneRenderer::UpdateHizDepthBinding(std::uint32_t frame_index, vk::ImageView depth_view) {
    auto& fr = frames_[frame_index % frames_in_flight_];
    if (!backend_) return;
    auto& dev = backend_->GetDevice();
    const vk::DescriptorImageInfo depth_info(
        nullptr, depth_view, vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet w{};
    w.dstSet = fr.hiz_set.GetHandle();
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = vk::DescriptorType::eSampledImage;
    w.pImageInfo = &depth_info;
    dev.updateDescriptorSets(w, nullptr);
}

SceneRenderer::FrameBlockArrays SceneRenderer::GetFrameBlockArrays(std::uint32_t frame_index) {
    auto& fr = frames_[frame_index % frames_in_flight_];
    return {
        &fr.dynamic_entries,
        &fr.static_entries,
        &fr.bounding_spheres,
        &fr.obb_entries
    };
}

void SceneRenderer::UploadLighting(const SceneHeader& header,
                                    std::span<const Light> lights,
                                    GpuResources::StagingPool& staging) {
    if (!backend_) return;

    // Grow BlockArray to fit all lights. The descriptor binding is a fixed
    // MAX_LIGHT_BLOCKS array, so the upload is clamped to what actually fits.
    constexpr std::size_t kLightCapacity =
        static_cast<std::size_t>(MAX_LIGHT_BLOCKS) * LIGHTS_PER_BLOCK;
    if (lights.size() > kLightCapacity) {
        static bool reported_overflow = false;
        if (!reported_overflow) {
            reported_overflow = true;
            LOGIFACE_LOG(error, "SceneRenderer::UploadLighting: " +
                std::to_string(lights.size()) + " lights exceed the " +
                std::to_string(kLightCapacity) +
                "-light descriptor capacity; extra lights are dropped");
        }
    }
    const std::size_t light_count = std::min<std::size_t>(lights.size(), kLightCapacity);
    if (!scene_lights.EnsureCapacity(static_cast<std::uint32_t>(light_count))) {
        LOGIFACE_LOG(error, "SceneRenderer::UploadLighting: light BlockArray could not grow to hold " +
            std::to_string(light_count) + " lights");
    }
    const std::size_t uploadable = std::min<std::size_t>(light_count,
        static_cast<std::size_t>(scene_lights.BlockCount()) * scene_lights.EntriesPerBlock());

    // Stage the header buffer
    {
        auto header_slice = staging.Allocate(sizeof(SceneHeader));
        if (header_slice.has_value()) {
            std::memcpy(header_slice->mapped_ptr, &header, sizeof(SceneHeader));
            staging.RecordBufferCopy(*header_slice,
                                     static_cast<vk::Buffer>(*scene_header.GetBuffer()), 0);
        }
    }

    // Stage each light via BlockArray::UploadEntry (uses staging internally)
    for (std::size_t i = 0; i < uploadable; ++i) {
        scene_lights.UploadEntry(static_cast<std::uint32_t>(i), &lights[i],
                                         sizeof(Light), staging);
    }

    // Write BlockArray buffers to descriptor set binding 1
    const auto& dev = backend_->GetDevice();
    for (std::uint32_t bi = 0; bi < scene_lights.BlockCount(); ++bi) {
        const vk::DescriptorBufferInfo buf_info(scene_lights.GetBlockArray(bi),
                                                 0, scene_lights.BlockSize());
        vk::WriteDescriptorSet w{};
        w.dstSet = **scene_uniform_set_;
        w.dstBinding = 1;
        w.dstArrayElement = bi;
        w.descriptorCount = 1;
        w.descriptorType = vk::DescriptorType::eStorageBuffer;
        w.pBufferInfo = &buf_info;
        dev.updateDescriptorSets(w, nullptr);
    }

    staging.FlushImmediate();
}

} // namespace VulkanEngine::SceneRenderer
