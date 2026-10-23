module;

#include <glm/glm.hpp> //NOLINT(misc-include-cleaner)
#include <glm/gtc/matrix_transform.hpp> //NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> //NOLINT(misc-include-cleaner)

#include <logging/logging_macros.hpp>

module VulkanEngine.SceneRenderer;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Components.Transform;
import VulkanEngine.Components.MeshReference;
import VulkanEngine.GpuResources;
import VulkanEngine.StandardMeshPipeline;
import VulkanEngine.TechniqueManager;
import VulkanEngine.BindlessManager;
import VulkanEngine.MaterialManager;
import VulkanEngine.PipelinePass;

namespace VulkanEngine::SceneRenderer {
    namespace {
        struct ExpandPC { glm::mat4 vp; std::uint32_t cnt; std::uint32_t p0; std::uint32_t p1; };
        struct HiZPC { std::uint32_t bl; std::uint32_t sw; std::uint32_t sh; std::uint32_t tc; };
        // projInfo = (|proj[0][0]|, |proj[1][1]|, proj[2][2], 1 if perspective else 0)
        struct OccPC { std::uint32_t cnt; std::uint32_t refineLevel; std::uint32_t hizWidth; std::uint32_t hizHeight; glm::vec4 projInfo; };
        struct CollectPC { std::uint32_t cnt; std::uint32_t p0; std::uint32_t mt; std::uint32_t pass; };
        struct WritePC { std::uint32_t cnt; std::uint32_t p0; std::uint32_t techniqueCount; std::uint32_t p1; };
        struct OccluderSelectPC { std::uint32_t cnt; std::uint32_t minAreaPx; std::uint32_t screenW; std::uint32_t screenH; };
        struct PreCullPC { std::uint32_t cnt; std::uint32_t hizW; std::uint32_t hizH; std::uint32_t p0; glm::vec4 projInfo; };

        template<typename Handle>
        std::uint64_t HandleToU64(Handle h) {
            return reinterpret_cast<std::uint64_t>(static_cast<typename Handle::CType>(h));
        }

        static void WriteBuffer(vk::DescriptorSet ds, std::uint32_t binding,
                                VulkanEngine::GpuResources::GpuBuffer& buf,
                                const vk::raii::Device& dev) {
            const vk::DescriptorBufferInfo bi(*buf.GetBuffer(), 0, vk::WholeSize);
            vk::WriteDescriptorSet w{};
            w.dstSet = ds;
            w.dstBinding = binding;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eStorageBuffer;
            w.pBufferInfo = &bi;
            dev.updateDescriptorSets(w, nullptr);
        }

        // Block-array descriptors are written incrementally: see the sync_blocks
        // lambda in PrepareCompute. There is deliberately no unconditional
        // WriteBlocks helper — rewriting every block every frame was the
        // steady-state cost this cache exists to remove.
    } // anonymous namespace

void SceneRenderer::PrepareCompute(vk::CommandBuffer /*cmd*/,
                                    VulkanEngine::ComponentRegistry& /*reg*/,
                                    const glm::mat4& vm, const glm::mat4& pm,
                                    std::uint32_t, std::uint32_t, std::uint32_t fi) {
    PollShaders(fi);
    view_proj_ = pm * vm;
    const std::uint32_t f = fi % frames_in_flight_;
    auto& fr = frames_[f];
    const auto& dev = backend_->GetDevice();

    // Projection coefficients for the occlusion culler. The x/y scales are the NDC
    // units per world unit (focal length for perspective, 1/half-extent for ortho),
    // z is the clip-space z scale, and the flag distinguishes the two: a perspective
    // projection derives w from view z (pm[3][2] = -1), an ortho projection keeps
    // w == 1 (pm[3][2] = 0).
    fr.proj_info = glm::vec4(
        std::abs(pm[0][0]), std::abs(pm[1][1]), pm[2][2],
        (pm[3][2] != 0.0f) ? 1.0f : 0.0f);

    const std::uint32_t total = current_entity_count_;
    if (total == 0) {
        LOGIFACE_LOG(trace, "PrepareCompute: current_entity_count_ is 0, no work to do");
    }

    // Grow the per-submesh block arrays. Each is bounded by Config::max_blocks
    // (the descriptor binding array size), so a failure here means the scene
    // exceeded the capacity budget; report once and let the caller clamp.
    const auto grow_blocks = [](GpuResources::BlockArray& buf, const std::uint32_t count,
                                const char* name) {
        if (!buf.EnsureCapacity(count)) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                LOGIFACE_LOG(error, std::string("PrepareCompute: BlockArray '") + name +
                    "' could not grow to hold " + std::to_string(count) +
                    " entries; it is capped at " + std::to_string(buf.BlockLimit()) +
                    " blocks");
            }
        }
    };
    grow_blocks(fr.dynamic_entries, total, "dynamic_entries");
    grow_blocks(fr.static_entries, total, "static_entries");
    grow_blocks(fr.bounding_spheres, total, "bounding_spheres");
    grow_blocks(fr.obb_entries, total, "obb_entries");
    grow_blocks(fr.submesh_vertex_entries, total, "submesh_vertex_entries");
    grow_blocks(fr.cull_entries, total, "cull_entries");

    // The six arrays are grown with the same `total` and the same
    // entries-per-block, so they always have the same block count. The
    // per-binding descriptor-write cursors below each track their own array,
    // and this lockstep is the invariant the block lookups in the shaders use.
#ifndef NDEBUG
    const std::uint32_t block_count = fr.dynamic_entries.BlockCount();
#endif
    assert(fr.static_entries.BlockCount() == block_count &&
           fr.bounding_spheres.BlockCount() == block_count &&
           fr.obb_entries.BlockCount() == block_count &&
           fr.submesh_vertex_entries.BlockCount() == block_count &&
           fr.cull_entries.BlockCount() == block_count &&
           "the six submesh block arrays must grow in lockstep");

    // Write only the block descriptors that do not exist yet. Blocks are only
    // appended and never removed, so a descriptor written in an earlier frame
    // stays valid; `binding_epoch_` changes when CreateFrameBuffers recreated
    // the underlying buffers, forcing a full rewrite.
    //
    // Each (set, binding) pair owns its own cursor. The same BlockArray is bound
    // into several sets (cull_entries alone appears in four) and each binding
    // still has to be written: a single shared cursor lets the first pair write
    // its blocks and then reports the frame's whole block budget written, so
    // every later binding is left with a descriptor that was never written.
    const std::uint32_t binding_epoch = binding_epoch_;
    const auto sync_blocks = [&dev, &fr, binding_epoch](
                                 FrameResources::BlockBindingSlot slot,
                                 vk::DescriptorSet ds, std::uint32_t binding,
                                 GpuResources::BlockArray& buf,
                                 vk::DescriptorType desc_type) {
        auto& cache = fr.block_binding_cache[static_cast<std::size_t>(slot)];
        if (cache.epoch != binding_epoch) {
            cache.written_blocks = 0;
            cache.epoch = binding_epoch;
        }
        for (std::uint32_t bi = cache.written_blocks; bi < buf.BlockCount(); ++bi) {
            const vk::DescriptorBufferInfo bii(buf.GetBlockArray(bi), 0, buf.BlockSize());
            vk::WriteDescriptorSet w{};
            w.dstSet = ds;
            w.dstBinding = binding;
            w.dstArrayElement = bi;
            w.descriptorCount = 1;
            w.descriptorType = desc_type;
            w.pBufferInfo = &bii;
            dev.updateDescriptorSets(w, nullptr);
        }
        cache.written_blocks = buf.BlockCount();
    };

    const bool mid = draw_mode_ == DrawMode::MID;

    // Per-frame counters/commands.
    {
        const std::uint32_t zero2[2] = {0u, 0u};
        fr.expand_counter.Upload(zero2, sizeof(zero2));
        const std::uint32_t zero = 0u;
        fr.occluder_candidate_count.Upload(&zero, sizeof(zero));
        if (mid) {
            fr.depth_out_command_count.Upload(&zero, sizeof(zero));
            fr.occluder_out_command_count.Upload(&zero, sizeof(zero));
        } else {
            const vk::DrawIndexedIndirectCommand zero_cmd{0u, 1u, 0u, 0, 0u};
            fr.depth_out_draw_command.Upload(&zero_cmd, sizeof(zero_cmd));
            fr.occluder_out_draw_command.Upload(&zero_cmd, sizeof(zero_cmd));
        }
    }

    // Zero CPU-consumed buffers. technique_draw_commands only exists in
    // CID mode; technique_counts is written by the MID compact pass.
    for (auto* buf : {&fr.technique_results, &fr.technique_counts}) {
        auto* p = buf->Map(0, buf->GetSize());
        if (p) {
            std::memset(p, 0, buf->GetSize());
            buf->Unmap();
        }
    }
    if (!mid) {
        auto* p = fr.technique_draw_commands.Map(0, fr.technique_draw_commands.GetSize());
        if (p) {
            std::memset(p, 0, fr.technique_draw_commands.GetSize());
            fr.technique_draw_commands.Unmap();
        }
    }

    LOGIFACE_LOG(trace, "PrepareCompute: binding dynamic_entries blocks=" +
                 std::to_string(fr.dynamic_entries.BlockCount()) +
                 " static_entries blocks=" + std::to_string(fr.static_entries.BlockCount()) +
                 " total=" + std::to_string(total));

    sync_blocks(FrameResources::BlockBindingSlot::expand_dynamic_entries,
                fr.expand_set.GetHandle(), 0, fr.dynamic_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::expand_static_entries,
                fr.expand_set.GetHandle(), 1, fr.static_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::expand_submesh_vertex,
                fr.expand_set.GetHandle(), 2, fr.submesh_vertex_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::expand_cull,
                fr.expand_set.GetHandle(), 3, fr.cull_entries,
                vk::DescriptorType::eStorageBuffer);
    WriteBuffer(fr.expand_set.GetHandle(), 4, fr.vertex_indirection, dev);
    WriteBuffer(fr.expand_set.GetHandle(), 5, fr.draw_indices, dev);
    WriteBuffer(fr.expand_set.GetHandle(), 6, fr.expand_counter, dev);

    // Occluder-select block arrays (block counts change with scene capacity).
    sync_blocks(FrameResources::BlockBindingSlot::occluder_select_cull,
                fr.occluder_select_set.GetHandle(), 0, fr.cull_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occluder_select_submesh_vertex,
                fr.occluder_select_set.GetHandle(), 1, fr.submesh_vertex_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occluder_select_obb,
                fr.occluder_select_set.GetHandle(), 2, fr.obb_entries,
                vk::DescriptorType::eStorageBuffer);
    WriteBuffer(fr.occluder_select_set.GetHandle(), 3, technique_flags, dev);
    WriteBuffer(fr.occluder_select_set.GetHandle(), 4, fr.draw_indices, dev);
    WriteBuffer(fr.occluder_select_set.GetHandle(), 5,
                mid ? fr.occluder_commands : fr.occluder_indices, dev);
    WriteBuffer(fr.occluder_select_set.GetHandle(), 6,
                mid ? fr.occluder_out_command_count : fr.occluder_out_draw_command, dev);
    WriteBuffer(fr.occluder_select_set.GetHandle(), 7, fr.occluder_candidate_count, dev);

    sync_blocks(FrameResources::BlockBindingSlot::submesh_vertex,
                fr.submesh_vertex_set.GetHandle(), 0, fr.submesh_vertex_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occlusion_submesh_vertex,
                fr.occlusion_set.GetHandle(), 0, fr.submesh_vertex_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occlusion_cull,
                fr.occlusion_set.GetHandle(), 1, fr.cull_entries,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occlusion_bounding_spheres,
                fr.occlusion_set.GetHandle(), 2, fr.bounding_spheres,
                vk::DescriptorType::eStorageBuffer);
    sync_blocks(FrameResources::BlockBindingSlot::occlusion_obb,
                fr.occlusion_set.GetHandle(), 4, fr.obb_entries,
                vk::DescriptorType::eStorageBuffer);
    {
        const vk::DescriptorImageInfo hiz_info(
            **hiz_sampler_, *fr.hiz_full_view,
            vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::WriteDescriptorSet w{};
        w.dstSet = fr.occlusion_set.GetHandle();
        w.dstBinding = 3;
        w.descriptorCount = 1;
        w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w.pImageInfo = &hiz_info;
        dev.updateDescriptorSets(w, nullptr);
    }
    {
        // Technique flag table consumed by the occlusion cull and pre-cull
        // shaders (receives_occlusion gate).
        const vk::DescriptorBufferInfo bi(
            *technique_flags.GetBuffer(), 0, vk::WholeSize);
        vk::WriteDescriptorSet w{};
        w.dstSet = fr.occlusion_set.GetHandle();
        w.dstBinding = 5;
        w.descriptorCount = 1;
        w.descriptorType = vk::DescriptorType::eStorageBuffer;
        w.pBufferInfo = &bi;
        dev.updateDescriptorSets(w, nullptr);
    }
    // Pre-cull survivor compaction bindings 6-8.
    WriteBuffer(fr.occlusion_set.GetHandle(), 6, fr.draw_indices, dev);
    WriteBuffer(fr.occlusion_set.GetHandle(), 7,
                mid ? fr.depth_commands : fr.depth_indices, dev);
    WriteBuffer(fr.occlusion_set.GetHandle(), 8,
                mid ? fr.depth_out_command_count : fr.depth_out_draw_command, dev);

    // Collect bindings 1-4.
    sync_blocks(FrameResources::BlockBindingSlot::collect_cull,
                fr.collect_set.GetHandle(), 0, fr.cull_entries,
                vk::DescriptorType::eStorageBuffer);
    WriteBuffer(fr.collect_set.GetHandle(), 1, fr.draw_indices, dev);
    WriteBuffer(fr.collect_set.GetHandle(), 2,
                mid ? fr.main_commands : fr.main_indices, dev);
    WriteBuffer(fr.collect_set.GetHandle(), 3, fr.technique_results, dev);
    WriteBuffer(fr.collect_set.GetHandle(), 4, fr.technique_region_bases, dev);
    WriteBuffer(fr.collect_set.GetHandle(), 5, fr.technique_counts, dev);

    // Collect-write bindings (CID only): intermediate + technique commands.
    if (!mid) {
        WriteBuffer(fr.collect_write_set.GetHandle(), 0, fr.technique_results, dev);
        WriteBuffer(fr.collect_write_set.GetHandle(), 1, fr.technique_draw_commands, dev);
    }

    // Single shared indirection set (set 3) bound to vertex_indirection.
    {
        const vk::DescriptorBufferInfo bi(*fr.vertex_indirection.GetBuffer(), 0, vk::WholeSize);
        vk::WriteDescriptorSet w{};
        w.dstSet = *fr.indirection_raw_set;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = vk::DescriptorType::eStorageBuffer;
        w.pBufferInfo = &bi;
        dev.updateDescriptorSets(w, nullptr);
    }
}

void SceneRenderer::SetTechniqueCommandRegions(
    std::span<const std::uint32_t> submeshes_per_technique) {
    region_base_.assign(MAX_DRAW_GROUPS, 0);
    region_count_.assign(MAX_DRAW_GROUPS, 0);
    std::uint32_t offset = 0;
    std::uint32_t highest = 0;
    const std::size_t n = std::min<std::size_t>(submeshes_per_technique.size(), MAX_DRAW_GROUPS);
    for (std::size_t t = 0; t < n; ++t) {
        region_base_[t] = offset;
        region_count_[t] = submeshes_per_technique[t];
        offset += submeshes_per_technique[t];
        if (submeshes_per_technique[t] != 0) highest = static_cast<std::uint32_t>(t) + 1;
    }
    region_total_ = offset;
    // The collect passes only need to scan keys that can appear; the highest
    // key with submeshes is the tight bound. UpdateTechniqueFlags may raise it
    // to the registered count before the dispatches run.
    technique_count_ = highest;

    if (!backend_) return;
    for (auto& fr : frames_) {
        if (fr.technique_region_bases.GetSize() >= region_base_.size() * sizeof(std::uint32_t)) {
            fr.technique_region_bases.Upload(region_base_.data(),
                region_base_.size() * sizeof(std::uint32_t));
        }
    }
}

void SceneRenderer::UpdateTechniqueFlags(
    VulkanEngine::TechniqueManager::TechniqueManager& tm) {
    if (!backend_) return;

    // Flags are per draw group, inheriting the owning technique's PipelineFlags.
    // The interning guarantee that every live group is below the registered
    // technique count is what lets the collect passes scan only up to that
    // count; assert it in debug builds.
    const std::uint32_t count =
        std::min<std::uint32_t>(tm.GetDrawGroupCount(), MAX_DRAW_GROUPS);
    if (tm.GetDrawGroupCount() > MAX_DRAW_GROUPS) {
        LOGIFACE_LOG(warn, "UpdateTechniqueFlags: " +
                     std::to_string(tm.GetDrawGroupCount()) +
                     " draw groups exceed flag-table capacity " +
                     std::to_string(MAX_DRAW_GROUPS));
    }
    // The live group count is the upper bound on any key, so the collect passes
    // never scan beyond it. Keep a tighter region bound when it is already
    // higher (region indices are group ids).
    technique_count_ = std::max(technique_count_, count);

    std::vector<std::uint32_t> flags(MAX_DRAW_GROUPS, 0);
    for (std::uint32_t g = 0; g < count; ++g) {
        const std::uint16_t tech_id = tm.GetDrawGroupTechnique(static_cast<std::uint16_t>(g));
        auto* tech = tm.GetTechnique(tech_id);
        if (!tech) continue;
        const auto& f = tech->pipeline_flags;
        std::uint32_t bits = 0;
        if (f.participates_in_depth_pass) bits |= TECHNIQUE_FLAG_DEPTH_PASS;
        if (f.receives_occlusion)        bits |= TECHNIQUE_FLAG_RECEIVES_OCCLUSION;
        if (f.participates_in_collect)   bits |= TECHNIQUE_FLAG_COLLECT;
        if (f.bounds_conservative)       bits |= TECHNIQUE_FLAG_OCCLUDER_SAFE;
        flags[g] = bits;
    }

    // Upload only when contents change (flags are static in practice).
    if (technique_flags_cache_ != flags) {
        technique_flags_cache_ = flags;
        technique_flags.Upload(flags.data(), sizeof(std::uint32_t) * flags.size());
    }
}

void SceneRenderer::DepthPrepass(vk::CommandBuffer cmd, std::uint32_t w, std::uint32_t h, std::uint32_t fi) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!depth_slot_.Get()) {
        LOGIFACE_LOG(warn, "DepthPrepass: depth pipeline is null, skipping");
        return;
    }
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DepthPrepass: current_entity_count_ is 0, skipping");
        return;
    }
    LOGIFACE_LOG(trace, "DepthPrepass: submesh_count=" + std::to_string(current_entity_count_) +
                 " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    cmd.setViewport(0, VulkanEngine::PipelinePass::SceneViewport(w, h));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {w, h}));
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, depth_slot_.Get());
    LOGIFACE_LOG(trace, std::format("DepthPrepass: bound pipeline 0x{:x}",
                                    HandleToU64(depth_slot_.Get())));
    const std::array<vk::DescriptorSet, 4> ds{
        empty_sets_[fi % frames_in_flight_].GetHandle(),
        fr.submesh_vertex_set.GetHandle(),
        static_cast<vk::DescriptorSet>(*fr.vertex_buffers_set),
        *fr.indirection_raw_set
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *depth_pipeline_layout_,
                             0, ds, {});
    if (draw_mode_ == DrawMode::MID) {
        cmd.bindIndexBuffer(*fr.draw_indices.GetBuffer(), 0, vk::IndexType::eUint32);
        cmd.drawIndexedIndirectCount(*fr.depth_commands.GetBuffer(), 0,
            *fr.depth_out_command_count.GetBuffer(), 0,
            scene_capacity_.submesh_count, sizeof(vk::DrawIndexedIndirectCommand));
    } else {
        cmd.bindIndexBuffer(*fr.depth_indices.GetBuffer(), 0, vk::IndexType::eUint32);
        cmd.drawIndexedIndirect(*fr.depth_out_draw_command.GetBuffer(), 0, 1,
            sizeof(vk::DrawIndexedIndirectCommand));
    }
}

void SceneRenderer::OccluderPrepass(vk::CommandBuffer cmd, std::uint32_t w, std::uint32_t h, std::uint32_t fi) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!depth_slot_.Get()) {
        LOGIFACE_LOG(warn, "OccluderPrepass: depth pipeline is null, skipping");
        return;
    }
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "OccluderPrepass: current_entity_count_ is 0, skipping");
        return;
    }
    LOGIFACE_LOG(trace, "OccluderPrepass: selected occluders (" + std::to_string(w) + "x" +
                 std::to_string(h) + ")");
    cmd.setViewport(0, VulkanEngine::PipelinePass::SceneViewport(w, h));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {w, h}));
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, depth_slot_.Get());
    const std::array<vk::DescriptorSet, 4> ds{
        empty_sets_[fi % frames_in_flight_].GetHandle(),
        fr.submesh_vertex_set.GetHandle(),
        static_cast<vk::DescriptorSet>(*fr.vertex_buffers_set),
        *fr.indirection_raw_set
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *depth_pipeline_layout_,
                             0, ds, {});
    if (draw_mode_ == DrawMode::MID) {
        cmd.bindIndexBuffer(*fr.draw_indices.GetBuffer(), 0, vk::IndexType::eUint32);
        cmd.drawIndexedIndirectCount(*fr.occluder_commands.GetBuffer(), 0,
            *fr.occluder_out_command_count.GetBuffer(), 0,
            scene_capacity_.submesh_count, sizeof(vk::DrawIndexedIndirectCommand));
    } else {
        cmd.bindIndexBuffer(*fr.occluder_indices.GetBuffer(), 0, vk::IndexType::eUint32);
        cmd.drawIndexedIndirect(*fr.occluder_out_draw_command.GetBuffer(), 0, 1,
            sizeof(vk::DrawIndexedIndirectCommand));
    }
}

void SceneRenderer::Render(vk::CommandBuffer cmd,
                            VulkanEngine::ComponentRegistry&,
                            VulkanEngine::TechniqueManager::TechniqueManager& tm,
                            VulkanEngine::BindlessManager::BindlessManager& bm,
                            const glm::mat4& proj, const glm::mat4& view,
                            std::uint32_t w, std::uint32_t h, std::uint32_t fi) {
    if (!w || !h) return;
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "Render: current_entity_count_ is 0, skipping");
        return;
    }
    auto& fr = frames_[fi % frames_in_flight_];

    cmd.setViewport(0, VulkanEngine::PipelinePass::SceneViewport(w, h));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {w, h}));

    // Engine descriptor set handles (used by both legacy and BaseTechnique paths)
    const vk::DescriptorSet engine_set0 = bm.GetDescriptorSet();
    const vk::DescriptorSet engine_set1 = fr.submesh_vertex_set.GetHandle();
    const vk::DescriptorSet engine_set2 = static_cast<vk::DescriptorSet>(*fr.vertex_buffers_set);
    const vk::DescriptorSet engine_set3 = *fr.indirection_raw_set;

    LOGIFACE_LOG(trace, "RenderMain: submesh_count=" + std::to_string(current_entity_count_) +
                 " techniques=" + std::to_string(tm.GetTechniqueCount()));

    // Scene uniform set (set 4) — lighting data (device-local storage)
    const vk::DescriptorSet scene_uniform_set = GetSceneUniformSet();

    // Camera position for fragment shader (push constant at offset 0)
    const glm::mat4 inv_view = glm::inverse(view);
    float camera_pos[4] = {inv_view[3][0], inv_view[3][1], inv_view[3][2], 0.0f};

    const bool mid = draw_mode_ == DrawMode::MID;
    cmd.bindIndexBuffer(
        mid ? *fr.draw_indices.GetBuffer() : *fr.main_indices.GetBuffer(),
        0, vk::IndexType::eUint32);

    // The collect arrays and command regions are indexed by draw group. A group
    // resolves to its owning technique, which owns the pipeline and the shared
    // layout/descriptor sets; render-state variants reuse the same set (variant
    // pipelines are added by the variant work, and the pipeline is selected per
    // group there).
    const std::uint32_t group_count =
        std::min<std::uint32_t>(tm.GetDrawGroupCount(), MAX_DRAW_GROUPS);
    for (std::uint32_t g = 0; g < group_count; ++g) {
        const std::uint16_t tech_id = tm.GetDrawGroupTechnique(static_cast<std::uint16_t>(g));
        if (tech_id == std::numeric_limits<std::uint16_t>::max()) continue;
        auto* tech = tm.GetTechnique(tech_id);
        if (!tech) continue;
        if (!tech->pipeline_flags.participates_in_collect) continue;

        // Materialize this group's pipeline variant on first use. The group
        // stores interface variant and render state separately; the composite
        // draw key rebuilds from them. A group with no matching variant (or the
        // base key 0) falls back to the base pipeline. The layout is shared
        // across variants, so the descriptor sets stay valid.
        const std::uint32_t draw_key = TechniqueManager::DrawKeyState::PackDrawKey(
            tm.GetDrawGroupVariant(g), tm.GetDrawGroupRenderState(g));
        if (!tech->HasVariant(draw_key)) {
            (void)tech->EnsureVariant(draw_key, *shader_mgr_, *pipeline_factory_);
        }
        auto pipeline = tech->GetPipeline(draw_key);
        auto layout = tech->GetPipelineLayout();
        if (!pipeline || !layout) {
            LOGIFACE_LOG(trace, std::format("RenderMain: group {} missing pipeline or layout, skipping", g));
            continue;
        }

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        LOGIFACE_LOG(trace, std::format("RenderMain: group {} bound pipeline 0x{:x}",
                                        g, HandleToU64(pipeline)));

        std::array<vk::DescriptorSet, 16> ds{};
        std::uint32_t slot = 0;
        ds[slot++] = engine_set0;  // set 0: bindless textures
        ds[slot++] = engine_set1;  // set 1: submesh vertex data
        ds[slot++] = engine_set2;  // set 2: raw vertex buffers
        ds[slot++] = engine_set3;  // set 3: indirection
        ds[slot++] = scene_uniform_set;  // set 4: scene uniforms (lighting, camera)

        // Bind technique custom descriptor sets (BlockArray/Shared buffers) at slots 5+
        for (const auto& tech_ds : tech->GetCustomDescriptorSets()) {
            ds[slot++] = tech_ds;
        }

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            layout,
            0,
            vk::ArrayProxy<const vk::DescriptorSet>(slot, ds.data()),
            {}
        );

        // Camera position (fragment-stage push constant at offset 0, 16 bytes)
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eFragment, 0, 16, camera_pos);

        if (mid) {
            const std::uint32_t base = (g < region_base_.size()) ? region_base_[g] : 0u;
            const std::uint32_t cnt = (g < region_count_.size()) ? region_count_[g] : 0u;
            if (cnt == 0) continue;
            cmd.drawIndexedIndirectCount(
                *fr.main_commands.GetBuffer(),
                static_cast<vk::DeviceSize>(base) * sizeof(vk::DrawIndexedIndirectCommand),
                *fr.technique_counts.GetBuffer(),
                static_cast<vk::DeviceSize>(g) * sizeof(std::uint32_t),
                cnt, sizeof(vk::DrawIndexedIndirectCommand));
        } else {
            const vk::DeviceSize draw_cmd_offset =
                static_cast<vk::DeviceSize>(g) * sizeof(vk::DrawIndexedIndirectCommand);
            cmd.drawIndexedIndirect(*fr.technique_draw_commands.GetBuffer(),
                                    draw_cmd_offset, 1,
                                    sizeof(vk::DrawIndexedIndirectCommand));
        }
    }
}

void SceneRenderer::DispatchExpand(vk::CommandBuffer cmd, std::uint32_t cnt,
                                    const glm::mat4& vp, std::uint32_t fi) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!cnt) {
        LOGIFACE_LOG(trace, "DispatchExpand: cnt is 0, skipping");
        return;
    }
    LOGIFACE_LOG(trace, "DispatchExpand: submeshes=" + std::to_string(cnt));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, expand_slot_.Get());
    const std::array<vk::DescriptorSet, 2> ds{
        fr.expand_set.GetHandle(),
        static_cast<vk::DescriptorSet>(*fr.index_buffers_set)
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *expand_pipeline_layout_,
                             0, ds, {});
    ExpandPC pc{ vp, cnt, 0, 0 };
    cmd.pushConstants(*expand_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                       0, sizeof(ExpandPC), &pc);
    cmd.dispatch((cnt + 63) / 64, 1, 1);
}

void SceneRenderer::DispatchHiZGen(vk::CommandBuffer cmd, std::uint32_t w, std::uint32_t h,
                                    std::uint32_t fi, std::uint32_t) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DispatchHiZGen: current_entity_count_ is 0, skipping");
        return;
    }
    if (!fr.hiz_set.IsValid()) {
        LOGIFACE_LOG(trace, "DispatchHiZGen: hiz_set invalid, skipping");
        return;
    }
    if (w != depth_width_ || h != depth_height_) {
        LOGIFACE_LOG(warn, "DispatchHiZGen: dimensions changed (" + std::to_string(w) + "x" +
                     std::to_string(h) + " != " + std::to_string(depth_width_) + "x" +
                     std::to_string(depth_height_) + "), skipping");
        return;
    }
    LOGIFACE_LOG(trace, "DispatchHiZGen: " + std::to_string(w) + "x" + std::to_string(h) +
                 " mips=" + std::to_string(hiz_mip_count_));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, hiz_slot_.Get());
    const std::array<vk::DescriptorSet, 1> ds{ fr.hiz_set.GetHandle() };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *hiz_pipeline_layout_,
                             0, ds, {});
    // Every level the cull shaders can sample must be produced here. Iterating
    // to hiz_mip_count_ (not hiz_mip_count_ - 1) lets the final iteration run
    // with bc == 1, whose second reduction the shader already guards out; the
    // old bound skipped the top level whenever hiz_mip_count_ was odd.
    for (std::uint32_t bl = 0; bl < hiz_mip_count_; bl += HIZ_BATCH) {
        const std::uint32_t bc = std::min(HIZ_BATCH, hiz_mip_count_ - bl);
        const std::uint32_t sw = w >> bl;
        const std::uint32_t sh = h >> bl;
        HiZPC pc{ bl, sw, sh, bc };
        cmd.pushConstants(*hiz_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                           0, sizeof(HiZPC), &pc);
        cmd.dispatch((sw + 15) / 16, (sh + 15) / 16, 1);
        vk::MemoryBarrier hiz_mb{};
        hiz_mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        hiz_mb.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {}, hiz_mb, {}, {});
    }
}

void SceneRenderer::DispatchOcclusion(vk::CommandBuffer cmd, std::uint32_t fi) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DispatchOcclusion: current_entity_count_ is 0, skipping");
        return;
    }
    if (!fr.occlusion_set.IsValid()) {
        LOGIFACE_LOG(trace, "DispatchOcclusion: occlusion_set invalid, skipping");
        return;
    }
    LOGIFACE_LOG(trace, "DispatchOcclusion: submeshes=" + std::to_string(current_entity_count_));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, occlusion_slot_.Get());
    const std::array<vk::DescriptorSet, 1> ds{ fr.occlusion_set.GetHandle() };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *occlusion_pipeline_layout_,
                             0, ds, {});
    const std::uint32_t hiz_w = (depth_width_ + 1) / 2;
    const std::uint32_t hiz_h = (depth_height_ + 1) / 2;
    OccPC pc{ current_entity_count_, 1, hiz_w, hiz_h, fr.proj_info };
    cmd.pushConstants(*occlusion_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                       0, sizeof(OccPC), &pc);
    cmd.dispatch((current_entity_count_ + 63) / 64, 1, 1);
}

void SceneRenderer::DispatchCollect(vk::CommandBuffer cmd, std::uint32_t fi) {
    auto& fr = frames_[fi % frames_in_flight_];
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DispatchCollect: current_entity_count_ is 0, skipping");
        return;
    }
    // Only keys below the live draw-key count can appear in cull entries; the
    // GPU tables are larger, but scanning the capacity would waste a serial
    // prefix pass and the collect-write loop. The count is a hard upper bound
    // (set from the registered technique count and the region table).
    const std::uint32_t key_count =
        std::min(technique_count_, MAX_DRAW_GROUPS);
    LOGIFACE_LOG(trace, "DispatchCollect: submeshes=" + std::to_string(current_entity_count_) +
                 " drawKeys=" + std::to_string(key_count));
    const bool mid = draw_mode_ == DrawMode::MID;

    const auto barrier = [&cmd]() {
        vk::MemoryBarrier mb{};
        mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        mb.dstAccessMask = vk::AccessFlagBits::eShaderRead |
                           vk::AccessFlagBits::eShaderWrite;
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, mb, {}, {});
    };

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, collect_count_slot_.Get());
    const std::array<vk::DescriptorSet, 1> ds1{ fr.collect_set.GetHandle() };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *collect_pipeline_layout_,
                            0, ds1, {});

    const std::uint32_t groups = (current_entity_count_ + 255) / 256;
    if (!mid) {
        // CID: count -> global prefix -> compact -> command emission.
        CollectPC pc0{ current_entity_count_, 0, key_count, 0 };
        cmd.pushConstants(*collect_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                          0, sizeof(CollectPC), &pc0);
        cmd.dispatch(groups, 1, 1);

        barrier();

        CollectPC pc1{ current_entity_count_, 0, key_count, 1 };
        cmd.pushConstants(*collect_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                          0, sizeof(CollectPC), &pc1);
        cmd.dispatch(1, 1, 1);

        barrier();

        CollectPC pc2{ current_entity_count_, 0, key_count, 2 };
        cmd.pushConstants(*collect_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                          0, sizeof(CollectPC), &pc2);
        cmd.dispatch(groups, 1, 1);

        barrier();

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, collect_write_slot_.Get());
        const std::array<vk::DescriptorSet, 1> ds2{ fr.collect_write_set.GetHandle() };
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *collect_write_pipeline_layout_,
                                0, ds2, {});
        WritePC pcw{ current_entity_count_, 0, key_count, 0 };
        cmd.pushConstants(*collect_write_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                          0, sizeof(WritePC), &pcw);
        cmd.dispatch(1, 1, 1);
    } else {
        // MID: single compact pass appends commands into CPU-prefixed regions
        // and publishes the per-technique alive count. No epilogue dispatch.
        CollectPC pcMid{ current_entity_count_, 0, key_count, 2 };
        cmd.pushConstants(*collect_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                          0, sizeof(CollectPC), &pcMid);
        cmd.dispatch(groups, 1, 1);
    }
}

void SceneRenderer::DispatchOccluderSelect(vk::CommandBuffer cmd,
                                            std::uint32_t w,
                                            std::uint32_t h,
                                            std::uint32_t fi) {
    // §5.1-A: OBB screen-area threshold selection + occluder indirection
    // compaction into the occluder-prepass draw command. Any set of large
    // real-geometry occluders is valid — overflow only misses culls.
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DispatchOccluderSelect: current_entity_count_ is 0, skipping");
        return;
    }
    auto& fr = frames_[fi % frames_in_flight_];
    if (!occluder_select_slot_.Get()) {
        LOGIFACE_LOG(warn, "DispatchOccluderSelect: pipeline is null, skipping");
        return;
    }
    LOGIFACE_LOG(trace, "DispatchOccluderSelect: submeshes=" + std::to_string(current_entity_count_));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, occluder_select_slot_.Get());
    const std::array<vk::DescriptorSet, 1> ds{ fr.occluder_select_set.GetHandle() };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *occluder_select_pipeline_layout_,
                             0, ds, {});
    const float screenArea = static_cast<float>(w) * static_cast<float>(h);
    OccluderSelectPC pc{
        current_entity_count_,
        static_cast<std::uint32_t>(screenArea * kOccluderMinAreaFraction),
        w, h };
    cmd.pushConstants(*occluder_select_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                       0, sizeof(OccluderSelectPC), &pc);
    cmd.dispatch((current_entity_count_ + 63) / 64, 1, 1);
}

void SceneRenderer::DispatchPreCull(vk::CommandBuffer cmd, std::uint32_t fi) {
    // §5.1-D / §5.3: frustum reject + Hi-Z test against the occluder-only
    // pyramid, then compact survivors into the depth-prepass indirection.
    // Culled submeshes get indexCount = 0 so the post-prepass occlusion pass
    // and collect skip them for free.
    if (!current_entity_count_) {
        LOGIFACE_LOG(trace, "DispatchPreCull: current_entity_count_ is 0, skipping");
        return;
    }
    auto& fr = frames_[fi % frames_in_flight_];
    LOGIFACE_LOG(trace, "DispatchPreCull: submeshes=" + std::to_string(current_entity_count_));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pre_cull_slot_.Get());
    const std::array<vk::DescriptorSet, 1> ds{ fr.occlusion_set.GetHandle() };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pre_cull_pipeline_layout_,
                             0, ds, {});
    const std::uint32_t hiz_w = (depth_width_ + 1) / 2;
    const std::uint32_t hiz_h = (depth_height_ + 1) / 2;
    PreCullPC pc{ current_entity_count_, hiz_w, hiz_h, 0, fr.proj_info };
    cmd.pushConstants(*pre_cull_pipeline_layout_, vk::ShaderStageFlagBits::eCompute,
                       0, sizeof(PreCullPC), &pc);
    cmd.dispatch((current_entity_count_ + 63) / 64, 1, 1);
}

void SceneRenderer::InitializeHizFirstFrame(vk::CommandBuffer cmd) {
    if (hiz_initialized_) return;
    for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
        auto& f = frames_[i];
        const vk::ImageSubresourceRange hiz_range(
            vk::ImageAspectFlagBits::eColor, 0, hiz_mip_count_, 0, 1);
        const vk::ImageMemoryBarrier imb(
            vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
            *f.hiz_image, hiz_range);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, imb);
        const vk::ClearColorValue clear(1.0f, 1.0f, 1.0f, 1.0f);
        cmd.clearColorImage(*f.hiz_image, vk::ImageLayout::eGeneral,
                            clear, hiz_range);
        const vk::ImageMemoryBarrier post_clear(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
            *f.hiz_image, hiz_range);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, post_clear);
    }
    hiz_initialized_ = true;
}

} // namespace VulkanEngine::SceneRenderer
