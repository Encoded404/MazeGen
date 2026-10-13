module;

#include <cassert>
#include <logging/logging_macros.hpp>

module VulkanEngine.PipelineFactory;

import std;
import std.compat;

import logiface;
import vulkan;

import VulkanEngine.ShaderManager;
import VulkanBackend.Vulkan.VulkanDebugUtils;

namespace VulkanEngine::ShaderSystem {

PipelineProduct PipelineProduct::Monolithic(vk::raii::Pipeline pipeline) {
    PipelineProduct p;
    p.linked_ = std::move(pipeline);
    return p;
}

PipelineProduct PipelineProduct::GPLLinked(
    vk::raii::Pipeline linked,
    std::shared_ptr<vk::raii::Pipeline> vertex_input_lib,
    std::shared_ptr<vk::raii::Pipeline> pre_raster_lib,
    std::shared_ptr<vk::raii::Pipeline> fragment_lib,
    std::shared_ptr<vk::raii::Pipeline> fragment_output_lib) {
    PipelineProduct p;
    p.linked_ = std::move(linked);
    p.vertex_input_lib_ = std::move(vertex_input_lib);
    p.pre_raster_lib_ = std::move(pre_raster_lib);
    p.fragment_lib_ = std::move(fragment_lib);
    p.fragment_output_lib_ = std::move(fragment_output_lib);
    return p;
}

void PipelineSlot::SetFramesInFlight(const std::uint32_t frames_in_flight) {
    frames_in_flight_ = std::max(frames_in_flight, 1u);
    retiring_.resize(frames_in_flight_);
}

void PipelineSlot::Swap(PipelineProduct product, std::uint32_t frame_index) {
    PipelineProduct retired = std::move(current_);
    current_ = std::move(product);
    if (retired.Get()) {
        retiring_[frame_index % frames_in_flight_].push_back(std::move(retired));
    }
}

void PipelineSlot::RetireFrame(std::uint32_t frame_index) {
    retiring_[frame_index % frames_in_flight_].clear();
}

vk::Pipeline PipelineSlot::Get() const {
    return current_.Get();
}

bool PipelineSlot::PollAndRebuild(
        ShaderManager& shaders, ShaderId id,
        std::function<std::optional<PipelineProduct>(ShaderManager&)> rebuild_fn,
        std::uint32_t frame_index) {
    return PollAndRebuild(shaders, id, static_cast<ShaderId>(-1),
                          std::move(rebuild_fn), frame_index);
}

bool PipelineSlot::PollAndRebuild(
        ShaderManager& shaders, ShaderId vert_id, ShaderId frag_id,
        std::function<std::optional<PipelineProduct>(ShaderManager&)> rebuild_fn,
        std::uint32_t frame_index) {
    const bool has_vert = vert_id != static_cast<ShaderId>(-1);
    const bool has_frag = frag_id != static_cast<ShaderId>(-1);
    const std::uint64_t vert_version = has_vert ? shaders.GetVersion(vert_id) : 0;
    const std::uint64_t frag_version = has_frag ? shaders.GetVersion(frag_id) : 0;
    const bool vert_changed = has_vert && vert_version != last_vert_version_;
    const bool frag_changed = has_frag && frag_version != last_frag_version_;
    if (!vert_changed && !frag_changed) return false;

    if (vert_changed && frag_changed) {
        LOGIFACE_LOG(debug, std::format("PipelineSlot: shaders {} and {} changed (v{} / v{}), rebuilding pipeline",
                                        vert_id, frag_id, vert_version, frag_version));
    } else if (vert_changed) {
        LOGIFACE_LOG(debug, std::format("PipelineSlot: shader {} changed (v{}), rebuilding pipeline",
                                        vert_id, vert_version));
    } else {
        LOGIFACE_LOG(debug, std::format("PipelineSlot: shader {} changed (v{}), rebuilding pipeline",
                                        frag_id, frag_version));
    }

    auto result = rebuild_fn(shaders);
    if (!result.has_value()) {
        // Keep the tracked versions unchanged so a transient rebuild failure is
        // retried on the next frame instead of permanently disabling hot reload
        // for this shader until the next edit.
        return false;
    }
    last_vert_version_ = has_vert ? vert_version : last_vert_version_;
    last_frag_version_ = has_frag ? frag_version : last_frag_version_;
    Swap(std::move(*result), frame_index);
    return true;
}

namespace {
    std::uint64_t fnv1a_hash(std::string_view s, std::uint64_t h = 0xcbf29ce484222325ULL) {
        for (char c : s) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001b3ULL;
        return h;
    }

    template<typename Handle>
    std::uint64_t HandleToU64(Handle h) {
        return reinterpret_cast<std::uint64_t>(static_cast<typename Handle::CType>(h));
    }

    // pNext-chained extension state becomes part of the compiled library, so
    // cache keys must reflect it. Pointer contents are opaque here, but the
    // chain *shape* (structure types in order) distinguishes pipelines that
    // use different extension state. Identical shapes with different payloads
    // still alias; no engine caller uses pNext-chained pipeline state today.
    void AppendPNNextChainTypes(std::string& data, const void* pnext) {
        const auto* node = static_cast<const vk::BaseOutStructure*>(pnext);
        while (node != nullptr) {
            data.append(reinterpret_cast<const char*>(&node->sType), sizeof(node->sType));
            node = node->pNext;
        }
    }

    // Internal throw type for a failed shader-module fetch. It is deliberately
    // not a vk::SystemError so the GPL path can tell "the asset is broken" (the
    // monolithic fallback would fail identically) apart from "the driver refused
    // the pipeline state" (where falling back to monolithic is worthwhile).
    struct PipelineCreationError {
        PipelineError error;
    };

    vk::ShaderModule FetchModule(ShaderManager& shaders, ShaderId id) {
        try {
            auto result = shaders.GetModule(id);
            if (!result) {
                throw PipelineCreationError{PipelineError{
                    PipelineErrorKind::ShaderModuleUnavailable,
                    vk::Result::eErrorInitializationFailed,
                    std::format("shader {} unavailable: {}", id, result.error())}};
            }
            return *result;
        } catch (const vk::SystemError& e) {
            throw PipelineCreationError{PipelineError{
                PipelineErrorKind::ShaderModuleUnavailable,
                vk::Result::eErrorInitializationFailed,
                std::format("shader {} module creation failed: {}", id, e.what())}};
        }
    }

    // Rebind the desc's owned color-blend attachments onto color_blend. A desc
    // may have been copied or moved into any container since it was populated;
    // the pAttachments pointer must always point at the desc's own vector, never
    // at a caller member. Idempotent, so every PipelineFactory entry calls it.
    void BindColorBlendAttachments(const GraphicsPipelineDesc& desc) {
        desc.color_blend.attachmentCount =
            static_cast<std::uint32_t>(desc.color_blend_attachments.size());
        desc.color_blend.pAttachments =
            desc.color_blend_attachments.empty() ? nullptr : desc.color_blend_attachments.data();
    }

    // One-line, log-friendly summary of the pipeline state a driver would reject.
    std::string Describe(const GraphicsPipelineDesc& d) {
        std::string color_formats;
        for (std::size_t i = 0; i < d.color_formats.size(); ++i) {
            if (i != 0) color_formats += ',';
            color_formats += vk::to_string(d.color_formats[i]);
        }
        return std::format(
            "vert={} frag={} vbindings={} topo={} samples={} cull={} front={} "
            "depth(test={},write={},cmp={}) color=[{}] depth={} stencil={} dynamic={} layout=0x{:x}",
            d.vertex_shader, d.fragment_shader,
            d.vertex_input.vertexBindingDescriptionCount,
            vk::to_string(d.input_assembly.topology),
            vk::to_string(d.multisample.rasterizationSamples),
            vk::to_string(d.rasterization.cullMode),
            vk::to_string(d.rasterization.frontFace),
            d.depth_stencil.depthTestEnable, d.depth_stencil.depthWriteEnable,
            vk::to_string(d.depth_stencil.depthCompareOp),
            color_formats,
            vk::to_string(d.depth_format), vk::to_string(d.stencil_format),
            d.dynamic_states.size(),
            HandleToU64(d.layout));
    }

    vk::raii::Pipeline createVertexInputLibrary(
        const vk::raii::Device& device, const vk::raii::PipelineCache& cache,
        const vk::PipelineVertexInputStateCreateInfo& vi,
        const vk::PipelineInputAssemblyStateCreateInfo& ia) {
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface;
        vk::GraphicsPipelineCreateInfo ci{};
        ci.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
        ci.pNext = &lib;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        return device.createGraphicsPipeline(cache, ci);
    }

    vk::raii::Pipeline createPreRasterLibrary(
        const vk::raii::Device& device, const vk::raii::PipelineCache& cache,
        vk::ShaderModule vert_module,
        const vk::PipelineViewportStateCreateInfo& vp,
        const vk::PipelineRasterizationStateCreateInfo& rs,
        const vk::PipelineMultisampleStateCreateInfo& ms,
        vk::PipelineLayout layout,
        const std::vector<vk::DynamicState>& dynamic_states,
        std::span<const vk::SpecializationMapEntry> spec_entries,
        std::span<const std::byte> spec_data,
        const std::string& entry_point) {
        vk::SpecializationInfo spec{};
        if (!spec_entries.empty()) {
            spec.mapEntryCount = static_cast<std::uint32_t>(spec_entries.size());
            spec.pMapEntries = spec_entries.data();
            spec.dataSize = spec_data.size();
            spec.pData = spec_data.data();
        }
        vk::PipelineShaderStageCreateInfo ss({}, vk::ShaderStageFlagBits::eVertex, vert_module, entry_point.c_str());
        if (!spec_entries.empty()) {
            ss.pSpecializationInfo = &spec;
        }
        vk::PipelineDynamicStateCreateInfo dyn_state({}, dynamic_states);
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;
        // Empty VkPipelineRenderingCreateInfo chained into the pre-raster
        // library — REQUIRED on RADV < Mesa 26: removing it re-breaks rendering
        // (doc §5.1.3, Appendix A t/u); mirrors DXVK's pre-raster library
        // (dxvk_shader.cpp).
        vk::PipelineRenderingCreateInfo ri{};
        lib.pNext = &ri;
        vk::GraphicsPipelineCreateInfo ci{};
        ci.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
        ci.pNext = &lib;
        ci.stageCount = 1;
        ci.pStages = &ss;
        ci.layout = layout;
        ci.pViewportState = &vp;
        ci.pRasterizationState = &rs;
        // VUID-VkGraphicsPipelineCreateInfo-pRasterizationState-09039: a
        // pre-rasterization library must define the multisample state (the
        // rasterizer needs the sample count when rasterization is enabled).
        ci.pMultisampleState = &ms;
        ci.pDynamicState = &dyn_state;
        return device.createGraphicsPipeline(cache, ci);
    }

    // Fragment-shader-only library (FRAGMENT_SHADER bit, NO FOI bit). The FOI
    // bit must be omitted — setting it flips RADV's `has_epilog` compile path
    // and the color output is dropped on Mesa < 26 (doc §3.2, §5.1.1). Empty
    // VkPipelineRenderingCreateInfo chained: REQUIRED on RADV < 26 (doc §5.1.3,
    // Appendix A u). No pMultisampleState: ms lives in the FOI library, so
    // VUID-06635/06636/06637 (FS-lib and FOI-lib ms must be identically
    // defined) is not triggered (doc §8).
    vk::raii::Pipeline createFragmentShaderLibrary(
        const vk::raii::Device& device, const vk::raii::PipelineCache& cache,
        vk::ShaderModule frag_module,
        const vk::PipelineDepthStencilStateCreateInfo& ds,
        vk::PipelineLayout layout,
        const std::string& entry_point) {
        vk::PipelineShaderStageCreateInfo ss({}, vk::ShaderStageFlagBits::eFragment, frag_module, entry_point.c_str());
        vk::PipelineRenderingCreateInfo ri{};
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader;
        lib.pNext = &ri;
        vk::GraphicsPipelineCreateInfo ci{};
        ci.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
        ci.pNext = &lib;
        ci.stageCount = 1;
        ci.pStages = &ss;
        ci.layout = layout;
        ci.pDepthStencilState = &ds;
        return device.createGraphicsPipeline(cache, ci);
    }

    // Fragment-output-interface library: pColorBlendState + pMultisampleState.
    // ms is mandatory — VUID-VkGraphicsPipelineCreateInfo-pMultisampleState-09026,
    // and an FOI library without it segfaults even on lavapipe (doc §7 crash
    // #2). No shader state, no layout. Chains the SAME rendering info as the
    // final link: VUID-VkGraphicsPipelineCreateInfo-renderPass-06055 requires
    // VkPipelineRenderingCreateInfo::colorAttachmentCount to equal
    // pColorBlendState->attachmentCount, so the output formats are baked into
    // this library (doc §5.3 pseudocode rinfo_fmt; variant w — no rinfo — still
    // renders on drivers but is not validation-clean).
    vk::raii::Pipeline createFragmentOutputLibrary(
        const vk::raii::Device& device, const vk::raii::PipelineCache& cache,
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const vk::PipelineColorBlendStateCreateInfo& cb,
        std::span<const vk::Format> color_formats,
        vk::Format depth_fmt,
        vk::Format stencil_fmt) {
        vk::PipelineRenderingCreateInfo ri{};
        ri.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
        ri.pColorAttachmentFormats = color_formats.data();
        ri.depthAttachmentFormat = depth_fmt;
        ri.stencilAttachmentFormat = stencil_fmt;
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;
        lib.pNext = &ri;
        vk::GraphicsPipelineCreateInfo ci{};
        ci.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
        ci.pNext = &lib;
        ci.pMultisampleState = &ms;
        ci.pColorBlendState = &cb;
        return device.createGraphicsPipeline(cache, ci);
    }

    // Combined fragment-shader + fragment-output library. Per
    // VUID-VkGraphicsPipelineCreateInfo-flags-08906/08907 a fragment shader
    // library must also define fragment output interface state, so the two
    // subsets are compiled into a single library. This combined structure is
    // BROKEN on RADV < Mesa 26 (doc §3.1, §4; Appendix A a/c/e/g) — it is only
    // reachable when ResolveGpl allowed it (non-RADV or Mesa ≥ 26, or the
    // forced footgun path). Carries the rendering info with formats, same as
    // the FOI library (VUID-06055) and the final link.
    vk::raii::Pipeline createFragmentLibrary(
        const vk::raii::Device& device, const vk::raii::PipelineCache& cache,
        vk::ShaderModule frag_module,
        const vk::PipelineDepthStencilStateCreateInfo& ds,
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const vk::PipelineColorBlendStateCreateInfo& cb,
        std::span<const vk::Format> color_formats,
        vk::Format depth_fmt,
        vk::Format stencil_fmt,
        vk::PipelineLayout layout,
        const std::string& entry_point) {
        vk::PipelineShaderStageCreateInfo ss({}, vk::ShaderStageFlagBits::eFragment, frag_module, entry_point.c_str());
        vk::PipelineRenderingCreateInfo ri{};
        ri.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
        ri.pColorAttachmentFormats = color_formats.data();
        ri.depthAttachmentFormat = depth_fmt;
        ri.stencilAttachmentFormat = stencil_fmt;
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader |
                    vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;
        lib.pNext = &ri;
        vk::GraphicsPipelineCreateInfo ci{};
        ci.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
        ci.pNext = &lib;
        ci.stageCount = 1;
        ci.pStages = &ss;
        ci.layout = layout;
        ci.pDepthStencilState = &ds;
        ci.pMultisampleState = &ms;
        ci.pColorBlendState = &cb;
        return device.createGraphicsPipeline(cache, ci);
    }
}

PipelineFactory::PipelineFactory(const vk::raii::Device& device, const VulkanBackend::Vulkan::VulkanCapabilities& caps,
                                   const vk::raii::PipelineCache& cache, GplPolicy policy, GplStructurePolicy structure)
    : device_(device), cache_(cache)
    , shared_(std::make_shared<SharedLibraries>())
{
    // Single decision point: resolve the GPL policy + structure up front and
    // log once at construction (zero render-time cost). ResolveGpl implements
    // the full semantics table (doc §3/§4/§6/§9): affected RADV auto-resolves
    // to the split structure; an explicit combined structure on affected RADV
    // is refused (monolithic) unless forced (footgun, testing only).
    resolution_ = ResolveGpl(
        policy, structure,
        caps.CanUse(VulkanBackend::Vulkan::DeviceExtension::GraphicsPipelineLibrary),
        static_cast<std::uint32_t>(caps.GetDriverProperties().driverID),
        caps.GetProperties().driverVersion);
    gpl_available_ = resolution_.use_gpl;
    const GplDriverVersion version = DecodeDriverVersion(caps.GetProperties().driverVersion);
    switch (resolution_.warning) {
        case GplResolution::Warning::CombinedDenied:
            LOGIFACE_LOG(error, "PipelineFactory: gpl.structure=combined unsupported on RADV < Mesa 26.0.0 "
                                "(docs/RADV-GPL-fast-linking-bug-and-workaround.md §3.2, §4) — using monolithic. "
                                "Use gpl.structure=split or upgrade Mesa.");
            break;
        case GplResolution::Warning::ForcedCombinedFootgun:
            LOGIFACE_LOG(warn, "PipelineFactory: forced combined structure on affected RADV — known-broken "
                               "(renders nothing), testing only (docs/RADV-GPL-fast-linking-bug-and-workaround.md §3.1, §6)");
            break;
        case GplResolution::Warning::ForcedUnsupported:
            LOGIFACE_LOG(warn, "PipelineFactory: GPL policy forced, but VK_EXT_graphics_pipeline_library "
                               "is unavailable or disabled; using monolithic");
            break;
        case GplResolution::Warning::None:
            break;
    }
    if (resolution_.warning == GplResolution::Warning::None) {
        LOGIFACE_LOG(info, std::format("PipelineFactory: GPL {} (structure={}), driver id=0x{:x}, version={}.{}.{}",
                                       resolution_.use_gpl ? "enabled" : "disabled",
                                       resolution_.structure == GplStructurePolicy::Split ? "split" : "combined",
                                       static_cast<std::uint32_t>(caps.GetDriverProperties().driverID),
                                       version.major, version.minor, version.patch));
    }
}

void PipelineFactory::DisableGplForRun(std::string_view reason) const {
    const bool first = !gpl_runtime_disabled_.exchange(true, std::memory_order_relaxed);
    if (first) {
        LOGIFACE_LOG(info, std::format("PipelineFactory: GPL disabled for the rest of this run after first "
                                       "failure ({}); subsequent pipelines use monolithic", reason));
    }
}

std::expected<PipelineProduct, PipelineError>
PipelineFactory::CreateGraphics(const GraphicsPipelineDesc& desc,
                                  ShaderManager& shaders) const {
    // The desc owns its attachments; refresh the pointer before anything reads
    // the blend state or hashes it for the GPL cache key.
    BindColorBlendAttachments(desc);
    const bool runtime_disabled = gpl_runtime_disabled_.load(std::memory_order_relaxed);
    try {
        if (gpl_available_ && !runtime_disabled) {
            LOGIFACE_LOG(debug, std::format("PipelineFactory: creating graphics pipeline via GPL (vert={}, frag={})",
                                            desc.vertex_shader, desc.fragment_shader));
            return CreateGraphicsGPL(desc, shaders);
        }
        if (gpl_available_ && runtime_disabled) {
            LOGIFACE_LOG(debug, std::format("PipelineFactory: GPL disabled at runtime; using monolithic (vert={}, frag={})",
                                            desc.vertex_shader, desc.fragment_shader));
        } else {
            LOGIFACE_LOG(debug, std::format("PipelineFactory: creating graphics pipeline via monolithic (vert={}, frag={})",
                                            desc.vertex_shader, desc.fragment_shader));
        }
        return CreateGraphicsMonolithic(desc, shaders);
    } catch (const PipelineCreationError& e) {
        LOGIFACE_LOG(error, std::format("PipelineFactory: {}", e.error.message));
        return std::unexpected(e.error);
    } catch (const vk::SystemError& e) {
        const auto result = static_cast<vk::Result>(e.code().value());
        const std::string message = std::format("{} ({}) [{}]", e.what(), vk::to_string(result), Describe(desc));
        LOGIFACE_LOG(error, std::format("PipelineFactory: graphics pipeline creation failed: {}", message));
        return std::unexpected(PipelineError{PipelineErrorKind::DriverRejectedPipeline, result, message});
    }
}

std::expected<PipelineProduct, PipelineError>
PipelineFactory::CreateCompute(const ComputePipelineDesc& desc,
                                 ShaderManager& shaders) const {
    try {
        vk::ShaderModule module = FetchModule(shaders, desc.shader);
        vk::PipelineShaderStageCreateInfo ss({}, vk::ShaderStageFlagBits::eCompute, module, desc.entry_point.c_str());
        vk::SpecializationInfo spec{};
        if (!desc.spec_entries.empty()) {
            spec.mapEntryCount = static_cast<std::uint32_t>(desc.spec_entries.size());
            spec.pMapEntries = desc.spec_entries.data();
            spec.dataSize = desc.spec_data.size();
            spec.pData = desc.spec_data.data();
            ss.pSpecializationInfo = &spec;
        }
        vk::ComputePipelineCreateInfo ci({}, ss, desc.layout);
        vk::raii::Pipeline pipeline = device_.createComputePipeline(cache_, ci);
        LOGIFACE_LOG(debug, std::format("PipelineFactory: compute pipeline created (shader={}): 0x{:x}",
                                        desc.shader, HandleToU64(*pipeline)));
        return PipelineProduct::Monolithic(std::move(pipeline));
    } catch (const PipelineCreationError& e) {
        LOGIFACE_LOG(error, std::format("PipelineFactory: {}", e.error.message));
        return std::unexpected(e.error);
    } catch (const vk::SystemError& e) {
        const auto result = static_cast<vk::Result>(e.code().value());
        const std::string message = std::format("{} ({}) [compute shader={}, layout=0x{:x}]",
                                                e.what(), vk::to_string(result), desc.shader, HandleToU64(desc.layout));
        LOGIFACE_LOG(error, std::format("PipelineFactory: compute pipeline creation failed: {}", message));
        return std::unexpected(PipelineError{PipelineErrorKind::DriverRejectedPipeline, result, message});
    }
}

PipelineProduct
PipelineFactory::CreateGraphicsMonolithic(const GraphicsPipelineDesc& desc,
                                            ShaderManager& shaders) const {
    auto vert_mod = FetchModule(shaders, desc.vertex_shader);
    auto frag_mod = FetchModule(shaders, desc.fragment_shader);

    // Vertex-stage specialization constants select the draw mode. The fragment
    // shader has no constant_id 0, so the map applies to the vertex stage only.
    vk::SpecializationInfo vs_spec{};
    if (!desc.spec_entries.empty()) {
        vs_spec.mapEntryCount = static_cast<std::uint32_t>(desc.spec_entries.size());
        vs_spec.pMapEntries = desc.spec_entries.data();
        vs_spec.dataSize = desc.spec_data.size();
        vs_spec.pData = desc.spec_data.data();
    }

    vk::PipelineShaderStageCreateInfo vs({}, vk::ShaderStageFlagBits::eVertex, vert_mod, desc.vertex_entry_point.c_str());
    if (!desc.spec_entries.empty()) {
        vs.pSpecializationInfo = &vs_spec;
    }
    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {
        vs,
        vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, frag_mod, desc.fragment_entry_point.c_str()}
    };

    vk::PipelineDynamicStateCreateInfo dyn_state({}, desc.dynamic_states);

    vk::PipelineRenderingCreateInfo ri{};
    ri.colorAttachmentCount = static_cast<std::uint32_t>(desc.color_formats.size());
    ri.pColorAttachmentFormats = desc.color_formats.data();
    ri.depthAttachmentFormat = desc.depth_format;
    ri.stencilAttachmentFormat = desc.stencil_format;

    vk::GraphicsPipelineCreateInfo pi({}, stages, &desc.vertex_input,
        &desc.input_assembly, nullptr, &desc.viewport,
        &desc.rasterization, &desc.multisample,
        &desc.depth_stencil, &desc.color_blend, &dyn_state,
        desc.layout, nullptr, 0, {}, 0);
    pi.setPNext(&ri);

    vk::raii::Pipeline pipeline = device_.createGraphicsPipeline(cache_, pi);
    LOGIFACE_LOG(debug, std::format("PipelineFactory: monolithic pipeline created: 0x{:x}",
                                    HandleToU64(*pipeline)));
    return PipelineProduct::Monolithic(std::move(pipeline));
}

PipelineProduct
PipelineFactory::CreateGraphicsGPL(const GraphicsPipelineDesc& desc,
                                     ShaderManager& shaders) const {
    // Defense-in-depth: the caller checked gpl_available_, which mirrors
    // resolution_.use_gpl.
    assert(resolution_.use_gpl);
    try {
        auto vert_mod = FetchModule(shaders, desc.vertex_shader);
        auto frag_mod = FetchModule(shaders, desc.fragment_shader);

        const auto vi_hash = HashVertexInput(desc.vertex_input, desc.input_assembly);
        auto pr_hash = HashPreRaster(desc.viewport, desc.rasterization, desc.multisample,
                                     desc.dynamic_states, desc.layout);
        pr_hash ^= static_cast<std::uint64_t>(desc.vertex_shader) << 32;
        pr_hash ^= shaders.GetVersion(desc.vertex_shader);
        // The pre-raster library is cached by vertex shader id/version only, so
        // the vertex-stage spec data must enter the key: without it two draw
        // modes that share a vertex shader alias to one library.
        pr_hash = fnv1a_hash(
            std::string_view(reinterpret_cast<const char*>(desc.spec_data.data()),
                             desc.spec_data.size()),
            pr_hash);
        // The old single fragment key splits into the shader-dependent FS part
        // and the shader-independent FOI part. Output formats/depth/stencil
        // enter the FOI key only: the FOI (and combined) libraries bake them
        // into their VkPipelineRenderingCreateInfo (VUID-06055, doc §5.3
        // rinfo_fmt), while the FS and PRE libraries carry empty rinfo
        // (doc §5.1.3) and the final link repeats the same formats.
        const auto fs_hash = HashFragmentShader(desc.depth_stencil, desc.layout);
        const auto foi_hash = HashFragmentOutput(desc.multisample, desc.color_blend,
                                                 desc.color_formats, desc.depth_format,
                                                 desc.stencil_format);
        LOGIFACE_LOG(debug, std::format("GPL: structure={}, hashes vi={:016x} pr={:016x} fs={:016x} foi={:016x}",
                                        resolution_.structure == GplStructurePolicy::Split ? "split" : "combined",
                                        vi_hash, pr_hash, fs_hash, foi_hash));

        // Shared-library lookup: reuse a cached library on a key hit, otherwise
        // compile a new one and publish it. Both branches return a shared_ptr,
        // so linked pipelines keep their libraries alive even after the cache
        // drops the entry (see InvalidateShader).
        const auto get_or_create =
            [this](auto& map, std::uint64_t key, ShaderId shader_id, std::uint64_t shader_version,
                   auto&& create) -> std::pair<std::shared_ptr<vk::raii::Pipeline>, bool> {
            {
                const std::shared_lock lock(shared_->mutex);
                const auto it = map.find(key);
                if (it != map.end()) {
                    return {it->second.pipeline, false};
                }
            }
            auto lib = std::make_shared<vk::raii::Pipeline>(create());
            bool created = false;
            std::shared_ptr<vk::raii::Pipeline> winner;
            {
                const std::unique_lock lock(shared_->mutex);
                const auto [it, inserted] =
                    map.try_emplace(key, SharedLibraries::LibraryEntry{shader_id, shader_version, lib});
                winner = it->second.pipeline;
                created = inserted;
            }
            return {std::move(winner), created};
        };

        const auto [vi_lib, vi_created] =
            get_or_create(shared_->vertex_input, vi_hash, {}, 0,
                          [&]() { return createVertexInputLibrary(device_, cache_, desc.vertex_input, desc.input_assembly); });
        LOGIFACE_LOG(debug, std::format("GPL: vertex-input library {}: 0x{:x}",
                                        vi_created ? "created" : "reused", HandleToU64(**vi_lib)));

        const auto [pr_lib, pr_created] =
            get_or_create(shared_->pre_raster, pr_hash, desc.vertex_shader, shaders.GetVersion(desc.vertex_shader),
                          [&]() { return createPreRasterLibrary(device_, cache_, vert_mod, desc.viewport,
                                                                 desc.rasterization, desc.multisample,
                                                                 desc.layout, desc.dynamic_states,
                                                                 desc.spec_entries, desc.spec_data,
                                                                 desc.vertex_entry_point); });
        LOGIFACE_LOG(debug, std::format("GPL: pre-raster library {}: 0x{:x}",
                                        pr_created ? "created" : "reused", HandleToU64(**pr_lib)));

        std::shared_ptr<vk::raii::Pipeline> fragment_shader_lib;
        std::shared_ptr<vk::raii::Pipeline> fragment_output_lib;
        if (resolution_.structure == GplStructurePolicy::Split) {
            // Split structure (doc §5, variant s/m): separate FS and FOI
            // libraries — the only structure that renders on RADV < 26.
            auto key = fs_hash;
            key ^= static_cast<std::uint64_t>(desc.fragment_shader) << 32;
            key ^= shaders.GetVersion(desc.fragment_shader);
            const auto [fs_lib, fs_created] =
                get_or_create(shared_->fragment_shader, key, desc.fragment_shader, shaders.GetVersion(desc.fragment_shader),
                              [&]() { return createFragmentShaderLibrary(device_, cache_, frag_mod,
                                                                         desc.depth_stencil, desc.layout,
                                                                         desc.fragment_entry_point); });
            LOGIFACE_LOG(debug, std::format("GPL: fragment-shader library {}: 0x{:x}",
                                            fs_created ? "created" : "reused", HandleToU64(**fs_lib)));
            const auto [foi_lib, foi_created] =
                get_or_create(shared_->fragment_output, foi_hash, {}, 0,
                              [&]() { return createFragmentOutputLibrary(device_, cache_,
                                                                         desc.multisample, desc.color_blend,
                                                                         desc.color_formats, desc.depth_format,
                                                                         desc.stencil_format); });
            LOGIFACE_LOG(debug, std::format("GPL: fragment-output library {}: 0x{:x}",
                                            foi_created ? "created" : "reused", HandleToU64(**foi_lib)));
            fragment_shader_lib = fs_lib;
            fragment_output_lib = foi_lib;
        } else {
            // Combined structure: single FS|FOI library. Only reachable when
            // ResolveGpl allowed it (non-RADV or Mesa ≥ 26, or the forced
            // footgun path).
            auto key = fs_hash ^ foi_hash;
            key ^= static_cast<std::uint64_t>(desc.fragment_shader) << 32;
            key ^= shaders.GetVersion(desc.fragment_shader);
            const auto [frag_lib, frag_created] =
                get_or_create(shared_->fragment_shader, key, desc.fragment_shader, shaders.GetVersion(desc.fragment_shader),
                              [&]() { return createFragmentLibrary(device_, cache_, frag_mod,
                                                                   desc.depth_stencil, desc.multisample,
                                                                   desc.color_blend, desc.color_formats,
                                                                    desc.depth_format, desc.stencil_format,
                                                                    desc.layout, desc.fragment_entry_point); });
            LOGIFACE_LOG(debug, std::format("GPL: combined fragment library {}: 0x{:x}",
                                            frag_created ? "created" : "reused", HandleToU64(**frag_lib)));
            fragment_shader_lib = frag_lib;
        }

        std::vector<vk::Pipeline> libraries;
        libraries.reserve(4);
        libraries.push_back(**vi_lib);
        libraries.push_back(**pr_lib);
        libraries.push_back(**fragment_shader_lib);
        if (fragment_output_lib) {
            libraries.push_back(**fragment_output_lib);
        }

        auto linked = CreateGraphicsGPLFinalLink(desc, libraries);
        if (!linked.has_value()) {
            DisableGplForRun("final link returned no pipeline");
            LOGIFACE_LOG(warn, std::format("GPL: fast link not possible (VK_PIPELINE_COMPILE_REQUIRED); "
                                           "falling back to monolithic [{}]", Describe(desc)));
            return CreateGraphicsMonolithic(desc, shaders);
        }
        return PipelineProduct::GPLLinked(std::move(*linked), vi_lib, pr_lib,
                                          fragment_shader_lib, fragment_output_lib);
    } catch (const PipelineCreationError&) {
        // Shader modules are the one failure the monolithic path cannot recover
        // from either — surface it as-is instead of pointlessly re-fetching.
        throw;
    } catch (const vk::SystemError& e) {
        const auto result = static_cast<vk::Result>(e.code().value());
        DisableGplForRun(e.what());
        LOGIFACE_LOG(warn, std::format("GPL pipeline creation failed: {} ({}) — falling back to monolithic [{}]",
                                       e.what(), vk::to_string(result), Describe(desc)));
        try {
            return CreateGraphicsMonolithic(desc, shaders);
        } catch (const PipelineCreationError&) {
            throw;
        } catch (const vk::SystemError& e2) {
            const auto result2 = static_cast<vk::Result>(e2.code().value());
            LOGIFACE_LOG(error, std::format("monolithic fallback failed too: {} ({})",
                                            e2.what(), vk::to_string(result2)));
            throw;
        }
    }
}

std::optional<vk::raii::Pipeline>
PipelineFactory::CreateGraphicsGPLFinalLink(const GraphicsPipelineDesc& desc,
                                            std::span<const vk::Pipeline> libraries) const {
    vk::PipelineLibraryCreateInfoKHR link_info{};
    link_info.libraryCount = static_cast<std::uint32_t>(libraries.size());
    link_info.pLibraries = libraries.data();

    vk::GraphicsPipelineCreateInfo linked_ci{};
    linked_ci.flags = vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
    linked_ci.layout = desc.layout;
    linked_ci.setPNext(&link_info);

    // The final link's rendering info declares the dynamic-rendering output
    // formats; the FOI/combined libraries carry the identical content in their
    // own VkPipelineRenderingCreateInfo (VUID-06055), so the link matches.
    vk::PipelineRenderingCreateInfo ri{};
    ri.colorAttachmentCount = static_cast<std::uint32_t>(desc.color_formats.size());
    ri.pColorAttachmentFormats = desc.color_formats.data();
    ri.depthAttachmentFormat = desc.depth_format;
    ri.stencilAttachmentFormat = desc.stencil_format;
    link_info.pNext = &ri;

    // Fast link without LINK_TIME_OPTIMIZATION (doc §6.3) — no LTO also means
    // crash #1 of doc §7 (LTO libs without RETAIN) cannot occur. FAIL_ON makes
    // fast-link failure deterministic: without it a conformant driver may
    // silently compile instead of returning VK_PIPELINE_COMPILE_REQUIRED
    // (doc §6.5, §9).
    //
    // VULKAN-HPP RESULT MASKING (verified against the vcpkg-built headers this
    // project compiles against): with exceptions enabled the raii vector form
    // returns std::vector<vk::raii::Pipeline> directly and DISCARDS the result
    // code. VK_PIPELINE_COMPILE_REQUIRED is a SUCCESS code (positive value
    // 1000297000, not an error), so nothing throws — instead the wrapper wraps
    // EVERY raw handle the driver wrote into the out-array, including
    // VK_NULL_HANDLE. On "cannot fast-link" the vector is therefore NOT empty:
    // it contains a null handle. Never test emptiness alone; check the handle.
    // (This differs from newer vulkan-hpp raii codegen, which only fills the
    // vector on eSuccess — see the note in docs/... §6.5.)
    //
    // DRIVER BEHAVIOR (verified on Mesa 25.3.6):
    //  - RADV: FAIL_ON + fast link → VK_SUCCESS + valid handle; COMPILE_REQUIRED
    //    only when a compile is genuinely required (e.g. LTO after cache miss).
    //  - llvmpipe/lavapipe: honors FAIL_ON literally — lvp_CreateGraphicsPipelines
    //    (lvp_pipeline.c) skips creation entirely whenever the flag is set and
    //    returns VK_PIPELINE_COMPILE_REQUIRED + VK_NULL_HANDLE unconditionally
    //    (llvmpipe always requires a compile). A FAIL_ON final link therefore
    //    NEVER succeeds on lavapipe.
    //
    // Strategy: treat "no usable pipeline" (empty or null handle) as "cannot
    // fast-link". Retry once WITHOUT FAIL_ON so drivers that refuse the flag
    // (lavapipe) can create the pipeline normally (silently compiled — the
    // exact degradation FAIL_ON exists to avoid, but only where the driver
    // leaves no other option); if that also fails, fall back to monolithic
    // (doc §6.5, §9).
    auto pipelines = device_.createGraphicsPipelines(cache_, std::array{linked_ci}, nullptr);
    if (pipelines.empty() || !*pipelines[0]) {
        LOGIFACE_LOG(warn, "GPL: final link returned no pipeline with VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED "
                           "(VK_PIPELINE_COMPILE_REQUIRED or driver refuses FAIL_ON links, e.g. llvmpipe/lavapipe) "
                           "— retrying without the flag");
        vk::GraphicsPipelineCreateInfo plain_ci = linked_ci;
        plain_ci.flags = {};
        pipelines = device_.createGraphicsPipelines(cache_, std::array{plain_ci}, nullptr);
        if (pipelines.empty() || !*pipelines[0]) {
            LOGIFACE_LOG(warn, "GPL: final link still returned no pipeline without FAIL_ON — cannot fast-link; "
                               "falling back to monolithic (doc §6.5, §9)");
            return std::nullopt;
        }
        LOGIFACE_LOG(warn, "GPL: fast link not possible (VK_PIPELINE_COMPILE_REQUIRED); pipeline was silently "
                           "compiled on the no-FAIL_ON retry");
    }
    LOGIFACE_LOG(debug, std::format("GPL: linked pipeline created: 0x{:x}",
                                    HandleToU64(*pipelines[0])));
    return std::move(pipelines[0]);
}

void PipelineFactory::InvalidateShader(ShaderId id) {
    // Only drop the shared-cache reference. Linked pipelines keep their
    // libraries alive through the PipelineProduct shared_ptr, so fast-linked
    // pipelines stay valid; the next creation compiles fresh libraries.
    const std::unique_lock lock(shared_->mutex);
    const auto matches_shader = [id](const auto& entry) { return entry.second.shader_id == id; };
    std::erase_if(shared_->pre_raster, matches_shader);
    std::erase_if(shared_->fragment_shader, matches_shader);
}

std::uint64_t PipelineFactory::HashVertexInput(const vk::PipelineVertexInputStateCreateInfo& vi,
                                               const vk::PipelineInputAssemblyStateCreateInfo& ia) {
    const auto append_bytes = [](std::string& data, const void* p, std::size_t n) {
        data.append(reinterpret_cast<const char*>(p), n);
    };
    std::string data;
    data.reserve(48);
    // Hash only value fields — never pointers into caller memory. The struct
    // fields are not contiguous (pointers sit between the counts), so hash
    // each field explicitly.
    append_bytes(data, &vi.flags, sizeof(vi.flags));
    append_bytes(data, &vi.vertexBindingDescriptionCount, sizeof(vi.vertexBindingDescriptionCount));
    for (uint32_t i = 0; i < vi.vertexBindingDescriptionCount && vi.pVertexBindingDescriptions; ++i) {
        append_bytes(data, &vi.pVertexBindingDescriptions[i], sizeof(vk::VertexInputBindingDescription));
    }
    append_bytes(data, &vi.vertexAttributeDescriptionCount, sizeof(vi.vertexAttributeDescriptionCount));
    for (uint32_t i = 0; i < vi.vertexAttributeDescriptionCount && vi.pVertexAttributeDescriptions; ++i) {
        append_bytes(data, &vi.pVertexAttributeDescriptions[i], sizeof(vk::VertexInputAttributeDescription));
    }
    // Input assembly is part of the vertex input state subset and lives in the vertex-input
    // library — include it in the cache key so pipelines with different topologies don't alias.
    append_bytes(data, &ia.flags, sizeof(ia.flags));
    append_bytes(data, &ia.topology, sizeof(ia.topology));
    append_bytes(data, &ia.primitiveRestartEnable, sizeof(ia.primitiveRestartEnable));
    AppendPNNextChainTypes(data, vi.pNext);
    AppendPNNextChainTypes(data, ia.pNext);
    return fnv1a_hash(data);
}

std::uint64_t PipelineFactory::HashPreRaster(
        const vk::PipelineViewportStateCreateInfo& vp,
        const vk::PipelineRasterizationStateCreateInfo& rs,
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const std::vector<vk::DynamicState>& dynamic_states,
        vk::PipelineLayout layout) {
    const auto append_bytes = [](std::string& data, const void* p, std::size_t n) {
        data.append(reinterpret_cast<const char*>(p), n);
    };
    std::string data;
    data.reserve(64);
    // Hash from `flags` onward, excluding the sType/pNext header (the pNext
    // pointer is heap-dependent and must not enter the key).
    append_bytes(data, &rs.flags,
                 reinterpret_cast<const char*>(&rs.lineWidth + 1) -
                     reinterpret_cast<const char*>(&rs.flags));
    append_bytes(data, &vp.flags, sizeof(vp.flags));
    append_bytes(data, &vp.viewportCount, sizeof(vp.viewportCount));
    // Static viewport/scissor rects are baked into the library when not
    // dynamic; hash their contents so pipelines using them don't alias.
    if (vp.pViewports) {
        for (std::uint32_t i = 0; i < vp.viewportCount; ++i) {
            append_bytes(data, &vp.pViewports[i], sizeof(vk::Viewport));
        }
    }
    append_bytes(data, &vp.scissorCount, sizeof(vp.scissorCount));
    if (vp.pScissors) {
        for (std::uint32_t i = 0; i < vp.scissorCount; ++i) {
            append_bytes(data, &vp.pScissors[i], sizeof(vk::Rect2D));
        }
    }
    for (const auto s : dynamic_states) {
        append_bytes(data, &s, sizeof(s));
    }
    // The pre-raster library carries the multisample state
    // (VUID-VkGraphicsPipelineCreateInfo-pRasterizationState-09039) — hash its
    // value fields, again excluding the sType/pNext header and the pSampleMask
    // pointer itself (mask contents are hashed below).
    append_bytes(data, &ms.flags, sizeof(ms.flags));
    append_bytes(data, &ms.rasterizationSamples, sizeof(ms.rasterizationSamples));
    append_bytes(data, &ms.sampleShadingEnable, sizeof(ms.sampleShadingEnable));
    append_bytes(data, &ms.minSampleShading, sizeof(ms.minSampleShading));
    append_bytes(data, &ms.alphaToCoverageEnable, sizeof(ms.alphaToCoverageEnable));
    append_bytes(data, &ms.alphaToOneEnable, sizeof(ms.alphaToOneEnable));
    if (ms.pSampleMask) {
        const std::uint32_t words = (static_cast<std::uint32_t>(ms.rasterizationSamples) + 31) / 32;
        append_bytes(data, ms.pSampleMask, sizeof(std::uint32_t) * words);
    }
    // pNext-chained state (depth clamp control, sample locations, ...) is part
    // of the compiled library; hash the chain shape so pipelines using
    // different extension state don't share the library.
    AppendPNNextChainTypes(data, rs.pNext);
    AppendPNNextChainTypes(data, vp.pNext);
    AppendPNNextChainTypes(data, ms.pNext);
    // The pre-raster library embeds the pipeline layout — include it in the key
    // so pipelines with identical state but different layouts don't share a library.
    append_bytes(data, &layout, sizeof(layout));
    return fnv1a_hash(data);
}

std::uint64_t PipelineFactory::HashFragmentShader(
        const vk::PipelineDepthStencilStateCreateInfo& ds,
        vk::PipelineLayout layout) {
    const auto append_bytes = [](std::string& data, const void* p, std::size_t n) {
        data.append(reinterpret_cast<const char*>(p), n);
    };
    std::string data;
    data.reserve(64);
    // Depth/stencil state — hash value fields explicitly (avoids padding
    // between stencil ops and the depth bounds doubles), then chain shape.
    append_bytes(data, &ds.flags, sizeof(ds.flags));
    append_bytes(data, &ds.depthTestEnable, sizeof(ds.depthTestEnable));
    append_bytes(data, &ds.depthWriteEnable, sizeof(ds.depthWriteEnable));
    append_bytes(data, &ds.depthCompareOp, sizeof(ds.depthCompareOp));
    append_bytes(data, &ds.depthBoundsTestEnable, sizeof(ds.depthBoundsTestEnable));
    append_bytes(data, &ds.stencilTestEnable, sizeof(ds.stencilTestEnable));
    append_bytes(data, &ds.front, sizeof(ds.front));
    append_bytes(data, &ds.back, sizeof(ds.back));
    append_bytes(data, &ds.minDepthBounds, sizeof(ds.minDepthBounds));
    append_bytes(data, &ds.maxDepthBounds, sizeof(ds.maxDepthBounds));
    AppendPNNextChainTypes(data, ds.pNext);
    // The fragment-shader library embeds the pipeline layout — include it in
    // the key so pipelines with identical state but different layouts don't
    // share a library.
    append_bytes(data, &layout, sizeof(layout));
    return fnv1a_hash(data);
}

std::uint64_t PipelineFactory::HashFragmentOutput(
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const vk::PipelineColorBlendStateCreateInfo& cb,
        std::span<const vk::Format> color_formats,
        vk::Format depth_fmt,
        vk::Format stencil_fmt) {
    const auto append_bytes = [](std::string& data, const void* p, std::size_t n) {
        data.append(reinterpret_cast<const char*>(p), n);
    };
    std::string data;
    data.reserve(96);
    // Multisample state — the FOI library carries it (VUID-09026; doc §7 crash
    // #2). Hash its value fields exactly like HashPreRaster's ms block so both
    // libraries agree on the same struct.
    append_bytes(data, &ms.flags, sizeof(ms.flags));
    append_bytes(data, &ms.rasterizationSamples, sizeof(ms.rasterizationSamples));
    append_bytes(data, &ms.sampleShadingEnable, sizeof(ms.sampleShadingEnable));
    append_bytes(data, &ms.minSampleShading, sizeof(ms.minSampleShading));
    append_bytes(data, &ms.alphaToCoverageEnable, sizeof(ms.alphaToCoverageEnable));
    append_bytes(data, &ms.alphaToOneEnable, sizeof(ms.alphaToOneEnable));
    if (ms.pSampleMask) {
        const std::uint32_t words = (static_cast<std::uint32_t>(ms.rasterizationSamples) + 31) / 32;
        append_bytes(data, ms.pSampleMask, sizeof(std::uint32_t) * words);
    }
    AppendPNNextChainTypes(data, ms.pNext);

    // Color blend state.
    append_bytes(data, &cb.flags, sizeof(cb.flags));
    append_bytes(data, &cb.logicOpEnable, sizeof(cb.logicOpEnable));
    append_bytes(data, &cb.logicOp, sizeof(cb.logicOp));
    append_bytes(data, &cb.attachmentCount, sizeof(cb.attachmentCount));
    for (std::uint32_t i = 0; i < cb.attachmentCount && cb.pAttachments; ++i) {
        append_bytes(data, &cb.pAttachments[i], sizeof(vk::PipelineColorBlendAttachmentState));
    }
    append_bytes(data, cb.blendConstants.data(), sizeof(cb.blendConstants));
    AppendPNNextChainTypes(data, cb.pNext);

    // The FOI library's VkPipelineRenderingCreateInfo embeds the dynamic-
    // rendering output formats (VUID-06055: colorAttachmentCount must equal
    // pColorBlendState->attachmentCount) — include them in the key. They are
    // NOT part of the FS library key (FS rinfo stays empty, doc §5.1.3).
    for (const auto f : color_formats) {
        append_bytes(data, &f, sizeof(f));
    }
    append_bytes(data, &depth_fmt, sizeof(depth_fmt));
    append_bytes(data, &stencil_fmt, sizeof(stencil_fmt));
    return fnv1a_hash(data);
}

} // namespace VulkanEngine::ShaderSystem
