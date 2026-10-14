module;

#include <cassert>
#include <logging/logging_macros.hpp>

module VulkanEngine.TechniqueManager.BaseTechnique;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.StandardMeshPipeline;
import VulkanEngine.GpuResources.BlockArray;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.PipelineFactory;
import VulkanEngine.PipelineTemplates;
import VulkanEngine.ShaderManager;

namespace {
    VulkanEngine::TechniqueManager::TechniqueId s_next_technique_id{0};

    template<typename Handle>
    std::uint64_t HandleToU64(Handle h) {
        return reinterpret_cast<std::uint64_t>(static_cast<typename Handle::CType>(h));
    }

    // The draw-mode spec writer lives in VulkanEngine.PipelineTemplates
    // (single copy shared with the scene renderer): ApplyDrawModeSpec
    // (constant_id 0, 0 = CID, 1 = MID). One writer keeps Compile and the
    // runtime draw-mode switch identical.
}

namespace VulkanEngine::TechniqueManager {

void BaseTechnique::Shutdown() {
    material_array_bindings_.clear();
    custom_descriptor_sets_.clear();
    custom_descriptor_set_handles_.clear();
    descriptor_pool_ = nullptr;
    custom_set_layouts_.clear();
    block_arrays_.clear();
    shared_buffers_.clear();
    shared_cpu_data_.clear();
    variants_.clear();
    pipeline_layout_ = nullptr;
    bindings_.clear();
    compiled_ = false;
    device_ = nullptr;
}

uint32_t BaseTechnique::PackMaterialData(uint32_t material_id, uint32_t draw_key) const {
    // Default: pack material_id and the resolved draw key into a single uint32
    return TechniquePacking::Pack(material_id, draw_key);
}

bool BaseTechnique::EnsureMaterialBlockBound(std::size_t binding_index,
                                             std::uint32_t block_index) {
    for (auto& mb : material_array_bindings_) {
        if (mb.binding_index != binding_index) continue;

        if (block_index >= mb.block_bound.size()) {
            LOGIFACE_LOG(error, "BaseTechnique::EnsureMaterialBlockBound: block " +
                         std::to_string(block_index) + " exceeds the descriptor array capacity (" +
                         std::to_string(mb.block_bound.size()) + ") for technique " +
                         std::to_string(id_.value));
            return false;
        }
        if (mb.block_bound[block_index] != 0) return true;

        if (mb.block_array_index >= block_arrays_.size()) return false;
        auto& ba = block_arrays_[mb.block_array_index];
        if (block_index >= ba.BlockCount()) {
            LOGIFACE_LOG(error, "BaseTechnique::EnsureMaterialBlockBound: block " +
                         std::to_string(block_index) + " does not exist in the BlockArray for "
                         "technique " + std::to_string(id_.value));
            return false;
        }
        if (device_ == nullptr) return false;

        const vk::DescriptorBufferInfo buffer_info(
            ba.GetBlockArray(block_index), 0, ba.BlockSize());
        vk::WriteDescriptorSet write{};
        write.dstSet = mb.set;
        write.dstBinding = mb.binding_number;
        write.dstArrayElement = block_index;
        write.descriptorCount = 1;
        write.descriptorType = vk::DescriptorType::eStorageBuffer;
        write.pBufferInfo = &buffer_info;
        device_->updateDescriptorSets(write, nullptr);

        mb.block_bound[block_index] = 1;
        return true;
    }
    return false;
}

void BaseTechnique::ValidateNoBindingCollision(std::uint32_t set, std::uint32_t binding) const {
    for (const auto& decl : bindings_) {
        assert(!(decl.set == set && decl.binding == binding) &&
               "Binding collision: set+binding already declared in this technique");
        if (decl.set == set && decl.binding == binding) break;
    }
}

void BaseTechnique::DeclareBindingImpl(BindingDecl decl) {
    if (id_.value == 0) {
        id_ = s_next_technique_id;
        s_next_technique_id = TechniqueId{static_cast<std::uint16_t>(s_next_technique_id.value + 1)};
    }
    bindings_.push_back(std::move(decl));
}

bool BaseTechnique::Compile(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                            ShaderSystem::ShaderManager& shader_mgr,
                            ShaderSystem::PipelineFactory& pipeline_factory,
                            ShaderSystem::ShaderId vert_id,
                            ShaderSystem::ShaderId frag_id,
                            const VulkanEngine::StandardMeshPipeline::PipelineConfig& config,
                            vk::DescriptorSetLayout bindless_layout,
                            vk::DescriptorSetLayout submesh_vertex_layout,
                            vk::DescriptorSetLayout raw_vertex_layout,
                            vk::DescriptorSetLayout indirection_layout,
                             vk::DescriptorSetLayout scene_uniform_layout) {
    const auto& device = bootstrap.GetBackend().GetDevice();
    device_ = &device;  // Valid for the lifetime of the backend; used by EnsureMaterialBlockBound()
    LOGIFACE_LOG(debug, std::format("BaseTechnique: compiling technique (vert={}, frag={})",
                                    vert_id, frag_id));

    // The base variant's retire ring is sized in step 5; any variant created
    // later by EnsureVariant is sized there.

    // ── 1. Build descriptor set layout array ──
    // Engine sets 0-4 are always at layout slots 0-4
    std::vector<vk::DescriptorSetLayout> set_layouts = {
        bindless_layout,           // set 0: bindless textures
        submesh_vertex_layout,     // set 1: submesh vertex data
        raw_vertex_layout,         // set 2: raw vertex buffers
        indirection_layout,        // set 3: indirection data
        scene_uniform_layout,      // set 4: scene uniforms (lighting, camera)
    };

    // ── 2. Validate and group custom bindings (sets >= 5) ──
    // The shared composer enforces the reserved engine sets and rejects
    // duplicates; the shared grouping helper gives a deterministic layout.
    VulkanEngine::Render::PipelineLayoutComposer composer;
    if (auto added = composer.AddAppBindings(bindings_); !added.has_value()) {
        LOGIFACE_LOG(error, "BaseTechnique: invalid descriptor declaration (error " +
                                std::to_string(static_cast<int>(added.error())) + ")");
        return false;
    }
    const auto custom_groups = VulkanEngine::Render::GroupBindingsBySet(bindings_);
    std::vector<vk::raii::DescriptorSetLayout> custom_set_layouts;
    custom_set_layouts.reserve(custom_groups.size());

    for (const auto& group : custom_groups) {
        std::vector<vk::DescriptorSetLayoutBinding> vk_bindings;
        vk_bindings.reserve(group.bindings.size());
        std::vector<vk::DescriptorBindingFlags> vk_binding_flags;
        vk_binding_flags.reserve(group.bindings.size());

        for (const auto& decl : group.bindings) {
            const bool per_material = decl.kind == BindingKind::PerMaterial;

            // Both kinds are StorageBuffer (StructuredBuffer in HLSL). A
            // PerMaterial binding is a BlockArray the shader indexes by
            // material_id / MATERIAL_BLOCK_SIZE, so it must expose the whole
            // descriptor array the material packing can name. Shared bindings
            // are a single buffer.
            vk::DescriptorSetLayoutBinding binding{};
            binding.binding = decl.binding;
            binding.descriptorType = vk::DescriptorType::eStorageBuffer;
            binding.descriptorCount = per_material
                ? TechniquePacking::MATERIAL_BLOCK_COUNT
                : 1u;
            binding.stageFlags = vk::ShaderStageFlagBits::eVertex |
                                 vk::ShaderStageFlagBits::eFragment;
            vk_bindings.push_back(binding);

            // Per-material arrays are partially bound (blocks appear as
            // materials are registered) and updated after bind (registration
            // can happen while frames are in flight).
            vk::DescriptorBindingFlags flags{};
            if (per_material) {
                flags = vk::DescriptorBindingFlagBits::ePartiallyBound |
                        vk::DescriptorBindingFlagBits::eUpdateAfterBind;
            }
            vk_binding_flags.push_back(flags);
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{};
        binding_flags_ci.bindingCount = static_cast<std::uint32_t>(vk_binding_flags.size());
        binding_flags_ci.pBindingFlags = vk_binding_flags.data();

        // The pool backing these sets is update-after-bind, and a set allocated
        // from such a pool must come from a layout created with the matching
        // flag (VUID-VkDescriptorSetAllocateInfo-descriptorPool-00308). Bindings
        // that do not opt into update-after-bind carry a zero flag above.
        vk::DescriptorSetLayoutCreateInfo layout_info{};
        layout_info.pNext = &binding_flags_ci;
        layout_info.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
        layout_info.bindingCount = static_cast<std::uint32_t>(vk_bindings.size());
        layout_info.pBindings = vk_bindings.data();

        custom_set_layouts.emplace_back(device, layout_info);
        set_layouts.push_back(*custom_set_layouts.back());
    }

    // ── 3. Determine push constant ranges ──
    // Camera position (fragment stage, 16 bytes at offset 0).
    std::vector<vk::PushConstantRange> push_constant_ranges;
    push_constant_ranges.emplace_back(
        vk::ShaderStageFlagBits::eFragment, 0, 16);

    // ── 4. Create VkPipelineLayout ──
    vk::PipelineLayoutCreateInfo layout_info{};
    layout_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
    layout_info.pSetLayouts = set_layouts.data();
    layout_info.pushConstantRangeCount = static_cast<std::uint32_t>(push_constant_ranges.size());
    layout_info.pPushConstantRanges = push_constant_ranges.data();

    pipeline_layout_ = vk::raii::PipelineLayout(device, layout_info);
    VulkanBackend::Vulkan::SetVulkanObjectName(device, pipeline_layout_, "base-technique-layout");

    // ── 5. Create the base pipeline (variant 0) via PipelineFactory ──
    {
        pipeline_config_ = config;

        // Fixed-state assembly comes from the shared builder (single copy of
        // the "pull-mode vertices + dynamic viewport/scissor + vertex-stage-
        // only spec constant" knowledge, shared with the scene pipelines).
        const std::array<vk::Format, 1> surface_formats{
            bootstrap.GetBackend().GetSurfaceFormat().format};
        ShaderSystem::GraphicsPipelineTemplate base_tmpl{
            .debug_name = "base-technique",
            .input_assembly = vk::PipelineInputAssemblyStateCreateInfo(
                {}, config.primitive_topology),
            .rasterization = vk::PipelineRasterizationStateCreateInfo(
                {}, false, false, config.polygon_mode, config.cull_mode,
                config.front_face, false, 0, 0, 0, config.line_width),
            .multisample = vk::PipelineMultisampleStateCreateInfo(
                {}, config.sample_count),
            .depth_stencil = vk::PipelineDepthStencilStateCreateInfo(
                {}, config.depth_test_enable, config.depth_write_enable,
                config.depth_compare_op),
            .blend_attachments = {vk::PipelineColorBlendAttachmentState(
                config.blend_enable,
                config.src_color_blend_factor, config.dst_color_blend_factor,
                config.color_blend_op, config.src_alpha_blend_factor,
                config.dst_alpha_blend_factor, config.alpha_blend_op,
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)},
        };
        // Attachment pointer/count are rebound by PipelineFactory from the
        // desc's owned color_blend_attachments vector; do not borrow a member.
        ShaderSystem::GraphicsPipelineDesc base = ShaderSystem::MakeGraphicsDesc(
            base_tmpl, vert_id, frag_id, *pipeline_layout_,
            surface_formats, bootstrap.GetBackend().GetDepthFormat());

        // Vertex-stage draw-mode specialization (constant_id 0). main_indir and
        // depth_indir read it to select the addressing path. Applied to the
        // vertex stage only; the fragment shader does not declare it.
        ApplyDrawModeSpec(base, config.draw_mode);

        auto result = pipeline_factory.CreateGraphics(base, shader_mgr);
        if (!result.has_value()) {
            LOGIFACE_LOG(error, std::format(
                "BaseTechnique {} ({}): pipeline creation failed: {} "
                "(vert={}, frag={}, topo={}, samples={}, color={}, depth={})",
                id_.value, typeid(*this).name(), result.error().message,
                vert_id, frag_id,
                vk::to_string(config.primitive_topology),
                vk::to_string(config.sample_count),
                vk::to_string(bootstrap.GetBackend().GetSurfaceFormat().format),
                vk::to_string(bootstrap.GetBackend().GetDepthFormat())));
            return false;
        }

        variants_.clear();
        TechniqueVariant base_variant{};
        base_variant.draw_key = 0;
        base_variant.vert = vert_id;
        base_variant.frag = frag_id;
        base_variant.desc = std::move(base);
        base_variant.slot.SetFramesInFlight(bootstrap.GetBackend().GetFramesInFlight());
        base_variant.slot.Swap(std::move(result.value()), 0);
        variants_.push_back(std::move(base_variant));

        VulkanBackend::Vulkan::SetVulkanObjectName(device, variants_[0].slot.Get(),
            vk::ObjectType::ePipeline, "technique-pipeline");
        LOGIFACE_LOG(debug, std::format("BaseTechnique: pipeline ready: 0x{:x} (layout 0x{:x})",
                                        HandleToU64(variants_[0].slot.Get()),
                                        HandleToU64(*pipeline_layout_)));
        // Mark compiled only once a usable pipeline exists: a failed creation
        // must not enable the hot-reload PollAndRebuild retry path.
        compiled_ = true;
    }

    // ── 6. Create BlockArrays for PerMaterial bindings ──
    block_arrays_.clear();
    block_arrays_.reserve(bindings_.size());
    for (const auto& decl : bindings_) {
        if (decl.kind == BindingKind::PerMaterial) {
            VulkanEngine::GpuResources::BlockArray ba;
            VulkanEngine::GpuResources::BlockArray::Config ba_cfg{};
            ba_cfg.entry_size = decl.stride;
            ba_cfg.entries_per_block = 256;
            ba_cfg.memory = vk::MemoryPropertyFlagBits::eDeviceLocal;
            ba_cfg.memory_mode = VulkanEngine::GpuResources::MemoryMode::DeviceLocal;
            if (!ba.Initialize(bootstrap.GetBackend(), ba_cfg)) {
                LOGIFACE_LOG(error, "BaseTechnique: Failed to initialize BlockArray for PerMaterial binding");
            }
            block_arrays_.push_back(std::move(ba));
        }
    }

    // ── 7. Create GpuBuffers for Shared bindings ──
    shared_buffers_.clear();
    shared_cpu_data_.clear();
    shared_buffers_.reserve(bindings_.size());
    shared_cpu_data_.reserve(bindings_.size());
    for (const auto& decl : bindings_) {
        if (decl.kind == BindingKind::Shared) {
            // Create device-local storage buffer
            // Shared data is small (typically < 256 bytes), but we allocate a reasonable size
            constexpr std::uint64_t SHARED_BUFFER_SIZE = 256;
            auto buf = VulkanEngine::GpuResources::GpuBuffer::Create(
                bootstrap.GetBackend(),
                SHARED_BUFFER_SIZE,
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            if (!buf.IsValid()) {
                LOGIFACE_LOG(error, "BaseTechnique: Failed to create shared buffer");
            }
            shared_buffers_.push_back(std::move(buf));

            // Allocate technique-local CPU buffer
            shared_cpu_data_.emplace_back(decl.stride > 0 ? decl.stride : 64, std::byte{0});
        }
    }

    // ── 8. Store custom set layouts as members (fixes lifetime bug — were local in Step 2) ──
    custom_set_layouts_ = std::move(custom_set_layouts);

    // ── 9. Create descriptor pool and sets for custom bindings ──
    // Pre-allocate at least 1 block in each PerMaterial BlockArray so block 0 is valid.
    for (auto& ba : block_arrays_) {
        if (!ba.EnsureCapacity(1)) {
            LOGIFACE_LOG(error, "BaseTechnique: could not pre-allocate block 0 for a per-material "
                         "BlockArray; materials for that binding will not be uploadable");
        }
    }

    // A PerMaterial binding needs one descriptor per block (the shader indexes
    // materialBuffer[materialId / MATERIAL_BLOCK_SIZE]); a Shared binding needs one.
    std::uint32_t total_descriptors = 0;
    for (const auto& group : custom_groups) {
        for (const auto& decl : group.bindings) {
            total_descriptors += (decl.kind == BindingKind::PerMaterial)
                ? TechniquePacking::MATERIAL_BLOCK_COUNT
                : 1u;
        }
    }

    material_array_bindings_.clear();

    if (total_descriptors > 0) {
        const vk::DescriptorPoolSize pool_size(
            vk::DescriptorType::eStorageBuffer, total_descriptors);
        vk::DescriptorPoolCreateInfo pool_info{};
        pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                          vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
        pool_info.maxSets = static_cast<std::uint32_t>(custom_groups.size());
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        descriptor_pool_ = vk::raii::DescriptorPool(device, pool_info);

        custom_descriptor_sets_.clear();
        custom_descriptor_sets_.reserve(custom_groups.size());
        custom_descriptor_set_handles_.clear();
        custom_descriptor_set_handles_.reserve(custom_groups.size());

        for (std::size_t gi = 0; gi < custom_groups.size(); ++gi) {
            const vk::DescriptorSetAllocateInfo alloc_info(
                *descriptor_pool_, *custom_set_layouts_[gi]);
            std::vector<vk::raii::DescriptorSet> ds_vector = device.allocateDescriptorSets(alloc_info);
            vk::raii::DescriptorSet ds = std::move(ds_vector[0]);
            const vk::DescriptorSet raw_ds = *ds;

            for (const auto& decl : custom_groups[gi].bindings) {
                // Find the matching declaration in bindings_ by identity (the
                // grouping helper copies declarations, so pointers no longer
                // alias the stored vector).
                const auto binding_it = std::ranges::find_if(bindings_, [&](const BindingDecl& candidate) {
                    return candidate.set == decl.set && candidate.binding == decl.binding;
                });
                if (binding_it == bindings_.end()) {
                    continue;
                }
                const std::size_t binding_idx =
                    static_cast<std::size_t>(std::distance(bindings_.begin(), binding_it));

                if (decl.kind == BindingKind::PerMaterial) {
                    // Per-material arrays are populated lazily as materials are
                    // registered (MaterialManager::Register -> EnsureMaterialBlockBound);
                    // only the bookkeeping is created here. Bind block 0 now so the
                    // fallback material is addressable before the first registration.
                    std::size_t pm_count = 0;
                    for (std::size_t i = 0; i < binding_idx; ++i) {
                        if (bindings_[i].kind == BindingKind::PerMaterial) ++pm_count;
                    }
                    material_array_bindings_.push_back(MaterialArrayBinding{
                        binding_idx, pm_count, raw_ds, decl.binding,
                        std::vector<std::uint8_t>(TechniquePacking::MATERIAL_BLOCK_COUNT, 0)});
                    if (!EnsureMaterialBlockBound(binding_idx, 0)) {
                        LOGIFACE_LOG(error, "BaseTechnique: failed to bind initial material block 0 "
                                     "for technique " + std::to_string(id_.value));
                    }
                } else /* Shared */ {
                    // Count Shared bindings up to binding_idx to find shared_buffers_ index
                    std::size_t sh_count = 0;
                    for (std::size_t i = 0; i < binding_idx; ++i) {
                        if (bindings_[i].kind == BindingKind::Shared) ++sh_count;
                    }
                    const vk::DescriptorBufferInfo buf_info(
                        *shared_buffers_[sh_count].GetBuffer(), 0, vk::WholeSize);
                    const vk::WriteDescriptorSet write(raw_ds, decl.binding, 0, 1,
                        vk::DescriptorType::eStorageBuffer, nullptr, &buf_info);
                    device.updateDescriptorSets(write, nullptr);
                }
            }

            custom_descriptor_set_handles_.push_back(raw_ds);
            custom_descriptor_sets_.push_back(std::move(ds));
        }
    }

    return true;
}

DrawKeyPipelineState DeriveDrawKeyPipelineState(std::uint32_t render_state_key,
                                                vk::CullModeFlags base_cull_mode) {
    using namespace DrawKeyState;
    DrawKeyPipelineState state{};
    state.blend_enable = BlendEnabled(render_state_key);
    state.cull_mode = DoubleSided(render_state_key)
                          ? vk::CullModeFlags(vk::CullModeFlagBits::eNone)
                          : base_cull_mode;
    // Transparent and explicitly no-depth-write keys keep testing but do not
    // write depth.
    state.depth_write = DepthWriteEnabled(render_state_key) && !BlendEnabled(render_state_key);
    return state;
}

ShaderSystem::GraphicsPipelineDesc BaseTechnique::MakeVariantDesc(
    const ShaderSystem::GraphicsPipelineDesc& base, std::uint32_t draw_key,
    const VulkanEngine::StandardMeshPipeline::PipelineConfig& config) const {
    ShaderSystem::GraphicsPipelineDesc desc = base;
    const std::uint32_t render_state_key = DrawKeyState::RenderStateOf(draw_key);
    const std::uint32_t variant_slot = DrawKeyState::VariantOf(draw_key);
    const DrawKeyPipelineState state = DeriveDrawKeyPipelineState(
        render_state_key, config.cull_mode);

    // Interface variant: rebind the shader rows. Each row is a distinct module
    // whose SPIR-V entry point is "main" (Slang emits "main" regardless of the
    // Slang wrapper name), so the entry points stay "main"; the desc's
    // entry-point fields exist for hand-authored modules and are not changed
    // here.
    ShaderSystem::ShaderId v_id{};
    ShaderSystem::ShaderId f_id{};
    if (GetVariantShaders(variant_slot, v_id, f_id)) {
        desc.vertex_shader = v_id;
        desc.fragment_shader = f_id;
    }

    // Color blend: the attachment's enable and factors come from the config for
    // the base variant; a blending key overrides them with standard
    // src-alpha / one-minus-src-alpha blending.
    vk::PipelineColorBlendAttachmentState attachment = config.blend_enable
        ? vk::PipelineColorBlendAttachmentState(
              true, config.src_color_blend_factor, config.dst_color_blend_factor,
              config.color_blend_op, config.src_alpha_blend_factor,
              config.dst_alpha_blend_factor, config.alpha_blend_op,
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        : vk::PipelineColorBlendAttachmentState(
              false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
              vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    if (state.blend_enable) {
        attachment.blendEnable = true;
        attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        attachment.colorBlendOp = vk::BlendOp::eAdd;
        attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        attachment.alphaBlendOp = vk::BlendOp::eAdd;
    }
    desc.color_blend_attachments = {attachment};

    desc.rasterization.cullMode = state.cull_mode;
    desc.depth_stencil.depthWriteEnable = state.depth_write;
    return desc;
}

bool BaseTechnique::EnsureVariant(std::uint32_t draw_key,
                                  ShaderSystem::ShaderManager& shaders,
                                  ShaderSystem::PipelineFactory& factory) {
    if (!compiled_) return false;
    if (HasVariant(draw_key)) return true;
    if (variants_.empty()) return false;

    // The interface variant must be one this technique owns; render state is
    // always buildable. Unknown interface slots fall back to the base pipeline.
    const std::uint32_t variant_slot = DrawKeyState::VariantOf(draw_key);
    if (variant_slot >= VariantCount()) return false;

    TechniqueVariant variant{};
    variant.draw_key = draw_key;
    if (!GetVariantShaders(variant_slot, variant.vert, variant.frag)) {
        return false;
    }
    variant.desc = MakeVariantDesc(variants_[0].desc, draw_key, pipeline_config_);
    variant.slot.SetFramesInFlight(variants_[0].slot.FramesInFlight());

    auto result = factory.CreateGraphics(variant.desc, shaders);
    if (!result.has_value()) {
        LOGIFACE_LOG(error, std::format(
            "BaseTechnique {}: variant 0x{:x} pipeline creation failed: {}",
            id_.value, draw_key, result.error().message));
        return false;
    }
    variant.slot.Swap(std::move(result.value()), 0);
    LOGIFACE_LOG(debug, std::format("BaseTechnique {}: materialized variant 0x{:x} (uv slot {})",
                                    id_.value, draw_key, variant_slot));
    variants_.push_back(std::move(variant));
    return true;
}

void BaseTechnique::PollAndRebuild(ShaderSystem::ShaderManager& shaders,
                                   ShaderSystem::PipelineFactory& factory,
                                   std::uint32_t frame_index) {
    if (!compiled_) return;
    // Each variant owns its slot, so a shader change rebuilds every variant
    // that references the changed stage without disturbing the others.
    for (auto& variant : variants_) {
        variant.slot.RetireFrame(frame_index);
        variant.slot.PollAndRebuild(shaders, variant.vert, variant.frag,
            [&factory, &variant](ShaderSystem::ShaderManager& s)
                -> std::optional<ShaderSystem::PipelineProduct> {
                auto result = factory.CreateGraphics(variant.desc, s);
                return result
                    ? std::optional<ShaderSystem::PipelineProduct>(std::move(*result))
                    : std::nullopt;
            }, frame_index);
    }
}

bool BaseTechnique::RebuildForDrawMode(ShaderSystem::ShaderManager& shaders,
                                       ShaderSystem::PipelineFactory& factory,
                                       std::uint32_t draw_mode,
                                       std::uint32_t frame_index) {
    if (!compiled_) return true;
    bool ok = true;
    for (auto& variant : variants_) {
        variant.slot.RetireFrame(frame_index);
        ApplyDrawModeSpec(variant.desc, draw_mode);
        auto result = factory.CreateGraphics(variant.desc, shaders);
        if (!result.has_value()) {
            LOGIFACE_LOG(error, std::format(
                "BaseTechnique {}: draw-mode pipeline rebuild failed for variant 0x{:x}: {}",
                id_.value, variant.draw_key, result.error().message));
            ok = false;
            continue;
        }
        variant.slot.Swap(std::move(result.value()), frame_index);
    }
    if (ok) {
        LOGIFACE_LOG(debug, std::format("BaseTechnique {}: re-specialized {} variant(s) for draw mode {}",
                                        id_.value, variants_.size(), draw_mode));
    }
    return ok;
}

} // namespace VulkanEngine::TechniqueManager
