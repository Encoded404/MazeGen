module;

#include <logging/logging_macros.hpp>

module VulkanEngine.RenderPipeline;

import std;

import logiface;

import vulkan;

import VulkanEngine.RenderGraph;
import VulkanBackend.Vulkan.GpuPassTimer;
import VulkanBackend.Vulkan.RenderGraphExecutor;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.PipelineCompiler;
import VulkanEngine.PipelineFactory;

namespace VulkanEngine::RenderPipeline {

namespace {

vk::ImageAspectFlags FormatToAspectFlags(vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eD32Sfloat:
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return vk::ImageAspectFlagBits::eDepth;
        default:
            return vk::ImageAspectFlagBits::eColor;
    }
}

} // namespace

RenderPipeline::RenderPipeline() = default;
RenderPipeline::~RenderPipeline() = default;

void RenderPipeline::Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                ShaderSystem::ShaderManager* shader_manager,
                                ShaderSystem::PipelineFactory* pipeline_factory) {
    bootstrap_ = &bootstrap;
    shader_manager_ = shader_manager;
    pipeline_factory_ = pipeline_factory;
    initialized_ = true;

    auto& backend = bootstrap.GetBackend();
    const auto surface_format = static_cast<vk::Format>(backend.GetSurfaceFormat().format);
    const auto depth_format = backend.GetDepthFormat();

    RegisterResourceResolver("swapchain-backbuffer",
        [&backend](std::uint32_t img_idx) { return backend.GetSwapchainImages()[img_idx]; },
        [&backend](std::uint32_t img_idx) { return *backend.GetSwapchainImageViews()[img_idx]; },
        surface_format);
    RegisterResourceResolver("depth-buffer",
        [&backend](std::uint32_t img_idx) { return *backend.GetDepthImage(img_idx); },
        [&backend](std::uint32_t img_idx) { return *backend.GetDepthImageView(img_idx); },
        depth_format);
}

void RenderPipeline::SetEngineDescriptorSetLayouts(std::array<vk::DescriptorSetLayout, 5> layouts) {
    if (engine_set_layouts_ == layouts) {
        return;
    }
    engine_set_layouts_ = layouts;
    // Built-in passes register (and the first Compile() runs) inside
    // Renderer::Initialize(), which its caller can only follow with this call:
    // every pass that declared a pipeline before the layouts existed was deferred
    // by BuildPassPipelines(), and nothing re-runs it until the pass model changes.
    // A deferred pass executes with a null pipeline and draws nothing, so build
    // what the new layouts unblocked here instead of waiting for a rebuild that a
    // frame that only re-records never triggers.
    BuildPassPipelines();
}

void RenderPipeline::Shutdown() {
    transient_image_descs_.clear();
    transient_buffer_descs_.clear();
    transient_allocator_.Shutdown();
    pass_pipelines_.clear();
    pass_pipeline_by_name_.clear();
    bootstrap_ = nullptr;
    shader_manager_ = nullptr;
    pipeline_factory_ = nullptr;
    initialized_ = false;
    compiled_ = false;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::ImportBackbuffer() {
    backbuffer_handle_ = graph_builder_.ImportResource("swapchain-backbuffer", VulkanEngine::RenderGraph::ResourceKind::Image);
    TrackImportedResource(backbuffer_handle_, "swapchain-backbuffer", VulkanEngine::RenderGraph::ResourceKind::Image);
    return backbuffer_handle_;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::ImportDepthBuffer() {
    depth_buffer_handle_ = graph_builder_.ImportResource("depth-buffer", VulkanEngine::RenderGraph::ResourceKind::Image);
    TrackImportedResource(depth_buffer_handle_, "depth-buffer", VulkanEngine::RenderGraph::ResourceKind::Image);
    return depth_buffer_handle_;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::ImportImage(const std::string& name) {
    const auto handle = graph_builder_.ImportResource(name, VulkanEngine::RenderGraph::ResourceKind::Image);
    TrackImportedResource(handle, name, VulkanEngine::RenderGraph::ResourceKind::Image);
    return handle;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::ImportBuffer(const std::string& name) {
    const auto handle = graph_builder_.ImportResource(name, VulkanEngine::RenderGraph::ResourceKind::Buffer);
    TrackImportedResource(handle, name, VulkanEngine::RenderGraph::ResourceKind::Buffer);
    return handle;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::CreateTransientImage(const TransientImageDesc& desc) {
    auto handle = graph_builder_.CreateTransientResource(desc.name, VulkanEngine::RenderGraph::ResourceKind::Image);
    resource_names_[handle.index] = desc.name;

    VulkanEngine::RenderGraph::TransientImageInfo info{};
    info.format = desc.format;
    const auto [image_width, image_height] =
        VulkanEngine::PipelinePass::ResolveTransientExtent(desc, render_width_, render_height_);
    info.width = image_width;
    info.height = image_height;
    info.mip_levels = 1;
    info.array_layers = 1;
    info.sample_count = vk::SampleCountFlagBits::e1;
    info.usage = desc.usage;
    info.tiling = vk::ImageTiling::eOptimal;
    info.aliasable = desc.aliasable;

    graph_builder_.SetTransientImageInfo(handle, info);

    // ContentsUndefined contract: an aliasable image must start Undefined, so a
    // requested non-Undefined initial layout is dropped rather than aliased.
    if (!desc.aliasable && desc.initial_layout != vk::ImageLayout::eUndefined) {
        // Both layout paths share one mapping (ToImageLayoutIntent), so the
        // initial and final chains cannot drift apart. An unrepresentable layout
        // is reported rather than silently becoming Undefined.
        const auto initial_layout = VulkanEngine::RenderGraph::ToImageLayoutIntent(desc.initial_layout);
        if (initial_layout == VulkanEngine::RenderGraph::ImageLayoutIntent::Undefined) {
            LOGIFACE_LOG(warn, "RenderPipeline: transient image '" + desc.name +
                                   "' requested an initial layout with no graph intent; using Undefined");
        }
        auto initial_state = VulkanEngine::RenderGraph::ResourceState::ImageState(
            VulkanEngine::RenderGraph::PipelineStageIntent::TopOfPipe,
            VulkanEngine::RenderGraph::AccessIntent::None,
            VulkanEngine::RenderGraph::QueueType::Graphics,
            initial_layout);
        graph_builder_.SetInitialState(handle, initial_state);
    }

    if (desc.final_layout != vk::ImageLayout::eUndefined) {
        const auto final_layout = VulkanEngine::RenderGraph::ToImageLayoutIntent(desc.final_layout);
        if (final_layout == VulkanEngine::RenderGraph::ImageLayoutIntent::Undefined) {
            LOGIFACE_LOG(warn, "RenderPipeline: transient image '" + desc.name +
                                   "' requested a final layout with no graph intent; using Undefined");
        }
        auto final_state = VulkanEngine::RenderGraph::ResourceState::ImageState(
            VulkanEngine::RenderGraph::PipelineStageIntent::BottomOfPipe,
            VulkanEngine::RenderGraph::AccessIntent::None,
            VulkanEngine::RenderGraph::QueueType::Graphics,
            final_layout);
        graph_builder_.SetFinalState(handle, final_state);
    }

    const std::uint32_t res_index = handle.index;
    transient_image_descs_[res_index] = desc;

    return handle;
}

VulkanEngine::RenderGraph::ResourceHandle RenderPipeline::CreateTransientBuffer(const TransientBufferDesc& desc) {
    auto handle = graph_builder_.CreateTransientResource(desc.name, VulkanEngine::RenderGraph::ResourceKind::Buffer);
    resource_names_[handle.index] = desc.name;

    VulkanEngine::RenderGraph::TransientBufferInfo info{};
    info.name = desc.name;
    info.size = desc.size;
    info.usage = desc.usage;
    info.memory_properties = desc.memory_properties;
    info.aliasable = desc.aliasable;

    graph_builder_.SetTransientBufferInfo(handle, info);

    const std::uint32_t res_index = handle.index;
    transient_buffer_descs_[res_index] = desc;

    return handle;
}

void RenderPipeline::RegisterResourceResolver(const std::string& name,
                                               ImageResolver resolve_image,
                                               ImageViewResolver resolve_image_view,
                                               vk::Format format) {
    resource_resolvers_[name] = ExternalResourceResolver{
        .resolve_image = std::move(resolve_image),
        .resolve_image_view = std::move(resolve_image_view),
        .format = format
    };
    image_resolver_names_.insert(name);
}

void RenderPipeline::RegisterBufferResolver(const std::string& name, BufferResolver resolve_buffer) {
    buffer_resolvers_[name] = std::move(resolve_buffer);
    buffer_resolver_names_.insert(name);
}

VulkanEngine::RenderGraph::PassHandle RenderPipeline::AddModelPass(ModelPass model) {
    const std::uint32_t slot = static_cast<std::uint32_t>(model_passes_.size());
    model.slot = slot;
    model.generation = 1;

    if (model.pipeline_request.IsDeclared() || !model.declared_bindings.empty()) {
        PassPipelineState state{};
        state.name = model.name;
        state.request = model.pipeline_request;
        state.bindings = model.declared_bindings;
        state.assignments = model.binding_assignments;
        state.attachments = model.attachments;
        pass_pipeline_by_name_[state.name] = slot;
        pass_pipelines_[slot] = std::move(state);
    }

    model_passes_.push_back(std::move(model));
    dirty_ = true;
    return VulkanEngine::RenderGraph::PassHandle{.index = slot, .generation = 1};
}

bool RenderPipeline::IsValidModelHandle(VulkanEngine::RenderGraph::PassHandle handle) const {
    return handle.IsValid() && handle.index < model_passes_.size() &&
           model_passes_[handle.index].generation == handle.generation &&
           model_passes_[handle.index].alive;
}

RenderPipeline::ModelPass* RenderPipeline::FindModelPass(VulkanEngine::RenderGraph::PassHandle handle) {
    if (!IsValidModelHandle(handle)) {
        return nullptr;
    }
    return &model_passes_[handle.index];
}

std::expected<VulkanEngine::RenderGraph::PassHandle, PassError> RenderPipeline::RegisterPass(
    std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass> pass) {
    if (!pass) {
        return std::unexpected(PassError{PassErrorCode::InvalidDeclaration, "null pass", {}});
    }
    if (executing_) {
        return std::unexpected(PassError{
            PassErrorCode::InvalidDeclaration,
            "passes cannot be registered while the frame is executing",
            std::string(pass->GetName())});
    }

    const std::string name = std::string(pass->GetName());
    if (name.empty()) {
        return std::unexpected(PassError{PassErrorCode::InvalidDeclaration, "pass has no name", {}});
    }
    const bool duplicate = std::ranges::any_of(model_passes_, [&](const ModelPass& model) {
        return model.alive && model.name == name;
    });
    if (duplicate) {
        return std::unexpected(PassError{PassErrorCode::DuplicateName, "duplicate pass name", name});
    }

    // Setup() creates graph resources directly (transients, imports), so snapshot
    // the resource table first: a registration rejected after Setup() rolls back
    // everything it created instead of leaving orphaned allocations behind.
    const std::size_t resource_checkpoint = graph_builder_.ResourceCount();
    const auto fail = [&](PassError error)
        -> std::expected<VulkanEngine::RenderGraph::PassHandle, PassError> {
        RollbackPassSetupResources(resource_checkpoint);
        return std::unexpected(std::move(error));
    };

    // The pipeline constructs the context (with the current extent) and calls
    // Setup() itself, so the app never has to.
    VulkanEngine::PipelinePass::PassSetupContext ctx(*this, render_width_, render_height_);
    if (!pass->Validate()) {
        return fail(PassError{PassErrorCode::ValidationFailed, "pass Validate() returned false", name});
    }
    pass->Setup(ctx);

    ModelPass model{};
    model.name = name;
    model.queue = ctx.GetQueueType();
    if (model.queue != VulkanEngine::RenderGraph::QueueType::Graphics && !async_compute_available_) {
        return fail(PassError{
            PassErrorCode::ValidationFailed,
            "pass requests a non-graphics queue but no async compute queue is available",
            name});
    }
    const std::uint32_t push_constant_size = ctx.GetPushConstantSize();
    const vk::ShaderStageFlags push_constant_stages = ctx.GetPushConstantStages();
    model.execute = [this, raw = pass.get(), captured_name = name,
                     push_constant_size, push_constant_stages](const void* user_data,
                                                               vk::CommandBuffer cmd) {
        const auto* frame_data = static_cast<const VulkanEngine::PipelinePass::RenderFrameData*>(user_data);
        if (frame_data == nullptr) {
            return;
        }
        VulkanEngine::PipelinePass::FrameContext frame = frame_data->frame;
        frame.declared_push_constant_size = push_constant_size;
        frame.declared_push_constant_stages = push_constant_stages;
        if (frame.pipeline_layout == nullptr) {
            frame.pipeline_layout = GetPassPipelineLayoutByName(captured_name);
        }
        const auto pipeline_it = pass_pipeline_by_name_.find(captured_name);
        if (pipeline_it != pass_pipeline_by_name_.end()) {
            RewirePassDescriptors(pipeline_it->second, frame);
        }
        raw->Execute(frame, cmd);
    };
    model.writes = ctx.GetWriteResources();
    model.attachments = ctx.GetAttachmentSetup();
    model.pipeline_request = ctx.GetPipelineRequest();
    model.pipeline_request.push_constant_size = ctx.GetPushConstantSize();
    model.pipeline_request.push_constant_stages = ctx.GetPushConstantStages();
    model.declared_bindings = ctx.GetDeclaredBindings();
    model.binding_assignments = ctx.GetBindingAssignments();
    const auto& register_reads = ctx.GetReadResources();
    const auto& register_stages = ctx.GetReadStages();
    const auto& register_accesses = ctx.GetReadAccesses();
    for (std::size_t i = 0; i < register_reads.size(); ++i) {
        model.reads.push_back(ModelRead{
            register_reads[i],
            i < register_stages.size() ? register_stages[i]
                                       : VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader,
            i < register_accesses.size() ? register_accesses[i]
                                         : VulkanEngine::RenderGraph::AccessIntent::Read,
        });
    }
    // A pass on a non-graphics queue is recorded into a command buffer on that
    // queue: dynamic rendering and graphics pipelines are not valid there, and
    // imported (engine-owned) resources are exclusive to the graphics family,
    // so only transients may be touched.
    if (model.queue != VulkanEngine::RenderGraph::QueueType::Graphics) {
        if (model.attachments.has_value() &&
            (!model.attachments->color_attachments.empty() ||
             model.attachments->depth_attachment.has_value())) {
            return fail(PassError{PassErrorCode::ValidationFailed,
                                  "pass on a non-graphics queue cannot declare render attachments",
                                  name});
        }
        if (model.pipeline_request.kind == VulkanEngine::PipelinePass::PassPipelineKind::Graphics) {
            return fail(PassError{PassErrorCode::ValidationFailed,
                                  "pass on a non-graphics queue cannot use a graphics pipeline",
                                  name});
        }
        const auto check_queue_resource = [&](VulkanEngine::RenderGraph::ResourceHandle resource)
            -> std::optional<PassError> {
            if (!IsImportedResource(resource)) {
                return std::nullopt;
            }
            const auto name_it = resource_names_.find(resource.index);
            return PassError{PassErrorCode::InvalidDeclaration,
                             "pass on a non-graphics queue cannot access imported resource '" +
                                 (name_it != resource_names_.end() ? name_it->second
                                                                   : std::string{"?"}) +
                                 "'; use a transient",
                             name};
        };
        for (const auto& read : model.reads) {
            if (auto error = check_queue_resource(read.resource); error.has_value()) {
                return fail(*error);
            }
        }
        for (const auto write : model.writes) {
            if (auto error = check_queue_resource(write); error.has_value()) {
                return fail(*error);
            }
        }
        for (const auto& assignment : model.binding_assignments) {
            if (auto error = check_queue_resource(assignment.resource); error.has_value()) {
                return fail(*error);
            }
        }
    }
    // Declared resources must be known, and imported images must be resolvable.
    // Imported buffers without a resolver are allowed: engine logical buffers
    // (e.g. "scene-buffers") are hazard-only and carry no Vulkan handle.
    const auto validate_resource = [&](VulkanEngine::RenderGraph::ResourceHandle resource)
        -> std::optional<PassError> {
        if (!resource.IsValid() || !resource_names_.contains(resource.index)) {
            return PassError{PassErrorCode::InvalidDeclaration, "invalid resource handle", name};
        }
        if (imported_resource_indices_.contains(resource.index)) {
            const auto kind_it = imported_resource_kinds_.find(resource.index);
            const bool is_image = kind_it != imported_resource_kinds_.end() &&
                                  kind_it->second == VulkanEngine::RenderGraph::ResourceKind::Image;
            if (is_image && !image_resolver_names_.contains(resource_names_[resource.index])) {
                return PassError{PassErrorCode::MissingResolver,
                                 "imported image '" + resource_names_[resource.index] +
                                     "' has no resolver",
                                 name};
            }
        }
        return std::nullopt;
    };
    for (const auto& read : model.reads) {
        if (auto error = validate_resource(read.resource); error.has_value()) {
            return fail(*error);
        }
    }
    for (const auto write : model.writes) {
        if (auto error = validate_resource(write); error.has_value()) {
            return fail(*error);
        }
    }
    // Every descriptor assignment must target a declared app binding and a
    // valid resource.
    for (const auto& assignment : model.binding_assignments) {
        if (assignment.set < VulkanEngine::Render::kFirstAppDescriptorSet) {
            return fail(PassError{
                PassErrorCode::InvalidDeclaration,
                "descriptor binding targets a reserved engine set", name});
        }
        const bool declared = std::ranges::any_of(
            model.declared_bindings, [&](const VulkanEngine::Render::DescriptorDecl& decl) {
                return decl.set == assignment.set && decl.binding == assignment.binding;
            });
        if (!declared) {
            return fail(PassError{
                PassErrorCode::InvalidDeclaration,
                "BindResource references an undeclared descriptor binding", name});
        }
        if (auto error = validate_resource(assignment.resource); error.has_value()) {
            return fail(*error);
        }
    }
    model.pass = std::move(pass);

    const auto handle = AddModelPass(std::move(model));

    // Application ordering around built-in anchors.
    for (const auto before : ctx.GetBeforeBuiltinPasses()) {
        const auto index = static_cast<std::size_t>(before);
        if (index < builtin_handles_.size() && builtin_handles_[index].IsValid()) {
            AddDependency(handle, builtin_handles_[index]);
        }
    }
    for (const auto after : ctx.GetAfterBuiltinPasses()) {
        const auto index = static_cast<std::size_t>(after);
        if (index < builtin_handles_.size() && builtin_handles_[index].IsValid()) {
            AddDependency(builtin_handles_[index], handle);
        }
    }

    return handle;
}

bool RenderPipeline::RemovePass(VulkanEngine::RenderGraph::PassHandle handle) {
    if (executing_) {
        return false;
    }
    ModelPass* model = FindModelPass(handle);
    if (model == nullptr) {
        return false;
    }
    model->alive = false;
    model->enabled = false;
    if (model->pass) {
        pending_removals_.emplace_back(std::move(model->pass), last_fif_slot_);
    }
    dirty_ = true;
    return true;
}

bool RenderPipeline::SetPassEnabled(VulkanEngine::RenderGraph::PassHandle handle, bool enabled) {
    if (executing_) {
        return false;
    }
    ModelPass* model = FindModelPass(handle);
    if (model == nullptr) {
        return false;
    }
    model->enabled = enabled;
    dirty_ = true;
    return true;
}

void RenderPipeline::RequestRebuild() {
    dirty_ = true;
}

void RenderPipeline::SetFramesInFlight(std::uint32_t frames_in_flight) {
    frames_in_flight_ = std::max(1u, frames_in_flight);
}

void RenderPipeline::SetQueueFamilies(std::span<const std::uint32_t> families) {
    transient_allocator_.SetQueueFamilies(families);
}

void RenderPipeline::SetRenderExtent(std::uint32_t width, std::uint32_t height) {
    if (width == render_width_ && height == render_height_) {
        return;
    }
    render_width_ = width;
    render_height_ = height;

    // Reallocate size-dependent (relative) transients against the new extent.
    // The compiled plan's resource identities do not change, so no rebuild is
    // needed; ResolveResources() picks up the new handles on the next frame.
    if (compiled_) {
        SyncTransients();
    }

    // Queue the app-owned-resource callback; drained once by ApplyChanges().
    resize_pending_ = true;
    resize_width_ = width;
    resize_height_ = height;
}

void RenderPipeline::OnSwapchainRecreated(std::uint32_t image_count) {
    if (image_count == 0) {
        image_count = 1;
    }
    // Every swapchain image (and its depth image/view) is new, so any recorded
    // end-of-frame layout is meaningless: force the next use to start from
    // Undefined. Sizes are re-established from the newly compiled graph.
    tracked_states_.assign(image_count,
                           std::vector<VulkanEngine::RenderGraph::ResourceState>(tracked_resource_count_));
    tracked_valid_.assign(image_count, std::vector<bool>(tracked_resource_count_, false));
}

const std::vector<VulkanEngine::RenderGraph::CompileDiagnostic>& RenderPipeline::GetDiagnostics() const {
    return compiled_graph_.diagnostics;
}

const std::vector<PassError>& RenderPipeline::GetValidationErrors() const {
    return validation_errors_;
}

void RenderPipeline::TrackImportedResource(VulkanEngine::RenderGraph::ResourceHandle handle,
                                           const std::string& name,
                                           VulkanEngine::RenderGraph::ResourceKind kind) {
    resource_names_[handle.index] = name;
    imported_resource_indices_.insert(handle.index);
    imported_resource_kinds_[handle.index] = kind;
}

bool RenderPipeline::IsImportedResource(VulkanEngine::RenderGraph::ResourceHandle handle) const {
    return handle.IsValid() && imported_resource_indices_.contains(handle.index);
}

void RenderPipeline::RollbackPassSetupResources(std::size_t resource_count) {
    graph_builder_.RollbackResources(resource_count);

    const auto erase_indexed = [resource_count](auto& map) {
        std::erase_if(map, [resource_count](const auto& entry) {
            return entry.first >= resource_count;
        });
    };
    erase_indexed(resource_names_);
    erase_indexed(transient_image_descs_);
    erase_indexed(transient_buffer_descs_);
    erase_indexed(imported_resource_kinds_);
    std::erase_if(imported_resource_indices_, [resource_count](std::uint32_t index) {
        return index >= resource_count;
    });
}

void RenderPipeline::ApplyChanges() {
    // Drain a queued resize at a frame boundary, before the frame is recorded.
    // This is not a graph mutation, so it does not dirty the model.
    if (resize_pending_) {
        resize_pending_ = false;
        for (auto& model : model_passes_) {
            if (model.alive && model.pass) {
                model.pass->OnRenderResize(resize_width_, resize_height_);
            }
        }
    }
    if (!dirty_ || applying_) {
        return;
    }
    RebuildFromModel();
}

void RenderPipeline::RebuildFromModel() {
    applying_ = true;
    graph_builder_.ResetPasses();

    std::vector<VulkanEngine::RenderGraph::PassHandle> builder_handles(model_passes_.size());
    std::vector<bool> present(model_passes_.size(), false);
    for (std::size_t i = 0; i < model_passes_.size(); ++i) {
        ModelPass& model = model_passes_[i];
        if (!model.alive || !model.enabled) {
            continue;
        }
        VulkanEngine::RenderGraph::PassExecutionCallback callback{};
        callback.callback = model.execute;
        const auto builder_handle = graph_builder_.AddPass(model.name, model.queue, true, callback);
        builder_handles[i] = builder_handle;
        present[i] = true;
        for (const auto& read : model.reads) {
            (void)graph_builder_.AddRead(builder_handle, read.resource, read.stage, read.access);
        }
        for (const auto write : model.writes) {
            (void)graph_builder_.AddWrite(builder_handle, write);
        }
        if (model.attachments) {
            (void)graph_builder_.SetPassAttachments(builder_handle, *model.attachments);
        }
    }

    for (const auto& [before_slot, after_slot] : model_dependencies_) {
        if (before_slot < present.size() && after_slot < present.size() && present[before_slot] &&
            present[after_slot]) {
            (void)graph_builder_.AddDependency(builder_handles[before_slot], builder_handles[after_slot]);
        }
    }
    applying_ = false;
    dirty_ = false;
    ++revision_;

    std::vector<PassError> errors;
    (void)ValidateModel(errors);

    compiled_graph_ = graph_builder_.Compile();
    compiled_ = compiled_graph_.success;

    // Partition the ordered passes into queue runs once, at compile time: the
    // plan is a pure function of the compiled graph. Reject (do not truncate) a
    // graph that needs more run slots than the device budgets, because the run
    // slots index per-run command buffers and semaphores.
    queue_runs_ = compiled_ ? VulkanEngine::RenderGraph::BuildQueueRuns(compiled_graph_)
                            : VulkanEngine::RenderGraph::QueueRunPlan{};
    if (compiled_ && queue_runs_.ExceedsLimit()) {
        errors.push_back(PassError{
            PassErrorCode::CompileFailed,
            "graph requires " + std::to_string(queue_runs_.runs.size()) +
                " queue runs; the device supports at most " +
                std::to_string(VulkanEngine::RenderGraph::kMaxQueueRuns),
            {}});
        compiled_ = false;
        queue_runs_ = {};
    }

    validation_errors_ = errors;
    for (const auto& error : errors) {
        LOGIFACE_LOG(error, "RenderPipeline: pass '" + error.pass + "': " + error.message);
    }
    if (!compiled_) {
        for (const auto& diagnostic : compiled_graph_.diagnostics) {
            LOGIFACE_LOG(error, "RenderPipeline: graph compile: " + diagnostic.message);
        }
    }

    if (compiled_) {
        SyncTransients();
        const std::uint32_t frames_in_flight =
            bootstrap_ ? std::max<std::uint32_t>(bootstrap_->GetBackend().GetFramesInFlight(), 1) : 1;
        for (auto& [index, state] : pass_pipelines_) {
            (void)index;
            state.slot.SetFramesInFlight(frames_in_flight);
        }
        BuildPassPipelines();

        const std::uint32_t image_count =
            bootstrap_ ? std::max<std::uint32_t>(bootstrap_->GetSnapshot().swapchain_image_count, 1) : 1;
        tracked_resource_count_ = static_cast<std::uint32_t>(compiled_graph_.resource_info.size());
        tracked_states_.assign(image_count,
                               std::vector<VulkanEngine::RenderGraph::ResourceState>(tracked_resource_count_));
        tracked_valid_.assign(image_count, std::vector<bool>(tracked_resource_count_, false));

        for (std::size_t i = 0; i < compiled_graph_.resource_lifetimes.size(); ++i) {
            const auto& lifetime = compiled_graph_.resource_lifetimes[i];
            if (lifetime.name == "swapchain-backbuffer") {
                backbuffer_resource_index_ = lifetime.handle.index;
            } else if (lifetime.name == "depth-buffer") {
                depth_buffer_resource_index_ = lifetime.handle.index;
            }
        }
    }
}

bool RenderPipeline::ValidateModel(std::vector<PassError>& errors) const {
    std::unordered_set<std::string> names;
    for (const auto& model : model_passes_) {
        if (!model.alive || !model.enabled) {
            continue;
        }
        if (!names.insert(model.name).second) {
            errors.push_back(PassError{PassErrorCode::DuplicateName, "duplicate pass name", model.name});
        }
        // Every imported image a pass touches must have a resolver; imported
        // buffers may be hazard-only and carry no Vulkan handle.
        const auto check_import = [&](VulkanEngine::RenderGraph::ResourceHandle resource) {
            if (!resource.IsValid() || resource.index >= compiled_graph_.resource_info.size()) {
                return;
            }
            const auto& info = compiled_graph_.resource_info[resource.index];
            if (!info.imported || info.kind != VulkanEngine::RenderGraph::ResourceKind::Image) {
                return;
            }
            if (!image_resolver_names_.contains(info.name)) {
                errors.push_back(PassError{PassErrorCode::MissingResolver,
                                           "imported image '" + info.name + "' has no resolver",
                                           model.name});
            }
        };
        for (const auto& read : model.reads) {
            check_import(read.resource);
        }
        for (const auto write : model.writes) {
            check_import(write);
        }
    }
    return errors.empty();
}

void RenderPipeline::CollectRemovedPasses(std::uint32_t fif_slot) {
    if (!bootstrap_) {
        return;
    }
    auto& backend = bootstrap_->GetBackend();
    for (auto it = pending_removals_.begin(); it != pending_removals_.end();) {
        const std::uint32_t retire_frame = it->second;
        if (fif_slot != retire_frame && backend.IsFrameComplete(retire_frame)) {
            it = pending_removals_.erase(it);
        } else {
            ++it;
        }
    }
}

void RenderPipeline::Compile() {
    ApplyChanges();
}

void RenderPipeline::BuildPassPipelines() {
    if (bootstrap_ == nullptr || shader_manager_ == nullptr || pipeline_factory_ == nullptr) {
        return;
    }
    auto& device = bootstrap_->GetBackend().GetDevice();

    // All pass pipelines compile in one batched submission at the end of this
    // function; the per-pass sections below only build layouts and descs.
    ShaderSystem::PipelineFactoryExecutor executor{*pipeline_factory_};
    ShaderSystem::PipelineCompiler compiler{executor};

    for (auto& [index, state] : pass_pipelines_) {
        (void)index;
        if (!state.request.IsDeclared() || state.built) {
            continue;
        }
        if (std::ranges::any_of(engine_set_layouts_, [](vk::DescriptorSetLayout layout) {
                return layout == nullptr;
            })) {
            // Engine set layouts were not supplied; defer until they are.
            LOGIFACE_LOG(debug, "RenderPipeline: pass pipeline '" + state.name +
                                    "' deferred until engine set layouts are set");
            return;
        }

        // App descriptor sets (>= 5) composed deterministically.
        VulkanEngine::Render::PipelineLayoutComposer composer;
        if (const auto added = composer.AddAppBindings(state.bindings); !added.has_value()) {
            LOGIFACE_LOG(error, "RenderPipeline: pass '" + state.name +
                                    "' has invalid descriptor declarations");
            continue;
        }
        for (const auto& group : VulkanEngine::Render::GroupBindingsBySet(state.bindings)) {
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            std::vector<vk::DescriptorBindingFlags> flags;
            bindings.reserve(group.bindings.size());
            flags.reserve(group.bindings.size());
            for (const auto& decl : group.bindings) {
                vk::DescriptorSetLayoutBinding binding{};
                binding.binding = decl.binding;
                binding.descriptorType = decl.descriptor_type;
                binding.descriptorCount = decl.count;
                binding.stageFlags = decl.stage_flags;
                bindings.push_back(binding);
                flags.push_back(decl.binding_flags);
            }
            vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info{};
            flags_info.bindingCount = static_cast<std::uint32_t>(flags.size());
            flags_info.pBindingFlags = flags.data();
            vk::DescriptorSetLayoutCreateInfo layout_info{};
            layout_info.pNext = &flags_info;
            layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layout_info.pBindings = bindings.data();
            state.app_set_layouts.emplace_back(device, layout_info);
            state.app_set_numbers.push_back(group.set);
        }

        std::vector<vk::DescriptorSetLayout> set_layouts(engine_set_layouts_.begin(),
                                                         engine_set_layouts_.end());
        for (auto& layout : state.app_set_layouts) {
            set_layouts.push_back(*layout);
        }

        std::vector<vk::PushConstantRange> push_ranges;
        if (state.request.push_constant_size > 0 &&
            state.request.push_constant_stages != vk::ShaderStageFlags{}) {
            push_ranges.push_back(vk::PushConstantRange{
                state.request.push_constant_stages, 0, state.request.push_constant_size});
        }

        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = static_cast<std::uint32_t>(push_ranges.size());
        pipeline_layout_info.pPushConstantRanges = push_ranges.data();
        state.layout = std::make_unique<vk::raii::PipelineLayout>(device, pipeline_layout_info);
        if (static_cast<vk::PipelineLayout>(**state.layout) == nullptr) {
            LOGIFACE_LOG(error, "RenderPipeline: failed to create layout for pass '" + state.name + "'");
            continue;
        }

        // One descriptor set per declared app set per frames-in-flight slot. A
        // frame rewrites only its own slot's sets before recording, so a set is
        // never updated while an in-flight frame references it.
        if (!state.app_set_layouts.empty()) {
            const std::uint32_t fif = frames_in_flight_;
            const std::uint32_t group_count = static_cast<std::uint32_t>(state.app_set_layouts.size());

            std::map<std::uint32_t, std::uint32_t> size_by_type;
            for (const auto& decl : state.bindings) {
                size_by_type[static_cast<std::uint32_t>(decl.descriptor_type)] += decl.count * fif;
            }
            std::vector<vk::DescriptorPoolSize> pool_sizes;
            pool_sizes.reserve(size_by_type.size());
            for (const auto& [type, count] : size_by_type) {
                pool_sizes.push_back(vk::DescriptorPoolSize{
                    static_cast<vk::DescriptorType>(type), count});
            }
            vk::DescriptorPoolCreateInfo pool_info{};
            // raii descriptor sets free themselves, which requires the pool flag.
            pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
            pool_info.maxSets = fif * group_count;
            pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
            pool_info.pPoolSizes = pool_sizes.data();
            state.app_pool = std::make_unique<vk::raii::DescriptorPool>(device, pool_info);

            state.app_sets.resize(fif);
            state.app_set_handles.resize(fif);
            state.last_written.resize(fif);
            for (std::uint32_t slot = 0; slot < fif; ++slot) {
                std::vector<vk::DescriptorSetLayout> app_layouts;
                app_layouts.reserve(state.app_set_layouts.size());
                for (const auto& app_layout : state.app_set_layouts) {
                    app_layouts.push_back(*app_layout);
                }
                vk::DescriptorSetAllocateInfo allocate_info{};
                allocate_info.descriptorPool = *state.app_pool;
                allocate_info.descriptorSetCount = static_cast<std::uint32_t>(app_layouts.size());
                allocate_info.pSetLayouts = app_layouts.data();
                state.app_sets[slot] = device.allocateDescriptorSets(allocate_info);
                state.app_set_handles[slot].clear();
                state.app_set_handles[slot].reserve(state.app_sets[slot].size());
                for (const auto& set : state.app_sets[slot]) {
                    state.app_set_handles[slot].push_back(*set);
                }
                state.last_written[slot].assign(state.assignments.size(),
                                                PassPipelineState::DescriptorBindingState{});
            }
        }

        // Infer attachment formats the pass did not specify (a graphics pass
        // rarely knows the swapchain/transient formats at Setup time).
        if (state.attachments.has_value()) {
            if (state.request.color_formats.empty()) {
                for (const auto& color : state.attachments->color_attachments) {
                    const vk::Format format = ResolveResourceFormat(color.resource);
                    if (format != vk::Format::eUndefined) {
                        state.request.color_formats.push_back(format);
                    }
                }
            }
            if (state.request.depth_format == vk::Format::eUndefined &&
                state.attachments->depth_attachment.has_value()) {
                state.request.depth_format =
                    ResolveResourceFormat(state.attachments->depth_attachment->resource);
            }
        }

        if (state.request.kind == VulkanEngine::PipelinePass::PassPipelineKind::Compute) {
            state.compute_desc = ShaderSystem::ComputePipelineDesc{
                .shader = static_cast<ShaderSystem::ShaderId>(state.request.compute_shader),
                .layout = *state.layout,
            };
            auto& st = state;
            compiler.SubmitCompute(
                state.compute_desc, st.name,
                [&st](ShaderSystem::PipelineOutcome&& outcome) {
                    if (!outcome.has_value()) {
                        LOGIFACE_LOG(error, "RenderPipeline: compute pipeline creation failed for pass '" +
                                                st.name + "': " + outcome.error().message);
                        return;
                    }
                    st.slot.Swap(std::move(*outcome), 0);
                    st.built = true;
                });
        } else if (state.request.kind == VulkanEngine::PipelinePass::PassPipelineKind::Graphics) {
            // Blend comes from the pass's declaration (a straight-alpha
            // source-over is the default when it enables blending); an
            // undeclared blend keeps the opaque write every other pass gets.
            const auto& blend = state.request.blend;
            vk::PipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
            blend_attachment.blendEnable = blend.enable ? vk::True : vk::False;
            blend_attachment.srcColorBlendFactor = blend.src_color_blend_factor;
            blend_attachment.dstColorBlendFactor = blend.dst_color_blend_factor;
            blend_attachment.colorBlendOp = blend.color_blend_op;
            blend_attachment.srcAlphaBlendFactor = blend.src_alpha_blend_factor;
            blend_attachment.dstAlphaBlendFactor = blend.dst_alpha_blend_factor;
            blend_attachment.alphaBlendOp = blend.alpha_blend_op;

            state.graphics_desc = ShaderSystem::GraphicsPipelineDesc{};
            state.graphics_desc.vertex_shader = static_cast<ShaderSystem::ShaderId>(state.request.vertex_shader);
            state.graphics_desc.fragment_shader = static_cast<ShaderSystem::ShaderId>(state.request.fragment_shader);
            state.graphics_desc.layout = *state.layout;
            state.graphics_desc.color_formats = state.request.color_formats;
            state.graphics_desc.depth_format = state.request.depth_format;
            // Depth state comes from the pass's declaration. The defaults leave
            // depth disabled, which is the overlay pipeline every pass that does
            // not declare one has always received; a depth-tested pass opts in.
            state.graphics_desc.depth_stencil.depthTestEnable =
                state.request.depth.test_enable ? vk::True : vk::False;
            state.graphics_desc.depth_stencil.depthWriteEnable =
                state.request.depth.write_enable ? vk::True : vk::False;
            state.graphics_desc.depth_stencil.depthCompareOp = state.request.depth.compare_op;
            state.graphics_desc.input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
            state.graphics_desc.viewport.viewportCount = 1;
            state.graphics_desc.viewport.scissorCount = 1;
            state.graphics_desc.rasterization.polygonMode = vk::PolygonMode::eFill;
            state.graphics_desc.rasterization.cullMode = vk::CullModeFlagBits::eNone;
            state.graphics_desc.rasterization.frontFace = vk::FrontFace::eCounterClockwise;
            state.graphics_desc.rasterization.lineWidth = 1.0f;
            state.graphics_desc.multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
            state.graphics_desc.color_blend_attachments = {blend_attachment};
            state.graphics_desc.dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

            auto& st = state;
            compiler.SubmitGraphics(
                state.graphics_desc, st.name,
                [&st](ShaderSystem::PipelineOutcome&& outcome) {
                    if (!outcome.has_value()) {
                        LOGIFACE_LOG(error, "RenderPipeline: graphics pipeline creation failed for pass '" +
                                                st.name + "': " + outcome.error().message);
                        return;
                    }
                    st.slot.Swap(std::move(*outcome), 0);
                    st.built = true;
                });
        }
    }

    // One batched driver call per kind for the whole pass set (compute and,
    // on the monolithic path, graphics). Per-pass failures log and leave the
    // pass unbuilt exactly as the per-pass creations did.
    compiler.Flush(*shader_manager_);
}

vk::Format RenderPipeline::ResolveResourceFormat(
    VulkanEngine::RenderGraph::ResourceHandle resource) const {
    const std::uint32_t index = resource.index;
    if (const auto transient = transient_image_descs_.find(index);
        transient != transient_image_descs_.end()) {
        return transient->second.format;
    }
    const auto name_it = resource_names_.find(index);
    if (name_it != resource_names_.end()) {
        const auto resolver = resource_resolvers_.find(name_it->second);
        if (resolver != resource_resolvers_.end()) {
            return resolver->second.format;
        }
    }
    return vk::Format::eUndefined;
}

void RenderPipeline::RewirePassDescriptors(std::uint32_t slot,
                                           VulkanEngine::PipelinePass::FrameContext& frame) {
    const auto pipeline_it = pass_pipelines_.find(slot);
    if (pipeline_it == pass_pipelines_.end()) {
        return;
    }
    auto& state = pipeline_it->second;
    frame.pass_pipeline = state.slot.Get();
    frame.first_app_descriptor_set = VulkanEngine::Render::kFirstAppDescriptorSet;
    if (state.app_sets.empty() || bootstrap_ == nullptr || frame.resource_lookup == nullptr) {
        return;
    }

    const std::uint32_t fif = frames_in_flight_ == 0 ? 0 : frame.ring_index % frames_in_flight_;
    if (fif >= state.app_set_handles.size()) {
        return;
    }
    frame.app_descriptor_sets = state.app_set_handles[fif];

    auto& device = bootstrap_->GetBackend().GetDevice();
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<vk::DescriptorImageInfo> image_infos(state.assignments.size());
    std::vector<vk::DescriptorBufferInfo> buffer_infos(state.assignments.size());
    auto& last_written = state.last_written[fif];

    for (std::size_t i = 0; i < state.assignments.size(); ++i) {
        const auto& assignment = state.assignments[i];
        const VulkanEngine::Render::DescriptorDecl* decl = nullptr;
        for (const auto& candidate : state.bindings) {
            if (candidate.set == assignment.set && candidate.binding == assignment.binding) {
                decl = &candidate;
                break;
            }
        }
        if (decl == nullptr) {
            continue;
        }
        const auto set_it = std::ranges::find(state.app_set_numbers, assignment.set);
        if (set_it == state.app_set_numbers.end()) {
            continue;
        }
        const auto set_index = static_cast<std::size_t>(
            std::distance(state.app_set_numbers.begin(), set_it));
        if (set_index >= state.app_set_handles[fif].size()) {
            continue;
        }
        const auto name_it = resource_names_.find(assignment.resource.index);
        if (name_it == resource_names_.end()) {
            continue;
        }
        const auto resolved = frame.resource_lookup->Find(name_it->second);
        if (!resolved.has_value()) {
            continue;
        }
        const auto& resource = *resolved;

        PassPipelineState::DescriptorBindingState key{};
        vk::WriteDescriptorSet write{};
        write.dstSet = state.app_set_handles[fif][set_index];
        write.dstBinding = assignment.binding;
        write.descriptorCount = 1;
        write.descriptorType = decl->descriptor_type;

        using vk::DescriptorType;
        switch (decl->descriptor_type) {
            case DescriptorType::eCombinedImageSampler:
            case DescriptorType::eSampledImage: {
                key.view = resource.AsImageView();
                key.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
                key.sampler = decl->descriptor_type == DescriptorType::eCombinedImageSampler
                                  ? frame.default_sampler
                                  : nullptr;
                image_infos[i] = vk::DescriptorImageInfo{key.sampler, key.view, key.layout};
                write.pImageInfo = &image_infos[i];
                break;
            }
            case DescriptorType::eStorageImage: {
                key.view = resource.AsImageView();
                key.layout = vk::ImageLayout::eGeneral;
                image_infos[i] = vk::DescriptorImageInfo{nullptr, key.view, key.layout};
                write.pImageInfo = &image_infos[i];
                break;
            }
            case DescriptorType::eStorageBuffer:
            case DescriptorType::eUniformBuffer: {
                key.buffer = resource.AsBuffer();
                key.offset = resource.GetOffset();
                key.size = resource.GetSize();
                buffer_infos[i] = vk::DescriptorBufferInfo{key.buffer, key.offset, key.size};
                write.pBufferInfo = &buffer_infos[i];
                break;
            }
            default:
                continue;
        }

        key.valid = true;
        auto& last = last_written[i];
        const bool changed =
            !last.valid || last.view != key.view || last.buffer != key.buffer ||
            last.sampler != key.sampler || last.offset != key.offset ||
            last.size != key.size || last.layout != key.layout;
        if (changed) {
            writes.push_back(write);
            last = key;
        }
    }

    if (!writes.empty()) {
        device.updateDescriptorSets(writes, {});
    }
}

void RenderPipeline::PollPassPipelines(std::uint32_t fif_slot) {
    for (auto& [index, state] : pass_pipelines_) {
        (void)index;
        state.slot.RetireFrame(fif_slot);
        if (!state.built || shader_manager_ == nullptr || pipeline_factory_ == nullptr) {
            continue;
        }

        const auto rebuild = [this, &state](ShaderSystem::ShaderManager& shaders)
            -> std::optional<ShaderSystem::PipelineProduct> {
            if (state.request.kind == VulkanEngine::PipelinePass::PassPipelineKind::Compute) {
                auto product = pipeline_factory_->CreateCompute(state.compute_desc, shaders);
                return product.has_value() ? std::optional<ShaderSystem::PipelineProduct>(std::move(*product))
                                           : std::nullopt;
            }
            auto product = pipeline_factory_->CreateGraphics(state.graphics_desc, shaders);
            return product.has_value() ? std::optional<ShaderSystem::PipelineProduct>(std::move(*product))
                                       : std::nullopt;
        };

        if (state.request.kind == VulkanEngine::PipelinePass::PassPipelineKind::Compute) {
            (void)state.slot.PollAndRebuild(*shader_manager_, static_cast<ShaderSystem::ShaderId>(state.request.compute_shader), rebuild, fif_slot);
        } else {
            (void)state.slot.PollAndRebuild(*shader_manager_, static_cast<ShaderSystem::ShaderId>(state.request.vertex_shader),
                                            static_cast<ShaderSystem::ShaderId>(state.request.fragment_shader), rebuild, fif_slot);
        }
    }
}

const VulkanEngine::PipelinePass::PassPipelineRequest* RenderPipeline::GetPassPipelineRequest(
    VulkanEngine::RenderGraph::PassHandle handle) const {
    const auto it = pass_pipelines_.find(handle.index);
    if (it == pass_pipelines_.end() || !it->second.request.IsDeclared()) {
        return nullptr;
    }
    return &it->second.request;
}

const VulkanEngine::PipelinePass::PassPipelineRequest* RenderPipeline::GetPassPipelineRequestByName(
    std::string_view name) const {
    const auto name_it = pass_pipeline_by_name_.find(std::string(name));
    if (name_it == pass_pipeline_by_name_.end()) {
        return nullptr;
    }
    const auto it = pass_pipelines_.find(name_it->second);
    if (it == pass_pipelines_.end() || !it->second.request.IsDeclared()) {
        return nullptr;
    }
    return &it->second.request;
}

vk::PipelineLayout RenderPipeline::GetPassPipelineLayout(VulkanEngine::RenderGraph::PassHandle handle) const {
    const auto it = pass_pipelines_.find(handle.index);
    if (it == pass_pipelines_.end() || it->second.layout == nullptr) {
        return nullptr;
    }
    return **it->second.layout;
}

vk::PipelineLayout RenderPipeline::GetPassPipelineLayoutByName(std::string_view name) const {
    const auto name_it = pass_pipeline_by_name_.find(std::string(name));
    if (name_it == pass_pipeline_by_name_.end()) {
        return nullptr;
    }
    return GetPassPipelineLayout(VulkanEngine::RenderGraph::PassHandle{
        .index = name_it->second, .generation = 1});
}

void RenderPipeline::SetBuiltinHandles(
    const std::array<VulkanEngine::RenderGraph::PassHandle,
                     VulkanEngine::PipelinePass::kBuiltinPassCount>& handles) {
    builtin_handles_ = handles;
}

const std::array<VulkanEngine::RenderGraph::PassHandle,
                 VulkanEngine::PipelinePass::kBuiltinPassCount>&
RenderPipeline::GetBuiltinHandles() const {
    return builtin_handles_;
}

bool RenderPipeline::AddDependency(VulkanEngine::RenderGraph::PassHandle before,
                                   VulkanEngine::RenderGraph::PassHandle after) {
    if (!IsValidModelHandle(before) || !IsValidModelHandle(after)) {
        return false;
    }
    model_dependencies_.emplace_back(before.index, after.index);
    dirty_ = true;
    return true;
}

bool RenderPipeline::SetInitialState(VulkanEngine::RenderGraph::ResourceHandle resource, VulkanEngine::RenderGraph::ResourceState state) {
    return graph_builder_.SetInitialState(resource, state).has_value();
}

bool RenderPipeline::SetFinalState(VulkanEngine::RenderGraph::ResourceHandle resource, VulkanEngine::RenderGraph::ResourceState state) {
    return graph_builder_.SetFinalState(resource, state).has_value();
}

void RenderPipeline::BeginFrame(const void* user_data, std::uint32_t image_index,
                                std::uint32_t fif_slot) {
    // Never leave the previous frame's plan or data visible if this frame bails
    // early. queue_runs_ is cached at compile time by RebuildFromModel, so it is
    // not cleared here.
    barrier_plan_ = {};
    frame_data_ = {};
    if (!compiled_ || !initialized_ || !bootstrap_) {
        return;
    }

    auto& backend = bootstrap_->GetBackend();

    executing_ = true;
    last_fif_slot_ = fif_slot;
    transient_allocator_.CollectGarbage(fif_slot);
    CollectRemovedPasses(fif_slot);

    // Retire pass pipelines swapped out FIF frames ago and hot-reload changed
    // shaders (old pipeline retained if a rebuild fails).
    PollPassPipelines(fif_slot);

    // The swapchain's per-image initialized flag is set when a present succeeds
    // and cleared on swapchain recreation, so it is the submission proof for
    // cross-frame layout tracking: an image that was never presented (or was
    // recreated) must start from Undefined.
    std::vector<bool>& initialized_flags = backend.GetSwapchainImageInitializedFlags();
    const bool image_initialized =
        image_index < initialized_flags.size() && initialized_flags[image_index];
    if (!image_initialized && image_index < tracked_valid_.size()) {
        std::ranges::fill(tracked_valid_[image_index], false);
    }

    VulkanEngine::RenderGraph::RuntimeResourceStates runtime_initial{};
    if (image_initialized && image_index < tracked_states_.size()) {
        runtime_initial.states = tracked_states_[image_index];
        runtime_initial.has_state = tracked_valid_[image_index];
    }

    ResolveResources(compiled_graph_, image_index, fif_slot);

    VulkanEngine::RenderGraph::ResolvedResourceHandles resolved{};
    resolved.images = compiled_graph_.resource_images;
    resolved.buffers = compiled_graph_.resource_buffers;
    resolved.buffer_offsets = compiled_graph_.resource_buffer_offsets;
    resolved.buffer_sizes = compiled_graph_.resource_buffer_sizes;
    resolved.formats = compiled_graph_.resource_formats;

    // Alias reuse decided by the transient planner becomes explicit ordering
    // dependencies in the barrier plan.
    VulkanEngine::RenderGraph::AliasIntervals alias_intervals{};
    for (const auto& alias : transient_allocator_.GetPlan().aliases) {
        alias_intervals.dependencies.push_back(VulkanEngine::RenderGraph::PlannedAliasDependency{
            .aliased_resource = alias.aliased_resource,
            .after_resource = alias.after_resource,
            .pass_index = alias.pass_index,
        });
    }

    const auto plan = VulkanEngine::RenderGraph::PlanBarriers(
        compiled_graph_, resolved, alias_intervals, &runtime_initial);

    // Record the actual end-of-frame state for this swapchain image. Use is
    // gated on the presented flag, so a failed submit cannot seed the next frame.
    if (image_index < tracked_states_.size() &&
        plan.end_states.size() == tracked_states_[image_index].size()) {
        tracked_states_[image_index] = plan.end_states;
        tracked_valid_[image_index] = plan.has_end_state;
    }

    // Per-frame name -> resolved-handle table for FrameContext::GetResource.
    frame_lookup_.Clear();
    for (const auto& lifetime : compiled_graph_.resource_lifetimes) {
        const std::uint32_t index = lifetime.handle.index;
        if (index >= compiled_graph_.resource_info.size()) {
            continue;
        }
        const auto& info = compiled_graph_.resource_info[index];

        VulkanEngine::PipelinePass::ResolvedResource entry{};
        entry.handle = lifetime.handle;
        entry.kind = info.kind;
        if (info.kind == VulkanEngine::RenderGraph::ResourceKind::Image) {
            if (index < compiled_graph_.resource_images.size()) {
                entry.image = compiled_graph_.resource_images[index];
            }
            if (index < compiled_graph_.resource_formats.size()) {
                entry.format = compiled_graph_.resource_formats[index];
            }
            if (info.imported) {
                const auto it = resource_resolvers_.find(info.name);
                if (it != resource_resolvers_.end()) {
                    entry.view = it->second.resolve_image_view(image_index);
                }
            } else {
                entry.view = transient_allocator_.GetImageView(index, fif_slot);
            }
            entry.resolved = entry.image != nullptr;
        } else {
            if (index < compiled_graph_.resource_buffers.size()) {
                entry.buffer = compiled_graph_.resource_buffers[index];
            }
            if (index < compiled_graph_.resource_buffer_offsets.size()) {
                entry.offset = compiled_graph_.resource_buffer_offsets[index];
            }
            if (index < compiled_graph_.resource_buffer_sizes.size()) {
                entry.size = compiled_graph_.resource_buffer_sizes[index];
            }
            entry.resolved = entry.buffer != nullptr;
        }
        frame_lookup_.Set(info.name, std::move(entry));
    }

    // Copy the incoming frame data so this frame's lookup can be attached
    // without mutating the caller's object. Runs are recorded by the caller
    // (one command buffer per queue run) via RecordRun().
    frame_data_ = VulkanEngine::PipelinePass::RenderFrameData{};
    if (user_data != nullptr) {
        frame_data_ = *static_cast<const VulkanEngine::PipelinePass::RenderFrameData*>(user_data);
    }
    frame_data_.frame.resource_lookup = &frame_lookup_;

    barrier_plan_ = plan;
}

void RenderPipeline::SetGpuPassTimer(VulkanBackend::Vulkan::GpuPassTimer* timer) {
    gpu_pass_timer_ = timer;
}

void RenderPipeline::RecordRun(std::uint32_t run_index, vk::CommandBuffer command_buffer) {
    if (run_index >= queue_runs_.runs.size()) {
        return;
    }
    const auto& run = queue_runs_.runs[run_index];
    // Timestamps cannot be written on a queue family without timestamp support,
    // and a pool cannot span queues: graphics runs get the timer, a separate
    // compute/transfer run gets null. The pool itself is created on the graphics
    // family (Renderer::Initialize).
    const bool async_compute = bootstrap_ != nullptr && bootstrap_->GetBackend().HasAsyncCompute();
    const bool timestamps_ok =
        !async_compute || run.queue == VulkanEngine::RenderGraph::QueueType::Graphics;
    VulkanBackend::Vulkan::ExecuteRenderGraphRange(barrier_plan_, compiled_graph_,
                                                   run.first_pass, run.last_pass,
                                                   &frame_data_, command_buffer,
                                                   timestamps_ok ? gpu_pass_timer_ : nullptr);
}

void RenderPipeline::EndFrame() {
    executing_ = false;
}

void RenderPipeline::SyncTransients() {
    if (!bootstrap_) {
        return;
    }

    auto& backend = bootstrap_->GetBackend();
    if (!transient_allocator_.IsInitialized()) {
        transient_allocator_.Initialize(backend, "render-pipeline-transients");
    }

    const std::uint32_t frames_in_flight = std::max<std::uint32_t>(backend.GetFramesInFlight(), 1);

    // One desc per graph resource index: the allocator keys lookups by
    // resource_index, so the vector must be dense (inactive placeholders for
    // imported/non-transient resources) rather than a filtered list.
    const std::size_t resource_count = compiled_graph_.resource_info.size();
    std::vector<VulkanEngine::GpuResources::TransientAllocator::Desc> descs(resource_count);
    for (auto& slot : descs) {
        slot.requirements.active = false;
    }

    for (const auto& lifetime : compiled_graph_.resource_lifetimes) {
        if (!lifetime.transient) {
            continue;
        }
        const std::uint32_t index = lifetime.handle.index;
        if (index >= descs.size()) {
            continue;
        }
        const bool is_buffer =
            index < compiled_graph_.resource_info.size() &&
            compiled_graph_.resource_info[index].kind == VulkanEngine::RenderGraph::ResourceKind::Buffer;

        VulkanEngine::GpuResources::TransientAllocator::Desc desc{};
        desc.requirements.name = lifetime.name;
        desc.requirements.first_pass = lifetime.first_pass;
        desc.requirements.last_pass = lifetime.last_pass;

        if (is_buffer) {
            const auto it = transient_buffer_descs_.find(index);
            if (it == transient_buffer_descs_.end() || it->second.size == 0) {
                continue;
            }
            desc.is_image = false;
            desc.requirements.kind = VulkanEngine::GpuResources::TransientKind::Buffer;
            desc.requirements.heap_key = 1;
            desc.requirements.size = it->second.size;
            desc.requirements.alignment = 256;
            desc.requirements.aliasable = it->second.aliasable;
            desc.buffer.usage = it->second.usage;
            desc.buffer.memory = it->second.memory_properties;
        } else {
            const auto it = transient_image_descs_.find(index);
            if (it == transient_image_descs_.end() || it->second.format == vk::Format::eUndefined) {
                continue;
            }
            const auto [image_width, image_height] =
                VulkanEngine::PipelinePass::ResolveTransientExtent(it->second,
                                                                   render_width_, render_height_);
            if (image_width == 0 || image_height == 0) {
                continue;
            }
            desc.is_image = true;
            desc.requirements.kind = VulkanEngine::GpuResources::TransientKind::Image;
            desc.requirements.heap_key = 0;
            // Sized from the real image memory requirements by the allocator.
            desc.requirements.size = 0;
            desc.requirements.alignment = 1;
            desc.requirements.aliasable = it->second.aliasable;
            desc.image.format = it->second.format;
            desc.image.width = image_width;
            desc.image.height = image_height;
            desc.image.usage = it->second.usage;
            desc.image.samples = vk::SampleCountFlagBits::e1;
            desc.image.aspect = FormatToAspectFlags(it->second.format);
        }

        desc.requirements.active = true;
        descs[index] = std::move(desc);
    }

    // A recompile happens at a frame boundary; frame 0 retires against the next
    // submitted frame, so nothing in flight is freed early.
    transient_allocator_.Sync(descs, frames_in_flight, 0);
}

void RenderPipeline::ResolveResources(VulkanEngine::RenderGraph::CompiledRenderGraph& graph,
                                      std::uint32_t image_index, std::uint32_t fif_slot) {
    if (!bootstrap_) {
        return;
    }

    auto& backend = bootstrap_->GetBackend();

    // Identity table: keyed by ResourceHandle.index, never by container size.
    const auto table = VulkanEngine::RenderGraph::BuildResourceResolutionTable(
        graph, image_resolver_names_, buffer_resolver_names_);

    using VulkanEngine::RenderGraph::ResourceResolutionKind;
    for (const auto& entry : table) {
        const std::uint32_t index = entry.resource_index;
        if (index >= graph.resource_info.size()) {
            continue;
        }

        switch (entry.resolution) {
            case ResourceResolutionKind::ImportedImage: {
                const auto it = resource_resolvers_.find(entry.name);
                if (it != resource_resolvers_.end()) {
                    graph.SetResourceImage(index, it->second.resolve_image(image_index));
                    graph.SetResourceFormat(index, it->second.format);
                }
                break;
            }
            case ResourceResolutionKind::ImportedBuffer: {
                const auto it = buffer_resolvers_.find(entry.name);
                if (it != buffer_resolvers_.end()) {
                    graph.SetResourceBuffer(index, it->second(image_index));
                }
                break;
            }
            case ResourceResolutionKind::TransientImage: {
                const vk::Image image = transient_allocator_.GetImage(index, fif_slot);
                if (image) {
                    graph.SetResourceImage(index, image);
                }
                const auto it = transient_image_descs_.find(index);
                if (it != transient_image_descs_.end()) {
                    graph.SetResourceFormat(index, it->second.format);
                }
                break;
            }
            case ResourceResolutionKind::TransientBuffer: {
                vk::Buffer buffer{};
                vk::DeviceSize offset = 0;
                vk::DeviceSize size = vk::WholeSize;
                if (transient_allocator_.GetBuffer(index, fif_slot, buffer, offset, size)) {
                    graph.SetResourceBuffer(index, buffer, offset, size);
                }
                break;
            }
            case ResourceResolutionKind::ImportedImageMissingResolver:
            case ResourceResolutionKind::ImportedBufferMissingResolver:
            case ResourceResolutionKind::Unused:
            case ResourceResolutionKind::InvalidIndex:
                break;
        }
    }

    // Resolve attachment views from the table and compute the render area from
    // the actual color/depth resources (imported or transient), so a depth-only
    // or imported setup sizes correctly.
    for (auto& pass : graph.passes) {
        if (!pass.attachment_setup || !pass.attachment_setup->auto_begin_rendering) {
            continue;
        }
        auto& setup = *pass.attachment_setup;

        const auto resolve_attachment_view = [&](VulkanEngine::RenderGraph::AttachmentInfo& attach) {
            const std::uint32_t resource = attach.resource.index;
            if (resource >= table.size() || table[resource].resource_index != resource) {
                return;
            }
            const auto& entry = table[resource];
            if (entry.resolution == ResourceResolutionKind::ImportedImage) {
                const auto it = resource_resolvers_.find(entry.name);
                if (it != resource_resolvers_.end()) {
                    attach.image_view = it->second.resolve_image_view(image_index);
                }
            } else if (entry.resolution == ResourceResolutionKind::TransientImage) {
                const vk::ImageView view = transient_allocator_.GetImageView(resource, fif_slot);
                if (view) {
                    attach.image_view = view;
                }
            }
        };

        for (auto& attach : setup.color_attachments) {
            resolve_attachment_view(attach);
        }
        if (setup.depth_attachment.has_value()) {
            resolve_attachment_view(*setup.depth_attachment); //NOLINT(bugprone-unchecked-optional-access)
        }

        std::uint32_t max_width = 0;
        std::uint32_t max_height = 0;
        const auto consider = [&](const VulkanEngine::RenderGraph::AttachmentInfo& attach) {
            const auto it = transient_image_descs_.find(attach.resource.index);
            if (it != transient_image_descs_.end()) {
                max_width = std::max(max_width, it->second.width);
                max_height = std::max(max_height, it->second.height);
            }
        };
        for (const auto& attach : setup.color_attachments) {
            consider(attach);
        }
        if (setup.depth_attachment.has_value()) {
            consider(*setup.depth_attachment); //NOLINT(bugprone-unchecked-optional-access)
        }
        if (max_width == 0) {
            (void)backend.GetSwapchainExtent(max_width, max_height);
        }
        setup.render_area = vk::Rect2D{{0, 0}, {max_width, max_height}};
    }
}

} // namespace VulkanEngine::RenderPipeline
