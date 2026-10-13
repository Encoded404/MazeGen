module;


module VulkanEngine.RenderGraph;

import std;
import std.compat;

import vulkan;


namespace VulkanEngine::RenderGraph {

// Module-linkage helpers (defined in RenderGraph.cpp)
using RenderGraph::ResourceHandle;
using ReadInfo = RenderGraphBuilder::ReadInfo;
using RenderGraph::ResourceKind;
using RenderGraph::ResourceState;

[[maybe_unused]] bool IsResourceStateCompatible(ResourceKind kind, const ResourceState& state);
[[maybe_unused]] bool ContainsResource(const std::vector<ResourceHandle>& handles, ResourceHandle value);

namespace {

// Default queue tag the compiler uses for resource states it synthesizes itself
// (undefined/imported initial states). Per-pass queue assignment is carried by
// each pass and copied into CompiledPass::queue, which drives queue-run
// partitioning; this constant is not a routing decision.
constexpr QueueType kCompilerQueue = QueueType::Graphics;

[[nodiscard]] ResourceState UndefinedStateFor(ResourceKind kind) {
    if (kind == ResourceKind::Image) {
        return ResourceState::ImageState(PipelineStageIntent::TopOfPipe, AccessIntent::None,
                                         kCompilerQueue, ImageLayoutIntent::Undefined);
    }
    return ResourceState::BufferState(PipelineStageIntent::TopOfPipe, AccessIntent::None,
                                      kCompilerQueue);
}

// The single destination state a pass leaves a resource in. A write always
// wins the layout; reads contribute only their stage/access scope so a
// read+write in one pass collapses to one transition with an OR'd scope.
struct CombinedUse {
    bool has_use = false;
    bool written = false;
    ResourceState target{};
    vk::PipelineStageFlags2 dst_stage{};
    vk::AccessFlags2 dst_access{};
};

}  // namespace

CompiledRenderGraph RenderGraphBuilder::Compile() const {
    CompiledRenderGraph result{};

    auto emit_diagnostic = [&](DiagnosticCode code, DiagnosticSeverity severity, std::string message) {
        result.diagnostics.push_back(CompileDiagnostic{
            .code = code,
            .severity = severity,
            .message = std::move(message),
        });
    };

    if (passes_.empty()) {
        result.success = true;
        emit_diagnostic(
            DiagnosticCode::EmptyGraph,
            DiagnosticSeverity::Info,
            "Render graph has no passes; compile produced an empty execution plan.");
        return result;
    }

    for (const auto& resource : resources_) {
        if (resource.has_initial_state && !IsResourceStateCompatible(resource.kind, resource.initial_state)) {
            emit_diagnostic(
                DiagnosticCode::InvalidInitialState,
                DiagnosticSeverity::Error,
                "Render graph compile failed: resource '" + resource.name + "' has an incompatible initial state.");
        }
        if (resource.has_final_state && !IsResourceStateCompatible(resource.kind, resource.final_state)) {
            emit_diagnostic(
                DiagnosticCode::InvalidFinalState,
                DiagnosticSeverity::Error,
                "Render graph compile failed: resource '" + resource.name + "' has an incompatible final state.");
        }
    }

    const std::size_t pass_count = passes_.size();
    std::vector<std::unordered_set<std::uint32_t>> edges(pass_count);
    std::vector<std::uint32_t> indegree(pass_count, 0);

    auto add_edge = [&](std::uint32_t from, std::uint32_t to) {
        if (from == to) {
            return;
        }
        if (edges[from].insert(to).second) {
            ++indegree[to];
        }
    };

    // True when `from` can already reach `to` through the edges added so far.
    const auto reachable = [&](std::uint32_t from, std::uint32_t to) {
        std::vector<std::uint32_t> stack{from};
        std::vector<bool> visited(pass_count, false);
        while (!stack.empty()) {
            const std::uint32_t current = stack.back();
            stack.pop_back();
            if (current == to) {
                return true;
            }
            if (visited[current]) {
                continue;
            }
            visited[current] = true;
            for (const std::uint32_t next : edges[current]) {
                stack.push_back(next);
            }
        }
        return false;
    };

    // Inferred write-after-read / write-after-write edges only fix an order; if
    // an explicit dependency already establishes the opposite order, adding the
    // inferred edge would create a spurious cycle. The explicit order wins.
    const auto add_hazard_order_edge = [&](std::uint32_t from, std::uint32_t to) {
        if (from == to) {
            return;
        }
        if (reachable(to, from)) {
            return;
        }
        add_edge(from, to);
    };

    for (const auto& [before, after] : explicit_dependencies_) {
        if (!IsValidPassHandle(before) || !IsValidPassHandle(after)) {
            emit_diagnostic(
                DiagnosticCode::InvalidExplicitDependency,
                DiagnosticSeverity::Error,
                "Render graph contains an invalid explicit pass dependency.");
            continue;
        }

        if (!passes_[before.index].enabled || !passes_[after.index].enabled) {
            continue;
        }

        add_edge(before.index, after.index);
    }

    struct ResourceTracker {
        std::int32_t last_writer = -1;
        std::unordered_set<std::uint32_t> readers{};
    };

    std::vector<ResourceTracker> resource_trackers(resources_.size());

    for (std::uint32_t pass_index = 0; pass_index < pass_count; ++pass_index) {
        const auto& pass = passes_[pass_index];
        if (!pass.enabled) {
            continue;
        }

        for (const auto& read : pass.reads) {
            if (!IsValidResourceHandle(read.resource)) {
                emit_diagnostic(
                    DiagnosticCode::InvalidReadHandle,
                    DiagnosticSeverity::Error,
                    "Pass '" + pass.name + "' references an invalid read resource handle.");
                continue;
            }

            auto& tracker = resource_trackers[read.resource.index];
            if (tracker.last_writer >= 0) {
                add_hazard_order_edge(static_cast<std::uint32_t>(tracker.last_writer), pass_index);
            }
            tracker.readers.insert(pass_index);
        }

        for (const auto& write : pass.writes) {
            if (!IsValidResourceHandle(write)) {
                emit_diagnostic(
                    DiagnosticCode::InvalidWriteHandle,
                    DiagnosticSeverity::Error,
                    "Pass '" + pass.name + "' references an invalid write resource handle.");
                continue;
            }

            auto& tracker = resource_trackers[write.index];
            if (tracker.last_writer >= 0) {
                add_hazard_order_edge(static_cast<std::uint32_t>(tracker.last_writer), pass_index);
            }
            for (const std::uint32_t reader_index : tracker.readers) {
                add_hazard_order_edge(reader_index, pass_index);
            }
            tracker.readers.clear();
            tracker.last_writer = static_cast<std::int32_t>(pass_index);
        }
    }

    // Index-ordered ready set: independent passes always run in ascending slot
    // order, and a newly unblocked lower slot is preferred over a higher one
    // that was already ready (the 0,3,5 interleaving case). A FIFO queue does
    // not provide this guarantee.
    std::set<std::uint32_t> ready{};
    for (std::uint32_t pass_index = 0; pass_index < pass_count; ++pass_index) {
        if (!passes_[pass_index].enabled) {
            continue;
        }
        if (indegree[pass_index] == 0) {
            ready.insert(pass_index);
        }
    }

    std::vector<std::uint32_t> sorted_indices{};
    sorted_indices.reserve(pass_count);
    while (!ready.empty()) {
        const std::uint32_t current = *ready.begin();
        ready.erase(ready.begin());
        sorted_indices.push_back(current);

        for (const std::uint32_t dependent : edges[current]) {
            if (--indegree[dependent] == 0) {
                ready.insert(dependent);
            }
        }
    }

    const std::uint32_t enabled_count = static_cast<std::uint32_t>(std::ranges::count_if(passes_, [](const PassNode& pass) {
        return pass.enabled;
    }));

    if (sorted_indices.size() != enabled_count) {
        emit_diagnostic(
            DiagnosticCode::CycleDetected,
            DiagnosticSeverity::Error,
            "Render graph compile failed: pass dependency cycle detected.");
        result.success = false;
        return result;
    }

    // Last ordered use of each resource, so end-of-frame final states are
    // attached to the pass that actually consumed the resource last.
    const auto uses_resource = [&](std::uint32_t pass_index, std::uint32_t resource_index) {
        const auto& pass = passes_[pass_index];
        const ResourceHandle handle{.index = resource_index,
                                    .generation = resources_[resource_index].generation};
        if (ContainsResource(pass.writes, handle)) {
            return true;
        }
        return std::ranges::any_of(pass.reads, [&](const ReadInfo& read) {
            return read.resource.index == resource_index;
        });
    };

    std::vector<std::int32_t> last_use(sorted_indices.size() > 0 ? resources_.size() : 0, -1);
    for (std::uint32_t resource_index = 0; resource_index < resources_.size(); ++resource_index) {
        for (std::size_t ordered = 0; ordered < sorted_indices.size(); ++ordered) {
            if (uses_resource(sorted_indices[ordered], resource_index)) {
                last_use[resource_index] = static_cast<std::int32_t>(ordered);
            }
        }
    }

    struct ResourcePassState {
        ResourceState state{};
        bool has_state = false;
    };

    std::vector<ResourcePassState> resource_states(resources_.size());
    for (std::uint32_t resource_index = 0; resource_index < resources_.size(); ++resource_index) {
        const auto& resource = resources_[resource_index];
        if (resource.has_initial_state) {
            resource_states[resource_index] = {resource.initial_state, true};
        }
    }

    std::vector<CompiledPass> compiled_passes;
    compiled_passes.reserve(sorted_indices.size());

    for (std::size_t ordered = 0; ordered < sorted_indices.size(); ++ordered) {
        const std::uint32_t pass_index = sorted_indices[ordered];
        const auto& pass = passes_[pass_index];

        CompiledPass compiled_pass{};
        compiled_pass.handle = PassHandle{.index = pass_index, .generation = pass.generation};
        compiled_pass.name = pass.name;
        compiled_pass.queue = pass.queue;
        compiled_pass.execute = pass.execute;
        compiled_pass.attachment_setup = pass.attachment_setup;

        // Collapse every read/write of one resource in this pass into a single
        // combined use.
        std::vector<std::pair<std::uint32_t, CombinedUse>> combined;
        const auto combined_for = [&](std::uint32_t resource_index) -> CombinedUse& {
            const auto it = std::ranges::find_if(combined, [&](const auto& entry) {
                return entry.first == resource_index;
            });
            if (it != combined.end()) {
                return it->second;
            }
            combined.emplace_back(resource_index, CombinedUse{});
            return combined.back().second;
        };

        for (const auto& read : pass.reads) {
            if (!IsValidResourceHandle(read.resource)) {
                continue;
            }
            auto& use = combined_for(read.resource.index);
            use.has_use = true;
            use.dst_stage |= IntentToPipelineStage(read.stage, read.access);
            use.dst_access |= IntentToAccessFlags(read.stage, read.access);

            const auto& resource = resources_[read.resource.index];
            if (resource.kind == ResourceKind::Image) {
                const ImageLayoutIntent layout = read.stage == PipelineStageIntent::DepthAttachment
                                                     ? ImageLayoutIntent::DepthReadOnly
                                                     : ImageLayoutIntent::ShaderReadOnly;
                if (!use.written) {
                    use.target = ResourceState::ImageState(read.stage, read.access, kCompilerQueue, layout);
                }
            } else if (!use.written) {
                use.target = ResourceState::BufferState(read.stage, read.access, kCompilerQueue);
            }
        }

        for (const auto& write : pass.writes) {
            if (!IsValidResourceHandle(write)) {
                continue;
            }
            auto& use = combined_for(write.index);
            use.has_use = true;
            use.written = true;

            const auto& resource = resources_[write.index];
            if (resource.kind == ResourceKind::Image) {
                ResourceState target = ResourceState::ImageState(
                    PipelineStageIntent::ComputeShader, AccessIntent::Write,
                    kCompilerQueue, ImageLayoutIntent::General);
                if (pass.attachment_setup) {
                    if (pass.attachment_setup->depth_attachment &&
                        pass.attachment_setup->depth_attachment->resource.index == write.index) {
                        target = ResourceState::ImageState(
                            PipelineStageIntent::DepthAttachment, AccessIntent::Write,
                            kCompilerQueue, ImageLayoutIntent::DepthAttachment);
                    } else {
                        for (const auto& color : pass.attachment_setup->color_attachments) {
                            if (color.resource.index == write.index) {
                                target = ResourceState::ImageState(
                                    PipelineStageIntent::ColorAttachment, AccessIntent::Write,
                                    kCompilerQueue, ImageLayoutIntent::ColorAttachment);
                                break;
                            }
                        }
                    }
                }
                use.target = target;
            } else {
                use.target = ResourceState::BufferState(
                    PipelineStageIntent::ComputeShader, AccessIntent::Write, kCompilerQueue);
            }
            use.dst_stage |= IntentToPipelineStage(use.target.stage, use.target.access);
            use.dst_access |= IntentToAccessFlags(use.target.stage, use.target.access);
        }

        std::ranges::sort(combined, [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& [resource_index, use] : combined) {
            if (!use.has_use) {
                continue;
            }

            const auto& resource = resources_[resource_index];
            auto& current = resource_states[resource_index];

            const ResourceState from = current.has_state
                                           ? current.state
                                           : UndefinedStateFor(resource.kind);
            if (!StatesEqual(from, use.target)) {
                ResourceTransition transition{};
                transition.resource_index = resource_index;
                transition.from_state = from;
                transition.from_known = current.has_state;
                transition.target_state = use.target;
                transition.src_stage = IntentToPipelineStage(from.stage, from.access);
                transition.src_access = IntentToAccessFlags(from.stage, from.access);
                transition.dst_stage = use.dst_stage;
                transition.dst_access = use.dst_access;
                compiled_pass.pre_pass_transitions.push_back(transition);
            }

            current.state = use.target;
            current.has_state = true;
        }

        // End-of-frame final state belongs to the last pass that touches the
        // resource (read or write), evaluated against that pass's state. A pass
        // whose last touch is a read (e.g. a capture pass sampling the
        // backbuffer) must still transition the resource to its final layout.
        for (std::uint32_t index = 0; index < resources_.size(); ++index) {
            const auto& resource = resources_[index];
            if (resource.kind != ResourceKind::Image || !resource.has_final_state) {
                continue;
            }
            if (last_use[index] != static_cast<std::int32_t>(ordered)) {
                continue;
            }

            auto& current = resource_states[index];
            if (!current.has_state || !StatesEqual(current.state, resource.final_state)) {
                ResourceTransition transition{};
                transition.resource_index = index;
                transition.from_state = current.has_state ? current.state : UndefinedStateFor(resource.kind);
                transition.from_known = current.has_state;
                transition.target_state = resource.final_state;
                transition.src_stage = IntentToPipelineStage(transition.from_state.stage, transition.from_state.access);
                transition.src_access = IntentToAccessFlags(transition.from_state.stage, transition.from_state.access);
                transition.dst_stage = IntentToPipelineStage(resource.final_state.stage, resource.final_state.access);
                transition.dst_access = IntentToAccessFlags(resource.final_state.stage, resource.final_state.access);
                compiled_pass.post_pass_transitions.push_back(transition);
                current.state = resource.final_state;
                current.has_state = true;
            }
        }

        compiled_passes.push_back(std::move(compiled_pass));
    }

    result.passes = std::move(compiled_passes);

    result.resource_lifetimes.reserve(resources_.size());
    result.resource_info.reserve(resources_.size());
    result.initial_states.resize(resources_.size());
    result.has_initial_state.resize(resources_.size(), false);
    result.resource_images.resize(resources_.size());
    result.resource_buffers.resize(resources_.size());
    result.resource_buffer_offsets.resize(resources_.size(), 0);
    result.resource_buffer_sizes.resize(resources_.size(), vk::WholeSize);
    result.resource_formats.resize(resources_.size(), vk::Format::eUndefined);

    for (std::uint32_t resource_index = 0; resource_index < resources_.size(); ++resource_index) {
        const auto& resource = resources_[resource_index];

        std::int32_t first = -1;
        std::int32_t last = -1;
        for (std::size_t ordered = 0; ordered < sorted_indices.size(); ++ordered) {
            if (uses_resource(sorted_indices[ordered], resource_index)) {
                if (first < 0) {
                    first = static_cast<std::int32_t>(ordered);
                }
                last = static_cast<std::int32_t>(ordered);
            }
        }

        result.resource_lifetimes.push_back(ResourceLifetime{
            .handle = ResourceHandle{.index = resource_index, .generation = resource.generation},
            .name = resource.name,
            .imported = resource.imported,
            .transient = resource.transient,
            .first_pass = first,
            .last_pass = last,
        });

        result.resource_info.push_back(ResourceInfo{
            .name = resource.name,
            .kind = resource.kind,
            .imported = resource.imported,
            .image_info = resource.image_info,
            .buffer_info = resource.buffer_info,
        });

        if (resource.image_info) {
            result.resource_formats[resource_index] = resource.image_info->format;
        }

        if (resource.has_initial_state) {
            result.initial_states[resource_index] = resource.initial_state;
            result.has_initial_state[resource_index] = true;
        }
    }

    const bool has_errors = std::ranges::any_of(result.diagnostics, [](const CompileDiagnostic& diagnostic) {
        return diagnostic.severity == DiagnosticSeverity::Error;
    });

    result.success = !has_errors;
    if (result.success) {
        emit_diagnostic(
            DiagnosticCode::CompileSuccess,
            DiagnosticSeverity::Info,
            "Render graph compile succeeded.");
    }

    return result;
}

}  // namespace VulkanEngine::RenderGraph
