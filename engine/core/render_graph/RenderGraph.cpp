module;

#include <logging/logging_macros.hpp>

module VulkanEngine.RenderGraph;

import std;
import std.compat;

import logiface;

import vulkan;

using VulkanEngine::RenderGraph::ResourceHandle;
using ReadInfo = VulkanEngine::RenderGraph::RenderGraphBuilder::ReadInfo;
using VulkanEngine::RenderGraph::PassHandle;
using VulkanEngine::RenderGraph::ResourceKind;
using VulkanEngine::RenderGraph::ResourceState;
using VulkanEngine::RenderGraph::PipelineStageIntent;
using VulkanEngine::RenderGraph::AccessIntent;
using VulkanEngine::RenderGraph::ImageLayoutIntent;
using VulkanEngine::RenderGraph::GraphBuildError;



namespace VulkanEngine::RenderGraph {

// we cant use internal linkage, because we are using modules they need to be visible at module level, not only TU level.
// NOLINTBEGIN(misc-use-internal-linkage)

[[maybe_unused]] bool ContainsResource(const std::vector<ResourceHandle>& handles, ResourceHandle value) {
    return std::ranges::find(handles, value) != handles.end();
}

[[maybe_unused]] bool ContainsReadResource(const std::vector<ReadInfo>& reads, ResourceHandle value) {
    return std::ranges::find_if(reads, [value](const ReadInfo& r) { return r.resource == value; }) != reads.end();
}

[[maybe_unused]] bool ContainsDependency(const std::vector<std::pair<PassHandle, PassHandle>>& deps,
                        const std::pair<PassHandle, PassHandle>& value) {
    return std::ranges::find(deps, value) != deps.end();
}

[[maybe_unused]] bool IsResourceStateCompatible(ResourceKind kind, const ResourceState& state) {
    if (kind == ResourceKind::Image) {
        return state.has_image_layout;
    }
    return !state.has_image_layout;
}

// NOLINTEND(misc-use-internal-linkage)

std::uint32_t RenderGraphBuilder::FindResourceByName(std::string_view name, ResourceKind kind) const {
    for (std::uint32_t index = 0; index < resources_.size(); ++index) {
        if (resources_[index].kind == kind && resources_[index].name == name) {
            return index;
        }
    }
    return std::numeric_limits<std::uint32_t>::max();
}

ResourceHandle RenderGraphBuilder::CreateTransientResource(std::string name, ResourceKind kind) {
    // Idempotent by (name, kind): asking twice yields the same slot, so a
    // rebuild-from-model never appends a duplicate.
    if (const std::uint32_t existing = FindResourceByName(name, kind);
        existing != std::numeric_limits<std::uint32_t>::max()) {
        return ResourceHandle{.index = existing, .generation = resources_[existing].generation};
    }

    const std::uint32_t index = static_cast<std::uint32_t>(resources_.size());
    resources_.push_back(ResourceNode{
        .name = std::move(name),
        .kind = kind,
        .generation = 1,
        .imported = false,
        .transient = true,
    });

    return ResourceHandle{.index = index, .generation = resources_.back().generation};
}

ResourceHandle RenderGraphBuilder::ImportResource(std::string name, ResourceKind kind) {
    if (const std::uint32_t existing = FindResourceByName(name, kind);
        existing != std::numeric_limits<std::uint32_t>::max()) {
        return ResourceHandle{.index = existing, .generation = resources_[existing].generation};
    }

    LOGIFACE_LOG(trace, "Importing resource '" + name + "'");

    const std::uint32_t index = static_cast<std::uint32_t>(resources_.size());
    resources_.push_back(ResourceNode{
        .name = std::move(name),
        .kind = kind,
        .generation = 1,
        .imported = true,
        .transient = false,
    });

    return ResourceHandle{.index = index, .generation = resources_.back().generation};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::SetTransientImageInfo(ResourceHandle resource, TransientImageInfo info) {
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    auto& resource_node = resources_[resource.index];
    if (!resource_node.transient || resource_node.kind != ResourceKind::Image) {
        return std::unexpected(GraphBuildError::IncompatibleResourceKind);
    }

    resource_node.image_info = info;
    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::SetTransientBufferInfo(ResourceHandle resource, TransientBufferInfo info) {
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    auto& resource_node = resources_[resource.index];
    if (!resource_node.transient || resource_node.kind != ResourceKind::Buffer) {
        return std::unexpected(GraphBuildError::IncompatibleResourceKind);
    }

    resource_node.buffer_info = info;
    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::SetInitialState(ResourceHandle resource, ResourceState state) {
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    auto& resource_node = resources_[resource.index];
    if (!IsResourceStateCompatible(resource_node.kind, state)) {
        return std::unexpected(GraphBuildError::IncompatibleResourceState);
    }

    resource_node.initial_state = state;
    resource_node.has_initial_state = true;
    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::SetFinalState(ResourceHandle resource, ResourceState state) {
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    auto& resource_node = resources_[resource.index];
    if (!IsResourceStateCompatible(resource_node.kind, state)) {
        return std::unexpected(GraphBuildError::IncompatibleResourceState);
    }

    resource_node.final_state = state;
    resource_node.has_final_state = true;
    return {};
}

PassHandle RenderGraphBuilder::AddPass(std::string name, QueueType queue, bool enabled, PassExecutionCallback execute) {
    LOGIFACE_LOG(trace, "Adding pass '" + name + "' with queue type " + std::to_string(static_cast<int>(queue)));

    const std::uint32_t index = static_cast<std::uint32_t>(passes_.size());
    passes_.push_back(PassNode{
        .name = std::move(name),
        .queue = queue,
        .generation = 1,
        .enabled = enabled,
        .execute = std::move(execute),
    });

    return PassHandle{.index = index, .generation = passes_.back().generation};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::AddRead(PassHandle pass, ResourceHandle resource) {
    return AddRead(pass, resource, PipelineStageIntent::FragmentShader, AccessIntent::Read);
}

std::expected<void, GraphBuildError> RenderGraphBuilder::AddRead(PassHandle pass, ResourceHandle resource,
                                                                 PipelineStageIntent stage, AccessIntent access) {
    if (!IsValidPassHandle(pass)) {
        return std::unexpected(GraphBuildError::InvalidPassHandle);
    }
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    // Multiple reads of the same resource with different stages are kept and
    // merged by the compiler; dropping the later ones would silently lose a
    // required synchronization scope (e.g. index fetch + shader read).
    passes_[pass.index].reads.push_back(ReadInfo{resource, stage, access});
    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::AddWrite(PassHandle pass, ResourceHandle resource) {
    // Validate before any indexing: the previous implementation read the pass
    // name out of bounds before checking the handle.
    if (!IsValidPassHandle(pass)) {
        return std::unexpected(GraphBuildError::InvalidPassHandle);
    }
    if (!IsValidResourceHandle(resource)) {
        return std::unexpected(GraphBuildError::InvalidResourceHandle);
    }

    LOGIFACE_LOG(trace, "Adding write to pass '" + passes_[pass.index].name + "' for resource '" + resources_[resource.index].name + "'");

    auto& pass_node = passes_[pass.index];
    if (!ContainsResource(pass_node.writes, resource)) {
        pass_node.writes.push_back(resource);
    }

    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::AddDependency(PassHandle before, PassHandle after) {
    if (!IsValidPassHandle(before)) {
        return std::unexpected(GraphBuildError::InvalidPassHandle);
    }
    if (!IsValidPassHandle(after)) {
        return std::unexpected(GraphBuildError::InvalidPassHandle);
    }
    if (before == after) {
        return std::unexpected(GraphBuildError::SelfDependency);
    }

    const std::pair<PassHandle, PassHandle> dependency{before, after};
    if (!ContainsDependency(explicit_dependencies_, dependency)) {
        explicit_dependencies_.emplace_back(before, after);
    }

    return {};
}

std::expected<void, GraphBuildError> RenderGraphBuilder::SetPassAttachments(PassHandle pass, PassAttachmentSetup setup) {
    if (!IsValidPassHandle(pass)) {
        return std::unexpected(GraphBuildError::InvalidPassHandle);
    }

    passes_[pass.index].attachment_setup = std::move(setup);
    return {};
}

void RenderGraphBuilder::Reset() {
    resources_.clear();
    passes_.clear();
    explicit_dependencies_.clear();
}

void RenderGraphBuilder::ResetPasses() {
    passes_.clear();
    explicit_dependencies_.clear();
}

void RenderGraphBuilder::RollbackResources(std::size_t count) {
    if (count < resources_.size()) {
        resources_.resize(count);
    }
}

bool RenderGraphBuilder::IsValidResourceHandle(ResourceHandle handle) const {
    if (!handle.IsValid()) {
        return false;
    }

    if (handle.index >= resources_.size()) {
        return false;
    }

    return resources_[handle.index].generation == handle.generation;
}

bool RenderGraphBuilder::IsValidPassHandle(PassHandle handle) const {
    if (!handle.IsValid()) {
        return false;
    }

    if (handle.index >= passes_.size()) {
        return false;
    }

    return passes_[handle.index].generation == handle.generation;
}

}  // namespace VulkanEngine::RenderGraph
