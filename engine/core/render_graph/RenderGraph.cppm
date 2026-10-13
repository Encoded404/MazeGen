module;

export module VulkanEngine.RenderGraph;

import std;
import std.compat;

import vulkan;

export import VulkanShared.RenderGraphTypes;

export namespace VulkanEngine::RenderGraph {

// Structured reason a builder mutation was rejected. Mutators validate before
// touching any container, so an invalid handle can never index out of bounds.
enum class GraphBuildError : std::uint8_t {
    None,
    InvalidPassHandle,
    InvalidResourceHandle,
    IncompatibleResourceState,
    IncompatibleResourceKind,
    SelfDependency,
};

class RenderGraphBuilder {
public:
    struct ReadInfo {
        ResourceHandle resource{};
        PipelineStageIntent stage = PipelineStageIntent::FragmentShader;
        AccessIntent access = AccessIntent::Read;
    };

    ResourceHandle CreateTransientResource(std::string name, ResourceKind kind);
    ResourceHandle ImportResource(std::string name, ResourceKind kind);

    std::expected<void, GraphBuildError> SetTransientImageInfo(ResourceHandle resource, TransientImageInfo info);
    std::expected<void, GraphBuildError> SetTransientBufferInfo(ResourceHandle resource, TransientBufferInfo info);

    std::expected<void, GraphBuildError> SetInitialState(ResourceHandle resource, ResourceState state);
    std::expected<void, GraphBuildError> SetFinalState(ResourceHandle resource, ResourceState state);

    PassHandle AddPass(std::string name,
                       QueueType queue = QueueType::Graphics,
                       bool enabled = true,
                       PassExecutionCallback execute = {});

    std::expected<void, GraphBuildError> AddRead(PassHandle pass, ResourceHandle resource);
    std::expected<void, GraphBuildError> AddRead(PassHandle pass, ResourceHandle resource,
                                                               PipelineStageIntent stage, AccessIntent access);
    std::expected<void, GraphBuildError> AddWrite(PassHandle pass, ResourceHandle resource);
    std::expected<void, GraphBuildError> AddDependency(PassHandle before, PassHandle after);

    std::expected<void, GraphBuildError> SetPassAttachments(PassHandle pass, PassAttachmentSetup setup);

    // Clears the model so the same builder can be rebuilt from scratch without
    // leaking earlier slots into the new pass/resource numbering.
    void Reset();

    // Clears only passes and explicit dependencies, keeping the resource table
    // (and therefore resource handles) intact. Used by model-driven rebuilds
    // where imported/transient resource identity must survive.
    void ResetPasses();

    // Number of resource slots registered so far. Snapshot before a transaction
    // (e.g. an application pass registration) that may create transients.
    [[nodiscard]] std::size_t ResourceCount() const { return resources_.size(); }

    // Drops every resource registered after `count`, restoring a snapshot taken
    // with ResourceCount(). Only valid when no surviving pass references the
    // dropped slots; used to roll back a failed pass registration.
    void RollbackResources(std::size_t count);

    [[nodiscard]] CompiledRenderGraph Compile() const;

private:
    struct ResourceNode {
        std::string name{};
        ResourceKind kind = ResourceKind::Image;
        // Slot identity policy: a slot's generation is bumped when a tombstoned
        // slot is reused (removal + tombstones arrive with the Phase 6
        // registration API), so a stale ResourceHandle fails IsValidResourceHandle
        // instead of aliasing a different resource.
        std::uint32_t generation = 1;
        bool imported = false;
        bool transient = false;
        bool has_initial_state = false;
        bool has_final_state = false;
        ResourceState initial_state{};
        ResourceState final_state{};
        std::optional<TransientImageInfo> image_info{};
        std::optional<TransientBufferInfo> buffer_info{};
    };

    struct PassNode {
        std::string name{};
        QueueType queue = QueueType::Graphics;
        std::uint32_t generation = 1;
        bool enabled = true;
        std::vector<ReadInfo> reads{};
        std::vector<ResourceHandle> writes{};
        PassExecutionCallback execute{};
        std::optional<PassAttachmentSetup> attachment_setup{};
    };

    [[nodiscard]] bool IsValidResourceHandle(ResourceHandle handle) const;
    [[nodiscard]] bool IsValidPassHandle(PassHandle handle) const;
    [[nodiscard]] std::uint32_t FindResourceByName(std::string_view name, ResourceKind kind) const;

    std::vector<ResourceNode> resources_{};
    std::vector<PassNode> passes_{};
    std::vector<std::pair<PassHandle, PassHandle>> explicit_dependencies_{};
};

}  // namespace VulkanEngine::RenderGraph
