module;

export module VulkanEngine.RenderPipeline;

import std;
import std.compat;

import vulkan;

export import VulkanEngine.RenderGraph;
export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.PipelinePass;

import VulkanEngine.GpuResources.TransientAllocator;
import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderManager;

export namespace VulkanEngine::RenderPipeline {

// TransientImageDesc is now provided by the VulkanEngine.PipelinePass module.
using VulkanEngine::PipelinePass::TransientImageDesc;
using VulkanEngine::PipelinePass::TransientBufferDesc;

// ── Structured registration errors ──
enum class PassErrorCode : std::uint8_t {
    None,
    DuplicateName,
    InvalidHandle,
    MissingResolver,
    InvalidDeclaration,
    SetupFailed,
    ValidationFailed,
    CompileFailed,
    PipelineFailed,
};

struct PassError {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    PassErrorCode code = PassErrorCode::None;
    std::string message{};
    std::string pass{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class RenderPipeline : public VulkanEngine::PipelinePass::IResourceRegistry {
public:
    RenderPipeline();
    ~RenderPipeline() override;

    void Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                    ShaderSystem::ShaderManager* shader_manager = nullptr,
                    ShaderSystem::PipelineFactory* pipeline_factory = nullptr);
    void Shutdown();

    // Engine-standard descriptor set layouts (sets 0-4) used to build pass
    // pipeline layouts. Passes that declared a pipeline before this call had it
    // deferred, so setting the layouts builds them; a pass registered afterwards
    // has them available from the next ApplyChanges(). Either order leaves every
    // declared pass with a pipeline before the first frame is recorded.
    void SetEngineDescriptorSetLayouts(std::array<vk::DescriptorSetLayout, 5> layouts);

    // ── IResourceRegistry overrides ──
    VulkanEngine::RenderGraph::ResourceHandle ImportBackbuffer() override;
    VulkanEngine::RenderGraph::ResourceHandle ImportDepthBuffer() override;
    VulkanEngine::RenderGraph::ResourceHandle ImportImage(const std::string& name) override;
    VulkanEngine::RenderGraph::ResourceHandle ImportBuffer(const std::string& name) override;
    VulkanEngine::RenderGraph::ResourceHandle CreateTransientImage(const TransientImageDesc& desc) override;
    VulkanEngine::RenderGraph::ResourceHandle CreateTransientBuffer(const TransientBufferDesc& desc) override;

    using ImageResolver = std::function<vk::Image(std::uint32_t image_index)>;
    using ImageViewResolver = std::function<vk::ImageView(std::uint32_t image_index)>;
    using BufferResolver = std::function<vk::Buffer(std::uint32_t image_index)>;
    void RegisterResourceResolver(const std::string& name,
                                  ImageResolver resolve_image,
                                  ImageViewResolver resolve_image_view,
                                  vk::Format format);

    // Registers the concrete buffer backing an imported Buffer resource. Until
    // a resolver exists the planned barrier for that resource falls back to a
    // conservative global memory barrier instead of being skipped.
    void RegisterBufferResolver(const std::string& name, BufferResolver resolve_buffer);

    // ── Engine-managed registration ──
    // The pipeline constructs the PassSetupContext (with the current render
    // extent) and calls Setup() itself, validates the result, and tracks the
    // pass in the stable model. Changes apply at the next ApplyChanges().
    [[nodiscard]] std::expected<VulkanEngine::RenderGraph::PassHandle, PassError> RegisterPass(
        std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass> pass);
    // Marks a pass dead (tombstone); its pass object is freed once the frame
    // that last used it is GPU-complete.
    [[nodiscard]] bool RemovePass(VulkanEngine::RenderGraph::PassHandle handle);
    [[nodiscard]] bool SetPassEnabled(VulkanEngine::RenderGraph::PassHandle handle, bool enabled);
    // Marks the model dirty; ApplyChanges() rebuilds and recompiles.
    void RequestRebuild();
    [[nodiscard]] const std::vector<VulkanEngine::RenderGraph::CompileDiagnostic>& GetDiagnostics() const;
    // Structured errors from the last model validation (missing resolvers,
    // duplicate names, invalid declarations).
    [[nodiscard]] const std::vector<PassError>& GetValidationErrors() const;
    // Increments on every model rebuild; unchanged when ApplyChanges() is a no-op.
    [[nodiscard]] std::uint32_t GetRevision() const { return revision_; }
    // Rebuilds the graph from the pass model if dirty and recompiles. Called at
    // a frame boundary (top of Renderer::RenderFrame).
    void ApplyChanges();
    void SetRenderExtent(std::uint32_t width, std::uint32_t height);
    // Number of frames-in-flight slots app-pass descriptor sets are allocated
    // for. Must be set before the first ApplyChanges/Compile.
    void SetFramesInFlight(std::uint32_t frames_in_flight);
    // Queue families transient resources are shared with; more than one enables
    // concurrent sharing so passes on multiple queues can access them.
    void SetQueueFamilies(std::span<const std::uint32_t> families);
    // Whether a dedicated async compute queue exists. Compute passes are
    // rejected when it does not.
    void SetAsyncComputeAvailable(bool available) { async_compute_available_ = available; }
    [[nodiscard]] bool IsAsyncComputeAvailable() const { return async_compute_available_; }
    // Swapchain images were destroyed/recreated (and possibly the image count
    // changed). Clears per-image imported-state tracking and forces the next
    // frame to start every imported image from Undefined.
    void OnSwapchainRecreated(std::uint32_t image_count);

    // ── Built-in pass handle access ──
    // The caller (Renderer) populates these after registering all built-in passes.
    void SetBuiltinHandles(const std::array<VulkanEngine::RenderGraph::PassHandle,
                                               VulkanEngine::PipelinePass::kBuiltinPassCount>& handles);
    [[nodiscard]] const std::array<VulkanEngine::RenderGraph::PassHandle,
                                   VulkanEngine::PipelinePass::kBuiltinPassCount>& GetBuiltinHandles() const;

    bool AddDependency(VulkanEngine::RenderGraph::PassHandle before,
                       VulkanEngine::RenderGraph::PassHandle after);

    bool SetInitialState(VulkanEngine::RenderGraph::ResourceHandle resource, VulkanEngine::RenderGraph::ResourceState state);
    bool SetFinalState(VulkanEngine::RenderGraph::ResourceHandle resource, VulkanEngine::RenderGraph::ResourceState state);

    void Compile();

    // Multi-queue recording: BeginFrame resolves resources and attaches this
    // frame's data; RecordRun records one queue run into its own command buffer;
    // EndFrame releases execution-time mutation blocking. A frame is always
    // recorded run-by-run so per-queue command buffers and barrier scopes stay
    // correct; there is deliberately no single-command-buffer convenience path.
    void BeginFrame(const void* user_data, std::uint32_t image_index, std::uint32_t fif_slot);
    void RecordRun(std::uint32_t run_index, vk::CommandBuffer command_buffer);
    void EndFrame();
    [[nodiscard]] const VulkanEngine::RenderGraph::QueueRunPlan& GetQueueRuns() const { return queue_runs_; }

    [[nodiscard]] bool IsCompiled() const { return compiled_; }
    [[nodiscard]] const VulkanEngine::RenderGraph::CompiledRenderGraph& GetCompiledGraph() const { return compiled_graph_; }

    // Engine-owned pipeline declaration for a pass (nullptr when none declared).
    [[nodiscard]] const VulkanEngine::PipelinePass::PassPipelineRequest* GetPassPipelineRequest(
        VulkanEngine::RenderGraph::PassHandle handle) const;
    [[nodiscard]] const VulkanEngine::PipelinePass::PassPipelineRequest* GetPassPipelineRequestByName(
        std::string_view name) const;
    [[nodiscard]] vk::PipelineLayout GetPassPipelineLayout(VulkanEngine::RenderGraph::PassHandle handle) const;
    [[nodiscard]] vk::PipelineLayout GetPassPipelineLayoutByName(std::string_view name) const;

private:
    // ── Pass model (stable identity, rebuildable) ──
    struct ModelRead {
        VulkanEngine::RenderGraph::ResourceHandle resource{};
        VulkanEngine::RenderGraph::PipelineStageIntent stage =
            VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader;
        VulkanEngine::RenderGraph::AccessIntent access = VulkanEngine::RenderGraph::AccessIntent::Read;
    };

    struct ModelPass {
        std::string name{};
        std::uint32_t slot = 0;
        std::uint32_t generation = 1;
        bool enabled = true;
        bool builtin = false;
        bool alive = true; // false == tombstone
        VulkanEngine::RenderGraph::QueueType queue = VulkanEngine::RenderGraph::QueueType::Graphics;
        std::vector<ModelRead> reads{};
        std::vector<VulkanEngine::RenderGraph::ResourceHandle> writes{};
        std::optional<VulkanEngine::RenderGraph::PassAttachmentSetup> attachments{};
        VulkanEngine::PipelinePass::PassPipelineRequest pipeline_request{};
        std::vector<VulkanEngine::Render::DescriptorDecl> declared_bindings{};
        std::vector<VulkanEngine::PipelinePass::BindingAssignment> binding_assignments{};
        std::function<void(const void*, vk::CommandBuffer)> execute{};
        std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass> pass{};
    };

    void SyncTransients();
    void BuildPassPipelines();
    void PollPassPipelines(std::uint32_t fif_slot);
    void RewirePassDescriptors(std::uint32_t slot, VulkanEngine::PipelinePass::FrameContext& frame);
    [[nodiscard]] vk::Format ResolveResourceFormat(VulkanEngine::RenderGraph::ResourceHandle resource) const;
    void RebuildFromModel();
    void ResolveResources(VulkanEngine::RenderGraph::CompiledRenderGraph& graph,
                          std::uint32_t image_index, std::uint32_t fif_slot);
    [[nodiscard]] bool ValidateModel(std::vector<PassError>& errors) const;
    void CollectRemovedPasses(std::uint32_t fif_slot);
    [[nodiscard]] ModelPass* FindModelPass(VulkanEngine::RenderGraph::PassHandle handle);
    [[nodiscard]] bool IsValidModelHandle(VulkanEngine::RenderGraph::PassHandle handle) const;
    [[nodiscard]] VulkanEngine::RenderGraph::PassHandle AddModelPass(ModelPass model);
    void TrackImportedResource(VulkanEngine::RenderGraph::ResourceHandle handle, const std::string& name,
                               VulkanEngine::RenderGraph::ResourceKind kind);
    [[nodiscard]] bool IsImportedResource(VulkanEngine::RenderGraph::ResourceHandle handle) const;
    // Rolls the graph builder and the pipeline's resource bookkeeping back to a
    // ResourceCount() snapshot taken before a pass's Setup(), so a registration
    // that is rejected after Setup() does not leave orphaned transients.
    void RollbackPassSetupResources(std::size_t resource_count);

    VulkanBackend::Vulkan::VulkanBootstrap* bootstrap_ = nullptr;
    ShaderSystem::ShaderManager* shader_manager_ = nullptr;
    ShaderSystem::PipelineFactory* pipeline_factory_ = nullptr;
    std::array<vk::DescriptorSetLayout, 5> engine_set_layouts_{};
    VulkanEngine::RenderGraph::RenderGraphBuilder graph_builder_{};
    VulkanEngine::RenderGraph::CompiledRenderGraph compiled_graph_{};
    // Set by BeginFrame(); consumed by RecordRun() while the frame is executing.
    VulkanEngine::RenderGraph::BarrierPlan barrier_plan_{};
    VulkanEngine::RenderGraph::QueueRunPlan queue_runs_{};
    VulkanEngine::PipelinePass::RenderFrameData frame_data_{};

    std::vector<ModelPass> model_passes_{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> model_dependencies_{}; // slots
    // Removed pass objects awaiting GPU completion before destruction.
    std::vector<std::pair<std::unique_ptr<VulkanEngine::PipelinePass::IPipelinePass>, std::uint32_t>>
        pending_removals_{};
    bool dirty_ = true;
    bool applying_ = false;
    // True while Execute() records the frame; mutations are rejected then.
    bool executing_ = false;
    std::uint32_t revision_ = 0;
    std::uint32_t render_width_ = 0;
    std::uint32_t render_height_ = 0;
    std::uint32_t frames_in_flight_ = 3;
    bool async_compute_available_ = false;
    std::uint32_t last_fif_slot_ = 0;
    // A render-extent change queues a one-shot OnRenderResize notification for
    // registered passes, drained at the next ApplyChanges().
    bool resize_pending_ = false;
    std::uint32_t resize_width_ = 0;
    std::uint32_t resize_height_ = 0;

    std::unordered_map<std::uint32_t, TransientImageDesc> transient_image_descs_{};
    std::unordered_map<std::uint32_t, TransientBufferDesc> transient_buffer_descs_{};
    VulkanEngine::GpuResources::TransientAllocator transient_allocator_{};

    struct ExternalResourceResolver {
        ImageResolver resolve_image;
        ImageViewResolver resolve_image_view;
        vk::Format format = vk::Format::eUndefined;
    };
    std::unordered_map<std::string, ExternalResourceResolver> resource_resolvers_{};
    std::unordered_map<std::string, BufferResolver> buffer_resolvers_{};
    std::unordered_set<std::string> image_resolver_names_{};
    std::unordered_set<std::string> buffer_resolver_names_{};

    // Resource identity: index -> registered name, for resolver validation.
    std::unordered_map<std::uint32_t, std::string> resource_names_{};
    std::unordered_set<std::uint32_t> imported_resource_indices_{};
    // Imported resource kind. Imported images require a resolver (they need a
    // real image/view); imported buffers may be hazard-only engine resources
    // with no Vulkan handle.
    std::unordered_map<std::uint32_t, VulkanEngine::RenderGraph::ResourceKind> imported_resource_kinds_{};
    std::vector<PassError> validation_errors_{};

    VulkanEngine::RenderGraph::ResourceHandle backbuffer_handle_{};
    VulkanEngine::RenderGraph::ResourceHandle depth_buffer_handle_{};

    std::uint32_t backbuffer_resource_index_ = 0;
    std::uint32_t depth_buffer_resource_index_ = 0;

    // Per swapchain image (outer index) and per resource index (inner), the
    // actual end-of-frame layout recorded last time that image was used. Seeded
    // back into the barrier plan so imported layouts are not reset to Undefined.
    std::vector<std::vector<VulkanEngine::RenderGraph::ResourceState>> tracked_states_{};
    std::vector<std::vector<bool>> tracked_valid_{};
    std::uint32_t tracked_resource_count_ = 0;

    // Built-in ordering anchors populated by the renderer.
    std::array<VulkanEngine::RenderGraph::PassHandle,
               VulkanEngine::PipelinePass::kBuiltinPassCount> builtin_handles_{};

    // Engine-owned pass pipelines: one retire ring per pass that declared a
    // pipeline request. The engine builds the VkPipeline from the request (and
    // hot-reloads it); the pass never owns a pipeline.
    struct PassPipelineState {
        std::string name{};
        VulkanEngine::PipelinePass::PassPipelineRequest request{};
        std::vector<VulkanEngine::Render::DescriptorDecl> bindings{};
        std::vector<VulkanEngine::PipelinePass::BindingAssignment> assignments{};
        std::optional<VulkanEngine::RenderGraph::PassAttachmentSetup> attachments{};
        ShaderSystem::PipelineSlot slot{};
        bool built = false;
        // Descriptor set layouts must outlive the pipeline layout that references
        // them, so the state owns both.
        std::vector<vk::raii::DescriptorSetLayout> app_set_layouts{};
        std::unique_ptr<vk::raii::PipelineLayout> layout{};
        // One descriptor set per declared app set per frame-in-flight slot. The
        // engine rewrites the slot's set from the resolved resources before that
        // frame is recorded, so no update-after-bind binding (and its device
        // descriptor limits) is required.
        std::unique_ptr<vk::raii::DescriptorPool> app_pool{};
        std::vector<std::vector<vk::raii::DescriptorSet>> app_sets{};
        std::vector<std::vector<vk::DescriptorSet>> app_set_handles{};
        std::vector<std::uint32_t> app_set_numbers{};
        struct DescriptorBindingState {
            vk::ImageView view = nullptr;
            vk::Buffer buffer = nullptr;
            vk::Sampler sampler = nullptr;
            vk::DeviceSize offset = 0;
            vk::DeviceSize size = 0;
            vk::ImageLayout layout = vk::ImageLayout::eUndefined;
            bool valid = false;
        };
        std::vector<std::vector<DescriptorBindingState>> last_written{};
        // The desc owns its color-blend attachments (PipelineFactory rebinds
        // pAttachments at each entry), so hot reload reuses it safely.
        ShaderSystem::GraphicsPipelineDesc graphics_desc{};
        ShaderSystem::ComputePipelineDesc compute_desc{};
    };
    std::unordered_map<std::uint32_t, PassPipelineState> pass_pipelines_{};
    std::unordered_map<std::string, std::uint32_t> pass_pipeline_by_name_{};

    // Per-frame name -> resolved-handle table handed to passes via FrameContext.
    VulkanEngine::PipelinePass::ResourceLookupTable frame_lookup_{};

    bool compiled_ = false;
    bool initialized_ = false;
};

} // namespace VulkanEngine::RenderPipeline
