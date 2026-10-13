module;
export module VulkanShared.RenderGraphTypes;
import std;
import vulkan;

export namespace VulkanEngine::RenderGraph {

enum class ResourceKind : std::uint8_t {
    Image,
    Buffer
};

struct ResourceHandle {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsValid() const noexcept {
        return index != std::numeric_limits<std::uint32_t>::max();
    }

    friend bool operator==(const ResourceHandle&, const ResourceHandle&) = default;
};

struct PassHandle {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsValid() const noexcept {
        return index != std::numeric_limits<std::uint32_t>::max();
    }

    friend bool operator==(const PassHandle&, const PassHandle&) = default;
};

// ── Execution model enums ──

enum class PipelineStageIntent : std::uint8_t {
    None,
    Transfer,
    ColorAttachment,
    DepthAttachment,
    VertexShader,
    // Vertex/index fetch: VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, distinct from the
    // vertex-shader stage. Also covers the vertex shader reads on draw passes
    // so one intent synchronizes a compute-written buffer consumed as both an
    // index buffer and a shader resource.
    IndexInput,
    IndirectDraw,
    FragmentShader,
    ComputeShader,
    Present,
    TopOfPipe,
    BottomOfPipe
};

enum class AccessIntent : std::uint8_t {
    None,
    Read,
    Write,
    ReadWrite
};

enum class ImageLayoutIntent : std::uint8_t {
    Undefined,
    General,
    ColorAttachment,
    DepthAttachment,
    ShaderReadOnly,
    TransferSource,
    TransferDestination,
    Present,
    DepthReadOnly
};

enum class QueueType : std::uint8_t {
    Graphics,
    Compute,
    Transfer
};

// ── Resource state tracking ──

struct ResourceState {
    PipelineStageIntent stage = PipelineStageIntent::None;
    AccessIntent access = AccessIntent::None;
    QueueType queue = QueueType::Graphics;
    ImageLayoutIntent layout = ImageLayoutIntent::Undefined;
    bool has_image_layout = false;

    [[nodiscard]] static ResourceState BufferState(PipelineStageIntent stage_intent,
                                                    AccessIntent access_intent,
                                                    QueueType queue_intent = QueueType::Graphics) {
        return ResourceState{
            .stage = stage_intent,
            .access = access_intent,
            .queue = queue_intent,
            .layout = ImageLayoutIntent::Undefined,
            .has_image_layout = false,
        };
    }

    [[nodiscard]] static ResourceState ImageState(PipelineStageIntent stage_intent,
                                                   AccessIntent access_intent,
                                                   QueueType queue_intent,
                                                   ImageLayoutIntent layout_intent) {
        return ResourceState{
            .stage = stage_intent,
            .access = access_intent,
            .queue = queue_intent,
            .layout = layout_intent,
            .has_image_layout = true,
        };
    }
};

// ── Compiler diagnostics ──

enum class DiagnosticSeverity : std::uint8_t {
    Info,
    Warning,
    Error
};

enum class DiagnosticCode : std::uint8_t {
    None,
    EmptyGraph,
    InvalidExplicitDependency,
    InvalidReadHandle,
    InvalidWriteHandle,
    InvalidInitialState,
    InvalidFinalState,
    CycleDetected,
    CompileSuccess
};

struct CompileDiagnostic {
    DiagnosticCode code = DiagnosticCode::None;
    DiagnosticSeverity severity = DiagnosticSeverity::Info;
    std::string message{};
};

// ── Transient resource descriptions ──

struct TransientImageInfo {
    vk::Format format = vk::Format::eUndefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
    // Reserved for Phase 2 interval aliasing. Aliasable images must be created
    // with VK_IMAGE_CREATE_ALIAS_BIT and must carry an Undefined initial
    // layout, which Phase 2 enforces.
    bool aliasable = false;
};

struct TransientBufferInfo {
    std::string name{};
    vk::DeviceSize size = 0;
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
    vk::MemoryPropertyFlags memory_properties = vk::MemoryPropertyFlagBits::eDeviceLocal;
    bool aliasable = false;
};

// ── Resource metadata ──

struct ResourceInfo {
    std::string name{};
    ResourceKind kind = ResourceKind::Image;
    bool imported = false;
    std::optional<TransientImageInfo> image_info{};
    std::optional<TransientBufferInfo> buffer_info{};
};

// ── Pass execution ──

struct PassExecutionCallback {
    std::function<void(const void* user_data, vk::CommandBuffer command_buffer)> callback{};
};

// ── Attachment types ──

struct AttachmentInfo {
    ResourceHandle resource{};
    vk::ImageView image_view = {};
    vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eClear;
    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore;
    vk::ClearColorValue clear_color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    vk::ClearDepthStencilValue clear_depth = vk::ClearDepthStencilValue(1.0f, 0);
};

struct PassAttachmentSetup {
    std::vector<AttachmentInfo> color_attachments{};
    std::optional<AttachmentInfo> depth_attachment{};
    vk::Rect2D render_area = {};
    bool auto_begin_rendering = false;
};

// ── Planned transitions ──
//
// A ResourceTransition is resolved entirely by the compiler: `from_state` is
// the state the resource was left in by the previous use (or the Undefined
// state when `from_known` is false), and the stage/access flag pairs are the
// OR of every use this pass makes of the resource. Read+write on the same
// resource in one pass therefore collapses into a single transition, which is
// what makes the emitted barrier correct for a combined read/write hazard.
struct ResourceTransition {
    std::uint32_t resource_index = std::numeric_limits<std::uint32_t>::max();
    ResourceState from_state{};
    ResourceState target_state{};
    bool from_known = false;
    vk::PipelineStageFlags2 src_stage{};
    vk::AccessFlags2 src_access{};
    vk::PipelineStageFlags2 dst_stage{};
    vk::AccessFlags2 dst_access{};
};

// ── Compiled pass description ──

struct CompiledPass {
    PassHandle handle{};
    std::string name{};
    QueueType queue = QueueType::Graphics;
    std::vector<ResourceTransition> pre_pass_transitions{};
    std::vector<ResourceTransition> post_pass_transitions{};
    PassExecutionCallback execute{};
    std::optional<PassAttachmentSetup> attachment_setup{};
};

// ── Resource lifetime tracking (compiler output) ──

struct ResourceLifetime {
    ResourceHandle handle{};
    std::string name{};
    bool imported = false;
    bool transient = false;
    std::int32_t first_pass = -1;
    std::int32_t last_pass = -1;
};

// ── Compiled render graph (execution plan) ──

struct CompiledRenderGraph {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    bool success = false;
    std::vector<CompileDiagnostic> diagnostics{};
    std::vector<CompiledPass> passes{};
    std::vector<ResourceLifetime> resource_lifetimes{};
    std::vector<ResourceInfo> resource_info{};
    mutable std::vector<ResourceState> initial_states{};
    mutable std::vector<bool> has_initial_state{};
    std::vector<vk::Image> resource_images{};
    std::vector<vk::Buffer> resource_buffers{};
    std::vector<vk::DeviceSize> resource_buffer_offsets{};
    std::vector<vk::DeviceSize> resource_buffer_sizes{};
    std::vector<vk::Format> resource_formats{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    void SetImportedResourceState(std::uint32_t resource_index, ResourceState state) const {
        if (resource_index < initial_states.size()) {
            initial_states[resource_index] = state;
            has_initial_state[resource_index] = true;
        }
    }

    void SetResourceImage(std::uint32_t resource_index, vk::Image image) {
        if (resource_index < resource_images.size()) {
            resource_images[resource_index] = image;
        }
    }

    void SetResourceBuffer(std::uint32_t resource_index, vk::Buffer buffer,
                           vk::DeviceSize offset = 0, vk::DeviceSize size = vk::WholeSize) {
        if (resource_index < resource_buffers.size()) {
            resource_buffers[resource_index] = buffer;
        }
        if (resource_index < resource_buffer_offsets.size()) {
            resource_buffer_offsets[resource_index] = offset;
        }
        if (resource_index < resource_buffer_sizes.size()) {
            resource_buffer_sizes[resource_index] = size;
        }
    }

    void SetResourceFormat(std::uint32_t resource_index, vk::Format format) {
        if (resource_index < resource_formats.size()) {
            resource_formats[resource_index] = format;
        }
    }
};

// ── Resolved handles + barrier plan (pure planner output) ──

// Vulkan handles resolved at the execute boundary, indexed by resource index.
// Kept separate from CompiledRenderGraph so the planner stays a pure function
// over graph data + handles and never touches the device.
struct ResolvedResourceHandles {
    std::vector<vk::Image> images{};
    std::vector<vk::Buffer> buffers{};
    // Sub-allocation window for buffers living inside a heap block; a whole
    // dedicated buffer leaves these empty (barrier defaults to offset 0/whole).
    std::vector<vk::DeviceSize> buffer_offsets{};
    std::vector<vk::DeviceSize> buffer_sizes{};
    std::vector<vk::Format> formats{};
};

// Reserved for Phase 2. Empty in Phase 1: no alias reuse is planned, but the
// planner signature and BarrierPlan carry the seam so Phase 2 can fill it
// without an API break.
struct PlannedAliasDependency {
    std::uint32_t aliased_resource = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t after_resource = std::numeric_limits<std::uint32_t>::max();
    std::int32_t pass_index = -1;
};

struct AliasIntervals {
    std::vector<PlannedAliasDependency> dependencies{};
};

struct PlannedImageBarrier {
    std::uint32_t resource_index = std::numeric_limits<std::uint32_t>::max();
    vk::Image image = {};
    vk::PipelineStageFlags2 src_stage{};
    vk::PipelineStageFlags2 dst_stage{};
    vk::AccessFlags2 src_access{};
    vk::AccessFlags2 dst_access{};
    vk::ImageLayout old_layout = vk::ImageLayout::eUndefined;
    vk::ImageLayout new_layout = vk::ImageLayout::eUndefined;
    vk::ImageSubresourceRange range{};
};

struct PlannedBufferBarrier {
    std::uint32_t resource_index = std::numeric_limits<std::uint32_t>::max();
    // Null when the engine could not resolve a concrete buffer for this
    // resource. The executor then falls back to a conservative global memory
    // barrier with the same scopes; correctness is preserved, only the
    // precision is lost until a resolver is registered.
    vk::Buffer buffer = {};
    vk::PipelineStageFlags2 src_stage{};
    vk::PipelineStageFlags2 dst_stage{};
    vk::AccessFlags2 src_access{};
    vk::AccessFlags2 dst_access{};
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = vk::WholeSize;
};

struct PlannedPassBarriers {
    std::uint32_t pass_index = std::numeric_limits<std::uint32_t>::max();
    std::string pass_name{};
    // Queue this pass (and therefore its barriers) runs on. Planned scopes are
    // clamped to it, so a compute/transfer run never names a graphics scope.
    QueueType queue = QueueType::Graphics;
    std::vector<PlannedImageBarrier> pre_image{};
    std::vector<PlannedBufferBarrier> pre_buffer{};
    std::vector<PlannedImageBarrier> post_image{};
    std::vector<PlannedBufferBarrier> post_buffer{};
};

struct BarrierPlan {
    bool valid = false;
    std::vector<PlannedPassBarriers> passes{};
    // Actual end-of-frame state produced by walking every planned transition.
    // The render pipeline records this and feeds it back as the next frame's
    // runtime old-state, so imported layouts survive across frames.
    std::vector<ResourceState> end_states{};
    std::vector<bool> has_end_state{};

    [[nodiscard]] bool HasBarriers() const {
        for (const auto& pass : passes) {
            if (!pass.pre_image.empty() || !pass.pre_buffer.empty() ||
                !pass.post_image.empty() || !pass.post_buffer.empty()) {
                return true;
            }
        }
        return false;
    }
};

// ── Helper functions for state mapping ──

inline bool StatesEqual(const ResourceState& a, const ResourceState& b) {
    if (a.stage != b.stage) return false;
    if (a.access != b.access) return false;
    if (a.queue != b.queue) return false;
    if (a.has_image_layout != b.has_image_layout) return false;
    if (a.has_image_layout && a.layout != b.layout) return false;
    return true;
}

// sync2 (core in Vulkan 1.3). TopOfPipe/BottomOfPipe/Present collapse to eNone:
// the deprecated top/bottom stages have no sync2 equivalent, and an Undefined
// source needs no source scope. Callers must OR these into 64-bit masks.
inline vk::PipelineStageFlags2 IntentToPipelineStage(PipelineStageIntent intent, AccessIntent access) {
    (void)access;
    switch (intent) {
        case PipelineStageIntent::None:
        case PipelineStageIntent::Present:
        case PipelineStageIntent::TopOfPipe:
        case PipelineStageIntent::BottomOfPipe:
            return vk::PipelineStageFlagBits2::eNone;
        case PipelineStageIntent::Transfer:
            return vk::PipelineStageFlagBits2::eTransfer;
        case PipelineStageIntent::ColorAttachment:
            return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        case PipelineStageIntent::DepthAttachment:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                   vk::PipelineStageFlagBits2::eLateFragmentTests;
        case PipelineStageIntent::VertexShader:
            return vk::PipelineStageFlagBits2::eVertexShader;
        case PipelineStageIntent::IndexInput:
            return vk::PipelineStageFlagBits2::eVertexInput |
                   vk::PipelineStageFlagBits2::eVertexShader;
        case PipelineStageIntent::FragmentShader:
            return vk::PipelineStageFlagBits2::eFragmentShader;
        case PipelineStageIntent::ComputeShader:
            return vk::PipelineStageFlagBits2::eComputeShader;
        case PipelineStageIntent::IndirectDraw:
            return vk::PipelineStageFlagBits2::eDrawIndirect;
    }
    return vk::PipelineStageFlagBits2::eNone;
}

inline vk::AccessFlags2 IntentToAccessFlags(PipelineStageIntent stage, AccessIntent access) {
    if (access == AccessIntent::None) return {};

    auto stage_access = [](PipelineStageIntent s, bool is_write) -> vk::AccessFlags2 {
        switch (s) {
            case PipelineStageIntent::Transfer:
                return is_write ? vk::AccessFlagBits2::eTransferWrite
                                : vk::AccessFlagBits2::eTransferRead;
            case PipelineStageIntent::ColorAttachment:
                return vk::AccessFlagBits2::eColorAttachmentWrite;
            case PipelineStageIntent::DepthAttachment:
                return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            case PipelineStageIntent::VertexShader:
            case PipelineStageIntent::FragmentShader:
            case PipelineStageIntent::ComputeShader:
                return is_write ? vk::AccessFlagBits2::eShaderWrite
                                : vk::AccessFlagBits2::eShaderRead;
            case PipelineStageIntent::IndexInput:
                return is_write ? vk::AccessFlagBits2::eShaderWrite
                                : (vk::AccessFlagBits2::eIndexRead |
                                   vk::AccessFlagBits2::eShaderRead);
            case PipelineStageIntent::IndirectDraw:
                return vk::AccessFlagBits2::eIndirectCommandRead;
            case PipelineStageIntent::Present:
            default:
                return {};
        }
    };

    return stage_access(stage, access == AccessIntent::Write || access == AccessIntent::ReadWrite);
}

// ── Queue synchronization scope support ──
//
// A barrier recorded into a command buffer may only name pipeline stages and
// access types supported by that command buffer's queue family. The barrier
// planner clamps each pass's planned scopes to the pass's queue so a compute or
// transfer run never emits a graphics-only scope. Cross-queue ordering is not
// lost by this clamp: the queue-run boundary semaphore carries the producer /
// consumer dependency, so a scope that refers to the other queue is vacuous in
// this command buffer and is replaced with eNone.
//
// Graphics queues support every stage this engine emits, so they are identity.

[[nodiscard]] inline bool QueueSupportsAllStages(QueueType queue) {
    return queue == QueueType::Graphics;
}

// Stage mask supported by a non-graphics queue. Callers must not use this as a
// support mask for a graphics queue (see QueueSupportsAllStages).
[[nodiscard]] inline vk::PipelineStageFlags2 QueueSupportedStages(QueueType queue) {
    switch (queue) {
        case QueueType::Compute:
            // Indirect dispatch consumes its parameters at the draw-indirect
            // stage; transfer and compute stages cover the rest of compute work.
            return vk::PipelineStageFlagBits2::eComputeShader |
                   vk::PipelineStageFlagBits2::eTransfer |
                   vk::PipelineStageFlagBits2::eDrawIndirect;
        case QueueType::Transfer:
            return vk::PipelineStageFlagBits2::eTransfer;
        case QueueType::Graphics:
        default:
            return vk::PipelineStageFlagBits2::eNone;
    }
}

// Access types that only a graphics command buffer may name. Indirect-command
// reads stay: dispatch-indirect is compute-valid.
[[nodiscard]] inline vk::AccessFlags2 QueueUnsupportedAccess() {
    return vk::AccessFlagBits2::eColorAttachmentRead |
           vk::AccessFlagBits2::eColorAttachmentWrite |
           vk::AccessFlagBits2::eDepthStencilAttachmentRead |
           vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
           vk::AccessFlagBits2::eInputAttachmentRead |
           vk::AccessFlagBits2::eIndexRead |
           vk::AccessFlagBits2::eVertexAttributeRead;
}

// Drops stages a `queue` cannot name. An emptied mask becomes eNone, which sync2
// permits (the barrier still performs its layout transition).
[[nodiscard]] inline vk::PipelineStageFlags2 ClampStagesToQueue(vk::PipelineStageFlags2 stages,
                                                               QueueType queue) {
    if (QueueSupportsAllStages(queue)) {
        return stages;
    }
    const vk::PipelineStageFlags2 clamped = stages & QueueSupportedStages(queue);
    return clamped == vk::PipelineStageFlags2{} ? vk::PipelineStageFlagBits2::eNone : clamped;
}

// Drops access types a `queue` cannot name.
[[nodiscard]] inline vk::AccessFlags2 ClampAccessToQueue(vk::AccessFlags2 access, QueueType queue) {
    if (QueueSupportsAllStages(queue)) {
        return access;
    }
    return access & ~QueueUnsupportedAccess();
}

// Pipeline stages that produce shader access. An access mask is only meaningful
// alongside a stage that can generate it (VUID-VkBufferMemoryBarrier2-srcAccessMask-07454
// and friends), so stage and access must be clamped as a pair.
[[nodiscard]] inline vk::PipelineStageFlags2 ShaderStages() {
    return vk::PipelineStageFlagBits2::eVertexShader |
           vk::PipelineStageFlagBits2::eTessellationControlShader |
           vk::PipelineStageFlagBits2::eTessellationEvaluationShader |
           vk::PipelineStageFlagBits2::eGeometryShader |
           vk::PipelineStageFlagBits2::eFragmentShader |
           vk::PipelineStageFlagBits2::eComputeShader;
}

// A queue-clamped barrier scope. Stage and access travel together so an access
// bit can never survive without a stage that supports it.
struct ClampedScopes {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    vk::PipelineStageFlags2 stage = vk::PipelineStageFlagBits2::eNone;
    vk::AccessFlags2 access{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Clamps a barrier's (stage, access) pair to what a command buffer on `queue`
// may name. Graphics is identity. On a non-graphics queue the stage mask is
// reduced to the queue's supported stages and the access mask is reduced to
// accesses those remaining stages can produce; an emptied stage mask clears the
// access mask entirely.
[[nodiscard]] inline ClampedScopes ClampScopesToQueue(vk::PipelineStageFlags2 stage,
                                                      vk::AccessFlags2 access,
                                                      QueueType queue) {
    if (QueueSupportsAllStages(queue)) {
        return ClampedScopes{stage, access};
    }

    const vk::PipelineStageFlags2 clamped_stage = ClampStagesToQueue(stage, queue);
    vk::AccessFlags2 clamped_access = ClampAccessToQueue(access, queue);
    if ((clamped_stage & ShaderStages()) == vk::PipelineStageFlags2{}) {
        clamped_access &= ~(vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }
    if ((clamped_stage & vk::PipelineStageFlagBits2::eTransfer) == vk::PipelineStageFlags2{}) {
        clamped_access &= ~(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
    }
    if ((clamped_stage & vk::PipelineStageFlagBits2::eDrawIndirect) == vk::PipelineStageFlags2{}) {
        clamped_access &= ~vk::AccessFlagBits2::eIndirectCommandRead;
    }
    return ClampedScopes{clamped_stage, clamped_access};
}

inline vk::ImageLayout IntentToImageLayout(ImageLayoutIntent intent) {
    switch (intent) {
        case ImageLayoutIntent::Undefined:
            return vk::ImageLayout::eUndefined;
        case ImageLayoutIntent::General:
            return vk::ImageLayout::eGeneral;
        case ImageLayoutIntent::ColorAttachment:
            return vk::ImageLayout::eColorAttachmentOptimal;
        case ImageLayoutIntent::DepthAttachment:
            return vk::ImageLayout::eDepthAttachmentOptimal;
        case ImageLayoutIntent::ShaderReadOnly:
            return vk::ImageLayout::eShaderReadOnlyOptimal;
        case ImageLayoutIntent::TransferSource:
            return vk::ImageLayout::eTransferSrcOptimal;
        case ImageLayoutIntent::TransferDestination:
            return vk::ImageLayout::eTransferDstOptimal;
        case ImageLayoutIntent::Present:
            return vk::ImageLayout::ePresentSrcKHR;
        case ImageLayoutIntent::DepthReadOnly:
            return vk::ImageLayout::eDepthReadOnlyOptimal;
    }
    return vk::ImageLayout::eUndefined;
}

// Inverse of IntentToImageLayout, and the only place a requested vk::ImageLayout
// is mapped to an intent. It is faithful by construction: a layout maps to an
// intent only when that intent's canonical layout is exactly that layout, so it
// never substitutes a different layout for the requested one. A layout the
// intent enum cannot represent (a vendor-extension layout, or the depth-stencil
// variants it does not distinguish) maps to Undefined; callers that must not
// silently drop a request compare the result against a non-Undefined input.
[[nodiscard]] inline ImageLayoutIntent ToImageLayoutIntent(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
            return ImageLayoutIntent::Undefined;
        case vk::ImageLayout::eGeneral:
            return ImageLayoutIntent::General;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return ImageLayoutIntent::ColorAttachment;
        case vk::ImageLayout::eDepthAttachmentOptimal:
            return ImageLayoutIntent::DepthAttachment;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return ImageLayoutIntent::ShaderReadOnly;
        case vk::ImageLayout::eTransferSrcOptimal:
            return ImageLayoutIntent::TransferSource;
        case vk::ImageLayout::eTransferDstOptimal:
            return ImageLayoutIntent::TransferDestination;
        case vk::ImageLayout::ePresentSrcKHR:
            return ImageLayoutIntent::Present;
        case vk::ImageLayout::eDepthReadOnlyOptimal:
            return ImageLayoutIntent::DepthReadOnly;
        default:
            return ImageLayoutIntent::Undefined;
    }
}

inline vk::ImageAspectFlags FormatToAspectFlags(vk::Format format) {
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

inline vk::ImageSubresourceRange DefaultImageSubresourceRange(vk::Format format) {
    return {FormatToAspectFlags(format), 0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers};
}

// How a resource will be resolved this frame. Building this table is pure:
// it keys everything by ResourceHandle.index and never indexes by position or
// with a non-const operator[], so a high-index transient can never be mistaken
// for a low-index one.
enum class ResourceResolutionKind : std::uint8_t {
    Unused,
    ImportedImage,
    ImportedBuffer,
    TransientImage,
    TransientBuffer,
    ImportedImageMissingResolver,
    ImportedBufferMissingResolver,
    InvalidIndex,
};

struct ResourceResolution {
    std::uint32_t resource_index = std::numeric_limits<std::uint32_t>::max();
    std::string name{};
    ResourceKind kind = ResourceKind::Image;
    ResourceResolutionKind resolution = ResourceResolutionKind::Unused;
};

[[nodiscard]] inline std::vector<ResourceResolution> BuildResourceResolutionTable(
    const CompiledRenderGraph& graph,
    const std::unordered_set<std::string>& image_resolvers,
    const std::unordered_set<std::string>& buffer_resolvers) {
    std::vector<ResourceResolution> table;
    table.reserve(graph.resource_lifetimes.size());

    for (const auto& lifetime : graph.resource_lifetimes) {
        ResourceResolution entry{};
        entry.resource_index = lifetime.handle.index;
        entry.name = lifetime.name;

        if (entry.resource_index >= graph.resource_info.size()) {
            entry.resolution = ResourceResolutionKind::InvalidIndex;
            table.push_back(std::move(entry));
            continue;
        }

        const auto& info = graph.resource_info[entry.resource_index];
        entry.kind = info.kind;

        if (lifetime.imported) {
            if (info.kind == ResourceKind::Image) {
                entry.resolution = image_resolvers.contains(info.name)
                                       ? ResourceResolutionKind::ImportedImage
                                       : ResourceResolutionKind::ImportedImageMissingResolver;
            } else {
                entry.resolution = buffer_resolvers.contains(info.name)
                                       ? ResourceResolutionKind::ImportedBuffer
                                       : ResourceResolutionKind::ImportedBufferMissingResolver;
            }
        } else if (lifetime.transient) {
            entry.resolution = info.kind == ResourceKind::Image
                                   ? ResourceResolutionKind::TransientImage
                                   : ResourceResolutionKind::TransientBuffer;
        } else {
            entry.resolution = ResourceResolutionKind::Unused;
        }

        table.push_back(std::move(entry));
    }

    return table;
}

// Per-resource actual old state supplied by the runtime (recorded end-of-frame
// state from a previous frame, keyed per swapchain image by the caller).
struct RuntimeResourceStates {
    std::vector<ResourceState> states{};
    std::vector<bool> has_state{};
};

// Pure barrier planning: translates the compiler-resolved transitions into
// concrete barriers with no device access. `resolved` supplies the handles;
// unresolved handles yield no image barrier / a conservative global memory
// barrier for buffers. `runtime_initial` overrides the first touch of a
// resource with its actual layout (cross-frame tracking); when that already
// equals the target, the barrier is dropped.
inline BarrierPlan PlanBarriers(const CompiledRenderGraph& graph,
                                const ResolvedResourceHandles& resolved,
                                const AliasIntervals& aliases,
                                const RuntimeResourceStates* runtime_initial = nullptr) {
    BarrierPlan plan{};
    plan.valid = graph.success;
    plan.passes.reserve(graph.passes.size());

    const std::size_t resource_count = graph.resource_info.size();
    std::vector<bool> seen(resource_count, false);
    std::vector<ResourceState> current(resource_count);
    std::vector<bool> current_known(resource_count, false);
    if (runtime_initial != nullptr) {
        const std::size_t count = std::min(resource_count, runtime_initial->states.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (i < runtime_initial->has_state.size() && runtime_initial->has_state[i]) {
                current[i] = runtime_initial->states[i];
                current_known[i] = true;
            }
        }
    }

    const auto image_range = [&](std::uint32_t index) {
        if (index < resolved.formats.size()) {
            return DefaultImageSubresourceRange(resolved.formats[index]);
        }
        return DefaultImageSubresourceRange(vk::Format::eUndefined);
    };

    for (std::size_t pass_index = 0; pass_index < graph.passes.size(); ++pass_index) {
        const auto& pass = graph.passes[pass_index];
        PlannedPassBarriers planned{};
        planned.pass_index = static_cast<std::uint32_t>(pass_index);
        planned.pass_name = pass.name;
        planned.queue = pass.queue;

        const auto emit = [&](const std::vector<ResourceTransition>& transitions,
                              std::vector<PlannedImageBarrier>& image_out,
                              std::vector<PlannedBufferBarrier>& buffer_out) {
            for (const auto& transition : transitions) {
                const std::uint32_t index = transition.resource_index;
                if (index >= resource_count) {
                    continue;
                }

                ResourceState from = transition.from_state;
                bool from_known = transition.from_known;
                vk::PipelineStageFlags2 src_stage = transition.src_stage;
                vk::AccessFlags2 src_access = transition.src_access;

                // The resource's actual old state wins over the compiler's
                // assumption on the first touch of the frame.
                if (!seen[index] && runtime_initial != nullptr &&
                    index < runtime_initial->has_state.size() && runtime_initial->has_state[index]) {
                    from = runtime_initial->states[index];
                    from_known = true;
                    src_stage = IntentToPipelineStage(from.stage, from.access);
                    src_access = IntentToAccessFlags(from.stage, from.access);
                }
                seen[index] = true;
                current[index] = transition.target_state;
                current_known[index] = true;

                // Already in the target state (e.g. a swapchain image left in
                // Present across frames): no barrier is required.
                if (from_known && StatesEqual(from, transition.target_state)) {
                    continue;
                }

                const ClampedScopes src_scopes = ClampScopesToQueue(src_stage, src_access, pass.queue);
                const ClampedScopes dst_scopes =
                    ClampScopesToQueue(transition.dst_stage, transition.dst_access, pass.queue);

                if (graph.resource_info[index].kind == ResourceKind::Image) {
                    PlannedImageBarrier barrier{};
                    barrier.resource_index = index;
                    if (index < resolved.images.size()) {
                        barrier.image = resolved.images[index];
                    }
                    barrier.src_stage = src_scopes.stage;
                    barrier.dst_stage = dst_scopes.stage;
                    barrier.src_access = src_scopes.access;
                    barrier.dst_access = dst_scopes.access;
                    barrier.old_layout = IntentToImageLayout(from.layout);
                    barrier.new_layout = IntentToImageLayout(transition.target_state.layout);
                    barrier.range = image_range(index);
                    image_out.push_back(barrier);
                } else {
                    PlannedBufferBarrier barrier{};
                    barrier.resource_index = index;
                    if (index < resolved.buffers.size()) {
                        barrier.buffer = resolved.buffers[index];
                    }
                    barrier.src_stage = src_scopes.stage;
                    barrier.dst_stage = dst_scopes.stage;
                    barrier.src_access = src_scopes.access;
                    barrier.dst_access = dst_scopes.access;
                    if (index < resolved.buffer_offsets.size()) {
                        barrier.offset = resolved.buffer_offsets[index];
                    }
                    if (index < resolved.buffer_sizes.size()) {
                        barrier.size = resolved.buffer_sizes[index];
                    }
                    buffer_out.push_back(barrier);
                }
            }
        };

        emit(pass.pre_pass_transitions, planned.pre_image, planned.pre_buffer);
        emit(pass.post_pass_transitions, planned.post_image, planned.post_buffer);
        plan.passes.push_back(std::move(planned));
    }

    plan.end_states = std::move(current);
    plan.has_end_state = std::move(current_known);

    // Alias ordering: when B reuses A's memory, B's first barrier must also wait
    // on A's last use. A discard barrier alone (srcStage=eNone) is not enough,
    // because A's writes may still be in flight.
    for (const auto& dependency : aliases.dependencies) {
        if (dependency.pass_index < 0 ||
            static_cast<std::size_t>(dependency.pass_index) >= plan.passes.size()) {
            continue;
        }
        const std::uint32_t after = dependency.after_resource;
        const std::uint32_t aliased = dependency.aliased_resource;

        // Only aliasable resources may overlap; ignore a dependency that names
        // a resource that was not declared aliasable.
        const auto is_aliasable = [&](std::uint32_t index) {
            if (index >= graph.resource_info.size()) {
                return false;
            }
            const auto& info = graph.resource_info[index];
            if (info.kind == ResourceKind::Image) {
                return info.image_info.has_value() && info.image_info->aliasable;
            }
            return info.buffer_info.has_value() && info.buffer_info->aliasable;
        };
        if (!is_aliasable(after) || !is_aliasable(aliased)) {
            continue;
        }

        vk::PipelineStageFlags2 after_stage{};
        vk::AccessFlags2 after_access{};
        vk::PipelineStageFlags2 aliased_stage{};
        vk::AccessFlags2 aliased_access{};
        ImageLayoutIntent aliased_layout = ImageLayoutIntent::Undefined;
        bool aliased_seen = false;

        for (const auto& pass : graph.passes) {
            for (const auto& transition : pass.pre_pass_transitions) {
                if (transition.resource_index == after) {
                    after_stage = transition.dst_stage;
                    after_access = transition.dst_access;
                }
                if (transition.resource_index == aliased && !aliased_seen) {
                    aliased_stage = transition.dst_stage;
                    aliased_access = transition.dst_access;
                    aliased_layout = transition.target_state.layout;
                    aliased_seen = true;
                }
            }
            for (const auto& transition : pass.post_pass_transitions) {
                if (transition.resource_index == after) {
                    after_stage = transition.dst_stage;
                    after_access = transition.dst_access;
                }
            }
        }

        auto& planned = plan.passes[static_cast<std::size_t>(dependency.pass_index)];
        if (aliased < graph.resource_info.size() &&
            graph.resource_info[aliased].kind == ResourceKind::Image) {
            PlannedImageBarrier* existing = nullptr;
            for (auto& barrier : planned.pre_image) {
                if (barrier.resource_index == aliased) {
                    existing = &barrier;
                    break;
                }
            }
            if (existing != nullptr) {
                const ClampedScopes alias_scopes = ClampScopesToQueue(
                    existing->src_stage | after_stage, existing->src_access | after_access, planned.queue);
                existing->src_stage = alias_scopes.stage;
                existing->src_access = alias_scopes.access;
                existing->old_layout = vk::ImageLayout::eUndefined;
            } else {
                const ClampedScopes alias_src = ClampScopesToQueue(after_stage, after_access, planned.queue);
                const ClampedScopes alias_dst = ClampScopesToQueue(
                    aliased_seen ? aliased_stage : vk::PipelineStageFlagBits2::eNone, aliased_access,
                    planned.queue);
                PlannedImageBarrier barrier{};
                barrier.resource_index = aliased;
                if (aliased < resolved.images.size()) {
                    barrier.image = resolved.images[aliased];
                }
                barrier.src_stage = alias_src.stage;
                barrier.src_access = alias_src.access;
                barrier.dst_stage = alias_dst.stage;
                barrier.dst_access = alias_dst.access;
                barrier.old_layout = vk::ImageLayout::eUndefined;
                barrier.new_layout = IntentToImageLayout(aliased_layout);
                barrier.range = image_range(aliased);
                planned.pre_image.push_back(barrier);
            }
        } else if (aliased < graph.resource_info.size()) {
            PlannedBufferBarrier* existing = nullptr;
            for (auto& barrier : planned.pre_buffer) {
                if (barrier.resource_index == aliased) {
                    existing = &barrier;
                    break;
                }
            }
            if (existing != nullptr) {
                const ClampedScopes alias_scopes = ClampScopesToQueue(
                    existing->src_stage | after_stage, existing->src_access | after_access, planned.queue);
                existing->src_stage = alias_scopes.stage;
                existing->src_access = alias_scopes.access;
            } else {
                const ClampedScopes alias_src = ClampScopesToQueue(after_stage, after_access, planned.queue);
                const ClampedScopes alias_dst = ClampScopesToQueue(
                    aliased_seen ? aliased_stage : vk::PipelineStageFlagBits2::eNone, aliased_access,
                    planned.queue);
                PlannedBufferBarrier barrier{};
                barrier.resource_index = aliased;
                if (aliased < resolved.buffers.size()) {
                    barrier.buffer = resolved.buffers[aliased];
                }
                barrier.src_stage = alias_src.stage;
                barrier.src_access = alias_src.access;
                barrier.dst_stage = alias_dst.stage;
                barrier.dst_access = alias_dst.access;
                planned.pre_buffer.push_back(barrier);
            }
        }
    }

    return plan;
}

// ── Queue runs (multi-queue submission) ──
// A run is a maximal contiguous span of ordered passes that execute on the same
// queue. Runs are submitted in order; a run on a different queue than the
// previous one is separated by a semaphore. Pure and device-free so the
// partitioning is unit-testable.

// Maximum number of graph queue runs a frame may be partitioned into. The
// device allocates one command buffer and one cross-queue semaphore per run
// slot per frame-in-flight, so the plan is rejected (not truncated) when it
// exceeds this. Kept here so the shared planner and the backend agree on the
// single source of truth.
inline constexpr std::uint32_t kMaxQueueRuns = 8;

struct QueueRun {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    QueueType queue = QueueType::Graphics;
    std::uint32_t first_pass = 0; // inclusive index into CompiledRenderGraph::passes
    std::uint32_t last_pass = 0;  // exclusive
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] std::uint32_t PassCount() const { return last_pass - first_pass; }
};

struct QueueRunPlan {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::vector<QueueRun> runs{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsSingleRun() const { return runs.size() <= 1; }
    [[nodiscard]] bool HasComputeRun() const {
        return std::ranges::any_of(runs, [](const QueueRun& run) {
            return run.queue != QueueType::Graphics;
        });
    }
    // True when the plan needs more run slots than the device budgets for.
    [[nodiscard]] bool ExceedsLimit() const { return runs.size() > kMaxQueueRuns; }
    // True when the very first run is on the graphics queue (the engine's scene
    // prep can then be folded into it); false when a graphics preamble must be
    // submitted ahead of the graph.
    [[nodiscard]] bool StartsWithGraphics() const {
        return !runs.empty() && runs.front().queue == QueueType::Graphics;
    }
};

[[nodiscard]] inline QueueRunPlan BuildQueueRuns(const CompiledRenderGraph& graph) {
    QueueRunPlan plan{};
    for (std::uint32_t i = 0; i < graph.passes.size(); ++i) {
        const QueueType queue = graph.passes[i].queue;
        if (!plan.runs.empty() && plan.runs.back().queue == queue) {
            plan.runs.back().last_pass = i + 1;
            continue;
        }
        plan.runs.push_back(QueueRun{.queue = queue, .first_pass = i, .last_pass = i + 1});
    }
    return plan;
}

}  // namespace VulkanEngine::RenderGraph
