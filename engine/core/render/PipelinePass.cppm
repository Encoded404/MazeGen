module;

#include <cassert> // NOLINT(misc-include-cleaner)
#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

export module VulkanEngine.PipelinePass;

import std;
import std.compat;

import vulkan;

export import VulkanEngine.RenderGraph;
export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.TechniqueManager;
export import VulkanEngine.BindlessManager;
export import VulkanEngine.DescriptorDecl;
export import VulkanEngine.ECS.ComponentRegistry;
export import VulkanEngine.ImGui;

export namespace VulkanEngine::PipelinePass {

// ── Built-in pass ordering points ──
// Exhaustive: every real built-in pass, so an application can order around any
// of them. `Count` is the array size for anchors.
enum class BuiltinPass : std::uint8_t {
    Expand,
    OccluderSelect,
    OccluderPrepass,
    HiZGenPre,
    PreCull,
    DepthPrepass,
    HiZGen,
    Occlusion,
    Collect,
    MainPass,
    // Depth-tested world-space text: scene content, so it sits after the main
    // pass and before the screen-space overlay.
    WorldText,
    Text,
    ImGui,
    Count,
};

inline constexpr std::size_t kBuiltinPassCount = static_cast<std::size_t>(BuiltinPass::Count);

// The viewport every pass that draws the y-up scene through the camera's
// view-projection must set.
//
// Vulkan's framebuffer origin is top-left and clip-space +y points *down* the
// framebuffer, while the engine's projection matrices are GLM's OpenGL-style y-up
// ones (GLM_FORCE_DEPTH_ZERO_TO_ONE moves z to [0, 1] and leaves y alone) and the
// scene camera's up is (0, 1, 0). A negative viewport height is what reconciles
// the two: it mirrors the viewport transform in y, so world +y -- and with it a
// world-space glyph's top edge -- lands at the top of the framebuffer.
//
// This is the one definition the main, depth and occluder passes and the
// world-text pass all draw through, so the scene and the text on it cannot drift
// apart again. A screen-space pass is the opposite case: it draws y-down pixels
// straight to clip space (see ui_text.slang) and deliberately sets a positive
// height.
[[nodiscard]] inline vk::Viewport SceneViewport(std::uint32_t width,
                                                std::uint32_t height) noexcept {
    return vk::Viewport(0.0f, static_cast<float>(height), static_cast<float>(width),
                        -static_cast<float>(height), 0.0f, 1.0f);
}

// ── Opaque typed handles for FrameContext ──
struct BindlessTextureSet  { vk::DescriptorSet handle = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct SubmeshVertexSet    { vk::DescriptorSet handle = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct RawVertexArray      { vk::DescriptorSet handle = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct IndirectionSet      { vk::DescriptorSet handle = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct SceneUniformSet     { vk::DescriptorSet handle = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct DepthPyramid        { vk::Image image = nullptr; vk::ImageView view = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)
struct DepthBufferView     { vk::ImageView view = nullptr; }; // NOLINT(misc-non-private-member-variables-in-classes)

// ── PassResource — a resolved resource handle for a pass ──
// Carries the Vulkan handles resolved for the current frame, keyed by resource
// identity. Typed accessors read the field matching the resource kind; raw()
// exposes everything for advanced use.
struct ResolvedResource {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    VulkanEngine::RenderGraph::ResourceHandle handle{};
    VulkanEngine::RenderGraph::ResourceKind kind = VulkanEngine::RenderGraph::ResourceKind::Image;
    vk::Image image = nullptr;
    vk::ImageView view = nullptr;
    vk::Buffer buffer = nullptr;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = vk::WholeSize;
    vk::Format format = vk::Format::eUndefined;
    bool resolved = false;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class PassResource {
public:
    PassResource() = default;
    explicit PassResource(ResolvedResource resource) : resource_(std::move(resource)) {}

    [[nodiscard]] bool IsValid() const { return resource_.resolved; }
    [[nodiscard]] bool IsImage() const { return IsValid() && resource_.kind == VulkanEngine::RenderGraph::ResourceKind::Image; }
    [[nodiscard]] bool IsBuffer() const { return IsValid() && resource_.kind == VulkanEngine::RenderGraph::ResourceKind::Buffer; }
    [[nodiscard]] vk::Image AsImage() const { return IsImage() ? resource_.image : nullptr; }
    [[nodiscard]] vk::ImageView AsImageView() const { return IsImage() ? resource_.view : nullptr; }
    [[nodiscard]] vk::Buffer AsBuffer() const { return IsBuffer() ? resource_.buffer : nullptr; }
    [[nodiscard]] vk::DeviceSize GetOffset() const { return resource_.offset; }
    [[nodiscard]] vk::DeviceSize GetSize() const { return resource_.size; }
    [[nodiscard]] vk::Format GetFormat() const { return resource_.format; }
    [[nodiscard]] VulkanEngine::RenderGraph::ResourceHandle GetHandle() const { return resource_.handle; }
    [[nodiscard]] const ResolvedResource& raw() const { return resource_; }

private:
    ResolvedResource resource_{};
};

enum class ResourceLookupError : std::uint8_t {
    NoLookupBound,
    NotFound,
};

// Maps a resource name to the handles resolved for this frame.
class IResourceLookup {
public:
    virtual ~IResourceLookup() = default;
    [[nodiscard]] virtual std::optional<PassResource> Find(std::string_view name) const = 0;
};

// Concrete lookup the pipeline fills each frame.
class ResourceLookupTable final : public IResourceLookup {
public:
    void Clear() { entries_.clear(); }
    void Set(std::string name, ResolvedResource resource) { entries_[std::move(name)] = std::move(resource); }
    [[nodiscard]] std::optional<PassResource> Find(std::string_view name) const override {
        const auto it = entries_.find(std::string(name));
        if (it == entries_.end()) {
            return std::nullopt;
        }
        return PassResource{it->second};
    }
    [[nodiscard]] std::size_t Size() const { return entries_.size(); }

private:
    std::unordered_map<std::string, ResolvedResource> entries_{};
};

enum class PushConstantError : std::uint8_t {
    NotDeclared,
    SizeMismatch,
    NoPipelineLayout,
};

// ── Binding helpers for PassResource ──
enum class BindingError : std::uint8_t {
    NotResolved,
    NotABuffer,
    NotAnImage,
    OffsetMisaligned,
    NoImageView,
};

// Alignment limits the caller reads from the device (pure inputs here so the
// rules stay unit-testable without a device).
struct DescriptorAlignment {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint64_t min_storage_buffer_offset_alignment = 1;
    std::uint64_t min_uniform_buffer_offset_alignment = 1;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

[[nodiscard]] inline std::expected<vk::DescriptorBufferInfo, BindingError> MakeBufferBinding(
    const PassResource& resource, std::uint64_t min_offset_alignment) {
    if (!resource.IsValid()) {
        return std::unexpected(BindingError::NotResolved);
    }
    if (!resource.IsBuffer()) {
        return std::unexpected(BindingError::NotABuffer);
    }
    if (min_offset_alignment > 1 && (resource.GetOffset() % min_offset_alignment) != 0) {
        return std::unexpected(BindingError::OffsetMisaligned);
    }
    vk::DescriptorBufferInfo info{};
    info.buffer = resource.AsBuffer();
    info.offset = resource.GetOffset();
    info.range = resource.GetSize();
    return info;
}

[[nodiscard]] inline std::expected<vk::DescriptorBufferInfo, BindingError> MakeStorageBufferBinding(
    const PassResource& resource, const DescriptorAlignment& alignment) {
    return MakeBufferBinding(resource, alignment.min_storage_buffer_offset_alignment);
}

[[nodiscard]] inline std::expected<vk::DescriptorBufferInfo, BindingError> MakeUniformBufferBinding(
    const PassResource& resource, const DescriptorAlignment& alignment) {
    return MakeBufferBinding(resource, alignment.min_uniform_buffer_offset_alignment);
}

// Sampled/combined image bindings require a view; a resource resolved to an
// image without a view is an error rather than a null descriptor.
[[nodiscard]] inline std::expected<vk::DescriptorImageInfo, BindingError> MakeSampledImageBinding(
    const PassResource& resource, vk::Sampler sampler = nullptr) {
    if (!resource.IsValid()) {
        return std::unexpected(BindingError::NotResolved);
    }
    if (!resource.IsImage()) {
        return std::unexpected(BindingError::NotAnImage);
    }
    if (resource.AsImageView() == nullptr) {
        return std::unexpected(BindingError::NoImageView);
    }
    vk::DescriptorImageInfo info{};
    info.sampler = sampler;
    info.imageView = resource.AsImageView();
    info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    return info;
}

// ── Pass pipeline request — declarative, engine-owned pipeline creation ──
enum class PassPipelineKind : std::uint8_t {
    None,
    Graphics,
    Compute,
};

// Colour-blend state for an engine-owned graphics pipeline. The defaults are the
// engine's standard straight-alpha source-over -- the same factors
// PipelineUtils::CreateAlphaBlendAttachment uses -- so a pass that declares
// blending gets a composited overlay without restating them, and a pass that
// leaves blending disabled keeps the opaque write every other pass gets.
struct PassBlendState {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    bool enable = false;
    vk::BlendFactor src_color_blend_factor = vk::BlendFactor::eSrcAlpha;
    vk::BlendFactor dst_color_blend_factor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::BlendOp color_blend_op = vk::BlendOp::eAdd;
    // Alpha accumulates as one/one-minus-src-alpha: the destination's coverage
    // is attenuated by the source's, so an overlay never makes an opaque target
    // transparent.
    vk::BlendFactor src_alpha_blend_factor = vk::BlendFactor::eOne;
    vk::BlendFactor dst_alpha_blend_factor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::BlendOp alpha_blend_op = vk::BlendOp::eAdd;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Depth state for an engine-owned graphics pipeline.
//
// The defaults are the engine's *disabled* state -- the depth attachment-less
// overlay every existing pass builds today -- so a pass that does not ask for
// depth keeps exactly the pipeline it had. A pass that renders into the scene's
// depth buffer declares test-only depth: depth_write stays false so the overlay
// never hides geometry a later pass would draw, and the compare is the engine's
// less-or-equal (the scene is depth 0..1 via GLM_FORCE_DEPTH_ZERO_TO_ONE and the
// camera techniques use eLessOrEqual). Enabling the test with an undefined
// depth format is the caller's to avoid: the engine infers the format from the
// declared depth attachment, and without one pipeline creation fails.
struct PassDepthState {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    bool test_enable = false;
    bool write_enable = false;
    vk::CompareOp compare_op = vk::CompareOp::eLessOrEqual;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// ── Pass pipeline request — declarative, engine-owned pipeline creation ──
struct PassPipelineRequest {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    PassPipelineKind kind = PassPipelineKind::None;
    std::uint64_t vertex_shader = 0;
    std::uint64_t fragment_shader = 0;
    std::uint64_t compute_shader = 0;
    std::vector<vk::Format> color_formats{};
    vk::Format depth_format = vk::Format::eUndefined;
    std::uint32_t push_constant_size = 0;
    vk::ShaderStageFlags push_constant_stages{};
    // Graphics only; ignored for a compute pipeline.
    PassBlendState blend{};
    // Graphics only; ignored for a compute pipeline.
    PassDepthState depth{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool IsDeclared() const { return kind != PassPipelineKind::None; }
};


// ── TransientImageDesc — description of a transient (pass-owned) image ──
struct TransientImageDesc {
    std::string name{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::Format format = vk::Format::eUndefined; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t width = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t height = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    // Relative size: when > 0 the dimension is derived from the render extent at
    // (re)allocation time and the absolute width/height is ignored. A scale of
    // 0 keeps the absolute dimension.
    float width_scale = 0.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    float height_scale = 0.0f; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::ImageLayout initial_layout = vk::ImageLayout::eUndefined; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::ImageLayout final_layout = vk::ImageLayout::eUndefined; // NOLINT(misc-non-private-member-variables-in-classes)
    // Aliasable images are created with VK_IMAGE_CREATE_ALIAS_BIT and must use
    // an Undefined initial layout (ContentsUndefined contract).
    bool aliasable = false; // NOLINT(misc-non-private-member-variables-in-classes)
};

// Resolves a transient image's real dimensions against the current render
// extent. Pure and device-free so relative sizing is unit-testable.
[[nodiscard]] inline std::pair<std::uint32_t, std::uint32_t> ResolveTransientExtent(
    const TransientImageDesc& desc, std::uint32_t render_width, std::uint32_t render_height) {
    const auto scaled = [](std::uint32_t base, float scale) -> std::uint32_t {
        if (scale <= 0.0f) {
            return base;
        }
        const auto value = std::lround(static_cast<double>(scale) * static_cast<double>(base));
        return value < 1 ? 1u : static_cast<std::uint32_t>(value);
    };
    const std::uint32_t width = desc.width_scale > 0.0f ? scaled(render_width, desc.width_scale) : desc.width;
    const std::uint32_t height = desc.height_scale > 0.0f ? scaled(render_height, desc.height_scale) : desc.height;
    return {width, height};
}

// ── TransientBufferDesc — description of a transient (pass-owned) buffer ──
struct TransientBufferDesc {
    std::string name{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::DeviceSize size = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::MemoryPropertyFlags memory_properties = vk::MemoryPropertyFlagBits::eDeviceLocal; // NOLINT(misc-non-private-member-variables-in-classes)
    bool aliasable = false; // NOLINT(misc-non-private-member-variables-in-classes)
};

// ── IResourceRegistry — abstract interface for resource registration ──
// Implemented by RenderPipeline to allow PassSetupContext to declare
// resources without depending on the RenderPipeline concrete type.
class IResourceRegistry {
public:
    virtual ~IResourceRegistry() = default;

    virtual VulkanEngine::RenderGraph::ResourceHandle ImportDepthBuffer() = 0;
    virtual VulkanEngine::RenderGraph::ResourceHandle ImportBackbuffer() = 0;
    virtual VulkanEngine::RenderGraph::ResourceHandle ImportImage(const std::string& name) = 0;
    virtual VulkanEngine::RenderGraph::ResourceHandle ImportBuffer(const std::string& name) = 0;
    virtual VulkanEngine::RenderGraph::ResourceHandle CreateTransientImage(const TransientImageDesc& desc) = 0;
    virtual VulkanEngine::RenderGraph::ResourceHandle CreateTransientBuffer(const TransientBufferDesc& desc) = 0;
};

// ── Forward declarations ──
class IPipelinePass;
class PassSetupContext;

// ── FrameContext — per-frame resources passed to IPipelinePass::Execute() ──
struct FrameContext {
    //NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    vk::Extent2D render_extent{};
    std::uint32_t frame_index = 0;
    // Frame-in-flight ring slot (frame_index % frames_in_flight): the index
    // every per-slot resource must use.
    std::uint32_t ring_index = 0;
    std::uint32_t swapchain_image_index = 0;

    // Camera data (populated by the renderer before dispatching passes)
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 view_proj{1.0f};

    // Engine-standard descriptor sets (sets 0-4)
    BindlessTextureSet bindless_textures{};
    SubmeshVertexSet   submesh_vertices{};
    RawVertexArray      raw_vertex_buffers{};
    IndirectionSet      indirection_data{};
    SceneUniformSet     scene_uniforms{};

    // Common GPU resources
    DepthPyramid    depth_pyramid{};
    DepthBufferView depth_buffer{};

    // Scene-level data
    VulkanEngine::TechniqueManager::TechniqueManager* techniques = nullptr;
    VulkanEngine::BindlessManager::BindlessManager* bindless = nullptr;
    VulkanEngine::ComponentRegistry* registry = nullptr;
    VulkanEngine::ImGui::ImGuiSystem* imgui = nullptr;
    std::uint32_t entity_count = 0;
    std::uint32_t render_width = 0;
    std::uint32_t render_height = 0;

    // Per-frame resource resolution, set by the pipeline before execution.
    const IResourceLookup* resource_lookup = nullptr;

    // App-pass descriptor sets (set index >= first_app_descriptor_set), written
    // by the engine from the pass's BindResource declarations this frame. Bind
    // them with `cmd.bindDescriptorSets(..., *pipeline_layout, first_app_descriptor_set,
    // app_descriptor_sets, {})`.
    std::span<const vk::DescriptorSet> app_descriptor_sets{};
    std::uint32_t first_app_descriptor_set =
        VulkanEngine::Render::kFirstAppDescriptorSet;
    // Engine-owned sampler for sampled/combined-image bindings.
    vk::Sampler default_sampler = nullptr;
    // Engine-owned pipeline for this pass (null when the pass declared none).
    // Bind it with cmd.bindPipeline().
    vk::Pipeline pass_pipeline = nullptr;

    // Pipeline layout for push constants (set per-pass by PassSetupContext)
    vk::PipelineLayout pipeline_layout = nullptr;

    // Push constant metadata for runtime validation
    std::uint32_t declared_push_constant_size = 0;
    vk::ShaderStageFlags declared_push_constant_stages{};
    //NOLINTEND(misc-non-private-member-variables-in-classes)

    // ── Push constant upload with runtime validation ──
    [[nodiscard]] std::expected<void, PushConstantError> ValidatePushConstants(std::size_t type_size) const {
        if (declared_push_constant_stages == vk::ShaderStageFlags{}) {
            return std::unexpected(PushConstantError::NotDeclared);
        }
        if (declared_push_constant_size != type_size) {
            return std::unexpected(PushConstantError::SizeMismatch);
        }
        if (pipeline_layout == nullptr) {
            return std::unexpected(PushConstantError::NoPipelineLayout);
        }
        return {};
    }

    template<typename T>
    [[nodiscard]] std::expected<void, PushConstantError> SetPushConstants(vk::CommandBuffer cmd, const T& src) const {
        auto valid = ValidatePushConstants(sizeof(T));
        if (!valid.has_value()) {
            return valid;
        }
        cmd.pushConstants(pipeline_layout, declared_push_constant_stages, 0, sizeof(T), &src);
        return {};
    }

    // ── Resolve a resource declared in Setup() ──
    // Looks the name up in the per-frame identity table.
    [[nodiscard]] std::expected<PassResource, ResourceLookupError> GetResource(std::string_view name) const {
        if (resource_lookup == nullptr) {
            return std::unexpected(ResourceLookupError::NoLookupBound);
        }
        auto found = resource_lookup->Find(name);
        if (!found.has_value()) {
            return std::unexpected(ResourceLookupError::NotFound);
        }
        return *found;
    }
};

// The executor's single opaque user_data. Every pass (built-in and app) reads
// the populated `frame`.
struct RenderFrameData {
    FrameContext frame{};
};

// ── PassSetupContext — resource and ordering declarations in Setup() ──
// A declared descriptor binding is associated with a graph resource so the
// engine can write the descriptor set from the resource resolved this frame.
struct BindingAssignment {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VulkanEngine::RenderGraph::ResourceHandle resource{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class PassSetupContext {
public:
    explicit PassSetupContext(IResourceRegistry& registry,
                              std::uint32_t render_width = 0,
                              std::uint32_t render_height = 0);
    ~PassSetupContext() = default;

    PassSetupContext(const PassSetupContext&) = delete;
    PassSetupContext& operator=(const PassSetupContext&) = delete;
    PassSetupContext(PassSetupContext&&) = default;
    PassSetupContext& operator=(PassSetupContext&&) = default;

    // ── Ordering relative to the fixed built-in pipeline ──
    void RunBefore(BuiltinPass pass);
    void RunAfter(BuiltinPass pass);

    // ── GPU resource imports and transient creation ──
    VulkanEngine::RenderGraph::ResourceHandle ReadDepthBuffer();
    VulkanEngine::RenderGraph::ResourceHandle ReadBackbuffer();
    VulkanEngine::RenderGraph::ResourceHandle ImportImage(std::string_view name);
    VulkanEngine::RenderGraph::ResourceHandle ImportBuffer(std::string_view name);
    VulkanEngine::RenderGraph::ResourceHandle CreateTransientImage(const TransientImageDesc& desc);
    VulkanEngine::RenderGraph::ResourceHandle CreateTransientBuffer(const TransientBufferDesc& desc);

    // ── Render graph resource usage declarations ──
    void AddRead(VulkanEngine::RenderGraph::ResourceHandle res,
                 VulkanEngine::RenderGraph::PipelineStageIntent stage,
                 VulkanEngine::RenderGraph::AccessIntent access);
    void AddWrite(VulkanEngine::RenderGraph::ResourceHandle res);

    // ── Attachment pass declaration ──
    void SetPassAttachments(VulkanEngine::RenderGraph::PassAttachmentSetup setup);

    // ── Typed push constant declaration ──
    template<typename T>
    void DeclarePushConstants(vk::ShaderStageFlags stages) {
        push_constant_size_ = static_cast<std::uint32_t>(sizeof(T));
        push_constant_stages_ = stages;
    }

    // ── Engine-owned pipeline declaration ──
    // The engine creates/owns the pipeline layout and pipeline, and hot-reloads
    // it; the pass never touches VkPipeline directly.
    void RequestGraphicsPipeline(std::uint64_t vertex_shader, std::uint64_t fragment_shader,
                                 std::vector<vk::Format> color_formats = {},
                                 vk::Format depth_format = vk::Format::eUndefined);
    void RequestComputePipeline(std::uint64_t compute_shader);
    // Colour blend for the requested graphics pipeline. Only meaningful for
    // RequestGraphicsPipeline; the defaults of PassBlendState are the engine's
    // straight-alpha source-over.
    void SetBlendState(PassBlendState blend) { pipeline_request_.blend = blend; }
    // Depth state for the requested graphics pipeline. Only meaningful for
    // RequestGraphicsPipeline; the defaults of PassDepthState keep depth
    // disabled, so a pass that renders over the scene is unchanged.
    void SetDepthState(PassDepthState depth) { pipeline_request_.depth = depth; }
    [[nodiscard]] const PassPipelineRequest& GetPipelineRequest() const { return pipeline_request_; }

    // Queue the pass runs on. Compute requires a dedicated async compute queue;
    // registration fails otherwise. Defaults to Graphics.
    void SetQueueType(VulkanEngine::RenderGraph::QueueType queue) { queue_type_ = queue; }
    [[nodiscard]] VulkanEngine::RenderGraph::QueueType GetQueueType() const { return queue_type_; }

    // ── Descriptor declaration (app sets >= 5) ──
    void DeclareBindings(std::vector<VulkanEngine::Render::DescriptorDecl> bindings);
    [[nodiscard]] const std::vector<VulkanEngine::Render::DescriptorDecl>& GetDeclaredBindings() const { return declared_bindings_; }

    // Associate a declared (set, binding) with a graph resource. The engine
    // allocates the set and rewrites the descriptor from the resource resolved
    // each frame; the pass only binds the set. The (set, binding) must have been
    // declared via DeclareBindings().
    void BindResource(std::uint32_t set, std::uint32_t binding,
                      VulkanEngine::RenderGraph::ResourceHandle resource);
    [[nodiscard]] const std::vector<BindingAssignment>& GetBindingAssignments() const {
        return binding_assignments_;
    }

    // ── Render extent query ──
    [[nodiscard]] std::uint32_t GetRenderWidth() const;
    [[nodiscard]] std::uint32_t GetRenderHeight() const;

    // ── Internal: accessors for RenderPipeline integration ──
    [[nodiscard]] std::uint32_t GetPushConstantSize() const { return push_constant_size_; }
    [[nodiscard]] vk::ShaderStageFlags GetPushConstantStages() const { return push_constant_stages_; }

    // ── Resource declaration accessors (populated during Setup()) ──
    [[nodiscard]] const std::vector<BuiltinPass>& GetBeforeBuiltinPasses() const { return before_builtin_passes_; }
    [[nodiscard]] const std::vector<BuiltinPass>& GetAfterBuiltinPasses() const { return after_builtin_passes_; }
    [[nodiscard]] const std::vector<VulkanEngine::RenderGraph::ResourceHandle>& GetReadResources() const { return read_resources_; }
    [[nodiscard]] const std::vector<VulkanEngine::RenderGraph::PipelineStageIntent>& GetReadStages() const { return read_stages_; }
    [[nodiscard]] const std::vector<VulkanEngine::RenderGraph::AccessIntent>& GetReadAccesses() const { return read_accesses_; }
    [[nodiscard]] const std::vector<VulkanEngine::RenderGraph::ResourceHandle>& GetWriteResources() const { return write_resources_; }
    [[nodiscard]] const std::optional<VulkanEngine::RenderGraph::PassAttachmentSetup>& GetAttachmentSetup() const { return attachment_setup_; }

private:
    IResourceRegistry* registry_ = nullptr;

    // Builtin pass ordering (stored as enums, resolved in RegisterPass)
    std::vector<BuiltinPass> before_builtin_passes_;
    std::vector<BuiltinPass> after_builtin_passes_;

    // Resource read/write declarations
    std::vector<VulkanEngine::RenderGraph::ResourceHandle> read_resources_;
    std::vector<VulkanEngine::RenderGraph::PipelineStageIntent> read_stages_;
    std::vector<VulkanEngine::RenderGraph::AccessIntent> read_accesses_;
    std::vector<VulkanEngine::RenderGraph::ResourceHandle> write_resources_;

    // Attachment setup
    std::optional<VulkanEngine::RenderGraph::PassAttachmentSetup> attachment_setup_;

    // Push constant declaration
    std::uint32_t push_constant_size_ = 0;
    vk::ShaderStageFlags push_constant_stages_{};

    // Engine-owned pipeline + app descriptor declaration
    PassPipelineRequest pipeline_request_{};
    VulkanEngine::RenderGraph::QueueType queue_type_ = VulkanEngine::RenderGraph::QueueType::Graphics;
    std::vector<VulkanEngine::Render::DescriptorDecl> declared_bindings_{};
    std::vector<BindingAssignment> binding_assignments_{};

    std::uint32_t render_width_ = 0;
    std::uint32_t render_height_ = 0;
};

// ── IPipelinePass — abstract base for all pipeline passes ──
// Built-in and application passes share this one interface: the engine
// constructs the PassSetupContext and calls Setup() during RegisterPass, then
// invokes Execute() through the render graph each frame.
class IPipelinePass {
public:
    virtual ~IPipelinePass() = default;

    // Declare GPU resources and ordering during pipeline setup
    virtual void Setup(PassSetupContext& ctx) = 0;

    // Execute the pass every frame
    virtual void Execute(const FrameContext& ctx, vk::CommandBuffer cmd) = 0;

    // Optional: validate configuration before compilation
    [[nodiscard]] virtual bool Validate() const { return true; }

    // Optional: react to a render-extent change. Called once per change at a
    // frame boundary (drained by RenderPipeline::ApplyChanges), never while the
    // frame is executing. Use it to resize app-owned resources; graph-declared
    // relative transients are reallocated by the engine itself.
    virtual void OnRenderResize(std::uint32_t width, std::uint32_t height) {
        (void)width;
        (void)height;
    }

    // Stable, human-readable name used for graph diagnostics, ordering, and
    // duplicate-name validation. Override for application passes.
    [[nodiscard]] virtual std::string_view GetName() const { return "AppPass"; }
};

} // namespace VulkanEngine::PipelinePass
