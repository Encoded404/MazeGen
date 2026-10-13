module;

export module VulkanEngine.PipelineFactory;

import std;
import vulkan;
import ShaderReflection;
import VulkanEngine.ShaderManager;
import VulkanEngine.GplPolicy;
import VulkanBackend.Vulkan.VulkanCapabilities;

export namespace VulkanEngine::ShaderSystem {

struct GraphicsPipelineDesc {
    ShaderId vertex_shader;
    ShaderId fragment_shader;
    // Named entry point compiled into each stage. Defaults to "main"; a
    // multi-variant shader (one source, several entry-point wrappers, one row
    // each) names each wrapper's entry point here. Both stages share this name
    // only when it is "main"; a variant pair sets them per stage.
    std::string vertex_entry_point{"main"};
    std::string fragment_entry_point{"main"};

    vk::PipelineVertexInputStateCreateInfo vertex_input;
    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    vk::PipelineViewportStateCreateInfo viewport;
    vk::PipelineRasterizationStateCreateInfo rasterization;
    vk::PipelineMultisampleStateCreateInfo multisample;
    vk::PipelineDepthStencilStateCreateInfo depth_stencil;
    // Color-blend attachments are OWNED by the desc, not borrowed from a caller
    // member. `color_blend.pAttachments` is rebound to this vector at every
    // PipelineFactory entry (see BindColorBlendAttachments), so a desc may be
    // stored in any container and moved/copied without dangling — variants may
    // live in a std::vector that reallocates. Callers populate
    // color_blend_attachments and leave color_blend's attachmentCount/pAttachments
    // unset. `color_blend` is mutable so the rebound pointer can be refreshed
    // through a const desc reference at each creation entry.
    mutable vk::PipelineColorBlendStateCreateInfo color_blend;
    std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments;
    std::vector<vk::DynamicState> dynamic_states;

    vk::PipelineLayout layout;

    std::vector<vk::Format> color_formats;
    vk::Format depth_format = vk::Format::eUndefined;
    vk::Format stencil_format = vk::Format::eUndefined;

    // Optional specialization constants for the vertex stage (the draw-mode
    // selector lives here). The desc owns the data; the vectors must not be
    // reallocated between storing the desc and pipeline creation. Applied to
    // the vertex stage only: the monolithic vertex shader and, for GPL, the
    // pre-rasterization library (which holds the vertex shader).
    std::vector<vk::SpecializationMapEntry> spec_entries{};
    std::vector<std::byte> spec_data{};
};

struct ComputePipelineDesc {
    ShaderId shader;
    vk::PipelineLayout layout;
    // Named entry point compiled into the module. Defaults to "main".
    std::string entry_point{"main"};
    // Optional specialization constants. The desc owns the data; the vectors
    // must not be reallocated between storing the desc and pipeline creation
    // (hot-reload passes the same desc to CreateCompute).
    std::vector<vk::SpecializationMapEntry> spec_entries{};
    std::vector<std::byte> spec_data{};
};

// Why pipeline creation failed. Both a missing/invalid shader module and a
// driver-side rejection surface as VK_ERROR_INITIALIZATION_FAILED, so the kind
// is what lets a caller tell "my asset is broken" from "the driver refused this
// pipeline state". `message` carries the human-readable context (which shader,
// which create call, the vk::to_string of the result, and a desc summary).
enum class PipelineErrorKind : std::uint8_t {
    ShaderModuleUnavailable,
    DriverRejectedPipeline,
};

struct PipelineError {
    PipelineErrorKind kind = PipelineErrorKind::DriverRejectedPipeline;
    vk::Result result = vk::Result::eErrorInitializationFailed;
    std::string message;
};

class PipelineProduct {
public:
    PipelineProduct() : linked_(nullptr) {}

    static PipelineProduct Monolithic(vk::raii::Pipeline pipeline);

    static PipelineProduct GPLLinked(
        vk::raii::Pipeline linked,
        std::shared_ptr<vk::raii::Pipeline> vertex_input_lib,
        std::shared_ptr<vk::raii::Pipeline> pre_raster_lib,
        std::shared_ptr<vk::raii::Pipeline> fragment_shader_lib,
        std::shared_ptr<vk::raii::Pipeline> fragment_output_lib);

    ~PipelineProduct() = default;

    PipelineProduct(PipelineProduct&&) = default;
    PipelineProduct& operator=(PipelineProduct&&) = default;

    [[nodiscard]] vk::Pipeline Get() const { return *linked_; }

    PipelineProduct(const PipelineProduct&) = delete;
    PipelineProduct& operator=(const PipelineProduct&) = delete;

private:
    vk::raii::Pipeline linked_;
    // Fast-linked pipelines (no VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT)
    // reference the library pipelines they were linked from. Each product keeps
    // its libraries alive — including while retired — so the shared cache can
    // drop entries (e.g. InvalidateShader) without dangling the linked pipeline.
    std::shared_ptr<vk::raii::Pipeline> vertex_input_lib_;
    std::shared_ptr<vk::raii::Pipeline> pre_raster_lib_;
    std::shared_ptr<vk::raii::Pipeline> fragment_lib_;
    std::shared_ptr<vk::raii::Pipeline> fragment_output_lib_;
};

class PipelineSlot {
public:
    PipelineSlot() = default;

    // Runtime-sized retire ring. Must be called before the first Swap/Poll with
    // the configured frames in flight; systems default to 3 until then. Retained
    // pipelines are retired one FIF depth later, so a ring sized >= FIF is safe.
    void SetFramesInFlight(std::uint32_t frames_in_flight);

    void Swap(PipelineProduct product, std::uint32_t frame_index);

    void RetireFrame(std::uint32_t frame_index);

    // Configured retire-ring depth (>= 1). Variants created later copy it so
    // every slot in a technique retires on the same schedule.
    [[nodiscard]] std::uint32_t FramesInFlight() const { return frames_in_flight_; }

    [[nodiscard]] vk::Pipeline Get() const;

    bool PollAndRebuild(ShaderManager& shaders, ShaderId id,
                        std::function<std::optional<PipelineProduct>(ShaderManager&)> rebuild_fn,
                        std::uint32_t frame_index);

    bool PollAndRebuild(ShaderManager& shaders, ShaderId vert_id, ShaderId frag_id,
                        std::function<std::optional<PipelineProduct>(ShaderManager&)> rebuild_fn,
                        std::uint32_t frame_index);

private:
    PipelineProduct current_;
    std::vector<std::vector<PipelineProduct>> retiring_{3};
    std::uint32_t frames_in_flight_ = 3;
    std::uint64_t last_vert_version_{0};
    std::uint64_t last_frag_version_{0};
};

class PipelineFactory {
public:
    PipelineFactory(const vk::raii::Device& device, const VulkanBackend::Vulkan::VulkanCapabilities& caps,
                    const vk::raii::PipelineCache& cache, GplPolicy policy = GplPolicy::Auto,
                    GplStructurePolicy structure = GplStructurePolicy::Auto);

    [[nodiscard]] std::expected<PipelineProduct, PipelineError>
        CreateGraphics(const GraphicsPipelineDesc& desc,
                       ShaderManager& shaders) const;

    [[nodiscard]] std::expected<PipelineProduct, PipelineError>
        CreateCompute(const ComputePipelineDesc& desc,
                      ShaderManager& shaders) const;

    [[nodiscard]] bool IsGPLAvailable() const { return gpl_available_; }
    [[nodiscard]] bool IsGPLDisabledAtRuntime() const {
        return gpl_runtime_disabled_.load(std::memory_order_relaxed);
    }

    void InvalidateShader(ShaderId id);

private:
    // Sticky runtime GPL off-switch. Set the first time a GPL pipeline creation
    // fails, so every later pipeline in the same run goes straight to monolithic
    // instead of re-paying the failing GPL attempt (and re-logging it) per
    // technique. `mutable` keeps CreateGraphics const; creation is single-threaded.
    void DisableGplForRun(std::string_view reason) const;

    [[nodiscard]] PipelineProduct
        CreateGraphicsMonolithic(const GraphicsPipelineDesc& desc,
                                 ShaderManager& shaders) const;

    [[nodiscard]] PipelineProduct
        CreateGraphicsGPL(const GraphicsPipelineDesc& desc,
                          ShaderManager& shaders) const;

    // Final fast-link (no LTO): returns nullopt on VK_PIPELINE_COMPILE_REQUIRED
    // ("cannot fast-link") — the caller falls back to monolithic.
    [[nodiscard]] std::optional<vk::raii::Pipeline>
        CreateGraphicsGPLFinalLink(const GraphicsPipelineDesc& desc,
                                   std::span<const vk::Pipeline> libraries) const;

    static std::uint64_t HashVertexInput(const vk::PipelineVertexInputStateCreateInfo& vi,
                                         const vk::PipelineInputAssemblyStateCreateInfo& ia);
    static std::uint64_t HashPreRaster(
        const vk::PipelineViewportStateCreateInfo& vp,
        const vk::PipelineRasterizationStateCreateInfo& rs,
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const std::vector<vk::DynamicState>& dynamic_states,
        vk::PipelineLayout layout);
    static std::uint64_t HashFragmentShader(
        const vk::PipelineDepthStencilStateCreateInfo& ds,
        vk::PipelineLayout layout);
    static std::uint64_t HashFragmentOutput(
        const vk::PipelineMultisampleStateCreateInfo& ms,
        const vk::PipelineColorBlendStateCreateInfo& cb,
        std::span<const vk::Format> color_formats,
        vk::Format depth_fmt,
        vk::Format stencil_fmt);

    const vk::raii::Device& device_;
    const vk::raii::PipelineCache& cache_;
    bool gpl_available_;
    GplResolution resolution_;
    mutable std::atomic<bool> gpl_runtime_disabled_{false};

    struct SharedLibraries {
        struct LibraryEntry {
            ShaderId shader_id{};
            std::uint64_t shader_version{0};
            std::shared_ptr<vk::raii::Pipeline> pipeline;
        };
        std::unordered_map<std::uint64_t, LibraryEntry> vertex_input;
        std::unordered_map<std::uint64_t, LibraryEntry> pre_raster;
        std::unordered_map<std::uint64_t, LibraryEntry> fragment_shader;
        // Shader-independent (no shader_id) — never erased by InvalidateShader.
        std::unordered_map<std::uint64_t, LibraryEntry> fragment_output;
        mutable std::shared_mutex mutex;
    };
    std::shared_ptr<SharedLibraries> shared_;
};

} // namespace VulkanEngine::ShaderSystem
