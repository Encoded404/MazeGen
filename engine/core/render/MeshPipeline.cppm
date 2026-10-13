module;

export module VulkanEngine.StandardMeshPipeline;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.GpuResources;

export namespace VulkanEngine::StandardMeshPipeline {

struct PipelineConfig {
    // Rasterization
    vk::PolygonMode polygon_mode = vk::PolygonMode::eFill;
    vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eFront;
    vk::FrontFace front_face = vk::FrontFace::eClockwise;
    float line_width = 1.0f;

    // Depth/stencil
    bool depth_test_enable = true;
    bool depth_write_enable = true;
    vk::CompareOp depth_compare_op = vk::CompareOp::eLess;

    // Color blend
    bool blend_enable = false;
    vk::BlendFactor src_color_blend_factor = vk::BlendFactor::eSrcAlpha;
    vk::BlendFactor dst_color_blend_factor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::BlendOp color_blend_op = vk::BlendOp::eAdd;
    vk::BlendFactor src_alpha_blend_factor = vk::BlendFactor::eOne;
    vk::BlendFactor dst_alpha_blend_factor = vk::BlendFactor::eZero;
    vk::BlendOp alpha_blend_op = vk::BlendOp::eAdd;

    // Multisample
    vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1;

    // Topology
    vk::PrimitiveTopology primitive_topology = vk::PrimitiveTopology::eTriangleList;

    // Resolved draw mode (SceneRenderer::DrawMode, after capability fallback).
    // Specialized into the vertex shader's kCompactionMode constant:
    // 0 = CID, 1 = MID. Must be the resolved mode, never the requested
    // one: a MID-specialized vertex shader would misread CID indices.
    std::uint32_t draw_mode = 0;
};



// Canonical vertex layout. Mirrored by the `Vertex` structs in
// main_indir.slang and depth_indir.slang (SSBO vertex fetch — there are no
// vertex attributes); those MUST be changed in lockstep with this struct.
struct Vertex {
    float px, py, pz;       // position  (12 bytes)
    std::uint32_t packedTBN; // packed tangent frame (4 bytes):
                            //   [31]    handedness (1 = -1.0)
                            //   [30..21] tangent diamond d (10 bits)
                            //   [19..10] octahedral normal y (10 bits)
                            //   [9..0]  octahedral normal x (10 bits)
                            //   [20]    spare (reserved)
                            // see VulkanEngine.Mesh.NormalEncoding (pack) and
                            // shaders/normal_encoding.slang (unpack)
    float u, v;             // texcoord  ( 8 bytes)
    // Total: 24 bytes (6 uints)
};
static_assert(sizeof(Vertex) == 24, "Vertex layout must match the Slang Vertex structs (main_indir.slang, depth_indir.slang)");

struct MeshGPUResources {
    VulkanEngine::GpuResources::GpuBuffer vertex_buffer{};
    VulkanEngine::GpuResources::GpuBuffer index_buffer{};
    VulkanEngine::GpuResources::GpuTexture texture{};
    std::uint32_t index_count = 0;
};

class GraphicsPipeline {
public:
    GraphicsPipeline();
    ~GraphicsPipeline();

    void Initialize(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                    const std::vector<std::uint32_t>& vertex_spirv,
                    const std::vector<std::uint32_t>& fragment_spirv,
                    const PipelineConfig& config = {},
                    vk::DescriptorSetLayout* bindless_layout = nullptr,
                    vk::DescriptorSetLayout* object_data_layout = nullptr,
                    vk::DescriptorSetLayout* raw_vertex_layout = nullptr,
                    vk::DescriptorSetLayout* indirection_layout = nullptr);
    void Shutdown();

    [[nodiscard]] vk::PipelineLayout* GetPipelineLayout();
    [[nodiscard]] const vk::PipelineLayout* GetPipelineLayout() const;
    [[nodiscard]] vk::raii::Pipeline* GetPipeline();
    [[nodiscard]] const vk::raii::Pipeline* GetPipeline() const;
    [[nodiscard]] vk::DescriptorSetLayout* GetDescriptorSetLayout();
    [[nodiscard]] const vk::DescriptorSetLayout* GetDescriptorSetLayout() const;

    // ── New: Create pipeline with pre-built layout (used by BaseTechnique::Compile()) ──
    // Creates a graphics pipeline using an externally provided VkPipelineLayout,
    // descriptor set layouts, and push constant ranges. Does NOT create its own layout.
    void CreatePipelineWithLayout(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                                  const std::vector<std::uint32_t>& vertex_spirv,
                                  const std::vector<std::uint32_t>& fragment_spirv,
                                  const PipelineConfig& config,
                                  const std::vector<vk::DescriptorSetLayout>& set_layouts,
                                  vk::PipelineLayout external_pipeline_layout);

private:
    void CreatePipeline(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
                        const std::vector<std::uint32_t>& vertex_spirv,
                        const std::vector<std::uint32_t>& fragment_spirv,
                        const PipelineConfig& config);
    void CreateDescriptorSetLayout(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap);
    void CreateDescriptorPool(VulkanBackend::Vulkan::VulkanBootstrap& bootstrap);

    VulkanBackend::Vulkan::VulkanBootstrap* bootstrap_ = nullptr;
    PipelineConfig config_{};
    vk::DescriptorSetLayout* external_layout_ = nullptr;
    vk::DescriptorSetLayout* object_data_layout_ = nullptr;
    vk::DescriptorSetLayout* raw_vertex_layout_ = nullptr;
    vk::DescriptorSetLayout* indirection_layout_ = nullptr;
    std::shared_ptr<VulkanEngine::GpuResources::DescriptorPool> pool_;
    std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_{};
    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    std::unique_ptr<vk::raii::Pipeline> pipeline_{};
};

} // namespace VulkanEngine::StandardMeshPipeline
