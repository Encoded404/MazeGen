module;

#include <vulkan/vulkan.h>

export module TestSupport.TextureRenderHarness;

import std;

import vulkan;

import ShaderReflection;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;
import TestSupport.HeadlessVulkanBackend;

// Offscreen graphics renderer shared by the compressed-format sampling tests and
// the normal-mapping tests.
//
// Both modes draw a full-screen triangle with the same test vertex shader and
// upload their textures through the production GpuTexture::CreateFromTextureData
// path, so the real subresource copies and barriers are exercised. The mode
// picks which fragment shader runs:
//
//   Sampled:  test_sample_frag reads bindless slot 0 raw, so a BCn upload can be
//             compared channel-for-channel against the same fixture uploaded as
//             RGBA8 with nothing else in the way.
//   Standard: the REAL standard_mesh.spv runs against a small hand-built
//             descriptor layout, so the normal decode under test is the shipped
//             fragment shader.
//
// The layout is declared here rather than through BaseTechnique: a test must not
// depend on the scene manager. A change to the shader's bindings must fail here.
export namespace TestSupport {

// Matches DefaultMeshPerMaterialData (engine/core/render/techniques/
// DefaultMeshTechnique.cppm), which is 64 bytes.
struct HarnessMaterialData {
    std::uint32_t albedo_texture{0};
    std::uint32_t normal_texture{0};
    std::uint32_t orm_texture{0};
    std::uint32_t emissive_texture{0};
    std::uint32_t flags{0};
    std::uint32_t uv_sets{0};
    float roughness_factor{1.0f};
    float metallic_factor{0.0f};
    float ao_factor{1.0f};
    float normal_scale{1.0f};
    float occlusion_strength{1.0f};
    float alpha_cutoff{0.5f};
    float emissive_factor[4]{0.0f, 0.0f, 0.0f, 1.0f};
};
static_assert(sizeof(HarnessMaterialData) == 64, "material layout must mirror the shader");

// Matches the shader's SceneHeader (SceneRenderer.cppm).
struct HarnessSceneHeader {
    float ambient_color[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float sun_direction[4]{0.0f, 0.0f, 1.0f, 0.0f};
    float sun_color[4]{0.0f, 0.0f, 0.0f, 1.0f};
    std::uint32_t light_count{0};
};

// Matches the shader's Light (4 x float4).
struct HarnessLight {
    float position[4]{0.0f, 0.0f, 0.0f, 0.0f};
    float color[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float direction[4]{0.0f, 0.0f, 1.0f, 0.0f};
    float params[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

enum class HarnessMode {
    Sampled,   // test_sample_frag, bindless set 0 only
    Standard,  // standard_mesh.spv, sets 0/4/5 + camera push constant
    Unlit,     // unlit.spv, set 0 + a trimmed 24-byte per-material set 5
};

// Matches UnlitTextureTechnique::UnlitPerMaterialData: albedo_texture, float4
// albedo_factor, float alpha_cutoff. C data layout, 24 bytes, no padding.
struct HarnessUnlitMaterialData {
    std::uint32_t albedo_texture{0};
    float albedo_factor[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float alpha_cutoff{0.5f};
};
static_assert(sizeof(HarnessUnlitMaterialData) == 24,
              "unlit material layout must mirror the shader");

struct TextureRenderHarnessConfig {
    std::uint32_t width = 64;
    std::uint32_t height = 64;
    HarnessMode mode = HarnessMode::Sampled;
    // Shader ids; the caller owns the ShaderManager so the harness never
    // compiles or reloads. Sampled mode uses vertex_shader + fragment_shader
    // (the test sample shader); Standard mode uses vertex_shader + the engine's
    // standard_mesh fragment shader.
    VulkanEngine::ShaderSystem::ShaderId vertex_shader{};
    VulkanEngine::ShaderSystem::ShaderId fragment_shader{};
};

// Registers the test shaders with a ShaderManager through RegisterManual (they
// have no generated Slang module). `fragment_path` is the sampled-mode shader.
struct HarnessShaderIds {
    VulkanEngine::ShaderSystem::ShaderId vertex{};
    VulkanEngine::ShaderSystem::ShaderId sample_fragment{};
};

class TextureRenderHarness {
public:
    TextureRenderHarness() = default;
    ~TextureRenderHarness();

    TextureRenderHarness(const TextureRenderHarness&) = delete;
    TextureRenderHarness& operator=(const TextureRenderHarness&) = delete;

    bool Initialize(TestSupport::HeadlessVulkanBackend& backend,
                    VulkanEngine::ShaderSystem::ShaderManager& shaders,
                    const TextureRenderHarnessConfig& config);
    void Shutdown();

    // Draws one frame, then reads the color image back as RGBA8
    // (width * height * 4). Sampled mode samples `albedo` at slot 0 and ignores
    // the rest. Standard mode runs standard_mesh with the given material, scene,
    // and light; an invalid texture leaves its slot unmapped (0 in the
    // material). The trailing arguments are meaningful only in Standard mode.
    std::vector<std::uint8_t> RenderAndReadBack(const VulkanEngine::GpuResources::GpuTexture& albedo,
                                                const VulkanEngine::GpuResources::GpuTexture& normal = {},
                                                const VulkanEngine::GpuResources::GpuTexture& orm = {},
                                                const HarnessMaterialData& material = {},
                                                const HarnessSceneHeader& scene = {},
                                                const HarnessLight& light = {});

    // Unlit mode: draws one frame with the REAL unlit.spv against slot 0 and a
    // trimmed 24-byte UnlitPerMaterialData in set 5. Exercises the shipped
    // albedo_factor multiply and alpha_cutoff discard.
    std::vector<std::uint8_t> RenderUnlitAndReadBack(
        const VulkanEngine::GpuResources::GpuTexture& albedo,
        const HarnessUnlitMaterialData& material = {});

private:
    bool CreateDescriptorLayouts();
    bool CreatePipeline();
    void WriteDescriptors(const VulkanEngine::GpuResources::GpuTexture& albedo,
                          const VulkanEngine::GpuResources::GpuTexture& normal,
                          const VulkanEngine::GpuResources::GpuTexture& orm,
                          const HarnessMaterialData& material,
                          const HarnessSceneHeader& scene,
                          const HarnessLight& light);
    void WriteUnlitDescriptors(const VulkanEngine::GpuResources::GpuTexture& albedo,
                               const HarnessUnlitMaterialData& material);
    std::vector<std::uint8_t> RenderFrame(
        const std::function<void(vk::raii::CommandBuffer&)>& bind_and_draw);

    TestSupport::HeadlessVulkanBackend* backend_ = nullptr;
    VulkanEngine::ShaderSystem::ShaderManager* shaders_ = nullptr;
    TextureRenderHarnessConfig config_{};

    std::unique_ptr<vk::raii::DescriptorSetLayout> bindless_set0_layout_{};
    std::unique_ptr<vk::raii::DescriptorSetLayout> scene_set4_layout_{};
    std::unique_ptr<vk::raii::DescriptorSetLayout> material_set5_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> bindless_set0_{};
    std::unique_ptr<vk::raii::DescriptorSet> scene_set4_{};
    std::unique_ptr<vk::raii::DescriptorSet> material_set5_{};

    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    // PipelineProduct (not a bare Pipeline) keeps GPL library pipelines alive.
    std::unique_ptr<VulkanEngine::ShaderSystem::PipelineProduct> pipeline_{};

    std::vector<std::unique_ptr<vk::raii::Buffer>> host_buffers_;
    std::vector<std::unique_ptr<vk::raii::DeviceMemory>> host_memories_;
};

// Registers the test-only shaders. `shader_dir` holds test_fullscreen_vs.spv and
// test_sample_frag.spv (the test shader output directory).
[[nodiscard]] HarnessShaderIds RegisterHarnessShaders(
    VulkanEngine::ShaderSystem::ShaderManager& shaders, std::string_view shader_dir);

}  // namespace TestSupport
