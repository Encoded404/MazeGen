#include <gtest/gtest.h>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

import std;
import std.compat;

import vulkan;

import test_gpu;

import ShaderReflection;
import Shaders.Engine.WorldTextVert;
import Shaders.Engine.WorldTextFrag;
import Shaders.Engine.SlugTextVert;
import Shaders.Engine.SlugTextFrag;
import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Components.Text;
import VulkanEngine.Components.Transform;
import VulkanEngine.Render.Passes.WorldTextPass;
import VulkanEngine.PipelinePass;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GpuTextBlobBuffer;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;
import VulkanEngine.TextureTypes;
import VulkanEngine.BindlessManager;
import VulkanEngine.BindlessManager.TextureSlot;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuTexture;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

namespace {

using VulkanEngine::ComponentRegistry;
using VulkanEngine::Entity;
using VulkanEngine::BindlessManager::BindlessManager;
using VulkanEngine::Components::Text;
using VulkanEngine::Components::Transform;
using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::SamplerCache;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;
using VulkanEngine::SceneRenderer::BlobUploader;
using VulkanEngine::SceneRenderer::BuildSlugTextInstances;
using VulkanEngine::SceneRenderer::BuildWorldTextInstances;
using VulkanEngine::SceneRenderer::SlugTextConstants;
using VulkanEngine::SceneRenderer::SlugTextInstance;
using VulkanEngine::SceneRenderer::WorldTextConstants;
using VulkanEngine::SceneRenderer::WorldTextInstance;
using VulkanEngine::ShaderSystem::GraphicsPipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::Text::GlyphBlobEncoder;
using VulkanEngine::Text::GpuTextBlobBuffer;
using VulkanEngine::Text::MsdfConfig;
using VulkanEngine::Text::MsdfGenerator;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapeText;

constexpr std::uint32_t kTargetSize = 256;
constexpr std::uint32_t kPageSize = 64;
// The instance buffer holds the text batch and the depth-writing occluder batch
// in one allocation, so each draw binds its own descriptor set at its own
// offset. 32 KiB is a multiple of every driver's storage-buffer offset
// alignment and holds 400 80-byte glyphs.
constexpr vk::DeviceSize kOccluderOffset = 32U * 1024U;
constexpr vk::DeviceSize kInstanceBytes = 64U * 1024U;

// World x < 0 maps to framebuffer x < 128 under the test view-projection below,
// which is exactly how far the occluder quad reaches.
constexpr std::uint32_t kOccluderRightPixel = 128;

// The scene colour the text composites over: opaque black, so an ink pixel is
// any pixel whose red channel rose above 0.
constexpr std::array<std::uint8_t, 4> kBackground{0u, 0u, 0u, 255u};

// A test camera in the engine's row-vector convention. `world_text.slang` does
// mul(float4(p, 1), view_proj), which applies the memcpy'd glm matrix as
// m * p -- the same convention every engine shader uses -- so a plain
// scale/translate glm matrix is an exact camera:
//   world x in [-1, 1]  -> framebuffer x in [0, 256]
//   world y in [-1, 1]  -> framebuffer y in [0, 256], +y up the image
//   world z             -> depth in [0, 1] (Vulkan's NDC z range)
//
// The matrix itself stays GLM's y-up form; the world-up-to-image-top half of the
// mapping comes from the viewport, which is the engine's scene viewport
// (PipelinePass::SceneViewport, a negative height). Emulating that flip in the
// matrix instead would hide exactly the convention the pass has to share with the
// scene, which is how the world text once shipped upside down.
[[nodiscard]] glm::mat4 TestViewProjection() {
    glm::mat4 m{1.0f};
    m[0][0] = 1.0f;
    m[1][1] = 1.0f;
    m[2][2] = 0.5f;
    m[3][2] = 0.5f;
    return m;
}

struct PixelPoint {
    float x = 0.0f;
    float y = 0.0f;
    float depth = 0.0f;
};

// Projects one world point through the test camera exactly as the viewport
// transform does: perspective divide, then NDC -> framebuffer with the scene
// viewport's negative height (framebuffer_y = (1 - ndc_y) * height/2), NDC z ->
// depth.
[[nodiscard]] PixelPoint Project(const glm::mat4& view_proj, const glm::vec3& world) {
    const glm::vec4 clip = view_proj * glm::vec4(world, 1.0f);
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    return PixelPoint{(ndc.x + 1.0f) * 0.5f * static_cast<float>(kTargetSize),
                      (1.0f - ndc.y) * 0.5f * static_cast<float>(kTargetSize), ndc.z};
}

struct PixelBounds {
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
};

// Draws WorldTextInstance quads with the shipped world_text shaders through a
// hand-built pipeline that carries the exact state WorldTextPass declares: a
// world-space instance storage buffer at set 5, the 80-byte camera/atlas push
// constant, straight-alpha source-over blending, and -- for the text pipeline --
// depth test on, depth write off, less-or-equal against a real depth
// attachment. The GPU harness cannot drive a real RenderPipeline, so this is the
// same approach the screen-space text test takes.
//
// The structural assertions deliberately avoid a whole-frame golden hash:
// antialiased MSDF coverage is driver-sensitive, and a pinned constant would
// mask a broken assertion rather than catch one. Depth order is asserted by
// running the identical scene through a control pipeline with depth testing
// disabled, so the occlusion assertion cannot pass vacuously.
class WorldTextGpuTest : public ::testing::Test {
protected:
    struct InkBounds {
        std::uint32_t min_x = kTargetSize;
        std::uint32_t min_y = kTargetSize;
        std::uint32_t max_x = 0;
        std::uint32_t max_y = 0;
        std::uint32_t count = 0;
        bool empty() const { return count == 0; }
        std::uint32_t width() const { return max_x - min_x + 1; }
        std::uint32_t height() const { return max_y - min_y + 1; }
    };

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        ASSERT_TRUE(heap_.Initialize(backend_, ImageHeapConfig{.block_size = 8ULL << 20},
                                     "world-text-gpu-tests"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 1ULL << 20;
        staging_config.capacity_bytes = 8ULL << 20;
        ASSERT_TRUE(staging_.Initialize(backend_, staging_config));
        ASSERT_TRUE(samplers_.Initialize(backend_));

        bindless_ = std::make_unique<BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity{};
        capacity.app_capacity = 16;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity));
        bindless_->SetFallback(MakeSolidTexture(), VulkanEngine::ResourceId{"world-text-fallback"});

        handle_ = manager_.LoadFromFile<VulkanEngine::FontResource>(
            std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf",
            VulkanEngine::ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        auto* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());
        face_ = FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);

        AtlasConfig atlas_config{};
        atlas_config.page_width = kPageSize;
        atlas_config.page_height = kPageSize;
        atlas_config.padding = 1;
        atlas_config.max_pages = 4;
        generator_ = std::make_unique<MsdfGenerator>(atlas_config, 64);

        atlas_gpu_ = std::make_unique<GlyphAtlasGpu>();
        ASSERT_TRUE(atlas_gpu_->Initialize(backend_, heap_, staging_, *bindless_, &samplers_));
        atlas_gpu_->SetPageFormat(VulkanEngine::Text::AtlasPageFormat::Rgba8);
        atlas_gpu_->SetByteSource(
            [this](std::uint64_t key) { return generator_->BitmapForAtlasKey(key); });

        // The Slug backend's blob storage buffer. Deliberately small so the
        // first glyphs exercise the growth path (new buffer, whole image
        // re-uploaded, old buffer retired frame-gated).
        blobs_ = std::make_unique<GpuTextBlobBuffer>();
        ASSERT_TRUE(blobs_->Initialize(backend_, 64U));

        view_proj_ = TestViewProjection();
        BuildPipelines();
        CreateTargets();
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        readback_buffer_.reset();
        readback_memory_.reset();
        depth_view_.reset();
        depth_memory_.reset();
        depth_image_.reset();
        color_view_.reset();
        color_memory_.reset();
        color_image_.reset();
        occluder_pipeline_.reset();
        no_depth_pipeline_.reset();
        text_pipeline_.reset();
        slug_pipeline_.reset();
        pipeline_layout_.reset();
        occluder_set_.reset();
        text_set_.reset();
        slug_set_.reset();
        instance_pool_.reset();
        instance_layout_.reset();
        instance_buffer_.reset();
        shaders_.reset();
        slug_encoder_.Clear();
        if (blobs_) {
            blobs_->Shutdown();
            blobs_.reset();
        }
        if (atlas_gpu_) {
            atlas_gpu_->Shutdown();
            atlas_gpu_.reset();
        }
        generator_.reset();
        if (bindless_) {
            bindless_->Shutdown();
            bindless_.reset();
        }
        samplers_.Shutdown();
        staging_.Shutdown();
        heap_.Shutdown();
        backend_.Shutdown();
    }

    [[nodiscard]] GpuTexture MakeSolidTexture() {
        std::vector<std::uint8_t> pixels(8u * 8u * 4u, 0x30u);
        return GpuTexture::CreateFromPixels(backend_, heap_, pixels.data(), 8, 8,
                                            vk::Format::eR8G8B8A8Unorm, &samplers_);
    }

    // A depth attachment format the device really supports. A device-bound test
    // must not silently skip, so the caller asserts a format was found.
    [[nodiscard]] vk::Format PickDepthFormat() const {
        for (const vk::Format candidate :
             {vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint, vk::Format::eD16Unorm}) {
            const vk::FormatProperties properties =
                backend_.GetPhysicalDevice().getFormatProperties(candidate);
            if (properties.optimalTilingFeatures &
                vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
                return candidate;
            }
        }
        return vk::Format::eUndefined;
    }

    void BuildPipelines() {
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(),
                                                   backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        // The shipped engine shader modules, through the same generated
        // registration the engine itself uses.
        const ShaderId vert = Shaders::Engine::WorldTextVert::Register(*shaders_,
                                                                       VKENGINE_ENGINE_SHADER_DIR);
        const ShaderId frag = Shaders::Engine::WorldTextFrag::Register(*shaders_,
                                                                       VKENGINE_ENGINE_SHADER_DIR);
        const ShaderId slug_vert = Shaders::Engine::SlugTextVert::Register(
            *shaders_, VKENGINE_ENGINE_SHADER_DIR);
        const ShaderId slug_frag = Shaders::Engine::SlugTextFrag::Register(
            *shaders_, VKENGINE_ENGINE_SHADER_DIR);

        // Set 5: the instance storage buffer at binding 0 (one buffer per draw)
        // and, for the Slug backend, the blob storage buffer at binding 1. The
        // binding-1 stage flags are fragment-only, exactly as the pass's
        // DeclareBindings sets them: the Slug vertex does not read the blob, so
        // a vertex stage flag here would hide a layout mismatch. The MSDF
        // shaders declare only binding 0 and never statically use binding 1.
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eFragment, nullptr)};
        instance_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            backend_.GetDevice(),
            vk::DescriptorSetLayoutCreateInfo{{}, bindings.size(), bindings.data()});

        const vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 6);
        instance_pool_ = std::make_unique<vk::raii::DescriptorPool>(
            backend_.GetDevice(), vk::DescriptorPoolCreateInfo({}, 3, 1, &pool_size));
        auto sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**instance_pool_, 1, &**instance_layout_});
        ASSERT_EQ(sets.size(), 1u);
        text_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));
        sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**instance_pool_, 1, &**instance_layout_});
        ASSERT_EQ(sets.size(), 1u);
        occluder_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));
        sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**instance_pool_, 1, &**instance_layout_});
        ASSERT_EQ(sets.size(), 1u);
        slug_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));

        instance_buffer_ = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            backend_, kInstanceBytes, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        ASSERT_TRUE(instance_buffer_->IsValid());

        const vk::Buffer buffer = static_cast<vk::Buffer>(*instance_buffer_->GetBuffer());
        vk::DescriptorBufferInfo text_info{};
        text_info.buffer = buffer;
        text_info.offset = 0;
        text_info.range = kOccluderOffset;
        vk::WriteDescriptorSet text_write{};
        text_write.dstSet = **text_set_;
        text_write.dstBinding = 0;
        text_write.descriptorCount = 1;
        text_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        text_write.pBufferInfo = &text_info;
        backend_.GetDevice().updateDescriptorSets({text_write}, {});

        vk::DescriptorBufferInfo occluder_info{};
        occluder_info.buffer = buffer;
        occluder_info.offset = kOccluderOffset;
        occluder_info.range = kInstanceBytes - kOccluderOffset;
        vk::WriteDescriptorSet occluder_write{};
        occluder_write.dstSet = **occluder_set_;
        occluder_write.dstBinding = 0;
        occluder_write.descriptorCount = 1;
        occluder_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        occluder_write.pBufferInfo = &occluder_info;
        backend_.GetDevice().updateDescriptorSets({occluder_write}, {});

        // The Slug set's binding 0 is the same instance buffer; binding 1 is the
        // blob storage buffer the fragment's adapted accessor reads.
        vk::WriteDescriptorSet slug_instance_write = text_write;
        slug_instance_write.dstSet = **slug_set_;
        vk::DescriptorBufferInfo blob_info{};
        blob_info.buffer = blobs_->Buffer();
        blob_info.offset = 0;
        blob_info.range = vk::WholeSize;
        vk::WriteDescriptorSet blob_write{};
        blob_write.dstSet = **slug_set_;
        blob_write.dstBinding = 1;
        blob_write.descriptorCount = 1;
        blob_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        blob_write.pBufferInfo = &blob_info;
        backend_.GetDevice().updateDescriptorSets({slug_instance_write, blob_write}, {});

        // The shader names set 0 (bindless) and set 5 (instances); sets 1-4 are
        // reserved and must exist as empty layouts so set 5 lands where the
        // shader says.
        const std::array<vk::DescriptorSetLayout, 6> set_layouts{
            *bindless_->GetLayout(), {}, {}, {}, {}, **instance_layout_};
        const vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eVertex |
                                                   vk::ShaderStageFlagBits::eFragment,
                                               0, sizeof(WorldTextConstants));
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        layout_info.setPushConstantRanges(push_range);
        pipeline_layout_ =
            std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

        depth_format_ = PickDepthFormat();
        ASSERT_NE(depth_format_, vk::Format::eUndefined)
            << "the device supports no depth attachment format";

        // The pass's documented blend: straight-alpha source-over with a full
        // colour write.
        const vk::ColorComponentFlags rgba = vk::ColorComponentFlagBits::eR |
                                             vk::ColorComponentFlagBits::eG |
                                             vk::ColorComponentFlagBits::eB |
                                             vk::ColorComponentFlagBits::eA;
        const vk::PipelineColorBlendAttachmentState text_blend(
            vk::True, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd, rgba);
        // The occluder only writes depth: blending is off and the colour mask is
        // empty, so it leaves the scene colour exactly as it found it.
        const vk::PipelineColorBlendAttachmentState occluder_blend(
            vk::False, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::ColorComponentFlags{});

        text_pipeline_ = MakeGraphicsPipeline(vert, frag, text_blend, /*depth_test=*/true,
                                              /*depth_write=*/false);
        no_depth_pipeline_ = MakeGraphicsPipeline(vert, frag, text_blend, /*depth_test=*/false,
                                                  /*depth_write=*/false);
        occluder_pipeline_ = MakeGraphicsPipeline(vert, frag, occluder_blend, /*depth_test=*/true,
                                                  /*depth_write=*/true);
        // The Slug backend, through the shipped slug_text modules and the same
        // depth-tested, straight-alpha state the pass declares.
        slug_pipeline_ = MakeGraphicsPipeline(slug_vert, slug_frag, text_blend, /*depth_test=*/true,
                                              /*depth_write=*/false);
        // A missing pipeline is fatal for every test in the suite, so fail SetUp
        // here rather than dereferencing null inside a draw.
        ASSERT_NE(text_pipeline_, nullptr);
        ASSERT_NE(no_depth_pipeline_, nullptr);
        ASSERT_NE(occluder_pipeline_, nullptr);
        ASSERT_NE(slug_pipeline_, nullptr);
    }

    [[nodiscard]] std::unique_ptr<PipelineProduct> MakeGraphicsPipeline(
        ShaderId vert, ShaderId frag, const vk::PipelineColorBlendAttachmentState& blend,
        bool depth_test, bool depth_write) {
        GraphicsPipelineDesc desc{};
        desc.vertex_shader = vert;
        desc.fragment_shader = frag;
        desc.vertex_input = vk::PipelineVertexInputStateCreateInfo({}, 0, nullptr, 0, nullptr);
        desc.input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);
        desc.viewport = vk::PipelineViewportStateCreateInfo({}, 1, nullptr, 1, nullptr);
        desc.rasterization = vk::PipelineRasterizationStateCreateInfo(
            {}, vk::False, vk::False, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
            vk::FrontFace::eCounterClockwise, vk::False, 0.0f, 0.0f, 0.0f, 1.0f);
        desc.multisample = vk::PipelineMultisampleStateCreateInfo({}, vk::SampleCountFlagBits::e1);
        // WorldTextPass::Setup declares test-only depth, less-or-equal; the
        // occluder variant writes depth so the text pass has something to test
        // against, exactly as scene geometry would.
        desc.depth_stencil = vk::PipelineDepthStencilStateCreateInfo(
            {}, depth_test ? vk::True : vk::False, depth_write ? vk::True : vk::False,
            vk::CompareOp::eLessOrEqual);
        desc.color_blend =
            vk::PipelineColorBlendStateCreateInfo({}, vk::False, vk::LogicOp::eCopy, 0, nullptr);
        desc.color_blend_attachments = {blend};
        desc.dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        desc.layout = **pipeline_layout_;
        desc.color_formats = {vk::Format::eR8G8B8A8Unorm};
        desc.depth_format = depth_format_;

        PipelineFactory factory(backend_.GetDevice(), backend_.GetCapabilities(),
                                shaders_->GetPipelineCacheRAII());
        auto product = factory.CreateGraphics(desc, *shaders_);
        EXPECT_TRUE(product.has_value()) << product.error().message;
        if (!product.has_value()) {
            return nullptr;
        }
        return std::make_unique<PipelineProduct>(std::move(*product));
    }

    void CreateTargets() {
        const vk::ImageCreateInfo color_info(
            {}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm,
            vk::Extent3D{kTargetSize, kTargetSize, 1}, 1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive);
        color_image_ = std::make_unique<vk::raii::Image>(backend_.GetDevice(), color_info);
        const vk::MemoryRequirements color_req = color_image_->getMemoryRequirements();
        color_memory_ = std::make_unique<vk::raii::DeviceMemory>(
            backend_.GetDevice(),
            vk::MemoryAllocateInfo(color_req.size,
                                   VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                                       backend_.GetPhysicalDevice(), color_req.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal)));
        color_image_->bindMemory(*color_memory_, 0);
        color_view_ = std::make_unique<vk::raii::ImageView>(
            backend_.GetDevice(),
            vk::ImageViewCreateInfo({}, **color_image_, vk::ImageViewType::e2D,
                                    vk::Format::eR8G8B8A8Unorm, vk::ComponentMapping{},
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1,
                                                              0, 1)));

        const vk::ImageCreateInfo depth_info(
            {}, vk::ImageType::e2D, depth_format_, vk::Extent3D{kTargetSize, kTargetSize, 1}, 1, 1,
            vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive);
        depth_image_ = std::make_unique<vk::raii::Image>(backend_.GetDevice(), depth_info);
        const vk::MemoryRequirements depth_req = depth_image_->getMemoryRequirements();
        depth_memory_ = std::make_unique<vk::raii::DeviceMemory>(
            backend_.GetDevice(),
            vk::MemoryAllocateInfo(depth_req.size,
                                   VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                                       backend_.GetPhysicalDevice(), depth_req.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal)));
        depth_image_->bindMemory(*depth_memory_, 0);
        depth_view_ = std::make_unique<vk::raii::ImageView>(
            backend_.GetDevice(),
            vk::ImageViewCreateInfo({}, **depth_image_, vk::ImageViewType::e2D, depth_format_,
                                    vk::ComponentMapping{},
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1,
                                                              0, 1)));

        const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(kTargetSize) * kTargetSize * 4U;
        readback_buffer_ = std::make_unique<vk::raii::Buffer>(
            backend_.GetDevice(),
            vk::BufferCreateInfo({}, bytes, vk::BufferUsageFlagBits::eTransferDst));
        const vk::MemoryRequirements read_req = readback_buffer_->getMemoryRequirements();
        readback_memory_ = std::make_unique<vk::raii::DeviceMemory>(
            backend_.GetDevice(),
            vk::MemoryAllocateInfo(read_req.size,
                                   VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                                       backend_.GetPhysicalDevice(), read_req.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eHostVisible |
                                           vk::MemoryPropertyFlagBits::eHostCoherent)));
        readback_buffer_->bindMemory(*readback_memory_, 0);
    }

    // Shapes `content`, builds the world-space instances for a component at
    // `position`, uploads the newly packed MSDF pages, and rebuilds once so the
    // instances sample real bindless slots instead of the fallback.
    [[nodiscard]] std::vector<WorldTextInstance> QueueText(
        const std::string& content, float world_height, const glm::vec3& position,
        const std::array<float, 4>& color, const MsdfConfig& config = {}) {
        Entity& entity = registry_.CreateEntity();
        Text& text = registry_.AddComponent<Text>(entity);
        text.font_id = 1;
        text.content = content;
        text.world_height = world_height;
        text.color = color;
        Transform& transform = registry_.AddComponent<Transform>(entity);
        transform.position = position;
        transform.rotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
        transform.scale = glm::vec3{1.0f, 1.0f, 1.0f};
        return BuildFromComponent(text, transform, config);
    }

    [[nodiscard]] std::vector<WorldTextInstance> BuildFromComponent(
        const Text& text, const Transform& transform, const MsdfConfig& config = {}) {
        const ShapedRun run =
            ShapeText(*face_, text.content, ShapeOptions{.kerning = true, .ligatures = true});
        const auto page_slot = [this](std::uint32_t page) {
            const auto page_handle = atlas_gpu_->PageHandle(page);
            return page_handle.IsValid() ? page_handle.slot
                                         : VulkanEngine::BindlessManager::kFallbackSlot;
        };

        std::vector<WorldTextInstance> instances;
        BuildWorldTextInstances(*generator_, page_slot, text, transform, *face_, run, instances,
                                config);
        UploadAtlas();
        instances.clear();
        BuildWorldTextInstances(*generator_, page_slot, text, transform, *face_, run, instances,
                                config);
        return instances;
    }

    void UploadAtlas() {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        atlas_gpu_->UploadDirty(generator_->MutableAtlas(), *cmd, 0);
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();
    }

    // The Slug counterpart of QueueText: shapes, encodes each glyph's blob into
    // the pass-owned storage buffer, and returns the em-box instances whose flat
    // offsets point at those blobs. A growth inside Store() replaces the buffer,
    // so the descriptor is rewritten from the current one; the engine's buffer
    // resolver does the same thing every frame.
    [[nodiscard]] std::vector<SlugTextInstance> QueueSlugText(const std::string& content,
                                                              float world_height,
                                                              const glm::vec3& position,
                                                              const std::array<float, 4>& color) {
        Entity& entity = registry_.CreateEntity();
        Text& text = registry_.AddComponent<Text>(entity);
        text.font_id = 1;
        text.content = content;
        text.world_height = world_height;
        text.color = color;
        Transform& transform = registry_.AddComponent<Transform>(entity);
        transform.position = position;
        transform.rotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
        transform.scale = glm::vec3{1.0f, 1.0f, 1.0f};
        return BuildSlugFromComponent(text, transform);
    }

    [[nodiscard]] std::vector<SlugTextInstance> BuildSlugFromComponent(
        const Text& text, const Transform& transform) {
        const ShapedRun run =
            ShapeText(*face_, text.content, ShapeOptions{.kerning = true, .ligatures = true});
        const BlobUploader upload =
            [this](std::span<const std::byte> bytes) -> std::optional<std::uint64_t> {
            const auto offset = blobs_->Store(bytes, /*recording_frame=*/0);
            if (!offset.has_value()) {
                return std::nullopt;
            }
            return blobs_->ElementOffset(*offset);
        };
        std::vector<SlugTextInstance> instances;
        BuildSlugTextInstances(slug_encoder_, upload, text, transform, *face_, run, instances);
        UpdateBlobDescriptor();
        return instances;
    }

    // Points the Slug set's binding 1 at whatever blob buffer is current. A
    // growth retires the old buffer but leaves its bytes readable, so the only
    // thing that has to change is the descriptor.
    void UpdateBlobDescriptor() {
        vk::DescriptorBufferInfo blob_info{};
        blob_info.buffer = blobs_->Buffer();
        blob_info.offset = 0;
        blob_info.range = vk::WholeSize;
        vk::WriteDescriptorSet blob_write{};
        blob_write.dstSet = **slug_set_;
        blob_write.dstBinding = 1;
        blob_write.descriptorCount = 1;
        blob_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        blob_write.pBufferInfo = &blob_info;
        backend_.GetDevice().updateDescriptorSets({blob_write}, {});
    }

    // One depth-writing quad covering framebuffer x < 128 (world x < 0) at
    // world z = -0.5, i.e. depth 0.25 -- nearer than any text the tests draw.
    // Its colour is masked off, so from the text pass's point of view it is
    // scene geometry that filled the depth buffer.
    [[nodiscard]] std::vector<WorldTextInstance> Occluders() const {
        WorldTextInstance occluder{};
        occluder.origin[0] = -1.0f;
        occluder.origin[1] = -1.0f;
        occluder.origin[2] = -0.5f;
        occluder.right[0] = 1.0f;
        occluder.right[1] = 0.0f;
        occluder.right[2] = 0.0f;
        occluder.up[0] = 0.0f;
        occluder.up[1] = 2.0f;
        occluder.up[2] = 0.0f;
        occluder.uv_min[0] = 0.0f;
        occluder.uv_min[1] = 0.0f;
        occluder.uv_max[0] = 1.0f;
        occluder.uv_max[1] = 1.0f;
        occluder.page = VulkanEngine::BindlessManager::kFallbackSlot;
        return {occluder};
    }

    struct RenderRequest {
        std::vector<WorldTextInstance> text{};
        std::vector<SlugTextInstance> slug_text{};
        std::vector<WorldTextInstance> occluders{};
        // false runs the identical scene through a pipeline with depth testing
        // disabled -- the control that makes the occlusion assertion real.
        bool depth_test = true;
    };

    [[nodiscard]] std::vector<std::uint8_t> Render(const RenderRequest& request) {
        if (!request.text.empty()) {
            instance_buffer_->UploadAt(request.text.data(),
                                       request.text.size() * sizeof(WorldTextInstance), 0);
        }
        if (!request.slug_text.empty()) {
            instance_buffer_->UploadAt(
                request.slug_text.data(),
                request.slug_text.size() * sizeof(SlugTextInstance), 0);
        }
        if (!request.occluders.empty()) {
            instance_buffer_->UploadAt(request.occluders.data(),
                                       request.occluders.size() * sizeof(WorldTextInstance),
                                       kOccluderOffset);
        }

        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        const vk::ImageSubresourceRange color_range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        const vk::ImageSubresourceRange depth_range(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
        Transition(cmd, **color_image_, color_range, color_layout_,
                   vk::ImageLayout::eColorAttachmentOptimal);
        Transition(cmd, **depth_image_, depth_range, depth_layout_,
                   vk::ImageLayout::eDepthAttachmentOptimal);

        // Pass 1: the main-pass proxy. Clears colour and depth, then fills the
        // depth buffer with the occluder quads exactly as scene geometry would.
        {
            const vk::RenderingAttachmentInfo color_attachment =
                MakeColorAttachment(vk::AttachmentLoadOp::eClear,
                                    vk::ImageLayout::eColorAttachmentOptimal);
            const vk::RenderingAttachmentInfo depth_attachment =
                MakeDepthAttachment(vk::AttachmentLoadOp::eClear, vk::ImageLayout::eDepthAttachmentOptimal);
            vk::RenderingInfo rendering{};
            rendering.renderArea = vk::Rect2D({0, 0}, {kTargetSize, kTargetSize});
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            rendering.pDepthAttachment = &depth_attachment;
            cmd.beginRendering(rendering);
            SetViewportAndScissor(cmd);
            if (!request.occluders.empty()) {
                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, occluder_pipeline_->Get());
                BindInstanceSet(cmd, **occluder_set_);
                PushConstants(cmd);
                cmd.draw(6, static_cast<std::uint32_t>(request.occluders.size()), 0, 0);
            }
            cmd.endRendering();
        }

        // The depth the occluder wrote must be visible to the text pass's depth
        // test, and the colour it did not touch must be visible to the load.
        BarrierBetweenPasses(cmd, color_range, depth_range);

        // Pass 2: depth-tested world text, loading what the main pass left.
        {
            const vk::RenderingAttachmentInfo color_attachment =
                MakeColorAttachment(vk::AttachmentLoadOp::eLoad,
                                    vk::ImageLayout::eColorAttachmentOptimal);
            const vk::RenderingAttachmentInfo depth_attachment =
                MakeDepthAttachment(vk::AttachmentLoadOp::eLoad, vk::ImageLayout::eDepthReadOnlyOptimal);
            vk::RenderingInfo rendering{};
            rendering.renderArea = vk::Rect2D({0, 0}, {kTargetSize, kTargetSize});
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            rendering.pDepthAttachment = &depth_attachment;
            cmd.beginRendering(rendering);
            SetViewportAndScissor(cmd);
            if (!request.slug_text.empty()) {
                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, slug_pipeline_->Get());
                BindInstanceSet(cmd, **slug_set_);
                PushSlugConstants(cmd);
                cmd.draw(6, static_cast<std::uint32_t>(request.slug_text.size()), 0, 0);
            } else if (!request.text.empty()) {
                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 request.depth_test ? text_pipeline_->Get()
                                                    : no_depth_pipeline_->Get());
                BindInstanceSet(cmd, **text_set_);
                PushConstants(cmd);
                cmd.draw(6, static_cast<std::uint32_t>(request.text.size()), 0, 0);
            }
            cmd.endRendering();
        }

        const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(kTargetSize) * kTargetSize * 4U;
        const vk::ImageMemoryBarrier to_src(
            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, **color_image_, color_range);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, to_src);
        const vk::BufferImageCopy region(
            0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            vk::Offset3D{0, 0, 0}, vk::Extent3D{kTargetSize, kTargetSize, 1});
        cmd.copyImageToBuffer(**color_image_, vk::ImageLayout::eTransferSrcOptimal,
                              **readback_buffer_, region);
        const vk::BufferMemoryBarrier to_host(
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, **readback_buffer_, 0, bytes);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, {}, to_host, {});
        cmd.end();

        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();

        color_layout_ = vk::ImageLayout::eTransferSrcOptimal;
        depth_layout_ = vk::ImageLayout::eDepthReadOnlyOptimal;

        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(bytes));
        const void* mapped = readback_memory_->mapMemory(0, bytes);
        std::memcpy(pixels.data(), mapped, pixels.size());
        readback_memory_->unmapMemory();
        return pixels;
    }

    void Transition(vk::CommandBuffer cmd, vk::Image image,
                    const vk::ImageSubresourceRange& range, vk::ImageLayout old_layout,
                    vk::ImageLayout new_layout) const {
        if (old_layout == new_layout) {
            return;
        }
        const bool first_use = old_layout == vk::ImageLayout::eUndefined;
        const vk::ImageMemoryBarrier barrier(
            first_use ? vk::AccessFlags{}
                      : (vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite),
            {}, old_layout, new_layout, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, image,
            range);
        cmd.pipelineBarrier(first_use ? vk::PipelineStageFlagBits::eTopOfPipe
                                      : vk::PipelineStageFlagBits::eAllCommands,
                            vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, {barrier});
    }

    void BarrierBetweenPasses(vk::CommandBuffer cmd, const vk::ImageSubresourceRange& color_range,
                              const vk::ImageSubresourceRange& depth_range) const {
        const vk::ImageMemoryBarrier color_barrier(
            vk::AccessFlagBits::eColorAttachmentWrite,
            vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, **color_image_, color_range);
        const vk::ImageMemoryBarrier depth_barrier(
            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits::eDepthStencilAttachmentRead,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, **depth_image_, depth_range);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {}, {}, {}, {color_barrier, depth_barrier});
    }

    [[nodiscard]] vk::RenderingAttachmentInfo MakeColorAttachment(
        vk::AttachmentLoadOp load_op, vk::ImageLayout layout) const {
        vk::RenderingAttachmentInfo attachment{};
        attachment.imageView = **color_view_;
        attachment.imageLayout = layout;
        attachment.resolveMode = vk::ResolveModeFlagBits::eNone;
        attachment.loadOp = load_op;
        attachment.storeOp = vk::AttachmentStoreOp::eStore;
        attachment.clearValue =
            vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
        return attachment;
    }

    [[nodiscard]] vk::RenderingAttachmentInfo MakeDepthAttachment(
        vk::AttachmentLoadOp load_op, vk::ImageLayout layout) const {
        vk::RenderingAttachmentInfo attachment{};
        attachment.imageView = **depth_view_;
        attachment.imageLayout = layout;
        attachment.resolveMode = vk::ResolveModeFlagBits::eNone;
        attachment.loadOp = load_op;
        attachment.storeOp = vk::AttachmentStoreOp::eStore;
        attachment.clearValue = vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0u));
        return attachment;
    }

    static void SetViewportAndScissor(vk::CommandBuffer cmd) {
        // The engine's scene viewport, not a test-local one, so this harness
        // renders through the same convention WorldTextPass has to use.
        cmd.setViewport(0, VulkanEngine::PipelinePass::SceneViewport(kTargetSize, kTargetSize));
        cmd.setScissor(0, vk::Rect2D({0, 0}, {kTargetSize, kTargetSize}));
    }

    void PushConstants(vk::CommandBuffer cmd) const {
        WorldTextConstants constants{};
        std::memcpy(constants.view_proj, &view_proj_, sizeof(constants.view_proj));
        constants.atlas_size[0] = static_cast<float>(kPageSize);
        constants.atlas_size[1] = static_cast<float>(kPageSize);
        cmd.pushConstants<WorldTextConstants>(
            **pipeline_layout_,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
    }

    void BindInstanceSet(vk::CommandBuffer cmd, vk::DescriptorSet instance_set) const {
        const std::array<vk::DescriptorSet, 1> set0{bindless_->GetDescriptorSet()};
        const std::array<vk::DescriptorSet, 1> set5{instance_set};
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 0, set0, {});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 5, set5, {});
    }

    void PushSlugConstants(vk::CommandBuffer cmd) const {
        SlugTextConstants constants{};
        std::memcpy(constants.view_proj, &view_proj_, sizeof(constants.view_proj));
        // The same viewport the dynamic state sets; hb_gpu_dilate needs it for
        // the half-pixel edge dilation.
        constants.viewport[0] = static_cast<float>(kTargetSize);
        constants.viewport[1] = static_cast<float>(kTargetSize);
        cmd.pushConstants<SlugTextConstants>(
            **pipeline_layout_,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
    }

    [[nodiscard]] static std::array<std::uint8_t, 4> Pixel(
        const std::vector<std::uint8_t>& pixels, std::uint32_t x, std::uint32_t y) {
        const std::size_t base = (static_cast<std::size_t>(y) * kTargetSize + x) * 4U;
        return {pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]};
    }

    // Bounding box of every pixel whose red channel rose above the background.
    [[nodiscard]] static InkBounds Ink(const std::vector<std::uint8_t>& pixels) {
        InkBounds bounds{};
        for (std::uint32_t y = 0; y < kTargetSize; ++y) {
            for (std::uint32_t x = 0; x < kTargetSize; ++x) {
                if (Pixel(pixels, x, y)[0] > kBackground[0]) {
                    bounds.min_x = std::min(bounds.min_x, x);
                    bounds.min_y = std::min(bounds.min_y, y);
                    bounds.max_x = std::max(bounds.max_x, x);
                    bounds.max_y = std::max(bounds.max_y, y);
                    ++bounds.count;
                }
            }
        }
        return bounds;
    }

    // Where the shader should put the quads: the four corners of every instance
    // projected through the same camera, so the assertion tests the GPU path
    // (instance layout, vertex pulling, view-projection) rather than the CPU
    // batch builder, which the device-free suite already covers.
    [[nodiscard]] PixelBounds ProjectInstances(
        const std::vector<WorldTextInstance>& instances) const {
        PixelBounds bounds{1.0e9f, 1.0e9f, -1.0e9f, -1.0e9f};
        for (const WorldTextInstance& instance : instances) {
            const glm::vec3 origin{instance.origin[0], instance.origin[1], instance.origin[2]};
            const glm::vec3 right{instance.right[0], instance.right[1], instance.right[2]};
            const glm::vec3 up{instance.up[0], instance.up[1], instance.up[2]};
            for (const float cx : {0.0f, 1.0f}) {
                for (const float cy : {0.0f, 1.0f}) {
                    const PixelPoint point = Project(view_proj_, origin + cx * right + cy * up);
                    bounds.min_x = std::min(bounds.min_x, point.x);
                    bounds.min_y = std::min(bounds.min_y, point.y);
                    bounds.max_x = std::max(bounds.max_x, point.x);
                    bounds.max_y = std::max(bounds.max_y, point.y);
                }
            }
        }
        return bounds;
    }

    [[nodiscard]] PixelBounds ProjectInstances(const std::vector<SlugTextInstance>& instances) const {
        PixelBounds bounds{1.0e9f, 1.0e9f, -1.0e9f, -1.0e9f};
        for (const SlugTextInstance& instance : instances) {
            const glm::vec3 origin{instance.origin[0], instance.origin[1], instance.origin[2]};
            const glm::vec3 right{instance.right[0], instance.right[1], instance.right[2]};
            const glm::vec3 up{instance.up[0], instance.up[1], instance.up[2]};
            for (const float cx : {0.0f, 1.0f}) {
                for (const float cy : {0.0f, 1.0f}) {
                    const PixelPoint point = Project(view_proj_, origin + cx * right + cy * up);
                    bounds.min_x = std::min(bounds.min_x, point.x);
                    bounds.min_y = std::min(bounds.min_y, point.y);
                    bounds.max_x = std::max(bounds.max_x, point.x);
                    bounds.max_y = std::max(bounds.max_y, point.y);
                }
            }
        }
        return bounds;
    }

    // Pixels with a partial red channel: the analytic rasterizer's coverage is a
    // fraction in (0, 1), so an edge pixel composites to something strictly
    // between the black background and the full white foreground. A hard 0/1
    // step would produce none of these.
    [[nodiscard]] static std::uint32_t FractionalInk(const std::vector<std::uint8_t>& pixels) {
        std::uint32_t count = 0;
        for (std::uint32_t y = 0; y < kTargetSize; ++y) {
            for (std::uint32_t x = 0; x < kTargetSize; ++x) {
                const std::uint8_t red = Pixel(pixels, x, y)[0];
                if (red > kBackground[0] && red < 255u) {
                    ++count;
                }
            }
        }
        return count;
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    StagingPool staging_{};
    SamplerCache samplers_{};
    std::unique_ptr<BindlessManager> bindless_{};
    std::unique_ptr<ShaderManager> shaders_{};

    std::unique_ptr<GpuBuffer> instance_buffer_{};
    std::unique_ptr<vk::raii::DescriptorSetLayout> instance_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> instance_pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> text_set_{};
    std::unique_ptr<vk::raii::DescriptorSet> occluder_set_{};
    std::unique_ptr<vk::raii::DescriptorSet> slug_set_{};
    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    std::unique_ptr<PipelineProduct> text_pipeline_{};
    std::unique_ptr<PipelineProduct> no_depth_pipeline_{};
    std::unique_ptr<PipelineProduct> occluder_pipeline_{};
    std::unique_ptr<PipelineProduct> slug_pipeline_{};
    vk::Format depth_format_ = vk::Format::eUndefined;
    std::unique_ptr<vk::raii::Image> color_image_{};
    std::unique_ptr<vk::raii::DeviceMemory> color_memory_{};
    std::unique_ptr<vk::raii::ImageView> color_view_{};
    std::unique_ptr<vk::raii::Image> depth_image_{};
    std::unique_ptr<vk::raii::DeviceMemory> depth_memory_{};
    std::unique_ptr<vk::raii::ImageView> depth_view_{};
    std::unique_ptr<vk::raii::Buffer> readback_buffer_{};
    std::unique_ptr<vk::raii::DeviceMemory> readback_memory_{};
    vk::ImageLayout color_layout_ = vk::ImageLayout::eUndefined;
    vk::ImageLayout depth_layout_ = vk::ImageLayout::eUndefined;
    glm::mat4 view_proj_{1.0f};

    VulkanEngine::ResourceManager manager_{};
    VulkanEngine::ResourceHandle<VulkanEngine::FontResource> handle_{};
    std::shared_ptr<FontFace> face_{};
    std::unique_ptr<MsdfGenerator> generator_{};
    std::unique_ptr<GlyphAtlasGpu> atlas_gpu_{};
    std::unique_ptr<GpuTextBlobBuffer> blobs_{};
    GlyphBlobEncoder slug_encoder_{};
    ComponentRegistry registry_{};
};

// Ink appears where the component's world transform says: the string's glyphs
// cover pixels inside the projected world-space AABB of the quads the builder
// produced, and the corners far from the text stay exactly the cleared
// background.
TEST_F(WorldTextGpuTest, InkAppearsInsideTheProjectedWorldRegion) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const auto instances = QueueText("Hi!", 0.5f, glm::vec3{-0.2f, 0.35f, 0.0f}, white);
    ASSERT_FALSE(instances.empty());

    const auto pixels = Render(RenderRequest{.text = instances});
    const InkBounds ink = Ink(pixels);
    ASSERT_FALSE(ink.empty()) << "the component drew no ink";

    // The quad world AABB (which includes the distance-field gutter) projected
    // through the camera bounds every fragment the shader can emit. Signed math
    // so a zero bound cannot wrap around and mask a failure.
    const PixelBounds expected = ProjectInstances(instances);
    EXPECT_GE(expected.min_x, 0.0f);
    EXPECT_LE(expected.max_x, static_cast<float>(kTargetSize));
    const int expected_min_x = static_cast<int>(std::floor(expected.min_x));
    const int expected_max_x = static_cast<int>(std::ceil(expected.max_x));
    const int expected_min_y = static_cast<int>(std::floor(expected.min_y));
    const int expected_max_y = static_cast<int>(std::ceil(expected.max_y));
    EXPECT_GE(static_cast<int>(ink.min_x), expected_min_x - 1);
    EXPECT_LE(static_cast<int>(ink.max_x), expected_max_x + 1);
    EXPECT_GE(static_cast<int>(ink.min_y), expected_min_y - 1);
    EXPECT_LE(static_cast<int>(ink.max_y), expected_max_y + 1);
    // The ink is a real glyph, not a stray pixel: the two lines above would also
    // hold for a single antialiased texel.
    EXPECT_GT(ink.count, 200u);
    EXPECT_GT(ink.width(), 20u);
    EXPECT_GT(ink.height(), 20u);

    // Far from the text, the target is byte-identical to the clear.
    for (std::uint32_t y = 0; y < 16; ++y) {
        for (std::uint32_t x = 0; x < 16; ++x) {
            EXPECT_EQ(Pixel(pixels, x, y), kBackground)
                << "untouched pixel changed at (" << x << ", " << y << ")";
        }
    }
    for (std::uint32_t y = kTargetSize - 32; y < kTargetSize; ++y) {
        for (std::uint32_t x = kTargetSize - 32; x < kTargetSize; ++x) {
            EXPECT_EQ(Pixel(pixels, x, y), kBackground)
                << "untouched pixel changed at (" << x << ", " << y << ")";
        }
    }
}

// Depth ordering is real. Scene geometry (a depth-writing quad) fills the depth
// buffer at world z = -0.5, and a string drawn at world z = +0.5 behind it is
// rejected: no ink survives where the geometry is. The identical scene through a
// pipeline with depth testing disabled draws ink there, so this assertion fails
// if depth testing were off -- it is not trivially true.
TEST_F(WorldTextGpuTest, NearerGeometryOccludesFartherText) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const auto text = QueueText("Hi!", 0.5f, glm::vec3{-0.2f, 0.35f, 0.5f}, white);
    ASSERT_FALSE(text.empty());
    const auto occluders = Occluders();
    ASSERT_FALSE(occluders.empty());

    // Reference: the same depth-tested text with nothing in front of it really
    // does cover the occluder's half of the target, so the occlusion below is
    // the depth test rejecting the fragments and not the string missing them.
    const InkBounds unoccluded =
        Ink(Render(RenderRequest{.text = text, .depth_test = true}));
    ASSERT_FALSE(unoccluded.empty());
    EXPECT_LT(unoccluded.min_x, kOccluderRightPixel)
        << "the string does not reach the occluder region; the test is vacuous";

    const InkBounds tested = Ink(Render(RenderRequest{
        .text = text, .occluders = occluders, .depth_test = true}));
    ASSERT_FALSE(tested.empty()) << "the unoccluded part of the string drew no ink";
    // The occluder covers framebuffer x < 128, so every surviving ink pixel is
    // to its right.
    EXPECT_GE(tested.min_x, kOccluderRightPixel)
        << "text survived in front of geometry written nearer to the camera";
    EXPECT_GT(tested.count, 50u);
    EXPECT_LT(tested.count, unoccluded.count);

    const InkBounds control = Ink(Render(RenderRequest{
        .text = text, .occluders = occluders, .depth_test = false}));
    ASSERT_FALSE(control.empty());
    // With the depth test disabled the very same fragments are drawn over the
    // occluder, which is exactly what the depth-tested assertion above forbids.
    EXPECT_LT(control.min_x, kOccluderRightPixel)
        << "the control did not draw over the occluder; the test is vacuous";
    EXPECT_GT(control.count, tested.count);
}

// Scaling: a larger world_height covers a strictly larger ink box for the same
// string and the same anchor.
TEST_F(WorldTextGpuTest, LargerWorldHeightCoversMoreInk) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const auto small = Ink(Render(RenderRequest{
        .text = QueueText("Hi!", 0.35f, glm::vec3{-0.2f, 0.35f, 0.0f}, white)}));
    const auto large = Ink(Render(RenderRequest{
        .text = QueueText("Hi!", 0.6f, glm::vec3{-0.2f, 0.35f, 0.0f}, white)}));
    ASSERT_FALSE(small.empty());
    ASSERT_FALSE(large.empty());
    EXPECT_GT(large.width(), small.width());
    EXPECT_GT(large.height(), small.height());
    EXPECT_GT(large.count, small.count);
}

// Nothing to draw draws nothing: a component whose content is empty or inkless
// contributes no instances, and the frame is byte-identical to the clear-only
// frame. A real string does change that frame, so the comparison is not vacuous.
TEST_F(WorldTextGpuTest, EmptyAndInklessComponentsLeaveTheTargetUnchanged) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec3 anchor{-0.2f, 0.35f, 0.0f};

    const auto cleared = Render(RenderRequest{});
    EXPECT_TRUE(Ink(cleared).empty());

    const auto empty_content = QueueText("", 0.5f, anchor, white);
    ASSERT_TRUE(empty_content.empty());
    const auto inkless_content = QueueText("   ", 0.5f, anchor, white);
    ASSERT_TRUE(inkless_content.empty());

    const auto empty_frame = Render(RenderRequest{.text = empty_content});
    const auto inkless_frame = Render(RenderRequest{.text = inkless_content});
    EXPECT_EQ(empty_frame, cleared);
    EXPECT_EQ(inkless_frame, cleared);

    const auto drawn = Render(RenderRequest{.text = QueueText("Hi!", 0.5f, anchor, white)});
    EXPECT_NE(drawn, cleared) << "the frame comparison is vacuous: nothing draws at all";
    EXPECT_FALSE(Ink(drawn).empty());
}

// The Slug backend renders ink where the glyphs' projected em boxes are, through
// the shipped slug_text modules and a real blob storage buffer. The buffer is
// deliberately smaller than the first glyph's blob, so the stores inside
// QueueSlugText grow it and the frame still reads every glyph.
TEST_F(WorldTextGpuTest, SlugInkAppearsInsideTheProjectedWorldRegion) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const auto instances = QueueSlugText("Hi!", 0.5f, glm::vec3{-0.2f, 0.35f, 0.0f}, white);
    ASSERT_FALSE(instances.empty());
    EXPECT_GT(blobs_->Capacity(), 64u) << "the blob buffer never grew past its initial size";

    const auto pixels = Render(RenderRequest{.slug_text = instances});
    const InkBounds ink = Ink(pixels);
    ASSERT_FALSE(ink.empty()) << "the Slug backend drew no ink";

    const PixelBounds expected = ProjectInstances(instances);
    const int expected_min_x = static_cast<int>(std::floor(expected.min_x));
    const int expected_max_x = static_cast<int>(std::ceil(expected.max_x));
    const int expected_min_y = static_cast<int>(std::floor(expected.min_y));
    const int expected_max_y = static_cast<int>(std::ceil(expected.max_y));
    // The em box is the *undilated* quad; hb_gpu_dilate pushes each corner out
    // by half a screen pixel along its outward normal, so the analytic ink may
    // extend a pixel or two past the projected box. The MSDF path does not need
    // this slack because its quad already carries the distance-field gutter.
    constexpr int kDilationSlack = 3;
    EXPECT_GE(static_cast<int>(ink.min_x), expected_min_x - kDilationSlack);
    EXPECT_LE(static_cast<int>(ink.max_x), expected_max_x + kDilationSlack);
    EXPECT_GE(static_cast<int>(ink.min_y), expected_min_y - kDilationSlack);
    EXPECT_LE(static_cast<int>(ink.max_y), expected_max_y + kDilationSlack);
    EXPECT_GT(ink.count, 200u);
    EXPECT_GT(ink.width(), 20u);
    EXPECT_GT(ink.height(), 20u);

    for (std::uint32_t y = 0; y < 16; ++y) {
        for (std::uint32_t x = 0; x < 16; ++x) {
            EXPECT_EQ(Pixel(pixels, x, y), kBackground)
                << "untouched pixel changed at (" << x << ", " << y << ")";
        }
    }
}

// Cross-check that the port is not wildly wrong: the same string at the same
// size and anchor renders through Slug and through MSDF into comparable ink
// bounding boxes. The assertion is deliberately loose (it is a cross-check, not
// a golden image) and structural -- every edge and both extents.
TEST_F(WorldTextGpuTest, SlugAndMsdfInkBoundsAreComparable) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec3 anchor{-0.2f, 0.35f, 0.0f};

    const auto msdf_instances = QueueText("Hi!", 0.5f, anchor, white);
    const auto slug_instances = QueueSlugText("Hi!", 0.5f, anchor, white);
    ASSERT_FALSE(msdf_instances.empty());
    ASSERT_FALSE(slug_instances.empty());

    const InkBounds msdf = Ink(Render(RenderRequest{.text = msdf_instances}));
    const InkBounds slug = Ink(Render(RenderRequest{.slug_text = slug_instances}));
    ASSERT_FALSE(msdf.empty()) << "the MSDF reference drew no ink";
    ASSERT_FALSE(slug.empty()) << "the Slug backend drew no ink";

    const float tolerance_x =
        std::max(4.0f, 0.2f * static_cast<float>(msdf.width()));
    const float tolerance_y =
        std::max(4.0f, 0.2f * static_cast<float>(msdf.height()));
    EXPECT_NEAR(static_cast<float>(slug.min_x), static_cast<float>(msdf.min_x), tolerance_x);
    EXPECT_NEAR(static_cast<float>(slug.max_x), static_cast<float>(msdf.max_x), tolerance_x);
    EXPECT_NEAR(static_cast<float>(slug.min_y), static_cast<float>(msdf.min_y), tolerance_y);
    EXPECT_NEAR(static_cast<float>(slug.max_y), static_cast<float>(msdf.max_y), tolerance_y);
    EXPECT_NEAR(static_cast<float>(slug.width()), static_cast<float>(msdf.width()), tolerance_x);
    EXPECT_NEAR(static_cast<float>(slug.height()), static_cast<float>(msdf.height()),
                tolerance_y);
}

// The property the analytic rasterizer exists to provide: edge coverage is a
// fraction, not a hard 0/1 step. Partial red pixels between the black background
// and the full foreground can only come from fractional coverage.
TEST_F(WorldTextGpuTest, SlugEdgeCoverageIsAntialiased) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const auto instances = QueueSlugText("Hi!", 0.5f, glm::vec3{-0.2f, 0.35f, 0.0f}, white);
    ASSERT_FALSE(instances.empty());

    const auto pixels = Render(RenderRequest{.slug_text = instances});
    ASSERT_FALSE(Ink(pixels).empty()) << "the Slug backend drew no ink";
    EXPECT_GT(FractionalInk(pixels), 30u)
        << "the Slug edge is a hard 0/1 step rather than analytic coverage";

    // The MSDF path is antialiased too, so the assertion above is not just
    // "the frame has some pixels".
    const auto msdf_pixels = Render(RenderRequest{
        .text = QueueText("Hi!", 0.5f, glm::vec3{-0.2f, 0.35f, 0.0f}, white)});
    EXPECT_GT(FractionalInk(msdf_pixels), 30u);
}

// Resolution independence: the encoded outline is size independent and the
// fragment evaluates it at the drawn size, so the ink box scales with the em
// size instead of being clamped to a fixed field resolution.
TEST_F(WorldTextGpuTest, SlugInkBoundsScaleWithSize) {
    const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec3 anchor{-0.2f, 0.35f, 0.0f};

    const InkBounds small = Ink(
        Render(RenderRequest{.slug_text = QueueSlugText("H", 0.25f, anchor, white)}));
    const InkBounds large = Ink(
        Render(RenderRequest{.slug_text = QueueSlugText("H", 0.5f, anchor, white)}));
    ASSERT_FALSE(small.empty());
    ASSERT_FALSE(large.empty());
    EXPECT_GT(large.width(), small.width());
    EXPECT_GT(large.height(), small.height());

    const float ratio_width =
        static_cast<float>(large.width()) / static_cast<float>(small.width());
    const float ratio_height =
        static_cast<float>(large.height()) / static_cast<float>(small.height());
    // Doubling the em size doubles the ink box, within the couple of pixels a
    // rasterized edge can differ by.
    EXPECT_NEAR(ratio_width, 2.0f, 0.5f);
    EXPECT_NEAR(ratio_height, 2.0f, 0.5f);
}

}  // namespace
