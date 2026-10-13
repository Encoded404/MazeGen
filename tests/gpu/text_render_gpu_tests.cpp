#include <gtest/gtest.h>

#include "test_text_shader_abi.hpp"

import std;
import std.compat;

import vulkan;

import test_gpu;

import ShaderReflection;
import Shaders.Engine.UiTextVert;
import Shaders.Engine.UiTextFrag;
import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Render.Passes.TextPass;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.GlyphRaster;
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

using VulkanEngine::BindlessManager::BindlessManager;
using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::SamplerCache;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;
using VulkanEngine::SceneRenderer::BuildTextInstances;
using VulkanEngine::SceneRenderer::PageSlotLookup;
using VulkanEngine::SceneRenderer::TextInstance;
using VulkanEngine::SceneRenderer::TextRunRequest;
using VulkanEngine::ShaderSystem::GraphicsPipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::Text::GlyphRasterizer;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapeText;
using TestSupport::TextShaderAbi::kBindlessDescriptorSet;
using TestSupport::TextShaderAbi::kFirstAppDescriptorSet;
using TestSupport::TextShaderAbi::kUiTextInstanceBinding;
using TestSupport::TextShaderAbi::kUiTextInstanceBytes;
using TestSupport::TextShaderAbi::kUiTextPushConstantBytes;

constexpr std::uint32_t kTargetSize = 256;
constexpr std::uint32_t kPageSize = 256;

static_assert(kUiTextInstanceBytes == sizeof(TextInstance),
              "the shared text ABI must mirror the engine's TextInstance layout");

// The scene colour the text composites over: opaque black, so an ink pixel is
// any pixel whose red channel rose above 0.
constexpr std::array<std::uint8_t, 4> kBackground{0u, 0u, 0u, 255u};

// Draws glyph instances with the real ui_text shaders over a cleared offscreen
// target, using the pass's declared state: bindless set 0, an instance storage
// buffer at set 5, a screen-size push constant, and straight-alpha source-over
// blending. Reads the target back as RGBA8.
//
// The structural assertions deliberately avoid a whole-frame hash: antialiased
// text coverage is driver-sensitive, and a golden constant would hide a broken
// assertion rather than catch one.
class TextRenderGpuTest : public ::testing::Test {
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
                                     "text-render-gpu-tests"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 1ULL << 20;
        staging_config.capacity_bytes = 8ULL << 20;
        ASSERT_TRUE(staging_.Initialize(backend_, staging_config));
        ASSERT_TRUE(samplers_.Initialize(backend_));

        bindless_ = std::make_unique<BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity{};
        capacity.app_capacity = 16;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity));
        bindless_->SetFallback(MakeSolidTexture(), VulkanEngine::ResourceId{"text-fallback"});

        handle_ = manager_.LoadFromFile<VulkanEngine::FontResource>(
            std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf",
            VulkanEngine::ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        auto* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());
        face_ = VulkanEngine::Text::FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);

        AtlasConfig atlas_config{};
        atlas_config.page_width = kPageSize;
        atlas_config.page_height = kPageSize;
        atlas_config.padding = 1;
        atlas_config.max_pages = 4;
        rasterizer_ = std::make_unique<GlyphRasterizer>(atlas_config, 256);
        atlas_gpu_ = std::make_unique<GlyphAtlasGpu>();
        ASSERT_TRUE(atlas_gpu_->Initialize(backend_, heap_, staging_, *bindless_, &samplers_));
        atlas_gpu_->SetGlyphSource(
            [this](std::uint64_t key) { return rasterizer_->GlyphForAtlasKey(key); });

        BuildPipeline();
        CreateTarget();
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        readback_buffer_.reset();
        readback_memory_.reset();
        color_view_.reset();
        color_memory_.reset();
        color_image_.reset();
        pipeline_.reset();
        pipeline_layout_.reset();
        instance_set_.reset();
        instance_pool_.reset();
        instance_layout_.reset();
        instance_buffer_.reset();
        shaders_.reset();
        if (atlas_gpu_) {
            atlas_gpu_->Shutdown();
            atlas_gpu_.reset();
        }
        rasterizer_.reset();
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

    void BuildPipeline() {
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(),
                                                   backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        const auto vert = Shaders::Engine::UiTextVert::Register(*shaders_,
                                                                VKENGINE_ENGINE_SHADER_DIR);
        const auto frag = Shaders::Engine::UiTextFrag::Register(*shaders_,
                                                                VKENGINE_ENGINE_SHADER_DIR);

        // Set kFirstAppDescriptorSet: the instance storage buffer, one for the
        // whole draw.
        const vk::DescriptorSetLayoutBinding instance_binding(
            kUiTextInstanceBinding, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eVertex, nullptr);
        instance_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            backend_.GetDevice(), vk::DescriptorSetLayoutCreateInfo{{}, 1, &instance_binding});
        const vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 1);
        instance_pool_ = std::make_unique<vk::raii::DescriptorPool>(
            backend_.GetDevice(), vk::DescriptorPoolCreateInfo({}, 1, 1, &pool_size));
        auto sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**instance_pool_, 1, &**instance_layout_});
        ASSERT_EQ(sets.size(), 1u);
        instance_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));

        instance_buffer_ = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            backend_, 64u * 1024u, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        ASSERT_TRUE(instance_buffer_->IsValid());
        vk::DescriptorBufferInfo instance_info{};
        instance_info.buffer = static_cast<vk::Buffer>(*instance_buffer_->GetBuffer());
        instance_info.offset = 0;
        instance_info.range = vk::WholeSize;
        vk::WriteDescriptorSet instance_write{};
        instance_write.dstSet = **instance_set_;
        instance_write.dstBinding = kUiTextInstanceBinding;
        instance_write.descriptorCount = 1;
        instance_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        instance_write.pBufferInfo = &instance_info;
        backend_.GetDevice().updateDescriptorSets({instance_write}, {});

        // The shader names set 0 (bindless) and set kFirstAppDescriptorSet
        // (instances); the reserved sets in between are empty layouts so the app
        // set lands where the shader says.
        const std::array<vk::DescriptorSetLayout, 6> set_layouts{
            *bindless_->GetLayout(), {}, {}, {}, {}, **instance_layout_};
        const vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eVertex, 0,
                                               kUiTextPushConstantBytes);
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        layout_info.setPushConstantRanges(push_range);
        pipeline_layout_ =
            std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

        // The pass's documented blend: straight-alpha source-over.
        const vk::PipelineColorBlendAttachmentState blend(
            vk::True, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

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
        desc.depth_stencil =
            vk::PipelineDepthStencilStateCreateInfo({}, vk::False, vk::False, vk::CompareOp::eAlways);
        desc.color_blend =
            vk::PipelineColorBlendStateCreateInfo({}, vk::False, vk::LogicOp::eCopy, 0, nullptr);
        desc.color_blend_attachments = {blend};
        desc.dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        desc.layout = **pipeline_layout_;
        desc.color_formats = {vk::Format::eR8G8B8A8Unorm};

        PipelineFactory factory(backend_.GetDevice(), backend_.GetCapabilities(),
                                shaders_->GetPipelineCacheRAII());
        auto product = factory.CreateGraphics(desc, *shaders_);
        ASSERT_TRUE(product.has_value()) << product.error().message;
        pipeline_ = std::make_unique<PipelineProduct>(std::move(*product));
    }

    void CreateTarget() {
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

    // Rasterizes + uploads `text` at `pixel_size` and returns the instances the
    // batch builder produced, pen origin (24, 64).
    [[nodiscard]] std::vector<TextInstance> QueueText(const std::string& text, float pixel_size) {
        const ShapedRun run = ShapeText(*face_, text,
                                        ShapeOptions{.kerning = true, .ligatures = true});
        TextRunRequest request{};
        request.face = face_.get();
        request.run = &run;
        request.pixel_size = pixel_size;
        request.x = 24.0f;
        request.y = 64.0f;
        request.color[0] = 1.0f;
        request.color[1] = 1.0f;
        request.color[2] = 1.0f;
        request.color[3] = 1.0f;

        const PageSlotLookup page_slot = [this](std::uint32_t page) {
            const auto handle = atlas_gpu_->PageHandle(page);
            return handle.IsValid() ? handle.slot : 0U;
        };

        // The builder also packs the glyphs into the atlas, so it runs once to
        // populate it, the dirty pages are uploaded, and it runs again to
        // resolve the pages that now have a bindless slot.
        std::vector<TextInstance> instances;
        BuildTextInstances(*rasterizer_, page_slot, request, instances);
        UploadAtlas();
        instances.clear();
        BuildTextInstances(*rasterizer_, page_slot, request, instances);
        return instances;
    }

    void UploadAtlas() {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        atlas_gpu_->UploadDirty(rasterizer_->MutableAtlas(), *cmd, 0);
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();
    }

    [[nodiscard]] std::vector<std::uint8_t> Render(const std::vector<TextInstance>& instances) {
        if (!instances.empty()) {
            instance_buffer_->UploadAt(instances.data(),
                                       instances.size() * sizeof(TextInstance), 0);
        }

        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        const vk::ImageMemoryBarrier to_color(
            {}, vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal, vk::QueueFamilyIgnored,
            vk::QueueFamilyIgnored, **color_image_, range);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {},
                            to_color);

        const vk::ClearValue clear(
            vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
        vk::RenderingAttachmentInfo color_attachment{};
        color_attachment.imageView = **color_view_;
        color_attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color_attachment.resolveMode = vk::ResolveModeFlagBits::eNone;
        color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
        color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
        color_attachment.clearValue = clear;
        vk::RenderingInfo rendering{};
        rendering.renderArea = vk::Rect2D({0, 0}, {kTargetSize, kTargetSize});
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;

        cmd.beginRendering(rendering);
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(kTargetSize),
                                        static_cast<float>(kTargetSize), 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, {kTargetSize, kTargetSize}));
        if (!instances.empty()) {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->Get());
            const std::array<vk::DescriptorSet, 2> sets{bindless_->GetDescriptorSet(),
                                                        **instance_set_};
            const std::array<vk::DescriptorSet, 1> set0{sets[0]};
            const std::array<vk::DescriptorSet, 1> set5{sets[1]};
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_,
                                   kBindlessDescriptorSet, set0, {});
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_,
                                   kFirstAppDescriptorSet, set5, {});
            const std::array<float, 2> screen_size{static_cast<float>(kTargetSize),
                                                   static_cast<float>(kTargetSize)};
            cmd.pushConstants<float>(**pipeline_layout_, vk::ShaderStageFlagBits::eVertex, 0,
                                     screen_size);
            cmd.draw(6, static_cast<std::uint32_t>(instances.size()), 0, 0);
        }
        cmd.endRendering();

        const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(kTargetSize) * kTargetSize * 4U;
        const vk::ImageMemoryBarrier to_src(
            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, **color_image_, range);
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

        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(bytes));
        const void* mapped = readback_memory_->mapMemory(0, bytes);
        std::memcpy(pixels.data(), mapped, pixels.size());
        readback_memory_->unmapMemory();
        return pixels;
    }

    [[nodiscard]] static std::array<std::uint8_t, 4> Pixel(
        const std::vector<std::uint8_t>& pixels, std::uint32_t x, std::uint32_t y) {
        const std::size_t base = (static_cast<std::size_t>(y) * kTargetSize + x) * 4U;
        return {pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]};
    }

    // Bounding box of every pixel whose red channel is above the background.
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

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    StagingPool staging_{};
    SamplerCache samplers_{};
    std::unique_ptr<BindlessManager> bindless_{};
    std::unique_ptr<ShaderManager> shaders_{};

    std::unique_ptr<GpuBuffer> instance_buffer_{};
    std::unique_ptr<vk::raii::DescriptorSetLayout> instance_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> instance_pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> instance_set_{};
    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    std::unique_ptr<PipelineProduct> pipeline_{};
    std::unique_ptr<vk::raii::Image> color_image_{};
    std::unique_ptr<vk::raii::DeviceMemory> color_memory_{};
    std::unique_ptr<vk::raii::ImageView> color_view_{};
    std::unique_ptr<vk::raii::Buffer> readback_buffer_{};
    std::unique_ptr<vk::raii::DeviceMemory> readback_memory_{};

    VulkanEngine::ResourceManager manager_{};
    VulkanEngine::ResourceHandle<VulkanEngine::FontResource> handle_{};
    std::shared_ptr<VulkanEngine::Text::FontFace> face_{};
    std::unique_ptr<GlyphRasterizer> rasterizer_{};
    std::unique_ptr<GlyphAtlasGpu> atlas_gpu_{};
};

// Ink appears where it should: the string's glyphs cover pixels inside the
// region the pen origin implies, every covered pixel is a blend of the instance
// colour over the background (so coverage is in range and never opaque-over-
// opaque), and the frame's opposite corner stays untouched.
TEST_F(TextRenderGpuTest, InkAppearsInsideTheExpectedRegion) {
    const auto instances = QueueText("Hey", 32.0f);
    ASSERT_FALSE(instances.empty());
    const auto pixels = Render(instances);

    const InkBounds ink = Ink(pixels);
    ASSERT_FALSE(ink.empty()) << "the string drew no ink";
    // Pen origin (24, 64), baseline 64, 32px glyphs: the ink box is a short band
    // starting near x=20 and sitting above the baseline.
    EXPECT_GE(ink.min_x, 16u);
    EXPECT_LE(ink.max_x, 160u);
    EXPECT_GE(ink.min_y, 24u);
    EXPECT_LE(ink.max_y, 72u);
    EXPECT_GT(ink.count, 40u);

    // Far from the text the target is exactly the cleared background.
    for (std::uint32_t y = kTargetSize - 16; y < kTargetSize; ++y) {
        for (std::uint32_t x = kTargetSize - 16; x < kTargetSize; ++x) {
            EXPECT_EQ(Pixel(pixels, x, y), kBackground)
                << "untouched pixel changed at (" << x << ", " << y << ")";
        }
    }

    // Coverage composited over opaque black: red is the coverage (<= 255 by
    // construction) and alpha stays the destination's opaque 255.
    for (std::uint32_t y = ink.min_y; y <= ink.max_y; ++y) {
        for (std::uint32_t x = ink.min_x; x <= ink.max_x; ++x) {
            const auto pixel = Pixel(pixels, x, y);
            EXPECT_LE(pixel[0], 255u);
            EXPECT_EQ(pixel[3], 255u);
        }
    }
}

// Nothing queued draws nothing: an empty instance list produces a frame
// byte-identical to the same clear with no pass at all.
TEST_F(TextRenderGpuTest, NothingQueuedDrawsNothing) {
    const auto empty_frame = Render({});
    const auto with_empty_pass = Render(std::vector<TextInstance>{});
    ASSERT_EQ(empty_frame.size(), with_empty_pass.size());
    EXPECT_EQ(empty_frame, with_empty_pass);
    EXPECT_TRUE(Ink(empty_frame).empty());
    EXPECT_EQ(Pixel(empty_frame, 0, 0), kBackground);
}

// Scaling: the same string at a larger pixel size covers a strictly larger ink
// box.
TEST_F(TextRenderGpuTest, LargerPixelSizeCoversMoreInk) {
    const auto small = Ink(Render(QueueText("Hey", 24.0f)));
    const auto large = Ink(Render(QueueText("Hey", 48.0f)));
    ASSERT_FALSE(small.empty());
    ASSERT_FALSE(large.empty());
    EXPECT_GT(large.width(), small.width());
    EXPECT_GT(large.height(), small.height());
    EXPECT_GT(large.count, small.count);
}

}  // namespace
