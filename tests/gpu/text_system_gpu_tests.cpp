#include <gtest/gtest.h>

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
import VulkanEngine.Text.TextSystem;
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
using VulkanEngine::BindlessManager::TextureHandle;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::SamplerCache;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;
using VulkanEngine::ResourceId;
using VulkanEngine::SceneRenderer::TextInstance;
using VulkanEngine::SceneRenderer::TextPass;
using VulkanEngine::ShaderSystem::GraphicsPipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::Text::TextSystem;

constexpr std::uint32_t kTargetSize = 256;
constexpr std::uint32_t kPageSize = 256;

// The scene colour the text composites over: opaque black, so an ink pixel is
// any pixel whose red channel rose above 0.
constexpr std::array<std::uint8_t, 4> kBackground{0u, 0u, 0u, 255u};

// Drives the *public* text path: a real TextSystem with a device, a real
// TextPass attached through the submission seam, and the pass's queued instances
// drawn with the shipped ui_text shaders over a cleared offscreen target. This
// is the same render harness tests/gpu/text_render_gpu_tests.cpp uses; what is
// new here is that the instances come from TextSystem::SubmitScreenText rather
// than from the batch builder directly.
class TextSystemGpuTest : public ::testing::Test {
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

    // A scratch copy of the shipped Lato, so the reload path points at a file
    // the test owns rather than the read-only fixture.
    class TempFontFile final {
    public:
        TempFontFile() {
            const auto stamp = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            std::random_device rd;
            path_ = std::filesystem::temp_directory_path() /
                    ("ve5_text_system_gpu_" + std::to_string(stamp) + "_" +
                     std::to_string(rd()) + ".ttf");
        }
        ~TempFontFile() {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
        [[nodiscard]] const std::filesystem::path& Path() const noexcept { return path_; }
        [[nodiscard]] bool Write(std::span<const std::byte> bytes) const {
            std::ofstream file(path_, std::ios::binary | std::ios::trunc);
            if (!file) {
                return false;
            }
            file.write(reinterpret_cast<const char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()));
            return static_cast<bool>(file);
        }

    private:
        std::filesystem::path path_;
    };

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        ASSERT_TRUE(heap_.Initialize(backend_, ImageHeapConfig{.block_size = 8ULL << 20},
                                     "text-system-gpu-tests"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 1ULL << 20;
        staging_config.capacity_bytes = 8ULL << 20;
        ASSERT_TRUE(staging_.Initialize(backend_, staging_config));
        ASSERT_TRUE(samplers_.Initialize(backend_));

        bindless_ = std::make_unique<BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity{};
        capacity.app_capacity = 16;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity));
        bindless_->SetFallback(MakeSolidTexture(), ResourceId{"text-system-fallback"});

        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(),
                                                   backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());

        // The shipped ui_text modules, exactly as the engine registers them.
        ui_text_vert_ = Shaders::Engine::UiTextVert::Register(*shaders_, VKENGINE_ENGINE_SHADER_DIR);
        ui_text_frag_ = Shaders::Engine::UiTextFrag::Register(*shaders_, VKENGINE_ENGINE_SHADER_DIR);

        // The pass the renderer would own; the test drives its queue directly.
        pass_ = std::make_unique<TextPass>(&backend_, ui_text_vert_, ui_text_frag_);

        TextSystem::Config config{};
        config.atlas.page_width = kPageSize;
        config.atlas.page_height = kPageSize;
        config.atlas.padding = 1;
        config.atlas.max_pages = 4;
        text_system_ = std::make_unique<TextSystem>(config);
        text_system_->SetResourceManager(manager_);
        ASSERT_TRUE(text_system_->InitializeGpu(backend_, heap_, staging_, *bindless_, &samplers_));

        // A writable copy of the fixture so a reload reads a real file.
        const auto source = std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
        std::ifstream file(source, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(file);
        const std::streamsize size = file.tellg();
        ASSERT_GT(size, 0);
        file.seekg(0, std::ios::beg);
        font_bytes_.resize(static_cast<std::size_t>(size));
        ASSERT_TRUE(file.read(reinterpret_cast<char*>(font_bytes_.data()), size));
        ASSERT_TRUE(font_file_.Write(font_bytes_));

        font_id_ = ResourceId{font_file_.Path().string()};
        face_ = text_system_->LoadFontFromPath(font_file_.Path());
        ASSERT_NE(face_, nullptr);

        // The engine's seam: the system queues into this pass.
        text_system_->SetTextPass(pass_.get());

        BuildPipeline();
        CreateTarget();
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        // Detach before the pass dies so the pass never holds a dangling hook.
        if (text_system_) {
            text_system_->SetTextPass(nullptr);
        }
        pass_.reset();
        if (text_system_) {
            text_system_->Shutdown();
            text_system_.reset();
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
        const vk::DescriptorSetLayoutBinding instance_binding(
            0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
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
        instance_write.dstBinding = 0;
        instance_write.descriptorCount = 1;
        instance_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        instance_write.pBufferInfo = &instance_info;
        backend_.GetDevice().updateDescriptorSets({instance_write}, {});

        const std::array<vk::DescriptorSetLayout, 6> set_layouts{
            *bindless_->GetLayout(), {}, {}, {}, {}, **instance_layout_};
        const vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eVertex, 0,
                                               2U * sizeof(float));
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        layout_info.setPushConstantRanges(push_range);
        pipeline_layout_ =
            std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

        const vk::PipelineColorBlendAttachmentState blend(
            vk::True, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

        GraphicsPipelineDesc desc{};
        desc.vertex_shader = ui_text_vert_;
        desc.fragment_shader = ui_text_frag_;
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

    // Submits through the public entry point, snapshots the instances the pass
    // queued, and clears the pass so the next submit starts from an empty frame.
    [[nodiscard]] std::vector<TextInstance> SubmitText(const std::string& text) {
        const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
        text_system_->SubmitScreenText(*face_, text, 32.0f, 24.0f, 64.0f, white);
        std::vector<TextInstance> instances(pass_->QueuedInstances().begin(),
                                            pass_->QueuedInstances().end());
        pass_->ClearQueue();
        return instances;
    }

    // Records the dirty atlas pages into its own submitted command buffer, the
    // same work the pass's pre-record hook does in the engine frame.
    void UploadAtlas() {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        (void)text_system_->UploadAtlas(*cmd, 0);
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
            const std::array<vk::DescriptorSet, 1> set0{bindless_->GetDescriptorSet()};
            const std::array<vk::DescriptorSet, 1> set5{**instance_set_};
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 0, set0,
                                   {});
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 5, set5,
                                   {});
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

    [[nodiscard]] static std::array<std::uint8_t, 4> Pixel(const std::vector<std::uint8_t>& pixels,
                                                           std::uint32_t x, std::uint32_t y) {
        const std::size_t base = (static_cast<std::size_t>(y) * kTargetSize + x) * 4U;
        return {pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]};
    }

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

    [[nodiscard]] std::uint32_t DrainFrame() const {
        return std::max<std::uint32_t>(backend_.GetFramesInFlight(), 1U);
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    StagingPool staging_{};
    SamplerCache samplers_{};
    std::unique_ptr<BindlessManager> bindless_{};
    std::unique_ptr<ShaderManager> shaders_{};
    ShaderId ui_text_vert_ = 0;
    ShaderId ui_text_frag_ = 0;
    std::unique_ptr<TextPass> pass_{};

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
    TempFontFile font_file_{};
    std::vector<std::byte> font_bytes_{};
    ResourceId font_id_{};
    std::unique_ptr<TextSystem> text_system_{};
    std::shared_ptr<const VulkanEngine::Text::FontFace> face_{};
};

// The public entry point produces instances that draw real ink: the string's
// glyphs cover pixels inside the region the pen origin implies and the far
// corner stays exactly the cleared background.
TEST_F(TextSystemGpuTest, SubmittedTextRendersInk) {
    const auto instances = SubmitText("Hey");
    ASSERT_FALSE(instances.empty()) << "the submission path queued no instances";
    UploadAtlas();
    const auto pixels = Render(instances);

    const InkBounds ink = Ink(pixels);
    ASSERT_FALSE(ink.empty()) << "the submitted string drew no ink";
    EXPECT_GE(ink.min_x, 16u);
    EXPECT_LE(ink.max_x, 160u);
    EXPECT_GE(ink.min_y, 24u);
    EXPECT_LE(ink.max_y, 72u);
    EXPECT_GT(ink.count, 40u);

    for (std::uint32_t y = kTargetSize - 16; y < kTargetSize; ++y) {
        for (std::uint32_t x = kTargetSize - 16; x < kTargetSize; ++x) {
            EXPECT_EQ(Pixel(pixels, x, y), kBackground)
                << "untouched pixel changed at (" << x << ", " << y << ")";
        }
    }
}

// After a font reload the old page is retired through the bindless ring drain --
// it stays bound until the drain so a recorded frame keeps sampling it, and its
// stale handle stops resolving once the drain applies -- and the next submission
// rasterizes into a fresh page that samples correctly.
TEST_F(TextSystemGpuTest, FontReloadRetiresPagesAtTheDrainAndRendersFreshInk) {
    GlyphAtlasGpu* atlas = text_system_->GetGlyphAtlasGpu();
    ASSERT_NE(atlas, nullptr);

    const auto before_instances = SubmitText("Hey");
    ASSERT_FALSE(before_instances.empty());
    UploadAtlas();
    ASSERT_GE(atlas->PageCount(), 1u);
    const TextureHandle old_handle = atlas->PageHandle(0);
    ASSERT_TRUE(old_handle.IsValid());
    const vk::Image old_image = atlas->PageImage(0);
    const std::uint64_t old_face_id = face_->UniqueId();

    ASSERT_TRUE(text_system_->ReloadFont(font_id_, font_file_.Path(), /*frame_index=*/0));
    const auto new_face = text_system_->GetFace(font_id_);
    ASSERT_NE(new_face, nullptr);
    ASSERT_NE(new_face->UniqueId(), old_face_id);

    // The pages are gone from the GPU layer immediately, but the slot is still
    // committed: a frame recorded against it has not drained yet.
    EXPECT_EQ(atlas->PageCount(), 0u);
    EXPECT_TRUE(bindless_->IsCommitted(old_handle.slot));
    const auto* still_bound = bindless_->GetTexture(old_handle.slot);
    ASSERT_NE(still_bound, nullptr);
    EXPECT_EQ(still_bound->GetImage(), old_image);

    // The drain applies the decommit and invalidates the generation, so the
    // stale handle can never resolve to the reused slot.
    bindless_->BeginFrame(DrainFrame(), [](std::uint32_t) { return true; });
    EXPECT_FALSE(bindless_->IsCommitted(old_handle.slot));
    const auto stale = bindless_->GetHandle(old_handle.slot);
    EXPECT_TRUE(!stale.has_value() || stale->generation != old_handle.generation);

    // The rebuilt face rasterizes into a fresh page with its own image, and the
    // text it produces still draws ink.
    face_ = new_face;
    const auto after_instances = SubmitText("Hey");
    ASSERT_FALSE(after_instances.empty());
    UploadAtlas();
    ASSERT_GE(atlas->PageCount(), 1u);
    const TextureHandle new_handle = atlas->PageHandle(0);
    ASSERT_TRUE(new_handle.IsValid());
    EXPECT_NE(atlas->PageImage(0), old_image);

    const auto pixels = Render(after_instances);
    const InkBounds ink = Ink(pixels);
    ASSERT_FALSE(ink.empty()) << "the reloaded font drew no ink";
    EXPECT_GE(ink.min_x, 16u);
    EXPECT_LE(ink.max_x, 160u);
    EXPECT_GE(ink.min_y, 24u);
    EXPECT_LE(ink.max_y, 72u);
}

// A rejected reload leaves the working page and face in place: nothing is
// retired and the text still draws.
TEST_F(TextSystemGpuTest, RejectedReloadKeepsTheWorkingPage) {
    GlyphAtlasGpu* atlas = text_system_->GetGlyphAtlasGpu();
    ASSERT_NE(atlas, nullptr);

    ASSERT_FALSE(SubmitText("Hey").empty());
    UploadAtlas();
    ASSERT_GE(atlas->PageCount(), 1u);
    const TextureHandle handle = atlas->PageHandle(0);
    ASSERT_TRUE(handle.IsValid());
    const std::uint64_t old_face_id = face_->UniqueId();

    // Corrupt the file the reload reads: the resource must refuse it and the
    // registry entry must keep the previous face.
    const std::vector<std::byte> junk(256, std::byte{0x5A});
    ASSERT_TRUE(font_file_.Write(junk));
    EXPECT_FALSE(text_system_->ReloadFont(font_id_, font_file_.Path(), /*frame_index=*/0));
    EXPECT_EQ(atlas->PageCount(), 1u);
    EXPECT_TRUE(bindless_->IsCommitted(handle.slot));
    EXPECT_EQ(text_system_->GetFace(font_id_)->UniqueId(), old_face_id);

    const auto instances = SubmitText("Hey");
    ASSERT_FALSE(instances.empty());
    UploadAtlas();
    EXPECT_FALSE(Ink(Render(instances)).empty());
}

}  // namespace
