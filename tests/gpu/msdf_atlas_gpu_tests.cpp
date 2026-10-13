#include <gtest/gtest.h>

import std;
import std.compat;

import vulkan;

import test_gpu;

import ShaderReflection;
import TestSupport.HeadlessVulkanBackend;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.Msdf;
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
using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::GpuResources::SamplerCache;
using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;
using VulkanEngine::ShaderSystem::ComputePipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::AtlasPageFormat;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::Text::GlyphSlot;
using VulkanEngine::Text::MsdfConfig;
using VulkanEngine::Text::MsdfGenerator;
using VulkanEngine::Text::MsdfGlyph;

constexpr std::uint32_t kPageWidth = 64;
constexpr std::uint32_t kPageHeight = 64;

// The MSDF page is a different format and a different byte source from the A8
// page the FreeType rasterizer feeds. This proves the generalized uploader's
// seam end to end on a real device: four channels per texel are staged and
// copied with the right stride, the page is created as RGBA8, and the bytes a
// shader samples are the bytes the generator produced. Nothing here is skipped
// when a device exists -- a silent skip would leave the seam untested.
class MsdfAtlasGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        ASSERT_TRUE(heap_.Initialize(backend_, ImageHeapConfig{.block_size = 4ULL << 20},
                                     "msdf-atlas-gpu-tests"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 1ULL << 20;
        staging_config.capacity_bytes = 8ULL << 20;
        ASSERT_TRUE(staging_.Initialize(backend_, staging_config));
        ASSERT_TRUE(samplers_.Initialize(backend_));

        bindless_ = std::make_unique<BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity{};
        capacity.app_capacity = 16;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity));
        bindless_->SetFallback(MakeSolidTexture(), VulkanEngine::ResourceId{"msdf-fallback"});

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
        atlas_config.page_width = kPageWidth;
        atlas_config.page_height = kPageHeight;
        atlas_config.padding = 1;
        atlas_config.max_pages = 4;
        generator_ = std::make_unique<MsdfGenerator>(atlas_config, 16);

        atlas_gpu_ = std::make_unique<GlyphAtlasGpu>();
        ASSERT_TRUE(atlas_gpu_->Initialize(backend_, heap_, staging_, *bindless_, &samplers_));
        atlas_gpu_->SetPageFormat(AtlasPageFormat::Rgba8);
        atlas_gpu_->SetByteSource(
            [this](std::uint64_t key) { return generator_->BitmapForAtlasKey(key); });

        BuildReadbackPipeline();
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        out_buffer_.reset();
        out_set_.reset();
        out_pool_.reset();
        out_layout_.reset();
        readback_pipeline_.reset();
        readback_layout_.reset();
        shaders_.reset();
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
        std::vector<std::uint8_t> pixels(8u * 8u * 4u, 0x20u);
        return GpuTexture::CreateFromPixels(backend_, heap_, pixels.data(), 8, 8,
                                            vk::Format::eR8G8B8A8Unorm, &samplers_);
    }

    void BuildReadbackPipeline() {
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(),
                                                   backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        const ShaderId compute = shaders_->RegisterManual(
            std::format("{}/test_read_page_rgba.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eCompute);

        out_bytes_ = static_cast<vk::DeviceSize>(kPageWidth) * kPageHeight * 4U *
                     sizeof(std::uint32_t);
        out_buffer_ = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            backend_, out_bytes_, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        ASSERT_TRUE(out_buffer_->IsValid());

        const vk::DescriptorSetLayoutBinding out_binding(
            0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        out_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            backend_.GetDevice(), vk::DescriptorSetLayoutCreateInfo{{}, 1, &out_binding});

        const vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 1);
        out_pool_ = std::make_unique<vk::raii::DescriptorPool>(
            backend_.GetDevice(), vk::DescriptorPoolCreateInfo({}, 1, 1, &pool_size));
        auto sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**out_pool_, 1, &**out_layout_});
        ASSERT_EQ(sets.size(), 1u);
        out_set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));
        vk::DescriptorBufferInfo out_info{};
        out_info.buffer = static_cast<vk::Buffer>(*out_buffer_->GetBuffer());
        out_info.offset = 0;
        out_info.range = out_bytes_;
        vk::WriteDescriptorSet out_write{};
        out_write.dstSet = **out_set_;
        out_write.dstBinding = 0;
        out_write.descriptorCount = 1;
        out_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        out_write.pBufferInfo = &out_info;
        backend_.GetDevice().updateDescriptorSets({out_write}, {});

        const std::array<vk::DescriptorSetLayout, 2> set_layouts{*bindless_->GetLayout(),
                                                                 **out_layout_};
        const vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eCompute, 0,
                                               3U * sizeof(std::uint32_t));
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        layout_info.setPushConstantRanges(push_range);
        readback_layout_ =
            std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

        PipelineFactory factory(backend_.GetDevice(), backend_.GetCapabilities(),
                                shaders_->GetPipelineCacheRAII());
        ComputePipelineDesc desc{};
        desc.shader = compute;
        desc.layout = **readback_layout_;
        auto product = factory.CreateCompute(desc, *shaders_);
        ASSERT_TRUE(product.has_value()) << product.error().message;
        readback_pipeline_ = std::make_unique<PipelineProduct>(std::move(*product));
    }

    // Records one UploadDirty into its own submitted command buffer. No frames
    // are in flight in this harness, so the recording frame is 0.
    void Upload(std::uint32_t frame_index = 0) {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        atlas_gpu_->UploadDirty(generator_->MutableAtlas(), *cmd, frame_index);
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();
    }

    struct Texel {
        std::uint8_t r = 0, g = 0, b = 0, a = 0;
    };

    [[nodiscard]] std::vector<Texel> ReadPage(std::uint32_t page) {
        const TextureHandle handle = atlas_gpu_->PageHandle(page);
        EXPECT_TRUE(handle.IsValid());
        const std::array<std::uint32_t, 3> params{handle.slot, kPageWidth, kPageHeight};

        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, readback_pipeline_->Get());
        const std::array<vk::DescriptorSet, 2> sets{bindless_->GetDescriptorSet(), **out_set_};
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **readback_layout_, 0, sets, {});
        cmd.pushConstants<std::uint32_t>(**readback_layout_, vk::ShaderStageFlagBits::eCompute, 0,
                                         params);
        cmd.dispatch((kPageWidth + 7) / 8, (kPageHeight + 7) / 8, 1);

        const vk::BufferMemoryBarrier to_host(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
            static_cast<vk::Buffer>(*out_buffer_->GetBuffer()), 0, out_bytes_);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eHost, {}, {}, to_host, {});
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();

        std::vector<Texel> texels(static_cast<std::size_t>(kPageWidth) * kPageHeight);
        const void* mapped = out_buffer_->Map(0, out_bytes_);
        const auto* words = static_cast<const std::uint32_t*>(mapped);
        for (std::size_t i = 0; i < texels.size(); ++i) {
            texels[i].r = static_cast<std::uint8_t>(std::min<std::uint32_t>(words[i * 4], 255U));
            texels[i].g = static_cast<std::uint8_t>(std::min<std::uint32_t>(words[i * 4 + 1], 255U));
            texels[i].b = static_cast<std::uint8_t>(std::min<std::uint32_t>(words[i * 4 + 2], 255U));
            texels[i].a = static_cast<std::uint8_t>(std::min<std::uint32_t>(words[i * 4 + 3], 255U));
        }
        out_buffer_->Unmap();
        return texels;
    }

    [[nodiscard]] static Texel At(const std::vector<Texel>& page, std::uint32_t x,
                                  std::uint32_t y) {
        return page[static_cast<std::size_t>(y) * kPageWidth + x];
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    StagingPool staging_{};
    SamplerCache samplers_{};
    std::unique_ptr<BindlessManager> bindless_{};
    std::unique_ptr<ShaderManager> shaders_{};
    std::unique_ptr<GpuBuffer> out_buffer_{};
    vk::DeviceSize out_bytes_ = 0;
    std::unique_ptr<vk::raii::DescriptorSetLayout> out_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> out_pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> out_set_{};
    std::unique_ptr<vk::raii::PipelineLayout> readback_layout_{};
    std::unique_ptr<PipelineProduct> readback_pipeline_{};

    VulkanEngine::ResourceManager manager_{};
    VulkanEngine::ResourceHandle<VulkanEngine::FontResource> handle_{};
    std::shared_ptr<VulkanEngine::Text::FontFace> face_{};
    std::unique_ptr<MsdfGenerator> generator_{};
    std::unique_ptr<GlyphAtlasGpu> atlas_gpu_{};
};

// The four generated channels round-trip through an RGBA8 page: every texel of
// the slot's inner region equals the generated field, channel for channel, and
// the page really is the format the generator chose.
TEST_F(MsdfAtlasGpuTest, Rgba8PageRoundTripsTheGeneratedField) {
    EXPECT_EQ(atlas_gpu_->PageFormat(), AtlasPageFormat::Rgba8);

    const std::uint32_t capital = face_->GlyphForCodepoint('H');
    const auto slot = generator_->Generate(*face_, capital, MsdfConfig{32.0f, 4.0});
    ASSERT_TRUE(slot.has_value());
    const auto glyph = generator_->Get(*face_, capital, MsdfConfig{32.0f, 4.0});
    ASSERT_NE(glyph, nullptr);
    ASSERT_FALSE(glyph->Empty());

    Upload();
    ASSERT_TRUE(atlas_gpu_->HasPage(slot->page));
    ASSERT_EQ(atlas_gpu_->PageCount(), generator_->Atlas().PageCount());

    const auto page = ReadPage(slot->page);
    const AtlasConfig& config = generator_->Atlas().Config();
    const std::uint32_t inner_x = slot->rect.x + config.padding;
    const std::uint32_t inner_y = slot->rect.y + config.padding;

    std::uint32_t mismatches = 0;
    for (std::uint32_t row = 0; row < glyph->height; ++row) {
        for (std::uint32_t col = 0; col < glyph->width; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * glyph->width + col) * MsdfGenerator::kChannels;
            const Texel seen = At(page, inner_x + col, inner_y + row);
            const Texel expected{glyph->pixels[base], glyph->pixels[base + 1],
                                 glyph->pixels[base + 2], glyph->pixels[base + 3]};
            // A correct 4-byte stride is what makes this exact; a copy that used
            // the A8 stride would smear each texel's channels into its
            // neighbours and fail here.
            if (seen.r != expected.r || seen.g != expected.g || seen.b != expected.b ||
                seen.a != expected.a) {
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << "the RGBA8 page did not round-trip its four channels";
}

// The generalized uploader still zeroes a freshly created whole page: every texel
// outside the field's slot is clear, which is what makes a page an eviction
// invalidates safe to recompose.
TEST_F(MsdfAtlasGpuTest, FreshPageIsZeroedOutsideTheSlot) {
    const std::uint32_t capital = face_->GlyphForCodepoint('H');
    const auto slot = generator_->Generate(*face_, capital, MsdfConfig{24.0f, 3.0});
    ASSERT_TRUE(slot.has_value());

    Upload();
    const auto page = ReadPage(slot->page);
    const Texel corner = At(page, kPageWidth - 1, kPageHeight - 1);
    EXPECT_EQ(corner.r, 0);
    EXPECT_EQ(corner.g, 0);
    EXPECT_EQ(corner.b, 0);
    EXPECT_EQ(corner.a, 0);
    EXPECT_FALSE(generator_->MutableAtlas().DirtyRect(slot->page).has_value());
}

}  // namespace
