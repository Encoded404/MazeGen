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
import VulkanEngine.Text.Font;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.GlyphAtlasGpu;
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
using VulkanEngine::Text::GlyphAtlas;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::Text::GlyphHinting;
using VulkanEngine::Text::GlyphRasterizer;
using VulkanEngine::Text::GlyphSlot;
using VulkanEngine::Text::RasterGlyph;

// A page small enough that a couple of glyphs already spill to a second page,
// so page growth and replacement are exercised without a large readback.
constexpr std::uint32_t kPageWidth = 48;
constexpr std::uint32_t kPageHeight = 48;

[[nodiscard]] AtlasConfig TestAtlasConfig() {
    AtlasConfig config{};
    config.page_width = kPageWidth;
    config.page_height = kPageHeight;
    config.padding = 1;
    config.max_pages = 4;
    return config;
}

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

// Reads one atlas page back through a compute shader that samples the bindless
// slot at texel centres. The page image is eTransferDst | eSampled (it is a
// streaming upload target), so a copyImageToBuffer readback is not available by
// design; sampling is also exactly how the text shader will consume it.
class GlyphAtlasGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        ASSERT_TRUE(heap_.Initialize(backend_, ImageHeapConfig{.block_size = 4ULL << 20},
                                     "glyph-atlas-gpu-tests"));
        StagingPoolConfig staging_config{};
        staging_config.block_size = 1ULL << 20;
        staging_config.capacity_bytes = 8ULL << 20;
        ASSERT_TRUE(staging_.Initialize(backend_, staging_config));
        ASSERT_TRUE(samplers_.Initialize(backend_));

        bindless_ = std::make_unique<BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity{};
        capacity.app_capacity = 16;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity));
        bindless_->SetFallback(MakeSolidTexture(8, 8, 0x20),
                               VulkanEngine::ResourceId{"glyph-fallback"});

        handle_ = manager_.LoadFromFile<VulkanEngine::FontResource>(
            TestFontPath(), VulkanEngine::ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        auto* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());
        face_ = VulkanEngine::Text::FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);

        rasterizer_ = std::make_unique<GlyphRasterizer>(TestAtlasConfig(), 64);
        atlas_gpu_ = std::make_unique<GlyphAtlasGpu>();
        ASSERT_TRUE(atlas_gpu_->Initialize(backend_, heap_, staging_, *bindless_, &samplers_));
        atlas_gpu_->SetGlyphSource(
            [this](std::uint64_t key) { return rasterizer_->GlyphForAtlasKey(key); });

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

    [[nodiscard]] GpuTexture MakeSolidTexture(std::uint32_t width, std::uint32_t height,
                                              std::uint8_t value) {
        const std::size_t bytes = static_cast<std::size_t>(width) * height * 4U;
        std::vector<std::uint8_t> pixels(bytes, value);
        return GpuTexture::CreateFromPixels(backend_, heap_, pixels.data(), width, height,
                                            vk::Format::eR8G8B8A8Unorm, &samplers_);
    }

    void BuildReadbackPipeline() {
        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(),
                                                   backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        const ShaderId compute = shaders_->RegisterManual(
            std::format("{}/test_read_page.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eCompute);

        out_bytes_ = static_cast<vk::DeviceSize>(kPageWidth) * kPageHeight * sizeof(std::uint32_t);
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

    [[nodiscard]] std::uint32_t GlyphId(char character) const {
        return face_->GlyphForCodepoint(static_cast<std::uint32_t>(character));
    }

    // Rasterizes `character` at `pixel_size` and returns its slot + bitmap.
    [[nodiscard]] std::pair<GlyphSlot, std::shared_ptr<const RasterGlyph>> Rasterize(
        char character, float pixel_size) {
        const std::uint32_t id = GlyphId(character);
        const auto slot = rasterizer_->Rasterize(*face_, id, pixel_size, GlyphHinting::Light);
        const auto glyph = rasterizer_->Get(*face_, id, pixel_size, GlyphHinting::Light);
        EXPECT_TRUE(slot.has_value()) << "glyph '" << character << "' did not rasterize";
        EXPECT_NE(glyph, nullptr);
        return {slot.value_or(GlyphSlot{}), glyph};
    }

    // Records one UploadDirty into its own submitted command buffer. No frames
    // are in flight in this harness, so the recording frame is 0 and the
    // bindless ring is drained for that frame afterwards.
    void Upload(std::uint32_t frame_index = 0) {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        atlas_gpu_->UploadDirty(rasterizer_->MutableAtlas(), *cmd, frame_index);
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();
    }

    [[nodiscard]] std::vector<std::uint8_t> ReadPage(std::uint32_t page) {
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

        std::vector<std::uint8_t> pixels(kPageWidth * kPageHeight);
        const void* mapped = out_buffer_->Map(0, out_bytes_);
        const auto* words = static_cast<const std::uint32_t*>(mapped);
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<std::uint8_t>(std::min<std::uint32_t>(words[i], 255U));
        }
        out_buffer_->Unmap();
        return pixels;
    }

    // The inner (ink) region of a slot: the atlas pads every side, so a glyph's
    // pixels live inset by the padding and are exactly width x height.
    struct Inner {
        std::uint32_t x, y, width, height;
    };
    [[nodiscard]] static Inner InnerRect(const GlyphSlot& slot) {
        return Inner{slot.rect.x + 1, slot.rect.y + 1, slot.rect.width - 2,
                     slot.rect.height - 2};
    }

    [[nodiscard]] static std::uint8_t At(const std::vector<std::uint8_t>& page,
                                         std::uint32_t x, std::uint32_t y) {
        return page[static_cast<std::size_t>(y) * kPageWidth + x];
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
    std::unique_ptr<GlyphRasterizer> rasterizer_{};
    std::unique_ptr<GlyphAtlasGpu> atlas_gpu_{};
};

// Upload -> sample the page back through the bindless slot -> the inner rect's
// bytes are the rasterizer's A8 coverage, channel for channel.
TEST_F(GlyphAtlasGpuTest, RoundTripMatchesRasterizedCoverage) {
    const auto [slot, glyph] = Rasterize('H', 24.0f);
    ASSERT_NE(glyph, nullptr);
    ASSERT_GT(glyph->width, 0u);
    ASSERT_GT(glyph->height, 0u);

    Upload();
    // One persistent GPU page per atlas page, created on the first upload.
    ASSERT_EQ(atlas_gpu_->PageCount(), rasterizer_->Atlas().PageCount());
    ASSERT_TRUE(atlas_gpu_->HasPage(slot.page));
    EXPECT_TRUE(atlas_gpu_->PageHandle(slot.page).IsValid());

    const auto page = ReadPage(slot.page);

    const Inner inner = InnerRect(slot);
    ASSERT_EQ(inner.width, glyph->width);
    ASSERT_EQ(inner.height, glyph->height);
    for (std::uint32_t row = 0; row < glyph->height; ++row) {
        for (std::uint32_t col = 0; col < glyph->width; ++col) {
            const std::uint8_t expected =
                glyph->coverage[static_cast<std::size_t>(row) * glyph->width + col];
            EXPECT_EQ(At(page, inner.x + col, inner.y + row), expected)
                << "coverage mismatch at (" << col << ", " << row << ")";
        }
    }
}

// A second upload must move only the second glyph's region. This is the test
// that proves dirty rects are precise: the first glyph's texels -- and every
// texel outside the new ink -- are byte-identical to the previous readback.
TEST_F(GlyphAtlasGpuTest, SecondUploadLeavesTheFirstGlyphByteIdentical) {
    const auto [first_slot, first] = Rasterize('H', 24.0f);
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first_slot.page, 0u);
    Upload();
    const auto before = ReadPage(first_slot.page);

    const auto [second_slot, second] = Rasterize('x', 24.0f);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second_slot.page, first_slot.page) << "the second glyph must share the page";
    Upload();
    const auto after = ReadPage(second_slot.page);

    const Inner first_inner = InnerRect(first_slot);
    for (std::uint32_t row = 0; row < first->height; ++row) {
        for (std::uint32_t col = 0; col < first->width; ++col) {
            EXPECT_EQ(At(after, first_inner.x + col, first_inner.y + row),
                      At(before, first_inner.x + col, first_inner.y + row))
                << "the first glyph changed at (" << col << ", " << row << ")";
        }
    }

    const Inner second_inner = InnerRect(second_slot);
    std::uint32_t changed = 0;
    for (std::uint32_t row = 0; row < second->height; ++row) {
        for (std::uint32_t col = 0; col < second->width; ++col) {
            const std::uint8_t expected =
                second->coverage[static_cast<std::size_t>(row) * second->width + col];
            const std::uint8_t seen = At(after, second_inner.x + col, second_inner.y + row);
            EXPECT_EQ(seen, expected) << "second glyph coverage mismatch at (" << col << ", "
                                      << row << ")";
            changed += seen != At(before, second_inner.x + col, second_inner.y + row) ? 1U : 0U;
        }
    }
    EXPECT_GT(changed, 0u) << "the second upload wrote nothing";

    // Everything outside both ink rectangles is untouched as well.
    for (std::uint32_t y = 0; y < kPageHeight; ++y) {
        for (std::uint32_t x = 0; x < kPageWidth; ++x) {
            const bool in_first = x >= first_inner.x && x < first_inner.x + first_inner.width &&
                                  y >= first_inner.y && y < first_inner.y + first_inner.height;
            const bool in_second =
                x >= second_inner.x && x < second_inner.x + second_inner.width &&
                y >= second_inner.y && y < second_inner.y + second_inner.height;
            if (in_first || in_second) {
                continue;
            }
            EXPECT_EQ(At(after, x, y), At(before, x, y))
                << "a texel outside the new ink changed at (" << x << ", " << y << ")";
        }
    }
}

// Page growth gives every atlas page a valid, distinct bindless slot, and a
// retired page's slot stays sampleable until the frame that could still sample
// it has drained -- then its generation rejects the stale handle.
TEST_F(GlyphAtlasGpuTest, PageReplacementRetiresTheSlotAtTheFrameDrain) {
    constexpr float kSize = 32.0f;
    // A 48x48 page cannot hold two 32px capitals, so rasterizing three distinct
    // glyphs is guaranteed to grow the atlas past one page.
    static_cast<void>(Rasterize('H', kSize));
    static_cast<void>(Rasterize('W', kSize));
    static_cast<void>(Rasterize('o', kSize));
    Upload();
    const std::size_t atlas_pages = rasterizer_->Atlas().PageCount();
    ASSERT_GE(atlas_pages, 2u);
    ASSERT_EQ(atlas_gpu_->PageCount(), atlas_pages);

    const std::uint32_t last_page = static_cast<std::uint32_t>(atlas_pages - 1);
    ASSERT_TRUE(atlas_gpu_->HasPage(last_page));
    const TextureHandle retired_handle = atlas_gpu_->PageHandle(last_page);
    const TextureHandle first_handle = atlas_gpu_->PageHandle(0);
    EXPECT_TRUE(retired_handle.IsValid());
    EXPECT_TRUE(first_handle.IsValid());
    EXPECT_NE(retired_handle.slot, first_handle.slot);
    EXPECT_GE(retired_handle.generation, 1u);

    // Force the atlas to drop its pages, then retire the GPU side for frame 0.
    const vk::Image retired_image = atlas_gpu_->PageImage(last_page);
    rasterizer_->MutableAtlas().Reset();
    atlas_gpu_->ReleaseAll(0);

    EXPECT_EQ(atlas_gpu_->PageCount(), 0u);
    EXPECT_EQ(atlas_gpu_->PageCount(), rasterizer_->Atlas().PageCount());

    // The slot is still committed: a frame recorded against it is still safe.
    EXPECT_TRUE(bindless_->IsCommitted(retired_handle.slot));
    const auto* still_bound = bindless_->GetTexture(retired_handle.slot);
    ASSERT_NE(still_bound, nullptr);
    EXPECT_EQ(still_bound->GetImage(), retired_image);

    // The drain applies the decommit, returns the slot and invalidates the
    // generation, so the stale handle can never resolve to the reused slot.
    bindless_->BeginFrame(DrainFrame(), [](std::uint32_t) { return true; });
    EXPECT_FALSE(bindless_->IsCommitted(retired_handle.slot));
    const auto stale = bindless_->GetHandle(retired_handle.slot);
    EXPECT_TRUE(!stale.has_value() || stale->generation != retired_handle.generation);
    const auto* rebound = bindless_->GetTexture(retired_handle.slot);
    ASSERT_NE(rebound, nullptr);
    EXPECT_NE(rebound->GetImage(), retired_image);

    // Neither stale handle can release the recycled slot out from under it.
    bindless_->ReleaseSlot(retired_handle, 1);
    bindless_->ReleaseSlot(first_handle, 1);
    EXPECT_FALSE(bindless_->IsCommitted(first_handle.slot));
}

// A rasterizer that evicts a glyph repaints the whole page; after the upload the
// page is clean on both sides.
TEST_F(GlyphAtlasGpuTest, UploadClearsTheDirtyRegionAndTracksPageCount) {
    const auto [slot, glyph] = Rasterize('A', 20.0f);
    ASSERT_NE(glyph, nullptr);
    EXPECT_TRUE(rasterizer_->MutableAtlas().DirtyRect(slot.page).has_value());

    Upload();
    EXPECT_FALSE(rasterizer_->MutableAtlas().DirtyRect(slot.page).has_value());
    EXPECT_EQ(atlas_gpu_->PageCount(), rasterizer_->Atlas().PageCount());

    // Nothing dirty draws nothing: a second upload writes no page.
    EXPECT_EQ(atlas_gpu_->UploadDirty(rasterizer_->MutableAtlas(), nullptr, 1), 0u);
}

}  // namespace
