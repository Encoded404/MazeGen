#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import ShaderReflection;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanEngine.ResourceSystem;
import VulkanEngine.TextureTypes;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuTexture;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

namespace {

using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::GpuResources::GpuImageHeap;
using VulkanEngine::GpuResources::GpuTexture;
using VulkanEngine::GpuResources::ImageHeapConfig;
using VulkanEngine::ShaderSystem::ComputePipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::Textures::GpuTextureInfo;

// ── The one shader-visibility mirror of GpuTextureInfo ─────────────────────
//
// The per-slot metadata element is six tightly packed 32-bit words, no padding
// (CDataLayout StructuredBuffer in the engine bindless shaders), mirroring
// `struct GpuTextureInfo` in engine/core/assets/TextureTypes.cppm. The word
// order is the struct's field order: width, height, mip_levels, array_layers,
// format (the numeric vk::Format), flags. Every place in this file that lays
// out or reads the shader-side element uses these constants, and the static
// assert pins the mirror to the engine struct so a divergence fails to compile.
inline constexpr std::uint32_t kGpuTextureInfoWordCount = 6;
inline constexpr std::uint32_t kGpuTextureInfoWidthWord = 0;
inline constexpr std::uint32_t kGpuTextureInfoHeightWord = 1;
inline constexpr std::uint32_t kGpuTextureInfoMipLevelsWord = 2;
inline constexpr std::uint32_t kGpuTextureInfoArrayLayersWord = 3;
inline constexpr std::uint32_t kGpuTextureInfoFormatWord = 4;
inline constexpr std::uint32_t kGpuTextureInfoFlagsWord = 5;

static_assert(sizeof(GpuTextureInfo) == kGpuTextureInfoWordCount * sizeof(std::uint32_t),
              "GpuTextureInfo must stay six packed 32-bit words for the shader mirror");

// The per-slot GpuTextureInfo buffer is shader-visible metadata: one element per
// bindless slot describing the extent, chain, layer count and device format the
// descriptor points at. These tests read it back through a real compute shader
// (the same binding the engine shaders use) so the C++ struct and the shader
// mirror cannot silently diverge.
class TextureInfoGpuTest : public ::testing::Test {
protected:
    static constexpr std::uint32_t kCapacity = 4;

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());
        ASSERT_TRUE(heap_.Initialize(backend_, ImageHeapConfig{}, "texture-info-tests"));
        bindless_ = std::make_unique<VulkanEngine::BindlessManager::BindlessManager>();
        VulkanEngine::BindlessManager::BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = kCapacity;
        ASSERT_TRUE(bindless_->Initialize(backend_, capacity_config));
        bindless_->SetFallback(MakeTexture(8, 8, 0xAA), VulkanEngine::ResourceId{"fallback"});

        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        const ShaderId compute_id = shaders_->RegisterManual(
            std::format("{}/test_texture_info.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eCompute);

        // Output buffer: one GpuTextureInfo element (kGpuTextureInfoWordCount
        // words) per slot, host-visible so the test reads it back.
        out_bytes_ = static_cast<vk::DeviceSize>(kCapacity) * kGpuTextureInfoWordCount *
                     sizeof(std::uint32_t);
        out_buffer_ = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            backend_, out_bytes_, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));

        // Set 1, binding 0: the output storage buffer.
        const vk::DescriptorSetLayoutBinding out_binding(
            0, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute, nullptr);
        out_set_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            backend_.GetDevice(), vk::DescriptorSetLayoutCreateInfo{{}, 1, &out_binding});

        const vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 1);
        out_pool_ = std::make_unique<vk::raii::DescriptorPool>(
            backend_.GetDevice(), vk::DescriptorPoolCreateInfo({}, 1, 1, &pool_size));
        auto sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**out_pool_, 1, &**out_set_layout_});
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

        // Pipeline layout: set 0 is the bindless set (owned by the manager), set
        // 1 is the test's output buffer.
        const std::array<vk::DescriptorSetLayout, 2> set_layouts{
            *bindless_->GetLayout(), **out_set_layout_};
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

        PipelineFactory factory(backend_.GetDevice(), backend_.GetCapabilities(),
                                shaders_->GetPipelineCacheRAII());
        ComputePipelineDesc desc{};
        desc.shader = compute_id;
        desc.layout = **pipeline_layout_;
        auto product = factory.CreateCompute(desc, *shaders_);
        ASSERT_TRUE(product.has_value()) << product.error().message;
        pipeline_product_ = std::make_unique<PipelineProduct>(std::move(*product));
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        pipeline_product_.reset();
        pipeline_layout_.reset();
        out_set_.reset();
        out_pool_.reset();
        out_set_layout_.reset();
        out_buffer_.reset();
        shaders_.reset();
        if (bindless_) {
            bindless_->Shutdown();
            bindless_.reset();
        }
        heap_.Shutdown();
        backend_.Shutdown();
    }

    [[nodiscard]] GpuTexture MakeTexture(std::uint32_t width, std::uint32_t height,
                                         std::uint8_t value) {
        VulkanEngine::Textures::TextureData data{};
        data.width = width;
        data.height = height;
        data.source_format = vk::Format::eR8G8B8A8Unorm;
        data.source_channels = 4;
        data.source_alpha = 1;
        const std::uint64_t bytes = static_cast<std::uint64_t>(width) * height * 4U;
        data.subresources.push_back(VulkanEngine::Textures::TextureSubresource{
            .mip = 0, .array_layer = 0, .offset = 0, .size = bytes,
            .width = width, .height = height, .depth = 1,
        });
        data.blob.assign(bytes, static_cast<std::byte>(value));
        return GpuTexture::CreateFromTextureData(backend_, heap_, data, vk::Format::eR8G8B8A8Unorm);
    }

    // Dispatches the metadata reader over every slot and returns the raw words.
    [[nodiscard]] std::vector<std::uint32_t> ReadAllSlotInfos() {
        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_product_->Get());
        const std::array<vk::DescriptorSet, 2> sets{bindless_->GetDescriptorSet(), **out_set_};
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **pipeline_layout_, 0, sets, {});
        cmd.dispatch(kCapacity, 1, 1);

        const vk::BufferMemoryBarrier to_host(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
            static_cast<vk::Buffer>(*out_buffer_->GetBuffer()), 0, out_bytes_);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost,
                            {}, {}, to_host, {});
        cmd.end();
        const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
        backend_.GetGraphicsQueue().submit(submit, {});
        backend_.GetGraphicsQueue().waitIdle();

        std::vector<std::uint32_t> words(kCapacity * kGpuTextureInfoWordCount);
        const void* mapped = out_buffer_->Map(0, out_bytes_);
        std::memcpy(words.data(), mapped, static_cast<std::size_t>(out_bytes_));
        out_buffer_->Unmap();
        return words;
    }

    TestSupport::HeadlessVulkanBackend backend_{};
    GpuImageHeap heap_{};
    std::unique_ptr<VulkanEngine::BindlessManager::BindlessManager> bindless_{};
    std::unique_ptr<ShaderManager> shaders_{};
    std::unique_ptr<GpuBuffer> out_buffer_{};
    vk::DeviceSize out_bytes_ = 0;
    std::unique_ptr<vk::raii::DescriptorSetLayout> out_set_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> out_pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> out_set_{};
    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    std::unique_ptr<PipelineProduct> pipeline_product_{};
};

namespace {

// Rebuilds one engine-struct element from the raw words the shader wrote, at
// the word offsets the single mirror block at the top of the file pins.
GpuTextureInfo WordsToInfo(const std::vector<std::uint32_t>& words, std::uint32_t slot) {
    GpuTextureInfo info{};
    const std::size_t base = static_cast<std::size_t>(slot) * kGpuTextureInfoWordCount;
    info.width = words[base + kGpuTextureInfoWidthWord];
    info.height = words[base + kGpuTextureInfoHeightWord];
    info.mip_levels = words[base + kGpuTextureInfoMipLevelsWord];
    info.array_layers = words[base + kGpuTextureInfoArrayLayersWord];
    info.format = words[base + kGpuTextureInfoFormatWord];
    info.flags = words[base + kGpuTextureInfoFlagsWord];
    return info;
}

}  // namespace

// The fallback's metadata is visible through the shader at slot 0.
TEST_F(TextureInfoGpuTest, FallbackSlotReportsItsExtentAndFormat) {
    const auto words = ReadAllSlotInfos();
    const GpuTextureInfo fallback = WordsToInfo(words, VulkanEngine::BindlessManager::kFallbackSlot);
    EXPECT_EQ(fallback.width, 8u);
    EXPECT_EQ(fallback.height, 8u);
    EXPECT_EQ(fallback.mip_levels, 1u);
    EXPECT_EQ(fallback.array_layers, 1u);
    EXPECT_EQ(fallback.format, static_cast<std::uint32_t>(vk::Format::eR8G8B8A8Unorm));
}

// A committed texture reports its own metadata, matching the C++ accessor.
TEST_F(TextureInfoGpuTest, CommittedSlotReportsItsExtentAndFormat) {
    const auto handle = bindless_->AllocateTextureSlot(
        MakeTexture(4, 2, 0x33), VulkanEngine::ResourceId{"info"});
    ASSERT_TRUE(handle.has_value());
    const auto* texture = bindless_->GetTexture(*handle);
    ASSERT_NE(texture, nullptr);

    const auto expected = texture->ToTextureInfo();
    const auto words = ReadAllSlotInfos();
    const GpuTextureInfo seen = WordsToInfo(words, handle->slot);

    EXPECT_EQ(seen.width, expected.width);
    EXPECT_EQ(seen.height, expected.height);
    EXPECT_EQ(seen.mip_levels, expected.mip_levels);
    EXPECT_EQ(seen.array_layers, expected.array_layers);
    EXPECT_EQ(seen.format, expected.format);
}

// A reserved-but-uncommitted slot points at the fallback, so its metadata must
// describe the fallback too (descriptor and metadata stay consistent).
TEST_F(TextureInfoGpuTest, ReservedSlotReportsTheFallbackMetadata) {
    const auto handle = bindless_->ReserveSlot(VulkanEngine::ResourceId{"pending"});
    ASSERT_TRUE(handle.has_value());

    const auto words = ReadAllSlotInfos();
    const GpuTextureInfo reserved = WordsToInfo(words, handle->slot);
    const GpuTextureInfo fallback = WordsToInfo(words, VulkanEngine::BindlessManager::kFallbackSlot);
    EXPECT_EQ(reserved.width, fallback.width);
    EXPECT_EQ(reserved.height, fallback.height);
    EXPECT_EQ(reserved.format, fallback.format);
}

// Host-side accessor agrees with the shader-visible element.
TEST_F(TextureInfoGpuTest, HostAccessorMatchesShaderView) {
    const auto handle = bindless_->AllocateTextureSlot(
        MakeTexture(2, 4, 0x44), VulkanEngine::ResourceId{"host"});
    ASSERT_TRUE(handle.has_value());

    const auto* host = bindless_->GetTextureInfo(handle->slot);
    ASSERT_NE(host, nullptr);
    const auto words = ReadAllSlotInfos();
    const GpuTextureInfo seen = WordsToInfo(words, handle->slot);
    EXPECT_EQ(host->width, seen.width);
    EXPECT_EQ(host->height, seen.height);
    EXPECT_EQ(host->mip_levels, seen.mip_levels);
    EXPECT_EQ(host->array_layers, seen.array_layers);
    EXPECT_EQ(host->format, seen.format);
}

}  // namespace
