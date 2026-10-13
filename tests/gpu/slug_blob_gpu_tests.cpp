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
import VulkanEngine.GpuBuffer;
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GpuTextBlobBuffer;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::GpuResources::GpuBuffer;
using VulkanEngine::ShaderSystem::ComputePipelineDesc;
using VulkanEngine::ShaderSystem::PipelineFactory;
using VulkanEngine::ShaderSystem::PipelineProduct;
using VulkanEngine::ShaderSystem::ShaderId;
using VulkanEngine::ShaderSystem::ShaderManager;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphBlob;
using VulkanEngine::Text::GlyphBlobEncoder;
using VulkanEngine::Text::GpuTextBlobBuffer;

struct ProbeParams {
    std::uint32_t base_element = 0;
    std::uint32_t element_count = 0;
    std::uint32_t probe_element = 0;
    std::uint32_t pad0 = 0;
};
static_assert(sizeof(ProbeParams) == 16, "the probe push constant must be 16 bytes");

// Device-bound: proves that the buffer GpuTextBlobBuffer creates is readable
// from a shader with the offsets the device-free allocator recorded, and that
// the element stride the shader sees is 8 bytes -- the specific mistake being a
// 16-byte `int4` stride copied from the upstream HLSL header comment.
class SlugBlobGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend_.Initialize());

        handle_ = manager_.LoadFromFile<FontResource>(
            std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf",
            ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        auto* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());
        face_ = FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);

        shaders_ = std::make_unique<ShaderManager>(backend_.GetDevice(), backend_.GetCapabilities(),
                                                   std::filesystem::temp_directory_path().string());
        const ShaderId compute_id = shaders_->RegisterManual(
            std::format("{}/test_slug_blob_probe.spv", VKENGINE_TEST_SHADER_DIR), "",
            ShaderStage::eCompute);

        // Output buffer: raw words plus the sign-extended int4, host-visible.
        out_bytes_ = static_cast<vk::DeviceSize>(kMaxInts) * sizeof(std::int32_t);
        out_buffer_ = std::make_unique<GpuBuffer>(GpuBuffer::Create(
            backend_, out_bytes_, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        ASSERT_TRUE(out_buffer_->IsValid());

        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr)};
        set_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            backend_.GetDevice(), vk::DescriptorSetLayoutCreateInfo{{}, bindings.size(), bindings.data()});

        const std::array<vk::DescriptorPoolSize, 1> pool_sizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2)};
        pool_ = std::make_unique<vk::raii::DescriptorPool>(
            backend_.GetDevice(),
            vk::DescriptorPoolCreateInfo({}, 1, pool_sizes.size(), pool_sizes.data()));
        auto sets = backend_.GetDevice().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{**pool_, 1, &**set_layout_});
        ASSERT_EQ(sets.size(), 1u);
        set_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));

        const vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eCompute, 0,
                                               sizeof(ProbeParams));
        const std::array<vk::DescriptorSetLayout, 1> set_layouts{**set_layout_};
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setSetLayouts(set_layouts);
        layout_info.setPushConstantRanges(push_range);
        pipeline_layout_ =
            std::make_unique<vk::raii::PipelineLayout>(backend_.GetDevice(), layout_info);

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
        set_.reset();
        pool_.reset();
        set_layout_.reset();
        out_buffer_.reset();
        shaders_.reset();
        blobs_.Shutdown();
        backend_.Shutdown();
    }

    // Binds the blob buffer at binding 0 and the output buffer at binding 1,
    // dispatches the probe, and reads back the words.
    [[nodiscard]] std::vector<std::int32_t> RunProbe(const ProbeParams& params) {
        vk::DescriptorBufferInfo blob_info{};
        blob_info.buffer = blobs_.Buffer();
        blob_info.offset = 0;
        blob_info.range = vk::WholeSize;
        vk::DescriptorBufferInfo out_info{};
        out_info.buffer = static_cast<vk::Buffer>(*out_buffer_->GetBuffer());
        out_info.offset = 0;
        out_info.range = out_bytes_;
        vk::WriteDescriptorSet blob_write{};
        blob_write.dstSet = **set_;
        blob_write.dstBinding = 0;
        blob_write.descriptorCount = 1;
        blob_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        blob_write.pBufferInfo = &blob_info;
        vk::WriteDescriptorSet out_write{};
        out_write.dstSet = **set_;
        out_write.dstBinding = 1;
        out_write.descriptorCount = 1;
        out_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        out_write.pBufferInfo = &out_info;
        backend_.GetDevice().updateDescriptorSets({blob_write, out_write}, {});

        vk::raii::CommandBuffer& cmd = backend_.GetCommandBuffer(0);
        cmd.reset({});
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_product_->Get());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **pipeline_layout_, 0,
                               std::array<vk::DescriptorSet, 1>{**set_}, {});
        cmd.pushConstants<ProbeParams>(**pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0,
                                       params);
        cmd.dispatch(1, 1, 1);
        const vk::BufferMemoryBarrier to_host(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
            static_cast<vk::Buffer>(*out_buffer_->GetBuffer()), 0, out_bytes_);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eHost, {}, {}, to_host, {});
        cmd.end();
        backend_.GetGraphicsQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &*cmd), {});
        backend_.GetGraphicsQueue().waitIdle();

        std::vector<std::int32_t> words(kMaxInts);
        const void* mapped = out_buffer_->Map(0, out_bytes_);
        std::memcpy(words.data(), mapped, static_cast<std::size_t>(out_bytes_));
        out_buffer_->Unmap();
        return words;
    }

    static constexpr std::uint32_t kMaxInts = 4096;

    TestSupport::HeadlessVulkanBackend backend_{};
    ResourceManager manager_{};
    ResourceHandle<FontResource> handle_{};
    std::shared_ptr<FontFace> face_{};
    std::unique_ptr<ShaderManager> shaders_{};
    std::unique_ptr<vk::raii::DescriptorSetLayout> set_layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> pool_{};
    std::unique_ptr<vk::raii::DescriptorSet> set_{};
    std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout_{};
    std::unique_ptr<PipelineProduct> pipeline_product_{};
    std::unique_ptr<GpuBuffer> out_buffer_{};
    vk::DeviceSize out_bytes_ = 0;
    GlyphBlobEncoder encoder_{};
    GpuTextBlobBuffer blobs_{};
};

// Reads the blob's 16-bit halves out of its bytes, the way the shader's
// `hb_gpu_fetch` reads them, so the expected sign extension is derived from the
// real bytes and not from a restatement of the shader.
[[nodiscard]] std::int32_t SignedHalf(const std::vector<std::byte>& bytes, std::size_t half) {
    std::int16_t value = 0;
    std::memcpy(&value, bytes.data() + half * 2U, sizeof(value));
    return value;
}

} // namespace

// The whole blob round-trips through a real shader at the offset the range
// allocator recorded, and the shader's element stride is 8 bytes: element i+1
// reads the bytes eight bytes after element i, so a 16-byte `int4` declaration
// would fail this. The buffer is deliberately created smaller than the first
// blob, so the store that fits it exercises the growth path (new buffer, whole
// image re-uploaded, old buffer retired) and the second blob still lands at the
// offset the allocator chose.
TEST_F(SlugBlobGpuTest, BlobBytesRoundTripAtTheRecordedOffsetWithAnEightByteStride) {
    // Smaller than either glyph's blob, so the first Store grows the buffer.
    ASSERT_TRUE(blobs_.Initialize(backend_, 64U));

    const auto capital = encoder_.Get(*face_, face_->GlyphForCodepoint(U'H'));
    const auto round = encoder_.Get(*face_, face_->GlyphForCodepoint(U'o'));
    ASSERT_NE(capital, nullptr);
    ASSERT_NE(round, nullptr);
    ASSERT_FALSE(capital->Empty());
    ASSERT_FALSE(round->Empty());

    const auto capital_offset = blobs_.Store(capital->bytes, /*recording_frame=*/1);
    ASSERT_TRUE(capital_offset.has_value());
    EXPECT_GT(blobs_.Capacity(), 64u) << "the first store did not grow the buffer";
    const auto round_offset = blobs_.Store(round->bytes, /*recording_frame=*/1);
    ASSERT_TRUE(round_offset.has_value());
    // Packing order: the second blob starts right after the first, 8-byte unit
    // for 8-byte unit.
    EXPECT_EQ(*round_offset, capital->bytes.size());
    EXPECT_EQ(blobs_.ElementOffset(*round_offset), capital->UnitCount());
    EXPECT_FALSE(blobs_.Store(std::span<const std::byte>{}, 1).has_value())
        << "a space's zero-length blob must take no range";
    const std::array<std::byte, 7> not_a_blob{};
    EXPECT_FALSE(blobs_.Store(not_a_blob, 1).has_value())
        << "a length that is not a whole number of units is not a blob";

    const std::uint32_t element_count = static_cast<std::uint32_t>(round->UnitCount());
    ASSERT_GT(element_count, 1u);
    ASSERT_LT(2u * element_count + 4u, kMaxInts);

    // The first unit whose second half is negative, so the sign-extension
    // assertion below cannot pass on positive values alone.
    std::uint32_t probe = 0;
    bool probe_has_negative = false;
    for (std::uint32_t unit = 0; unit < element_count && !probe_has_negative; ++unit) {
        for (std::uint32_t half = 0; half < 4; ++half) {
            if (SignedHalf(round->bytes, static_cast<std::size_t>(unit) * 4U + half) < 0) {
                probe = unit;
                probe_has_negative = true;
                break;
            }
        }
    }
    ASSERT_TRUE(probe_has_negative)
        << "no negative 16-bit half in the blob; the sign-extension check would be vacuous";

    ProbeParams params{};
    params.base_element = static_cast<std::uint32_t>(blobs_.ElementOffset(*round_offset));
    params.element_count = element_count;
    params.probe_element = probe;
    const std::vector<std::int32_t> words = RunProbe(params);

    // Every raw word round-trips. Elements are adjacent 8-byte units, so word
    // 2i/2i+1 is element i and word 2i+2/2i+3 is element i+1: a 16-byte stride
    // would put the wrong bytes there and fail this comparison.
    for (std::uint32_t unit = 0; unit < element_count; ++unit) {
        std::int32_t low = 0;
        std::int32_t high = 0;
        std::memcpy(&low, round->bytes.data() + (unit * 8U), 4U);
        std::memcpy(&high, round->bytes.data() + (unit * 8U) + 4U, 4U);
        EXPECT_EQ(words[2u * unit], low) << "low word of unit " << unit;
        EXPECT_EQ(words[2u * unit + 1u], high) << "high word of unit " << unit;
    }

    // The stride-specific assertion, stated on its own so the failure message
    // names the mistake: element i and i+1 are eight bytes apart.
    std::int32_t next_low = 0;
    std::memcpy(&next_low, round->bytes.data() + 8U, 4U);
    EXPECT_EQ(words[2], next_low)
        << "element 1 did not read the bytes eight bytes after element 0: the shader's "
           "element stride is not 8";

    // And the adapted accessor's sign extension matches the real bytes.
    const std::size_t probe_half = static_cast<std::size_t>(probe) * 4U;
    const std::uint32_t base = 2u * element_count;
    EXPECT_EQ(words[base + 0u], SignedHalf(round->bytes, probe_half + 0u));
    EXPECT_EQ(words[base + 1u], SignedHalf(round->bytes, probe_half + 1u));
    EXPECT_EQ(words[base + 2u], SignedHalf(round->bytes, probe_half + 2u));
    EXPECT_EQ(words[base + 3u], SignedHalf(round->bytes, probe_half + 3u));
    // The probe unit really does contain a negative half, so the equality above
    // is a sign-extension check and not four zeros comparing equal.
    EXPECT_TRUE(std::min({words[base + 0u], words[base + 1u], words[base + 2u], words[base + 3u]}) <
                0);
}
