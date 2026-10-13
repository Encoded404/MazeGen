#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.TextureTypes;

namespace {

using VulkanEngine::GpuResources::SamplerCache;
using VulkanEngine::Textures::SamplerDesc;

// The sampler cache needs a real device (VkSampler creation + the capability
// snapshot used for clamping), so these stay on the gpu label.
class SamplerCacheGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    SamplerCache cache{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        ASSERT_TRUE(cache.Initialize(backend, backend.GetCapabilities().GetMaxSamplerAllocationCount()));
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        cache.Shutdown();
        backend.Shutdown();
    }
};

// Identical descriptions share one sampler; the cache owns a single object.
TEST_F(SamplerCacheGpuTest, IdenticalDescriptionsShareOneSampler) {
    SamplerDesc desc{};
    const vk::Sampler first = cache.Get(desc, /*mip_levels=*/1);
    const vk::Sampler second = cache.Get(desc, /*mip_levels=*/1);
    ASSERT_NE(first, vk::Sampler{nullptr});
    EXPECT_EQ(first, second);
    EXPECT_EQ(cache.Size(), 1u);
}

// A clamped anisotropy request maps onto the same entry as its clamped form.
TEST_F(SamplerCacheGpuTest, ClampedRequestsCollapseToOneEntry) {
    SamplerDesc a{};
    a.anisotropy = true;
    a.max_anisotropy = 64; // clamped to the device maximum
    const vk::Sampler a_sampler = cache.Get(a, 1);

    SamplerDesc b = a;
    b.max_anisotropy = std::max(1u, static_cast<std::uint32_t>(
                                        backend.GetCapabilities().GetMaxSamplerAnisotropy()));
    const vk::Sampler b_sampler = cache.Get(b, 1);

    EXPECT_EQ(a_sampler, b_sampler);
    EXPECT_EQ(cache.Size(), 1u);
}

// Distinct descriptions get distinct samplers.
TEST_F(SamplerCacheGpuTest, DistinctDescriptionsCreateDistinctSamplers) {
    SamplerDesc linear{};
    SamplerDesc nearest{};
    nearest.min_filter = vk::Filter::eNearest;
    nearest.mag_filter = vk::Filter::eNearest;
    nearest.mip_mode = vk::SamplerMipmapMode::eNearest;

    const vk::Sampler a = cache.Get(linear, 1);
    const vk::Sampler b = cache.Get(nearest, 1);
    EXPECT_NE(a, b);
    EXPECT_EQ(cache.Size(), 2u);
}

// A different mip chain length changes max_lod, so it is a distinct entry.
TEST_F(SamplerCacheGpuTest, MipLevelsParticipateInTheKey) {
    SamplerDesc desc{};
    desc.max_lod = 1000.0f;
    const vk::Sampler no_mips = cache.Get(desc, /*mip_levels=*/1);
    const vk::Sampler with_mips = cache.Get(desc, /*mip_levels=*/4);
    EXPECT_NE(no_mips, with_mips);
    EXPECT_EQ(cache.Size(), 2u);
}

// At the allocation ceiling the cache substitutes the nearest sampler instead
// of failing.
TEST_F(SamplerCacheGpuTest, CapacitySubstitutesNearestSampler) {
    SamplerCache tiny{};
    ASSERT_TRUE(tiny.Initialize(backend, /*max_sampler_allocations=*/1));
    SamplerDesc first{};
    const vk::Sampler only = tiny.Get(first, 1);
    ASSERT_NE(only, vk::Sampler{nullptr});

    SamplerDesc second{};
    second.address_u = vk::SamplerAddressMode::eClampToEdge;
    const vk::Sampler substituted = tiny.Get(second, 1);
    EXPECT_EQ(substituted, only);
    EXPECT_EQ(tiny.Size(), 1u);
    EXPECT_EQ(tiny.Substitutions(), 1u);
    tiny.Shutdown();
}

} // namespace
