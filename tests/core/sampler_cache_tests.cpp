#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.TextureTypes;

namespace {

using VulkanEngine::Textures::ClampSamplerDesc;
using VulkanEngine::Textures::FindNearestSamplerIndex;
using VulkanEngine::Textures::HashSamplerDesc;
using VulkanEngine::Textures::SamplerDesc;
using VulkanEngine::Textures::SamplerDescDistance;

// A SamplerDesc is clamped to the device limits before it is hashed, so equal
// requests collapse to one cache entry and no two devices share a key by
// accident. These tests are device-free: only the pure policy lives here.

TEST(SamplerClampTest, AnisotropyDisabledForcesOne) {
    SamplerDesc desired{};
    desired.anisotropy = true;
    desired.max_anisotropy = 16;
    const auto clamped = ClampSamplerDesc(desired, /*anisotropy_supported=*/false,
                                          /*device_max_anisotropy=*/16, /*mip_levels=*/1);
    EXPECT_FALSE(clamped.anisotropy);
    EXPECT_EQ(clamped.max_anisotropy, 1u);
}

TEST(SamplerClampTest, AnisotropyRequestedOffStaysOffEvenWhenSupported) {
    SamplerDesc desired{};
    desired.anisotropy = false;
    desired.max_anisotropy = 8;
    const auto clamped = ClampSamplerDesc(desired, /*anisotropy_supported=*/true,
                                          /*device_max_anisotropy=*/16, /*mip_levels=*/1);
    EXPECT_FALSE(clamped.anisotropy);
    EXPECT_EQ(clamped.max_anisotropy, 1u);
}

TEST(SamplerClampTest, AnisotropyClampedToDeviceMaximum) {
    SamplerDesc desired{};
    desired.anisotropy = true;
    desired.max_anisotropy = 32;
    const auto clamped = ClampSamplerDesc(desired, /*anisotropy_supported=*/true,
                                          /*device_max_anisotropy=*/16, /*mip_levels=*/4);
    EXPECT_TRUE(clamped.anisotropy);
    EXPECT_EQ(clamped.max_anisotropy, 16u);
}

TEST(SamplerClampTest, MaxLodNeverExceedsTheChain) {
    SamplerDesc desired{};
    desired.max_lod = 1000.0f;
    const auto single = ClampSamplerDesc(desired, false, 1, /*mip_levels=*/1);
    EXPECT_FLOAT_EQ(single.max_lod, 0.0f);
    const auto chained = ClampSamplerDesc(desired, false, 1, /*mip_levels=*/5);
    EXPECT_FLOAT_EQ(chained.max_lod, 4.0f);
}

TEST(SamplerClampTest, DoesNotRaiseAnExplicitlyLowerMaxLod) {
    SamplerDesc desired{};
    desired.max_lod = 2.0f;
    const auto clamped = ClampSamplerDesc(desired, false, 1, /*mip_levels=*/8);
    EXPECT_FLOAT_EQ(clamped.max_lod, 2.0f);
}

TEST(SamplerHashTest, EqualDescriptionsHashEqually) {
    SamplerDesc a{};
    a.min_filter = vk::Filter::eNearest;
    a.address_u = vk::SamplerAddressMode::eClampToEdge;
    a.max_anisotropy = 4;
    SamplerDesc b = a;
    EXPECT_EQ(HashSamplerDesc(a), HashSamplerDesc(b));
    EXPECT_TRUE(a == b);
}

TEST(SamplerHashTest, DistinctDescriptionsHashDifferently) {
    SamplerDesc a{};
    SamplerDesc b = a;
    b.address_u = vk::SamplerAddressMode::eClampToEdge;
    EXPECT_NE(HashSamplerDesc(a), HashSamplerDesc(b));
}

// The nearest-sampler substitution is what keeps a draw alive at capacity: the
// cache never fails for want of a sampler, it degrades.

TEST(SamplerNearestTest, EmptyCandidatesReturnZero) {
    const std::span<const SamplerDesc> none{};
    EXPECT_EQ(FindNearestSamplerIndex(none, SamplerDesc{}), 0u);
}

TEST(SamplerNearestTest, ExactMatchHasZeroDistance) {
    SamplerDesc desired{};
    desired.min_filter = vk::Filter::eNearest;
    std::array<SamplerDesc, 2> candidates{};
    candidates[1] = desired;
    EXPECT_EQ(SamplerDescDistance(candidates[1], desired), 0u);
    EXPECT_EQ(FindNearestSamplerIndex(candidates, desired), 1u);
}

TEST(SamplerNearestTest, PicksTheClosestDistinctSampler) {
    // The two candidates are identical in every field except min_filter, so the
    // outcome rests on that single field alone: the candidate whose filter
    // equals the requested one has zero distance and must win, whatever the
    // weights of the remaining fields are.
    SamplerDesc desired{};
    desired.min_filter = vk::Filter::eNearest;
    desired.address_u = vk::SamplerAddressMode::eClampToEdge;
    std::array<SamplerDesc, 2> candidates{};
    candidates[0] = desired;
    candidates[1] = desired;
    candidates[0].min_filter = vk::Filter::eLinear;
    EXPECT_EQ(FindNearestSamplerIndex(candidates, desired), 1u);
}

}  // namespace
