#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanBackend.Vulkan.VulkanCapabilities;

namespace {

using VulkanBackend::Vulkan::Feature;
using VulkanBackend::Vulkan::SupportedDeviceState;
using VulkanBackend::Vulkan::VulkanCapabilities;
using VulkanBackend::Vulkan::VulkanCapabilitiesBuilder;
namespace Caps = VulkanBackend::Vulkan;

// MID requires drawIndirectFirstInstance (core 1.0), catalogued and mapped to
// the core VkPhysicalDeviceFeatures bit, so these tests are device-free: they
// drive the builder with a synthetic SupportedDeviceState.
//
// kFeatureCatalog is a positional table over the Feature enum by design (there
// is no name->catalog lookup in the engine), so indexing it with
// static_cast<std::size_t>(Feature::X) here is intentional. The requirement /
// policy values each row carries are engine-side table data driven by the
// MID policy, not a test contract; they are pinned against the capability walk
// on device, not here.

TEST(VulkanCapabilitiesDrawIndirectFirstInstance, NameMatchesTheVulkanSpecString) {
    // The catalog name must stay the exact VkPhysicalDeviceFeatures field the
    // Vulkan spec names: it feeds extension/feature string comparisons.
    const auto& spec = Caps::kFeatureCatalog[
        static_cast<std::size_t>(Feature::DrawIndirectFirstInstance)];
    EXPECT_EQ(spec.name, std::string_view("drawIndirectFirstInstance"));
}

TEST(VulkanCapabilitiesDrawIndirectFirstInstance, SupportedMapsToCoreFeatureBit) {
    VulkanCapabilities caps;
    SupportedDeviceState supported{};
    supported.features2.features.drawIndirectFirstInstance = vk::True;

    VulkanCapabilitiesBuilder builder(caps, supported);
    builder.SetFeatureRequest(Feature::DrawIndirectFirstInstance, /*required=*/false);
    builder.FinalizeFeatures();

    EXPECT_TRUE(caps.IsFeatureEnabled(Feature::DrawIndirectFirstInstance));
    EXPECT_FALSE(caps.HasUnmetRequirements());
    EXPECT_EQ(builder.GetFeatureChain().features.drawIndirectFirstInstance, vk::True);
}

TEST(VulkanCapabilitiesDrawIndirectFirstInstance, UnsupportedOptionalStaysOff) {
    VulkanCapabilities caps;
    SupportedDeviceState supported{};
    supported.features2.features.drawIndirectFirstInstance = vk::False;

    VulkanCapabilitiesBuilder builder(caps, supported);
    builder.SetFeatureRequest(Feature::DrawIndirectFirstInstance, /*required=*/false);
    builder.FinalizeFeatures();

    EXPECT_FALSE(caps.IsFeatureEnabled(Feature::DrawIndirectFirstInstance));
    EXPECT_FALSE(caps.HasUnmetRequirements());
}

// The candidate set is the closed list the format resolver may consult. Every
// entry must be a real format: a value-initialized (eUndefined) tail would be
// matched by Find and would answer queries for the wrong format.
TEST(FormatCapabilitiesTest, CandidateSetHasNoUndefinedFormat) {
    for (const vk::Format format : Caps::kCandidateFormats) {
        EXPECT_NE(format, vk::Format::eUndefined);
    }
}

TEST(FormatCapabilitiesTest, FlagsComeFromOptimalTiling) {
    const std::array<Caps::FormatSupport, 1> formats{{
        Caps::FormatSupport{
            .format = vk::Format::eR8G8B8A8Unorm,
            .linear_features = {},
            .optimal_features = vk::FormatFeatureFlagBits2::eSampledImage,
        },
    }};
    const Caps::FormatCapabilities caps{formats};

    EXPECT_TRUE(caps.sampled(vk::Format::eR8G8B8A8Unorm));
    EXPECT_FALSE(caps.transfer_dst(vk::Format::eR8G8B8A8Unorm));
    EXPECT_FALSE(caps.linear_filter(vk::Format::eR8G8B8A8Unorm));
}

// A format the snapshot does not carry (e.g. a native source the resolver has
// no candidate for) reports unsupported instead of asserting.
TEST(FormatCapabilitiesTest, FormatOutsideTheSnapshotReportsUnsupported) {
    const std::array<Caps::FormatSupport, 1> formats{{
        Caps::FormatSupport{
            .format = vk::Format::eR8G8B8A8Unorm,
            .optimal_features = vk::FormatFeatureFlagBits2::eSampledImage |
                                vk::FormatFeatureFlagBits2::eTransferDst,
        },
    }};
    const Caps::FormatCapabilities caps{formats};

    EXPECT_FALSE(caps.sampled(vk::Format::eBc2UnormBlock));
    EXPECT_FALSE(caps.transfer_dst(vk::Format::eBc2UnormBlock));
    EXPECT_FALSE(caps.sampled(vk::Format::eUndefined));
}

}  // namespace
