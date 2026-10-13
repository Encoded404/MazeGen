#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanEngine.TechniqueManager.BaseTechnique;
import VulkanEngine.MaterialManager;

namespace {

using VulkanEngine::MaterialManager::BlendMode;
using VulkanEngine::MaterialManager::MaterialDesc;
using VulkanEngine::MaterialManager::RenderStateKey;
using VulkanEngine::TechniqueManager::DeriveDrawKeyPipelineState;

// A draw group's render-state key must select pipeline state consistent with the
// material it came from: blend for Transparent, no depth write for blending or
// explicit no-depth-write, and no culling for double-sided. The base key 0 must
// leave the technique's configured state untouched.

vk::CullModeFlags BaseCull() { return vk::CullModeFlagBits::eBack; }

TEST(VariantPipelineStateTest, DefaultKeyLeavesStateUnchanged) {
    const auto state = DeriveDrawKeyPipelineState(0, BaseCull());
    EXPECT_FALSE(state.blend_enable);
    EXPECT_EQ(state.cull_mode, BaseCull());
    EXPECT_TRUE(state.depth_write);
}

TEST(VariantPipelineStateTest, TransparentEnablesBlendAndDisablesDepthWrite) {
    const auto key = RenderStateKey(MaterialDesc{.blend = BlendMode::Transparent});
    const auto state = DeriveDrawKeyPipelineState(key, BaseCull());
    EXPECT_TRUE(state.blend_enable);
    EXPECT_FALSE(state.depth_write);
    EXPECT_EQ(state.cull_mode, BaseCull());
}

TEST(VariantPipelineStateTest, CutoutDoesNotBlend) {
    // Cutout is an alpha-mask discard, not blending; it still writes depth.
    const auto key = RenderStateKey(MaterialDesc{.blend = BlendMode::Cutout});
    const auto state = DeriveDrawKeyPipelineState(key, BaseCull());
    EXPECT_FALSE(state.blend_enable);
    EXPECT_TRUE(state.depth_write);
}

TEST(VariantPipelineStateTest, DoubleSidedDisablesCulling) {
    MaterialDesc desc{};
    desc.double_sided = true;
    const auto state = DeriveDrawKeyPipelineState(RenderStateKey(desc), BaseCull());
    EXPECT_EQ(state.cull_mode, vk::CullModeFlags(vk::CullModeFlagBits::eNone));
    EXPECT_TRUE(state.depth_write);
}

TEST(VariantPipelineStateTest, ExplicitNoDepthWriteIsHonored) {
    MaterialDesc desc{};
    desc.depth_write = false;
    const auto state = DeriveDrawKeyPipelineState(RenderStateKey(desc), BaseCull());
    EXPECT_FALSE(state.depth_write);
    EXPECT_FALSE(state.blend_enable);
}

TEST(VariantPipelineStateTest, DefaultKeyMatchesMaterialDefaults) {
    // The default material must produce the base variant's state, so an
    // existing Opaque scene renders through variant 0 exactly as before.
    EXPECT_EQ(RenderStateKey(MaterialDesc{}), 0u);
    const auto state = DeriveDrawKeyPipelineState(RenderStateKey(MaterialDesc{}), BaseCull());
    EXPECT_FALSE(state.blend_enable);
    EXPECT_TRUE(state.depth_write);
    EXPECT_EQ(state.cull_mode, BaseCull());
}

}  // namespace
