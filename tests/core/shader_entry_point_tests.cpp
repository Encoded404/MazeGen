#include <gtest/gtest.h>

import std;

import vulkan;

import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

// Device-free: a ShaderManager needs a device to build, but the slot list and
// the enumeration lookups it drives are plain data. These tests use a fake
// device handle only where the constructor touches the device; the enumeration
// itself is exercised through the manager once the entry-point plumbing is in.

// The named-entry-point plumbing is: a slot records its entry point, the find
// helpers enumerate every slot for one source, and a reload targets the slot's
// own entry point. The last two matter because one source now backs several
// variant modules; resolving only the first would clobber the others.

TEST(ShaderEntryPointTest, SlotDefaultsToMain) {
    VulkanEngine::ShaderSystem::ShaderModuleSlot slot{};
    EXPECT_EQ(slot.entry_point, "main");
}

TEST(ShaderEntryPointTest, SlotCarriesNamedEntryPoint) {
    VulkanEngine::ShaderSystem::ShaderModuleSlot slot{};
    slot.entry_point = "main_uv1";
    EXPECT_EQ(slot.entry_point, "main_uv1");
}

// The pipeline descs default both stages to "main" so every existing caller is
// behavior-identical without naming an entry point.
TEST(ShaderEntryPointTest, GraphicsDescDefaultsBothStagesToMain) {
    VulkanEngine::ShaderSystem::GraphicsPipelineDesc desc{};
    EXPECT_EQ(desc.vertex_entry_point, "main");
    EXPECT_EQ(desc.fragment_entry_point, "main");
}

TEST(ShaderEntryPointTest, ComputeDescDefaultsToMain) {
    VulkanEngine::ShaderSystem::ComputePipelineDesc desc{};
    EXPECT_EQ(desc.entry_point, "main");
}
