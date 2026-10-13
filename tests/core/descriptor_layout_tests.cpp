#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.RenderPipeline;

namespace {

using namespace VulkanEngine::Render;
using VulkanEngine::PipelinePass::BindingError;
using VulkanEngine::PipelinePass::DescriptorAlignment;
using VulkanEngine::PipelinePass::MakeSampledImageBinding;
using VulkanEngine::PipelinePass::MakeStorageBufferBinding;
using VulkanEngine::PipelinePass::MakeUniformBufferBinding;
using VulkanEngine::PipelinePass::PassResource;
using VulkanEngine::PipelinePass::ResolvedResource;

DescriptorDecl Decl(std::uint32_t set, std::uint32_t binding) {
    DescriptorDecl decl{};
    decl.set = set;
    decl.binding = binding;
    decl.kind = DescriptorKind::Shared;
    decl.descriptor_type = vk::DescriptorType::eStorageBuffer;
    return decl;
}

vk::Buffer DummyBuffer(std::uintptr_t value) {
    return vk::Buffer(reinterpret_cast<vk::Buffer::CType>(value));
}

TEST(DescriptorLayoutTest, GroupingIsDeterministic) {
    std::vector<DescriptorDecl> decls{
        Decl(7, 2), Decl(5, 1), Decl(7, 0), Decl(5, 0),
    };

    const auto groups = GroupBindingsBySet(decls);
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].set, 5u);
    EXPECT_EQ(groups[1].set, 7u);
    ASSERT_EQ(groups[0].bindings.size(), 2u);
    EXPECT_EQ(groups[0].bindings[0].binding, 0u);
    EXPECT_EQ(groups[0].bindings[1].binding, 1u);
    ASSERT_EQ(groups[1].bindings.size(), 2u);
    EXPECT_EQ(groups[1].bindings[0].binding, 0u);
    EXPECT_EQ(groups[1].bindings[1].binding, 2u);
}

TEST(DescriptorLayoutTest, ComposesEngineAndAppSetsContiguously) {
    PipelineLayoutComposer composer;

    std::vector<DescriptorDecl> engine;
    for (std::uint32_t set = 0; set < kEngineDescriptorSetCount; ++set) {
        engine.push_back(Decl(set, 0));
    }
    ASSERT_TRUE(composer.AddEngineBindings(engine).has_value());

    ASSERT_TRUE(composer.AddAppBindings({Decl(6, 1), Decl(5, 0)}).has_value());

    const auto layout = composer.Compose();
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(layout->sets.size(), 7u);
    for (std::uint32_t set = 0; set < 7; ++set) {
        EXPECT_EQ(layout->sets[set].set, set);
    }
    EXPECT_EQ(layout->sets[5].bindings.front().binding, 0u);
    EXPECT_EQ(layout->sets[6].bindings.front().binding, 1u);
}

TEST(DescriptorLayoutTest, RejectsReservedSetFromApp) {
    PipelineLayoutComposer composer;
    const auto added = composer.AddAppBindings({Decl(3, 0)});
    ASSERT_FALSE(added.has_value());
    EXPECT_EQ(added.error(), LayoutError::ReservedSetUsedByApp);
}

TEST(DescriptorLayoutTest, RejectsDuplicateBinding) {
    PipelineLayoutComposer composer;
    ASSERT_TRUE(composer.AddAppBindings({Decl(5, 0)}).has_value());
    const auto duplicate = composer.AddAppBindings({Decl(5, 0)});
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error(), LayoutError::DuplicateBinding);
}

TEST(DescriptorLayoutTest, RejectsSetGap) {
    PipelineLayoutComposer composer;
    std::vector<DescriptorDecl> engine;
    for (std::uint32_t set = 0; set < kEngineDescriptorSetCount; ++set) {
        engine.push_back(Decl(set, 0));
    }
    ASSERT_TRUE(composer.AddEngineBindings(engine).has_value());
    // Set 6 without 5 leaves a hole that the layout slots cannot represent.
    ASSERT_TRUE(composer.AddAppBindings({Decl(6, 0)}).has_value());

    const auto layout = composer.Compose();
    ASSERT_FALSE(layout.has_value());
    EXPECT_EQ(layout.error(), LayoutError::InvalidSetGap);
}

TEST(DescriptorLayoutTest, PushConstantRangesSortedAndOverlapRejected) {
    PipelineLayoutComposer composer;
    ASSERT_TRUE(composer.AddPushConstantRange({
        .stages = vk::ShaderStageFlagBits::eFragment, .offset = 64, .size = 16}).has_value());
    ASSERT_TRUE(composer.AddPushConstantRange({
        .stages = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = 64}).has_value());

    const auto layout = composer.Compose();
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(layout->push_ranges.size(), 2u);
    EXPECT_EQ(layout->push_ranges[0].offset, 0u);
    EXPECT_EQ(layout->push_ranges[1].offset, 64u);

    PipelineLayoutComposer empty;
    EXPECT_EQ(empty.AddPushConstantRange({.stages = vk::ShaderStageFlags{}, .offset = 0, .size = 4}).error(),
              LayoutError::EmptyPushConstantRange);

    PipelineLayoutComposer overlapping;
    ASSERT_TRUE(overlapping.AddPushConstantRange({
        .stages = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = 32}).has_value());
    ASSERT_TRUE(overlapping.AddPushConstantRange({
        .stages = vk::ShaderStageFlagBits::eFragment, .offset = 16, .size = 32}).has_value());
    const auto bad = overlapping.Compose();
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error(), LayoutError::OverlappingPushConstantRange);
}

ResolvedResource BufferResource(std::uint64_t offset, std::uint64_t size) {
    ResolvedResource resource{};
    resource.kind = VulkanEngine::RenderGraph::ResourceKind::Buffer;
    resource.buffer = DummyBuffer(0xABC);
    resource.offset = offset;
    resource.size = size;
    resource.resolved = true;
    return resource;
}

TEST(PassBindingHelperTest, BufferOffsetAlignmentIsHonored) {
    const PassResource aligned{BufferResource(256, 512)};
    const PassResource misaligned{BufferResource(128, 512)};

    const DescriptorAlignment alignment{.min_storage_buffer_offset_alignment = 256,
                                        .min_uniform_buffer_offset_alignment = 64};

    const auto ok = MakeStorageBufferBinding(aligned, alignment);
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->offset, 256u);
    EXPECT_EQ(ok->range, 512u);

    const auto bad = MakeStorageBufferBinding(misaligned, alignment);
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error(), BindingError::OffsetMisaligned);

    // The uniform limit is lower, so the same resource is valid there.
    EXPECT_TRUE(MakeUniformBufferBinding(misaligned, alignment).has_value());
    const PassResource uniform_misaligned{BufferResource(32, 512)};
    EXPECT_EQ(MakeUniformBufferBinding(uniform_misaligned, alignment).error(),
              BindingError::OffsetMisaligned);
}

TEST(PassBindingHelperTest, SampledImageRequiresViewAndImageKind) {
    ResolvedResource image{};
    image.kind = VulkanEngine::RenderGraph::ResourceKind::Image;
    image.image = vk::Image(reinterpret_cast<vk::Image::CType>(0x1));
    image.resolved = true;

    const auto no_view = MakeSampledImageBinding(PassResource{image});
    ASSERT_FALSE(no_view.has_value());
    EXPECT_EQ(no_view.error(), BindingError::NoImageView);

    image.view = vk::ImageView(reinterpret_cast<vk::ImageView::CType>(0x2));
    const auto ok = MakeSampledImageBinding(PassResource{image});
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->imageView, image.view);
    EXPECT_EQ(ok->imageLayout, vk::ImageLayout::eShaderReadOnlyOptimal);

    const PassResource buffer{BufferResource(0, 16)};
    EXPECT_EQ(MakeSampledImageBinding(buffer).error(), BindingError::NotAnImage);
    EXPECT_EQ(MakeStorageBufferBinding(PassResource{image}, DescriptorAlignment{}).error(),
              BindingError::NotABuffer);
}

}  // namespace
