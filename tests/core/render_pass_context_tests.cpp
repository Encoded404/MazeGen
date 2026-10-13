#include <gtest/gtest.h>

#include <glm/glm.hpp>

import std;

import vulkan;

import VulkanEngine.RenderPipeline;

namespace {

using namespace VulkanEngine::PipelinePass;

vk::Image DummyImage(std::uintptr_t value) {
    return vk::Image(reinterpret_cast<vk::Image::CType>(value));
}

vk::Buffer DummyBuffer(std::uintptr_t value) {
    return vk::Buffer(reinterpret_cast<vk::Buffer::CType>(value));
}

TEST(RenderPassContextTest, PassResourceTypedAccessors) {
    ResolvedResource image{};
    image.kind = VulkanEngine::RenderGraph::ResourceKind::Image;
    image.image = DummyImage(0x1111);
    image.view = vk::ImageView(reinterpret_cast<vk::ImageView::CType>(0x2222));
    image.format = vk::Format::eR8G8B8A8Unorm;
    image.resolved = true;

    const PassResource image_resource{image};
    EXPECT_TRUE(image_resource.IsValid());
    EXPECT_TRUE(image_resource.IsImage());
    EXPECT_FALSE(image_resource.IsBuffer());
    EXPECT_EQ(image_resource.AsImage(), image.image);
    EXPECT_EQ(image_resource.AsImageView(), image.view);
    EXPECT_EQ(image_resource.AsBuffer(), nullptr);
    EXPECT_EQ(image_resource.GetFormat(), vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(image_resource.raw().handle, image.handle);

    ResolvedResource buffer{};
    buffer.kind = VulkanEngine::RenderGraph::ResourceKind::Buffer;
    buffer.buffer = DummyBuffer(0x3333);
    buffer.offset = 256;
    buffer.size = 1024;
    buffer.resolved = true;

    const PassResource buffer_resource{buffer};
    EXPECT_TRUE(buffer_resource.IsBuffer());
    EXPECT_FALSE(buffer_resource.IsImage());
    EXPECT_EQ(buffer_resource.AsBuffer(), buffer.buffer);
    EXPECT_EQ(buffer_resource.GetOffset(), 256u);
    EXPECT_EQ(buffer_resource.GetSize(), 1024u);
    EXPECT_EQ(buffer_resource.AsImage(), nullptr);
}

TEST(RenderPassContextTest, GetResourceResolvesByLookupWithErrors) {
    ResourceLookupTable lookup;
    ResolvedResource image{};
    image.kind = VulkanEngine::RenderGraph::ResourceKind::Image;
    image.image = DummyImage(0x10);
    image.format = vk::Format::eR8G8B8A8Unorm;
    image.resolved = true;
    lookup.Set("scene-color", image);

    FrameContext context{};
    EXPECT_EQ(context.GetResource("scene-color").error(), ResourceLookupError::NoLookupBound);

    context.resource_lookup = &lookup;
    auto found = context.GetResource("scene-color");
    ASSERT_TRUE(found.has_value());
    EXPECT_TRUE(found->IsImage());
    EXPECT_EQ(found->GetFormat(), vk::Format::eR8G8B8A8Unorm);

    const auto missing = context.GetResource("does-not-exist");
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error(), ResourceLookupError::NotFound);
}

TEST(RenderPassContextTest, PushConstantValidationReturnsErrorsNotAsserts) {
    FrameContext context{};

    const auto undeclared = context.ValidatePushConstants(sizeof(float));
    ASSERT_FALSE(undeclared.has_value());
    EXPECT_EQ(undeclared.error(), PushConstantError::NotDeclared);

    context.declared_push_constant_stages = vk::ShaderStageFlagBits::eFragment;
    context.declared_push_constant_size = sizeof(glm::vec4);
    const auto mismatch = context.ValidatePushConstants(sizeof(float));
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error(), PushConstantError::SizeMismatch);

    // Correct size/stages but no layout yet (Phase 5 owns layouts).
    const auto no_layout = context.ValidatePushConstants(sizeof(glm::vec4));
    ASSERT_FALSE(no_layout.has_value());
    EXPECT_EQ(no_layout.error(), PushConstantError::NoPipelineLayout);

    context.pipeline_layout = vk::PipelineLayout(reinterpret_cast<vk::PipelineLayout::CType>(0x1));
    EXPECT_TRUE(context.ValidatePushConstants(sizeof(glm::vec4)).has_value());
}

TEST(RenderPassContextTest, SetupContextUsesInjectedExtent) {
    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    PassSetupContext context(pipeline, 1920, 1080);
    EXPECT_EQ(context.GetRenderWidth(), 1920u);
    EXPECT_EQ(context.GetRenderHeight(), 1080u);
}

// A custom pass that records what the populated context looked like.
class CapturingPass final : public IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return "capturing"; }

    void Setup(PassSetupContext& ctx) override {
        transient_ = ctx.CreateTransientImage(TransientImageDesc{
            .name = "captured-transient",
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = 32,
            .height = 32,
        });
        ctx.AddWrite(transient_);
        ctx.DeclarePushConstants<float>(vk::ShaderStageFlagBits::eFragment);
    }

    void Execute(const FrameContext& ctx, vk::CommandBuffer) override {
        executed = true;
        captured_extent = ctx.render_extent;
        captured_view = ctx.view;
        captured_proj = ctx.proj;
        captured_push_size = ctx.declared_push_constant_size;
        captured_push_stages = ctx.declared_push_constant_stages;
        const auto resource = ctx.GetResource("captured-transient");
        resource_found = resource.has_value();
        const auto missing = ctx.GetResource("missing-resource");
        missing_reports_not_found =
            !missing.has_value() && missing.error() == ResourceLookupError::NotFound;
    }

    bool executed = false;
    vk::Extent2D captured_extent{};
    glm::mat4 captured_view{0.0f};
    glm::mat4 captured_proj{0.0f};
    std::uint32_t captured_push_size = 0;
    vk::ShaderStageFlags captured_push_stages{};
    bool resource_found = false;
    bool missing_reports_not_found = false;

private:
    VulkanEngine::RenderGraph::ResourceHandle transient_{};
};

TEST(RenderPassContextTest, RegisteredPassReceivesPopulatedFrameContext) {
    VulkanEngine::RenderPipeline::RenderPipeline pipeline;

    auto pass = std::make_unique<CapturingPass>();
    auto* raw_pass = pass.get();
    ASSERT_TRUE(pipeline.RegisterPass(std::move(pass)).has_value());
    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());

    ResourceLookupTable lookup;
    ResolvedResource entry{};
    entry.kind = VulkanEngine::RenderGraph::ResourceKind::Image;
    entry.image = DummyImage(0x44);
    entry.resolved = true;
    lookup.Set("captured-transient", entry);

    RenderFrameData data{};
    data.frame.render_extent = vk::Extent2D{800, 600};
    data.frame.view = glm::mat4(2.0f);
    data.frame.proj = glm::mat4(3.0f);
    data.frame.resource_lookup = &lookup;

    bool invoked = false;
    for (const auto& compiled_pass : pipeline.GetCompiledGraph().passes) {
        if (compiled_pass.name == "capturing") {
            ASSERT_TRUE(compiled_pass.execute.callback);
            compiled_pass.execute.callback(&data, {});
            invoked = true;
        }
    }
    ASSERT_TRUE(invoked);

    EXPECT_TRUE(raw_pass->executed);
    EXPECT_EQ(raw_pass->captured_extent.width, 800u);
    EXPECT_EQ(raw_pass->captured_extent.height, 600u);
    EXPECT_EQ(raw_pass->captured_view[0][0], 2.0f);
    EXPECT_EQ(raw_pass->captured_proj[0][0], 3.0f);
    // Push-constant declaration from Setup() reached the executing context.
    EXPECT_EQ(raw_pass->captured_push_size, sizeof(float));
    EXPECT_TRUE((raw_pass->captured_push_stages & vk::ShaderStageFlagBits::eFragment) !=
                vk::ShaderStageFlags{});
    EXPECT_TRUE(raw_pass->resource_found);
    EXPECT_TRUE(raw_pass->missing_reports_not_found);
}

}  // namespace
