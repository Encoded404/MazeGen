#include <gtest/gtest.h>

import std;
import std.compat;

import vulkan;

import FileLoader.Types;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.RenderPipeline;
import VulkanEngine.PipelinePass;
import VulkanEngine.Render.Passes.TextPass;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.Shaping;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::PipelinePass::PassPipelineKind;
using VulkanEngine::PipelinePass::PassSetupContext;
using VulkanEngine::Render::DescriptorDecl;
using VulkanEngine::RenderPipeline::RenderPipeline;
using VulkanEngine::SceneRenderer::BuildTextInstances;
using VulkanEngine::SceneRenderer::TextInstance;
using VulkanEngine::SceneRenderer::TextPass;
using VulkanEngine::SceneRenderer::TextRunRequest;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphRasterizer;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapeText;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

// Device-free: the batch builder and the pass's Setup declarations are pure CPU
// work. Only the pipeline and the instance buffer need a device, and neither is
// touched here.
class TextPassTest : public ::testing::Test {
protected:
    void SetUp() override {
        handle_ = manager_.LoadFromFile<FontResource>(TestFontPath(),
                                                      ResourceManager::LoadSpeed::Instant);
        ASSERT_TRUE(handle_.IsValid());
        FontResource* resource = handle_.Get();
        ASSERT_NE(resource, nullptr);
        ASSERT_TRUE(resource->IsLoaded());
        face_ = FontFace::Create(*resource);
        ASSERT_NE(face_, nullptr);
        rasterizer_ = std::make_unique<GlyphRasterizer>(AtlasConfig{}, 64);
    }

    [[nodiscard]] std::vector<TextInstance> Build(const std::string& text, float pixel_size,
                                                  std::uint32_t page = 3) {
        const ShapedRun run = ShapeText(*face_, text, ShapeOptions{});
        TextRunRequest request{};
        request.face = face_.get();
        request.run = &run;
        request.pixel_size = pixel_size;
        request.x = 10.0f;
        request.y = 40.0f;
        request.color[0] = 0.25f;
        request.color[1] = 0.5f;
        request.color[2] = 0.75f;
        request.color[3] = 1.0f;
        std::vector<TextInstance> instances;
        BuildTextInstances(*rasterizer_, [page](std::uint32_t) { return page; }, request,
                           instances);
        return instances;
    }

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    std::shared_ptr<FontFace> face_;
    std::unique_ptr<GlyphRasterizer> rasterizer_;
};

// The pass declares a load-into-the-existing-backbuffer overlay with the
// engine-owned pipeline, a straight-alpha blend, an eight-byte screen-size push
// constant, and one instance storage buffer at application set 5.
TEST_F(TextPassTest, SetupDeclaresOverlayBlendAndInstanceBinding) {
    RenderPipeline pipeline;
    PassSetupContext ctx(pipeline, 640, 480);
    TextPass pass(nullptr, /*vertex_shader=*/7, /*fragment_shader=*/9);
    pass.Setup(ctx);

    const auto attachments = ctx.GetAttachmentSetup();
    ASSERT_TRUE(attachments.has_value());
    EXPECT_FALSE(attachments->auto_begin_rendering);
    ASSERT_EQ(attachments->color_attachments.size(), 1u);
    EXPECT_EQ(attachments->color_attachments.front().load_op, vk::AttachmentLoadOp::eLoad);
    EXPECT_EQ(attachments->color_attachments.front().store_op, vk::AttachmentStoreOp::eStore);
    EXPECT_FALSE(attachments->depth_attachment.has_value());

    const auto& request = ctx.GetPipelineRequest();
    EXPECT_EQ(request.kind, PassPipelineKind::Graphics);
    EXPECT_EQ(request.vertex_shader, 7u);
    EXPECT_EQ(request.fragment_shader, 9u);
    // The pass's push constant is the screen size; the context carries the size
    // and stages until RegisterPass merges them into the pipeline request.
    EXPECT_EQ(ctx.GetPushConstantSize(), 2u * sizeof(float));
    EXPECT_EQ(ctx.GetPushConstantStages(), vk::ShaderStageFlagBits::eVertex);

    // The documented composite: straight-alpha source-over, matching
    // PipelineUtils::CreateAlphaBlendAttachment.
    EXPECT_TRUE(request.blend.enable);
    EXPECT_EQ(request.blend.src_color_blend_factor, vk::BlendFactor::eSrcAlpha);
    EXPECT_EQ(request.blend.dst_color_blend_factor, vk::BlendFactor::eOneMinusSrcAlpha);
    EXPECT_EQ(request.blend.color_blend_op, vk::BlendOp::eAdd);
    EXPECT_EQ(request.blend.src_alpha_blend_factor, vk::BlendFactor::eOne);
    EXPECT_EQ(request.blend.dst_alpha_blend_factor, vk::BlendFactor::eOneMinusSrcAlpha);
    EXPECT_EQ(request.blend.alpha_blend_op, vk::BlendOp::eAdd);

    ASSERT_EQ(ctx.GetDeclaredBindings().size(), 1u);
    const DescriptorDecl& decl = ctx.GetDeclaredBindings().front();
    EXPECT_EQ(decl.set, VulkanEngine::Render::kFirstAppDescriptorSet);
    EXPECT_EQ(decl.binding, 0u);
    EXPECT_EQ(decl.descriptor_type, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(decl.stage_flags, vk::ShaderStageFlagBits::eVertex);
    EXPECT_EQ(decl.count, 1u);

    ASSERT_EQ(ctx.GetBindingAssignments().size(), 1u);
    EXPECT_EQ(ctx.GetBindingAssignments().front().set,
              VulkanEngine::Render::kFirstAppDescriptorSet);
    EXPECT_EQ(ctx.GetBindingAssignments().front().binding, 0u);

    // The imported instance buffer is read in the vertex stage and the
    // backbuffer is written; nothing else is declared.
    EXPECT_EQ(ctx.GetReadResources().size(), 1u);
    EXPECT_EQ(ctx.GetReadStages().front(),
              VulkanEngine::RenderGraph::PipelineStageIntent::VertexShader);
    ASSERT_EQ(ctx.GetWriteResources().size(), 1u);
    EXPECT_EQ(ctx.GetWriteResources().front(), attachments->color_attachments.front().resource);
}

// Every drawable glyph becomes one instance whose quad is its ink box, whose uv
// rect is the atlas slot's inner region, and whose page comes from the lookup.
TEST_F(TextPassTest, BuildsOneInstancePerDrawableGlyph) {
    const auto instances = Build("Hi", 24.0f);
    ASSERT_EQ(instances.size(), 2u);

    const AtlasConfig& config = rasterizer_->Atlas().Config();
    const std::uint32_t glyph_h = face_->GlyphForCodepoint('H');
    const auto bitmap = rasterizer_->Get(*face_, glyph_h, 24.0f);
    const auto slot = rasterizer_->Rasterize(*face_, glyph_h, 24.0f);
    ASSERT_NE(bitmap, nullptr);
    ASSERT_TRUE(slot.has_value());

    const TextInstance& first = instances.front();
    EXPECT_EQ(first.page, 3u);
    EXPECT_FLOAT_EQ(first.size[0], static_cast<float>(bitmap->width));
    EXPECT_FLOAT_EQ(first.size[1], static_cast<float>(bitmap->height));
    EXPECT_FLOAT_EQ(first.position[0], 10.0f + static_cast<float>(bitmap->left));
    EXPECT_FLOAT_EQ(first.position[1], 40.0f + static_cast<float>(bitmap->top));
    EXPECT_FLOAT_EQ(first.uv_min[0],
                    static_cast<float>(slot->rect.x + config.padding) /
                        static_cast<float>(config.page_width));
    EXPECT_FLOAT_EQ(first.uv_min[1],
                    static_cast<float>(slot->rect.y + config.padding) /
                        static_cast<float>(config.page_height));
    EXPECT_FLOAT_EQ(first.uv_max[0] - first.uv_min[0],
                    static_cast<float>(bitmap->width) / static_cast<float>(config.page_width));
    EXPECT_LT(first.uv_min[0], first.uv_max[0]);
    EXPECT_LT(first.uv_min[1], first.uv_max[1]);
    EXPECT_GE(first.uv_min[0], 0.0f);
    EXPECT_LE(first.uv_max[0], 1.0f);

    EXPECT_FLOAT_EQ(first.color[0], 0.25f);
    EXPECT_FLOAT_EQ(first.color[3], 1.0f);

    // Hinting snaps a bitmap to the pixel grid, so the placement must not undo
    // that by landing on a fractional pixel.
    for (const TextInstance& instance : instances) {
        EXPECT_FLOAT_EQ(instance.position[0], std::round(instance.position[0]));
        EXPECT_FLOAT_EQ(instance.position[1], std::round(instance.position[1]));
        EXPECT_GT(instance.size[0], 0.0f);
        EXPECT_GT(instance.size[1], 0.0f);
    }
}

// A larger pixel size covers a strictly larger ink box for the same string.
TEST_F(TextPassTest, LargerPixelSizeProducesLargerInk) {
    const auto small = Build("Hey", 20.0f);
    const auto large = Build("Hey", 40.0f);
    ASSERT_FALSE(small.empty());
    ASSERT_FALSE(large.empty());

    const auto extent = [](const std::vector<TextInstance>& instances) {
        float min_x = instances.front().position[0];
        float max_x = instances.front().position[0] + instances.front().size[0];
        float max_y = instances.front().position[1] + instances.front().size[1];
        for (const TextInstance& instance : instances) {
            min_x = std::min(min_x, instance.position[0]);
            max_x = std::max(max_x, instance.position[0] + instance.size[0]);
            max_y = std::max(max_y, instance.position[1] + instance.size[1]);
        }
        return std::pair{max_x - min_x, max_y};
    };
    const auto [small_width, small_bottom] = extent(small);
    const auto [large_width, large_bottom] = extent(large);
    EXPECT_GT(large_width, small_width);
    EXPECT_GT(large_bottom, small_bottom);
}

// An empty run, a space (inkless glyph) and an unset request all contribute no
// quad rather than a degenerate one.
TEST_F(TextPassTest, InklessAndEmptyRunsProduceNoInstances) {
    EXPECT_TRUE(Build("", 24.0f).empty());
    EXPECT_TRUE(Build(" ", 24.0f).empty());

    TextRunRequest unset{};
    std::vector<TextInstance> instances;
    BuildTextInstances(*rasterizer_, [](std::uint32_t) { return 1u; }, unset, instances);
    EXPECT_TRUE(instances.empty());
}

}  // namespace
