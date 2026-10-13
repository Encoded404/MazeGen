#include <gtest/gtest.h>

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

import std;
import std.compat;

import vulkan;

import FileLoader.Types;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.RenderPipeline;
import VulkanEngine.PipelinePass;
import VulkanEngine.Components.Text;
import VulkanEngine.Components.Transform;
import VulkanEngine.Render.Passes.WorldTextPass;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;

namespace {

using VulkanEngine::ComponentRegistry;
using VulkanEngine::Entity;
using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Components::Text;
using VulkanEngine::Components::TextAlign;
using VulkanEngine::Components::TextWrap;
using VulkanEngine::Components::Transform;
using VulkanEngine::PipelinePass::PassPipelineKind;
using VulkanEngine::PipelinePass::PassSetupContext;
using VulkanEngine::Render::DescriptorDecl;
using VulkanEngine::RenderPipeline::RenderPipeline;
using VulkanEngine::SceneRenderer::BuildWorldTextInstances;
using VulkanEngine::SceneRenderer::WorldModelMatrix;
using VulkanEngine::SceneRenderer::WorldTextConstants;
using VulkanEngine::SceneRenderer::WorldTextInstance;
using VulkanEngine::SceneRenderer::WorldTextPass;
using VulkanEngine::Text::AtlasConfig;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphSlot;
using VulkanEngine::Text::MsdfConfig;
using VulkanEngine::Text::MsdfGenerator;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapeText;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

[[nodiscard]] AtlasConfig TestAtlasConfig() {
    AtlasConfig config{};
    config.page_width = 512;
    config.page_height = 512;
    config.padding = 1;
    config.max_pages = 4;
    return config;
}

// Device-free: the component, the batch builder and the pass's Setup
// declarations are pure CPU work. Only the pipeline, the instance buffer and the
// GPU pages need a device, and none is touched here.
class WorldTextTest : public ::testing::Test {
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
        generator_ = std::make_unique<MsdfGenerator>(TestAtlasConfig(), 64);
    }

    // Places a Text component on a fresh entity with an explicit world transform.
    [[nodiscard]] Text& AddText(ComponentRegistry& registry, Entity& entity,
                                std::string content, float world_height) {
        Text& text = registry.AddComponent<Text>(entity);
        text.font_id = 1;
        text.content = std::move(content);
        text.world_height = world_height;
        text.color = {0.25f, 0.5f, 0.75f, 1.0f};
        Transform& transform = registry.AddComponent<Transform>(entity);
        transform.position = glm::vec3{1.0f, 2.0f, 3.0f};
        transform.rotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
        transform.scale = glm::vec3{1.0f, 1.0f, 1.0f};
        return text;
    }

    [[nodiscard]] std::vector<WorldTextInstance> Build(const Text& text,
                                                       const Transform& transform,
                                                       const ShapedRun& run,
                                                       std::uint32_t page = 7) {
        std::vector<WorldTextInstance> instances;
        BuildWorldTextInstances(*generator_, [page](std::uint32_t) { return page; }, text, transform,
                                *face_, run, instances);
        return instances;
    }

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    std::shared_ptr<FontFace> face_;
    std::unique_ptr<MsdfGenerator> generator_;
};

// Text is a data component: addressable by type through the registry, a second
// add of the same type throws rather than silently replacing the first, and
// removal clears both the entity's mask bit and the stored column. The
// members' default values are the component's own business, not this test's.
TEST_F(WorldTextTest, ComponentRegistersAndRemoves) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    Text& text = registry.AddComponent<Text>(entity);
    EXPECT_TRUE(entity.HasComponent<Text>());
    EXPECT_EQ(entity.GetComponent<Text>(), &text);

    // A second component of the same type does not silently replace the first.
    EXPECT_THROW(registry.AddComponent<Text>(entity), std::logic_error);

    // Removal clears the entity's mask bit and the stored column.
    registry.RemoveComponent<Text>(entity);
    EXPECT_FALSE(entity.HasComponent<Text>());
    EXPECT_EQ(entity.GetComponent<Text>(), nullptr);
}

// A known string at a known transform produces one quad per drawable glyph. The
// quads hang off the transform's anchor and advance with the pen; how the
// engine turns the field into local units is its own placement math, asserted
// here only through relations against the engine's own field/slot outputs.
TEST_F(WorldTextTest, BuildsOneQuadPerDrawableGlyphAtTheTransform) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "Hi", 0.5f);
    const Transform& transform = *entity.GetComponent<Transform>();

    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});
    const auto instances = Build(text, transform, run);
    ASSERT_EQ(instances.size(), 2u);

    for (const auto& instance : instances) {
        // The quad sits at the anchor's depth and above it in a y-up frame (a
        // drawable glyph's field is inset by the font's own range gutter, so its
        // top edge is strictly above the baseline).
        EXPECT_NEAR(instance.origin[2], transform.position.z, 1e-4f);
        EXPECT_GT(instance.origin[1], transform.position.y)
            << "the field's top edge must sit above the baseline in a y-up world";

        // Identity rotation and unit scale keep the field's local axes
        // world-axis aligned: right runs +x, and the field's top-to-bottom edge
        // runs down (-y) because world space is y-up while the field's rows run
        // top-down.
        EXPECT_NEAR(instance.right[1], 0.0f, 1e-4f);
        EXPECT_GT(instance.right[0], 0.0f);
        EXPECT_NEAR(instance.up[0], 0.0f, 1e-4f);
        EXPECT_LT(instance.up[1], 0.0f);
    }

    // The pen advances right: the second glyph's pen slot starts strictly right
    // of the first (a glyph's own left bearing still offsets the field within
    // its slot).
    EXPECT_GT(instances[1].origin[0], instances[0].origin[0]);

    const MsdfConfig config{};
    for (std::size_t i = 0; i < run.glyphs.size(); ++i) {
        const auto field = generator_->Get(*face_, run.glyphs[i].glyph_id, config);
        ASSERT_NE(field, nullptr);
        ASSERT_FALSE(field->Empty());

        const WorldTextInstance& instance = instances[i];

        // The uv rect is the atlas slot's inner region -- exactly the slot the
        // generator reports for this field, so it is inside the page and
        // non-degenerate.
        const auto slot = generator_->Generate(*face_, run.glyphs[i].glyph_id, config);
        ASSERT_TRUE(slot.has_value());
        const AtlasConfig& atlas = generator_->Atlas().Config();
        EXPECT_GE(instance.uv_min[0], 0.0f);
        EXPECT_LE(instance.uv_max[1], 1.0f);
        EXPECT_NEAR(instance.uv_min[0],
                    static_cast<float>(slot->rect.x + atlas.padding) / atlas.page_width, 1e-6f);
        EXPECT_NEAR(instance.uv_min[1],
                    static_cast<float>(slot->rect.y + atlas.padding) / atlas.page_height, 1e-6f);
        EXPECT_NEAR(instance.uv_max[0] - instance.uv_min[0],
                    static_cast<float>(field->width) / atlas.page_width, 1e-6f);

        EXPECT_FLOAT_EQ(instance.px_range, field->range);
        EXPECT_EQ(instance.page, 7u);
        EXPECT_FLOAT_EQ(instance.color[0], 0.25f);
        EXPECT_FLOAT_EQ(instance.color[3], 1.0f);
    }
}

// An empty string, a string of spaces and a zero em size all contribute no
// quads rather than degenerate ones.
TEST_F(WorldTextTest, EmptyAndInklessStringsProduceNoQuads) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    Text& text = AddText(registry, entity, "", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();

    EXPECT_TRUE(Build(text, transform, ShapeText(*face_, "", ShapeOptions{})).empty());

    text.content = "   ";
    EXPECT_TRUE(Build(text, transform, ShapeText(*face_, "   ", ShapeOptions{})).empty());

    text.content = "Hi";
    text.world_height = 0.0f;
    EXPECT_TRUE(Build(text, transform, ShapeText(*face_, "Hi", ShapeOptions{})).empty());
}

// Alignment and wrap are honoured: a centred line shifts the block left by half
// its width, and word wrapping at a narrow width pushes glyphs onto more lines.
TEST_F(WorldTextTest, AlignAndWrapChangeTheQuadLayout) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    Text& text = AddText(registry, entity, "Hi", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});

    // Alignment only has slack to distribute when the box is wider than the
    // line. With no wrap width the block is exactly as wide as its widest line,
    // so centred and left coincide -- the box has to be widened first.
    Text aligned = text;
    aligned.wrap = TextWrap::Word;
    aligned.max_width = 8.0f; // comfortably wider than "Hi" at this em size
    aligned.align = TextAlign::Left;
    const auto left = Build(aligned, transform, run);
    ASSERT_FALSE(left.empty());

    aligned.align = TextAlign::Center;
    const auto centered = Build(aligned, transform, run);
    ASSERT_EQ(centered.size(), left.size());

    aligned.align = TextAlign::Right;
    const auto right = Build(aligned, transform, run);
    ASSERT_EQ(right.size(), left.size());

    const float centered_shift = centered.front().origin[0] - left.front().origin[0];
    const float right_shift = right.front().origin[0] - left.front().origin[0];
    // The box's left edge is the anchor, so alignment offsets are positive:
    // centring pushes the line right by half the slack and right-aligning by all
    // of it. Comparing the two shifts pins the ratio without the test needing the
    // line's measured width.
    EXPECT_GT(centered_shift, 0.0f)
        << "a centred line in a wider box must move right by half the slack";
    EXPECT_GT(right_shift, centered_shift);
    EXPECT_NEAR(right_shift, centered_shift * 2.0f, 1e-3f);

    // Word wrapping at a width narrower than the string puts the glyphs on
    // separate lines, so their world y differs.
    Text wrapped = text;
    wrapped.align = TextAlign::Left;
    wrapped.wrap = TextWrap::Word;
    wrapped.max_width = wrapped.world_height; // narrower than "Hi"
    const auto lines = Build(wrapped, transform, run);
    ASSERT_GE(lines.size(), 2u);
    EXPECT_NE(lines.front().origin[1], lines.back().origin[1]);
}

// The transform is applied to the quad's basis, not just its origin: a rotated
// transform turns the right/up vectors with the quad.
TEST_F(WorldTextTest, RotationTurnsTheQuadBasis) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "H", 1.0f);
    Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});

    const auto unrotated = Build(text, transform, run);
    ASSERT_EQ(unrotated.size(), 1u);
    EXPECT_NEAR(unrotated.front().right[1], 0.0f, 1e-4f);

    // 90 degrees about z: local +x becomes world +y, and local -y (the field's
    // top-to-bottom edge) becomes world +x.
    transform.rotation = glm::angleAxis(glm::radians(90.0f), glm::vec3{0.0f, 0.0f, 1.0f});
    const auto rotated = Build(text, transform, run);
    ASSERT_EQ(rotated.size(), 1u);
    EXPECT_NEAR(rotated.front().right[0], 0.0f, 1e-4f);
    EXPECT_GT(rotated.front().right[1], 0.0f);
    EXPECT_NEAR(rotated.front().up[1], 0.0f, 1e-4f);
    EXPECT_GT(rotated.front().up[0], 0.0f);
}

// World space is y-up (the camera's up vector is (0, 1, 0)) while the field's
// rows run top-down, so a glyph's field must sit ABOVE the baseline with its up
// edge running downward. Getting that frame wrong mirrors every glyph
// vertically, which no assertion about the quads' magnitudes can catch -- only
// the signs can.
TEST_F(WorldTextTest, PlacesGlyphFieldsAboveTheBaselineInAYUpFrame) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "H", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});

    const auto instances = Build(text, transform, run);
    ASSERT_EQ(instances.size(), 1u);
    const WorldTextInstance& instance = instances.front();

    // The first line's baseline is the block origin, so a capital's field top
    // must be above it and the field must extend downward from there.
    EXPECT_GT(instance.origin[1], 0.0f)
        << "the field's top edge must sit above the baseline in a y-up world";
    EXPECT_LT(instance.up[1], 0.0f)
        << "the field's top-to-bottom edge must run downward in a y-up world";

    // A mirrored build passes both of the above with the opposite signs, so this
    // is the assertion that actually fails on the bug.
    EXPECT_LT(instance.origin[1] + instance.up[1], instance.origin[1]);
}

// WorldModelMatrix composes translate * rotation * scale, the engine's
// column-vector convention.
TEST_F(WorldTextTest, ModelMatrixComposesTheTransform) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    Transform& transform = registry.AddComponent<Transform>(entity);
    transform.position = glm::vec3{4.0f, 5.0f, 6.0f};
    transform.rotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    transform.scale = glm::vec3{2.0f, 3.0f, 4.0f};

    const glm::mat4 model = WorldModelMatrix(transform);
    const glm::vec4 origin = model * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(origin.x, 4.0f);
    EXPECT_FLOAT_EQ(origin.y, 5.0f);
    EXPECT_FLOAT_EQ(origin.z, 6.0f);
    const glm::vec4 scaled = model * glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    EXPECT_FLOAT_EQ(scaled.x, 2.0f);
    EXPECT_FLOAT_EQ(scaled.y, 3.0f);
    EXPECT_FLOAT_EQ(scaled.z, 4.0f);
}

// The pass declares itself as depth-tested scene content: a read of the depth
// buffer in the depth stage and a depth attachment on its rendering scope, a
// load-into-the-existing-backbuffer colour attachment, a straight-alpha blend,
// an 80-byte camera/atlas push constant and one instance storage buffer at
// application set 5. Depth testing against the scene depth is what makes scene
// geometry occlude it; where it runs in the frame is the renderer's wiring, not
// a declaration detail this test pins.
TEST_F(WorldTextTest, SetupDeclaresDepthTestedOverlayAndInstanceBinding) {
    RenderPipeline pipeline;
    PassSetupContext ctx(pipeline, 640, 480);
    WorldTextPass pass(nullptr, /*vertex_shader=*/11, /*fragment_shader=*/13);
    pass.Setup(ctx);

    const auto attachments = ctx.GetAttachmentSetup();
    ASSERT_TRUE(attachments.has_value());
    EXPECT_FALSE(attachments->auto_begin_rendering);
    ASSERT_EQ(attachments->color_attachments.size(), 1u);
    EXPECT_EQ(attachments->color_attachments.front().load_op, vk::AttachmentLoadOp::eLoad);
    EXPECT_EQ(attachments->color_attachments.front().store_op, vk::AttachmentStoreOp::eStore);
    ASSERT_TRUE(attachments->depth_attachment.has_value());
    EXPECT_EQ(attachments->depth_attachment->load_op, vk::AttachmentLoadOp::eLoad);

    const auto& request = ctx.GetPipelineRequest();
    EXPECT_EQ(request.kind, PassPipelineKind::Graphics);
    EXPECT_EQ(request.vertex_shader, 11u);
    EXPECT_EQ(request.fragment_shader, 13u);
    EXPECT_EQ(ctx.GetPushConstantSize(), sizeof(WorldTextConstants));
    EXPECT_EQ(ctx.GetPushConstantStages(),
              vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    // The fragment emits non-premultiplied colour, so the source contributes its
    // own alpha; how that is composited over the backbuffer is the blend state's
    // full expression elsewhere.
    EXPECT_TRUE(request.blend.enable);
    EXPECT_EQ(request.blend.src_color_blend_factor, vk::BlendFactor::eSrcAlpha);

    EXPECT_TRUE(request.depth.test_enable);
    EXPECT_FALSE(request.depth.write_enable);
    EXPECT_EQ(request.depth.compare_op, vk::CompareOp::eLessOrEqual);

    // Two reads: the depth buffer in the depth stage, and the instance buffer in
    // the vertex stage. The backbuffer is the only write.
    ASSERT_EQ(ctx.GetReadResources().size(), 2u);
    EXPECT_EQ(ctx.GetReadStages()[0],
              VulkanEngine::RenderGraph::PipelineStageIntent::DepthAttachment);
    EXPECT_EQ(ctx.GetReadAccesses()[0], VulkanEngine::RenderGraph::AccessIntent::Read);
    EXPECT_EQ(ctx.GetReadStages()[1],
              VulkanEngine::RenderGraph::PipelineStageIntent::VertexShader);
    EXPECT_EQ(ctx.GetReadResources()[0], attachments->depth_attachment->resource);
    ASSERT_EQ(ctx.GetWriteResources().size(), 1u);
    EXPECT_EQ(ctx.GetWriteResources().front(), attachments->color_attachments.front().resource);

    ASSERT_EQ(ctx.GetDeclaredBindings().size(), 1u);
    const DescriptorDecl& decl = ctx.GetDeclaredBindings().front();
    EXPECT_EQ(decl.set, VulkanEngine::Render::kFirstAppDescriptorSet);
    EXPECT_EQ(decl.binding, 0u);
    EXPECT_EQ(decl.descriptor_type, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(decl.stage_flags, vk::ShaderStageFlagBits::eVertex);
}

} // namespace
