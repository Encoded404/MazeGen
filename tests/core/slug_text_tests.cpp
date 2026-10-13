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
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;

namespace {

using VulkanEngine::ComponentRegistry;
using VulkanEngine::Entity;
using VulkanEngine::FontResource;
using VulkanEngine::ResourceHandle;
using VulkanEngine::ResourceManager;
using VulkanEngine::Components::Text;
using VulkanEngine::Components::Transform;
using VulkanEngine::PipelinePass::PassPipelineKind;
using VulkanEngine::PipelinePass::PassSetupContext;
using VulkanEngine::Render::DescriptorDecl;
using VulkanEngine::RenderPipeline::RenderPipeline;
using VulkanEngine::SceneRenderer::BlobUploader;
using VulkanEngine::SceneRenderer::BuildSlugTextInstances;
using VulkanEngine::SceneRenderer::SlugTextConstants;
using VulkanEngine::SceneRenderer::SlugTextInstance;
using VulkanEngine::SceneRenderer::TextBackend;
using VulkanEngine::SceneRenderer::WorldTextPass;
using VulkanEngine::Text::BlobRangeAllocator;
using VulkanEngine::Text::FontFace;
using VulkanEngine::Text::GlyphAtlasGpu;
using VulkanEngine::Text::GlyphBlobEncoder;
using VulkanEngine::Text::MsdfGenerator;
using VulkanEngine::Text::ShapedRun;
using VulkanEngine::Text::ShapeOptions;
using VulkanEngine::Text::ShapeText;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

// Device-free: the Slug instance builder, the backend routing predicates and the
// pass's Setup declarations are pure CPU work. The buffer the offsets point into
// is the one device-bound piece, and the tests here supply a CPU allocator
// through the BlobUploader seam instead.
class SlugTextTest : public ::testing::Test {
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
    }

    // Places a Text component on a fresh entity with an explicit transform, the
    // same shape the world-text tests use.
    [[nodiscard]] Text& AddText(ComponentRegistry& registry, Entity& entity, std::string content,
                                float world_height) {
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

    // A CPU stand-in for GpuTextBlobBuffer: the same 8-byte-unit allocator the
    // device buffer uses, so the offsets the builder records are exactly the ones
    // the real buffer would hand out.
    [[nodiscard]] BlobUploader CpuUploader() {
        return [this](std::span<const std::byte> bytes) -> std::optional<std::uint64_t> {
            const auto offset = allocator_.Allocate(bytes.size());
            if (!offset.has_value()) {
                return std::nullopt;
            }
            return *offset / BlobRangeAllocator::kAlignment;
        };
    }

    ResourceManager manager_;
    ResourceHandle<FontResource> handle_;
    std::shared_ptr<FontFace> face_;
    GlyphBlobEncoder encoder_{};
    BlobRangeAllocator allocator_{};
};

// A run of glyphs resolves to contiguous blob offsets in 8-byte units, and each
// instance's em box is exactly the encoded blob's ink box.
TEST_F(SlugTextTest, BuilderRecordsContiguousBlobOffsetsForARunOfGlyphs) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "Hi!", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});
    ASSERT_EQ(run.glyphs.size(), 3u);

    std::vector<SlugTextInstance> instances;
    BuildSlugTextInstances(encoder_, CpuUploader(), text, transform, *face_, run, instances);
    ASSERT_EQ(instances.size(), run.glyphs.size())
        << "every glyph of 'Hi!' has ink and must produce a quad";

    const float design_to_local = face_->ScaleForSize(text.world_height);
    std::uint64_t expected_element = 0;
    float pen_x = 0.0f;
    for (std::size_t i = 0; i < run.glyphs.size(); ++i) {
        const auto blob = encoder_.Get(*face_, run.glyphs[i].glyph_id);
        ASSERT_NE(blob, nullptr);
        ASSERT_FALSE(blob->Empty());

        // Bookkeeping: the next blob is placed at the running unit offset, one
        // 8-byte unit per UnitCount, with no gaps and no reuse.
        EXPECT_EQ(instances[i].blob_offset, expected_element);
        expected_element += blob->UnitCount();

        // The em box is the blob's design-unit ink box, y-up.
        EXPECT_FLOAT_EQ(instances[i].em_min[0], static_cast<float>(blob->x_bearing));
        EXPECT_FLOAT_EQ(instances[i].em_max[0],
                        static_cast<float>(blob->x_bearing + blob->width));
        EXPECT_FLOAT_EQ(instances[i].em_min[1],
                        static_cast<float>(blob->y_bearing + blob->height));
        EXPECT_FLOAT_EQ(instances[i].em_max[1], static_cast<float>(blob->y_bearing));
        EXPECT_LT(instances[i].em_min[1], instances[i].em_max[1]);

        // Placement: the world origin is the box's minimum corner under the
        // transform, and the two world edge vectors are the box's spans.
        const float glyph_x = pen_x + run.glyphs[i].offset_x * design_to_local;
        const float baseline_y = run.glyphs[i].offset_y * design_to_local;
        EXPECT_NEAR(instances[i].origin[0],
                    1.0f + glyph_x + static_cast<float>(blob->x_bearing) * design_to_local,
                    1e-4f);
        EXPECT_NEAR(instances[i].origin[1],
                    2.0f + baseline_y +
                        static_cast<float>(blob->y_bearing + blob->height) * design_to_local,
                    1e-4f);
        EXPECT_NEAR(instances[i].origin[2], 3.0f, 1e-4f);
        EXPECT_NEAR(instances[i].right[0], static_cast<float>(blob->width) * design_to_local,
                    1e-4f);
        EXPECT_NEAR(instances[i].right[1], 0.0f, 1e-4f);
        EXPECT_NEAR(instances[i].up[0], 0.0f, 1e-4f);
        // -height: the y-up box height, positive, so the quad's up vector points
        // up the screen rather than down it.
        EXPECT_NEAR(instances[i].up[1], -static_cast<float>(blob->height) * design_to_local,
                    1e-4f);
        EXPECT_GT(instances[i].up[1], 0.0f);

        EXPECT_FLOAT_EQ(instances[i].color[0], 0.25f);
        EXPECT_FLOAT_EQ(instances[i].color[3], 1.0f);
        pen_x += run.glyphs[i].advance_x * design_to_local;
    }
    EXPECT_EQ(expected_element, allocator_.Capacity() / BlobRangeAllocator::kAlignment);
}

// The Slug frame is y-up like the rest of the world: a capital's box sits above
// the baseline and its up edge runs upward. This is the sign the MSDF builder's
// down edge points the other way on, so it is worth pinning directly.
TEST_F(SlugTextTest, BuilderPlacesTheGlyphAboveTheBaselineInAYUpFrame) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "H", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});

    std::vector<SlugTextInstance> instances;
    BuildSlugTextInstances(encoder_, CpuUploader(), text, transform, *face_, run, instances);
    ASSERT_EQ(instances.size(), 1u);
    const SlugTextInstance& instance = instances.front();

    EXPECT_GE(instance.origin[1], 0.0f) << "the box's bottom must not sit below the baseline";
    EXPECT_GT(instance.up[1], 0.0f) << "the box's up edge must run upward in a y-up world";
    // The top of the box is strictly above its bottom, so the quad is not a
    // degenerate line and the sign is not accidental.
    EXPECT_GT(instance.origin[1] + instance.up[1], instance.origin[1]);
}

// A space encodes to a zero-length blob, so it advances the pen and consumes no
// blob offset: the glyphs after it stay contiguous.
TEST_F(SlugTextTest, BuilderSkipsTheSpaceAndKeepsOffsetsContiguous) {
    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "H i", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});
    ASSERT_EQ(run.glyphs.size(), 3u);

    std::vector<SlugTextInstance> instances;
    BuildSlugTextInstances(encoder_, CpuUploader(), text, transform, *face_, run, instances);
    ASSERT_EQ(instances.size(), 2u) << "the space must contribute no quad";

    const auto capital = encoder_.Get(*face_, run.glyphs[0].glyph_id);
    const auto letter = encoder_.Get(*face_, run.glyphs[2].glyph_id);
    ASSERT_NE(capital, nullptr);
    ASSERT_NE(letter, nullptr);
    EXPECT_EQ(instances[0].blob_offset, 0u);
    EXPECT_EQ(instances[1].blob_offset, capital->UnitCount())
        << "the space consumed a blob range";
}

// The backend selector is real: the default constructor is MSDF, the
// backend-selecting constructor reports what it was given, and a queue call for
// the other backend is a no-op instead of instances reinterpreted through the
// wrong struct.
TEST_F(SlugTextTest, BackendSelectorRoutesToTheRightBackend) {
    WorldTextPass msdf(nullptr, /*vertex_shader=*/11, /*fragment_shader=*/13);
    EXPECT_EQ(msdf.Backend(), TextBackend::Msdf);
    EXPECT_TRUE(msdf.AcceptsMsdfRuns());
    EXPECT_FALSE(msdf.AcceptsSlugRuns());

    WorldTextPass slug(nullptr, TextBackend::Slug, /*vertex_shader=*/21, /*fragment_shader=*/23);
    EXPECT_EQ(slug.Backend(), TextBackend::Slug);
    EXPECT_TRUE(slug.AcceptsSlugRuns());
    EXPECT_FALSE(slug.AcceptsMsdfRuns());

    ComponentRegistry registry;
    Entity& entity = registry.CreateEntity();
    const Text& text = AddText(registry, entity, "Hi", 1.0f);
    const Transform& transform = *entity.GetComponent<Transform>();
    const ShapedRun run = ShapeText(*face_, text.content, ShapeOptions{});

    MsdfGenerator generator;
    GlyphAtlasGpu atlas;
    // The MSDF queue on a Slug pass and the Slug queue on an MSDF pass both
    // route away rather than filling the other layout's queue.
    slug.QueueRun(generator, atlas, text, transform, *face_, run);
    EXPECT_EQ(slug.QueuedInstanceCount(), 0u);
    msdf.QueueSlugRun(encoder_, text, transform, *face_, run, /*recording_frame=*/0);
    EXPECT_EQ(msdf.QueuedSlugInstanceCount(), 0u);
}

// Setup declares the Slug backend's own binding and push constant: the instance
// buffer at set 5 binding 0 in the vertex stage, the blob storage buffer at set
// 5 binding 1 in the fragment stage, a read of the blob buffer, and the 80-byte
// camera/viewport constant. Everything else (depth test, blend, attachments) is
// the same seamless world-text behaviour.
TEST_F(SlugTextTest, PassSetupDeclaresTheBlobBindingAndViewportConstants) {
    RenderPipeline pipeline;
    PassSetupContext ctx(pipeline, 640, 480);
    WorldTextPass pass(nullptr, TextBackend::Slug, /*vertex_shader=*/21, /*fragment_shader=*/23);
    pass.Setup(ctx);

    const auto& request = ctx.GetPipelineRequest();
    EXPECT_EQ(request.kind, PassPipelineKind::Graphics);
    EXPECT_EQ(request.vertex_shader, 21u);
    EXPECT_EQ(request.fragment_shader, 23u);
    EXPECT_EQ(ctx.GetPushConstantSize(), sizeof(SlugTextConstants));
    EXPECT_EQ(ctx.GetPushConstantStages(),
              vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    ASSERT_EQ(ctx.GetDeclaredBindings().size(), 2u);
    const DescriptorDecl& instances = ctx.GetDeclaredBindings()[0];
    EXPECT_EQ(instances.set, VulkanEngine::Render::kFirstAppDescriptorSet);
    EXPECT_EQ(instances.binding, 0u);
    EXPECT_EQ(instances.descriptor_type, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(instances.stage_flags, vk::ShaderStageFlagBits::eVertex);
    const DescriptorDecl& blobs = ctx.GetDeclaredBindings()[1];
    EXPECT_EQ(blobs.set, VulkanEngine::Render::kFirstAppDescriptorSet);
    EXPECT_EQ(blobs.binding, 1u);
    EXPECT_EQ(blobs.descriptor_type, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(blobs.stage_flags, vk::ShaderStageFlagBits::eFragment);

    // Depth buffer, instance buffer and blob buffer are the three reads; the
    // backbuffer is still the only write.
    ASSERT_EQ(ctx.GetReadResources().size(), 3u);
    EXPECT_EQ(ctx.GetReadStages()[0],
              VulkanEngine::RenderGraph::PipelineStageIntent::DepthAttachment);
    EXPECT_EQ(ctx.GetReadStages()[1],
              VulkanEngine::RenderGraph::PipelineStageIntent::VertexShader);
    EXPECT_EQ(ctx.GetReadStages()[2],
              VulkanEngine::RenderGraph::PipelineStageIntent::FragmentShader);
    ASSERT_EQ(ctx.GetWriteResources().size(), 1u);

    // The world-text depth/blend behaviour is unchanged by the backend.
    EXPECT_TRUE(request.blend.enable);
    EXPECT_TRUE(request.depth.test_enable);
    EXPECT_FALSE(request.depth.write_enable);
    EXPECT_EQ(request.depth.compare_op, vk::CompareOp::eLessOrEqual);
}

} // namespace
