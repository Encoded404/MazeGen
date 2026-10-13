#include <gtest/gtest.h>

#include <glm/glm.hpp>

import std;

import vulkan;

import VulkanEngine.RenderPipeline;
import VulkanEngine.SceneRenderer;
import VulkanEngine.PipelinePass;
import VulkanEngine.Render.Passes.ExpandPass;
import VulkanEngine.Render.Passes.OccluderSelectPass;
import VulkanEngine.Render.Passes.OccluderPrePass;
import VulkanEngine.Render.Passes.PreCullPass;
import VulkanEngine.Render.Passes.HiZPass;
import VulkanEngine.Render.Passes.DepthPrePass;
import VulkanEngine.Render.Passes.OcclusionPass;
import VulkanEngine.Render.Passes.CollectPass;
import VulkanEngine.Render.Passes.MainPass;
import VulkanEngine.Render.Passes.WorldTextPass;
import VulkanEngine.Render.Passes.TextPass;
import VulkanEngine.Render.Passes.ImGuiPass;

namespace {

using namespace VulkanEngine::PipelinePass;
using VulkanEngine::RenderGraph::PassHandle;
using VulkanEngine::RenderPipeline::RenderPipeline;

const VulkanEngine::RenderGraph::CompiledPass* FindPass(const RenderPipeline& pipeline,
                                                        PassHandle handle) {
    for (const auto& pass : pipeline.GetCompiledGraph().passes) {
        if (pass.handle == handle) {
            return &pass;
        }
    }
    return nullptr;
}

// Registers the import resolvers the built-in passes require, then every
// built-in through the public RegisterPass path and checks the compiled graph
// covers each BuiltinPass enumerator exactly once, plus the clear/load depth
// contract the renderer's correctness depends on.
TEST(BuiltinPassDeclarationsTest, SetupDrivesGraphDeclarations) {
    VulkanEngine::SceneRenderer::SceneRenderer scene_renderer; // Setup() does not touch the device
    RenderPipeline pipeline;
    pipeline.SetRenderExtent(1280, 720);

    const auto register_image = [&pipeline](std::string name) {
        pipeline.RegisterResourceResolver(std::move(name),
            [](std::uint32_t) { return vk::Image{}; },
            [](std::uint32_t) { return vk::ImageView{}; },
            vk::Format::eR8G8B8A8Unorm);
    };
    register_image("swapchain-backbuffer");
    register_image("depth-buffer");
    register_image("hiz-image");

    const auto reg = [&pipeline](std::unique_ptr<IPipelinePass> pass) -> PassHandle {
        auto result = pipeline.RegisterPass(std::move(pass));
        EXPECT_TRUE(result.has_value()) << (result ? std::string{} : result.error().message);
        return result.has_value() ? *result : PassHandle{};
    };

    namespace SR = VulkanEngine::SceneRenderer;
    const auto expand = reg(std::make_unique<SR::ExpandPass>(scene_renderer));
    const auto occluder_select = reg(std::make_unique<SR::OccluderSelectPass>(scene_renderer));
    const auto occluder_prepass =
        reg(std::make_unique<SR::OccluderPrePass>(scene_renderer, vk::ClearDepthStencilValue{1.0f, 0}));
    const auto hiz_pre = reg(std::make_unique<SR::HiZPass>(scene_renderer, "hiz-gen-pre"));
    const auto pre_cull = reg(std::make_unique<SR::PreCullPass>(scene_renderer));
    const auto depth = reg(std::make_unique<SR::DepthPrePass>(scene_renderer));
    const auto hiz = reg(std::make_unique<SR::HiZPass>(scene_renderer, "hiz-gen"));
    const auto occlusion = reg(std::make_unique<SR::OcclusionPass>(scene_renderer));
    const auto collect = reg(std::make_unique<SR::CollectPass>(scene_renderer));
    const auto main_pass =
        reg(std::make_unique<SR::MainPass>(scene_renderer, glm::vec4{0.1f, 0.1f, 0.1f, 1.0f}));
    const auto world_text = reg(std::make_unique<SR::WorldTextPass>(nullptr, 15, 17));
    const auto text = reg(std::make_unique<SR::TextPass>(nullptr, 19, 23));
    const auto imgui = reg(std::make_unique<SR::ImGuiPass>(nullptr));

    // Key the anchor table by the BuiltinPass enumerator each registration
    // backs: the Count sentinel then drives completeness, so a builtin that is
    // added to (or dropped from) the enum without a registered pass fails below
    // without this test naming it. Pass order itself is the renderer's wiring.
    std::array<PassHandle, kBuiltinPassCount> anchors{};
    anchors[static_cast<std::size_t>(BuiltinPass::Expand)] = expand;
    anchors[static_cast<std::size_t>(BuiltinPass::OccluderSelect)] = occluder_select;
    anchors[static_cast<std::size_t>(BuiltinPass::OccluderPrepass)] = occluder_prepass;
    anchors[static_cast<std::size_t>(BuiltinPass::HiZGenPre)] = hiz_pre;
    anchors[static_cast<std::size_t>(BuiltinPass::PreCull)] = pre_cull;
    anchors[static_cast<std::size_t>(BuiltinPass::DepthPrepass)] = depth;
    anchors[static_cast<std::size_t>(BuiltinPass::HiZGen)] = hiz;
    anchors[static_cast<std::size_t>(BuiltinPass::Occlusion)] = occlusion;
    anchors[static_cast<std::size_t>(BuiltinPass::Collect)] = collect;
    anchors[static_cast<std::size_t>(BuiltinPass::MainPass)] = main_pass;
    anchors[static_cast<std::size_t>(BuiltinPass::WorldText)] = world_text;
    anchors[static_cast<std::size_t>(BuiltinPass::Text)] = text;
    anchors[static_cast<std::size_t>(BuiltinPass::ImGui)] = imgui;

    pipeline.AddDependency(expand, occluder_select);
    pipeline.AddDependency(occluder_select, occluder_prepass);
    pipeline.AddDependency(occluder_prepass, hiz_pre);
    pipeline.AddDependency(hiz_pre, pre_cull);
    pipeline.AddDependency(pre_cull, depth);
    pipeline.AddDependency(depth, hiz);
    pipeline.AddDependency(hiz, occlusion);
    pipeline.AddDependency(occlusion, collect);
    pipeline.AddDependency(collect, main_pass);
    pipeline.AddDependency(main_pass, world_text);
    pipeline.AddDependency(world_text, text);
    pipeline.AddDependency(text, imgui);

    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());
    const auto& passes = pipeline.GetCompiledGraph().passes;

    // Every BuiltinPass enumerator has exactly one compiled pass: no builtin is
    // missing from the graph and no registration silently duplicated one.
    for (std::size_t i = 0; i < kBuiltinPassCount; ++i) {
        ASSERT_TRUE(anchors[i].IsValid())
            << "no pass registered for BuiltinPass enumerator " << i;
        std::size_t matches = 0;
        for (const auto& pass : passes) {
            if (pass.handle == anchors[i]) {
                ++matches;
            }
        }
        EXPECT_EQ(matches, 1u)
            << "BuiltinPass enumerator " << i << " must back exactly one compiled pass";
    }

    // The only ordering relation the docs pin for the anchors themselves: the
    // scene overlays must run after the main pass (see "Ordering" in
    // docs/app-render-passes.md, RunAfter(BuiltinPass::MainPass)). The rest of
    // the chain is hazard-resolved, not pinned by name here.
    const auto position_of = [&](PassHandle handle) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < passes.size(); ++i) {
            if (passes[i].handle == handle) {
                return i;
            }
        }
        return std::nullopt;
    };
    const auto main_pos = position_of(anchors[static_cast<std::size_t>(BuiltinPass::MainPass)]);
    const auto imgui_pos = position_of(anchors[static_cast<std::size_t>(BuiltinPass::ImGui)]);
    ASSERT_TRUE(main_pos.has_value());
    ASSERT_TRUE(imgui_pos.has_value());
    EXPECT_LT(*main_pos, *imgui_pos)
        << "the overlay composites over the main pass's output";

    // Occluder pre-pass: the depth the later contract pair starts from.
    const auto* occluder_pass =
        FindPass(pipeline, anchors[static_cast<std::size_t>(BuiltinPass::OccluderPrepass)]);
    ASSERT_NE(occluder_pass, nullptr);
    ASSERT_TRUE(occluder_pass->attachment_setup.has_value());
    ASSERT_TRUE(occluder_pass->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(occluder_pass->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eClear);

    // Depth pre-pass: loads the occluder depth (no clear).
    const auto* depth_pass =
        FindPass(pipeline, anchors[static_cast<std::size_t>(BuiltinPass::DepthPrepass)]);
    ASSERT_NE(depth_pass, nullptr);
    ASSERT_TRUE(depth_pass->attachment_setup.has_value());
    ASSERT_TRUE(depth_pass->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(depth_pass->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eLoad);

    // Main pass: clears its one colour attachment and loads (not clears) the
    // depth the depth pre-pass produced. That clear/load pair is the renderer's
    // correctness contract, so it is the only attachment state pinned here.
    const auto* main =
        FindPass(pipeline, anchors[static_cast<std::size_t>(BuiltinPass::MainPass)]);
    ASSERT_NE(main, nullptr);
    ASSERT_TRUE(main->attachment_setup.has_value());
    ASSERT_EQ(main->attachment_setup->color_attachments.size(), 1u);
    EXPECT_EQ(main->attachment_setup->color_attachments.front().load_op,
              vk::AttachmentLoadOp::eClear);
    ASSERT_TRUE(main->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(main->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eLoad);

    // ImGui overlay: continues the main pass's rendering area, so it must not
    // auto-begin (which would clear the frame).
    const auto* imgui_pass =
        FindPass(pipeline, anchors[static_cast<std::size_t>(BuiltinPass::ImGui)]);
    ASSERT_NE(imgui_pass, nullptr);
    ASSERT_TRUE(imgui_pass->attachment_setup.has_value());
    EXPECT_FALSE(imgui_pass->attachment_setup->auto_begin_rendering);
}

// SceneViewport is the y mirror that reconciles GLM's y-up projection with
// Vulkan's y-down framebuffer: clip-space +y (world up) has to land at
// framebuffer row 0, which only a negative height does. World text draws through
// this viewport as scene content; the screen-space overlay deliberately does not,
// which is exactly why the two must not share one call site.
TEST(SceneViewportTest, MirrorsClipSpaceYIntoTheFramebuffer) {
    const vk::Viewport viewport = SceneViewport(1280, 720);
    EXPECT_FLOAT_EQ(viewport.x, 0.0f);
    EXPECT_FLOAT_EQ(viewport.y, 720.0f);
    EXPECT_FLOAT_EQ(viewport.width, 1280.0f);
    EXPECT_FLOAT_EQ(viewport.height, -720.0f);
    EXPECT_FLOAT_EQ(viewport.minDepth, 0.0f);
    EXPECT_FLOAT_EQ(viewport.maxDepth, 1.0f);

    // The viewport transform's y term, framebuffer_y = (ndc_y + 1) * height / 2 + y.
    const auto framebuffer_y = [&viewport](float ndc_y) {
        return (ndc_y + 1.0f) * viewport.height * 0.5f + viewport.y;
    };
    EXPECT_FLOAT_EQ(framebuffer_y(1.0f), 0.0f)
        << "clip-space +y (world up) must map to the top framebuffer row";
    EXPECT_FLOAT_EQ(framebuffer_y(-1.0f), 720.0f);
}

} // namespace