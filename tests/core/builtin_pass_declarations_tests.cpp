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
import VulkanEngine.Render.Passes.ImGuiPass;

namespace {

using namespace VulkanEngine::PipelinePass;
using VulkanEngine::RenderGraph::PassHandle;
using VulkanEngine::RenderPipeline::RenderPipeline;

const VulkanEngine::RenderGraph::CompiledPass* FindPass(const RenderPipeline& pipeline,
                                                        std::string_view name) {
    for (const auto& pass : pipeline.GetCompiledGraph().passes) {
        if (pass.name == name) {
            return &pass;
        }
    }
    return nullptr;
}

// Registers the import resolvers the built-in passes require, then every
// built-in through the public RegisterPass path and checks the declarations
// reach the compiled graph.
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
    const auto imgui = reg(std::make_unique<SR::ImGuiPass>(nullptr));

    pipeline.AddDependency(expand, occluder_select);
    pipeline.AddDependency(occluder_select, occluder_prepass);
    pipeline.AddDependency(occluder_prepass, hiz_pre);
    pipeline.AddDependency(hiz_pre, pre_cull);
    pipeline.AddDependency(pre_cull, depth);
    pipeline.AddDependency(depth, hiz);
    pipeline.AddDependency(hiz, occlusion);
    pipeline.AddDependency(occlusion, collect);
    pipeline.AddDependency(collect, main_pass);
    pipeline.AddDependency(main_pass, imgui);

    pipeline.Compile();
    ASSERT_TRUE(pipeline.IsCompiled());

    // Order matches the built-in chain.
    static constexpr std::array<std::string_view, 11> kExpectedOrder{
        "expand", "occluder-select", "occluder-prepass", "hiz-gen-pre", "pre-cull",
        "depth-prepass", "hiz-gen", "occlusion", "collect", "main-pass", "imgui-overlay",
    };
    ASSERT_EQ(pipeline.GetCompiledGraph().passes.size(), kExpectedOrder.size());
    for (std::size_t i = 0; i < kExpectedOrder.size(); ++i) {
        EXPECT_EQ(pipeline.GetCompiledGraph().passes[i].name, kExpectedOrder[i]);
    }

    // Occluder pre-pass: auto render end, depth clear + store.
    const auto* occluder_pass = FindPass(pipeline, "occluder-prepass");
    ASSERT_NE(occluder_pass, nullptr);
    ASSERT_TRUE(occluder_pass->attachment_setup.has_value());
    EXPECT_TRUE(occluder_pass->attachment_setup->auto_begin_rendering);
    ASSERT_TRUE(occluder_pass->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(occluder_pass->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eClear);
    EXPECT_EQ(occluder_pass->attachment_setup->depth_attachment->store_op,
              vk::AttachmentStoreOp::eStore);

    // Depth pre-pass: loads the occluder depth (no clear).
    const auto* depth_pass = FindPass(pipeline, "depth-prepass");
    ASSERT_NE(depth_pass, nullptr);
    ASSERT_TRUE(depth_pass->attachment_setup.has_value());
    ASSERT_TRUE(depth_pass->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(depth_pass->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eLoad);

    // Main pass: one clear colour attachment plus the loaded depth.
    const auto* main = FindPass(pipeline, "main-pass");
    ASSERT_NE(main, nullptr);
    ASSERT_TRUE(main->attachment_setup.has_value());
    ASSERT_EQ(main->attachment_setup->color_attachments.size(), 1u);
    EXPECT_EQ(main->attachment_setup->color_attachments.front().load_op,
              vk::AttachmentLoadOp::eClear);
    ASSERT_TRUE(main->attachment_setup->depth_attachment.has_value());
    EXPECT_EQ(main->attachment_setup->depth_attachment->load_op,
              vk::AttachmentLoadOp::eLoad);

    // ImGui overlay: draws into the existing colour, so it must not auto-begin.
    const auto* imgui_pass = FindPass(pipeline, "imgui-overlay");
    ASSERT_NE(imgui_pass, nullptr);
    ASSERT_TRUE(imgui_pass->attachment_setup.has_value());
    EXPECT_FALSE(imgui_pass->attachment_setup->auto_begin_rendering);
    ASSERT_EQ(imgui_pass->attachment_setup->color_attachments.size(), 1u);
    EXPECT_EQ(imgui_pass->attachment_setup->color_attachments.front().load_op,
              vk::AttachmentLoadOp::eLoad);
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
