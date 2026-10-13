#include <gtest/gtest.h>

#include <glm/glm.hpp>

import std;

import vulkan;
import VulkanEngine.RenderPipeline;

namespace {

using namespace VulkanEngine::PipelinePass;

// A pass that declares an engine-owned compute pipeline, push constants, and an
// app descriptor binding, so the declaration plumbing can be inspected.
class DeclaringComputePass final : public IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return "declaring-compute"; }

    void Setup(PassSetupContext& ctx) override {
        ctx.RequestComputePipeline(/*compute_shader=*/42);

        VulkanEngine::Render::DescriptorDecl binding{};
        binding.set = 5;
        binding.binding = 0;
        binding.kind = VulkanEngine::Render::DescriptorKind::StorageImage;
        binding.descriptor_type = vk::DescriptorType::eStorageImage;
        binding.stage_flags = vk::ShaderStageFlagBits::eCompute;
        binding.count = 1;
        ctx.DeclareBindings({binding});

        ctx.DeclarePushConstants<glm::vec4>(vk::ShaderStageFlagBits::eCompute);

        const auto target = ctx.CreateTransientImage(TransientImageDesc{
            .name = "declaring-target",
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = 320,
            .height = 240,
        });
        ctx.AddWrite(target);
    }

    void Execute(const FrameContext&, vk::CommandBuffer) override {}
};

class DeclaringGraphicsPass final : public IPipelinePass {
public:
    [[nodiscard]] std::string_view GetName() const override { return "declaring-graphics"; }

    void Setup(PassSetupContext& ctx) override {
        ctx.RequestGraphicsPipeline(/*vertex=*/7, /*fragment=*/9,
                                    {vk::Format::eR8G8B8A8Unorm}, vk::Format::eD32Sfloat);
        const auto target = ctx.CreateTransientImage(TransientImageDesc{
            .name = "graphics-target",
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = 128,
            .height = 128,
        });
        ctx.AddWrite(target);
    }

    void Execute(const FrameContext&, vk::CommandBuffer) override {}
};

TEST(PassPipelineTest, ComputeRequestReachesPipelineDeclaration) {
    VulkanEngine::RenderPipeline::RenderPipeline pipeline;

    const auto handle = pipeline.RegisterPass(std::make_unique<DeclaringComputePass>());
    ASSERT_TRUE(handle.has_value());

    const auto* request = pipeline.GetPassPipelineRequest(*handle);
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->kind, PassPipelineKind::Compute);
    EXPECT_EQ(request->compute_shader, 42u);
    EXPECT_EQ(request->push_constant_size, sizeof(glm::vec4));
    EXPECT_TRUE((request->push_constant_stages & vk::ShaderStageFlagBits::eCompute) !=
                vk::ShaderStageFlags{});

    // No device/shader manager was supplied, so nothing is built (and nothing
    // crashes trying).
    EXPECT_EQ(pipeline.GetPassPipelineLayout(*handle), nullptr);
    EXPECT_EQ(pipeline.GetPassPipelineLayoutByName("declaring-compute"), nullptr);
}

TEST(PassPipelineTest, GraphicsRequestReachesPipelineDeclaration) {
    VulkanEngine::RenderPipeline::RenderPipeline pipeline;

    const auto handle = pipeline.RegisterPass(std::make_unique<DeclaringGraphicsPass>());
    ASSERT_TRUE(handle.has_value());

    const auto* request = pipeline.GetPassPipelineRequest(*handle);
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->kind, PassPipelineKind::Graphics);
    EXPECT_EQ(request->vertex_shader, 7u);
    EXPECT_EQ(request->fragment_shader, 9u);
    EXPECT_EQ(request->depth_format, vk::Format::eD32Sfloat);
    ASSERT_EQ(request->color_formats.size(), 1u);
    EXPECT_EQ(request->color_formats.front(), vk::Format::eR8G8B8A8Unorm);
}

TEST(PassPipelineTest, UndeclaredPassHasNoPipelineRequest) {
    class PlainPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "plain"; }
        void Setup(PassSetupContext&) override {}
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    const auto handle = pipeline.RegisterPass(std::make_unique<PlainPass>());
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(pipeline.GetPassPipelineRequest(*handle), nullptr);
}

// A declared binding may be associated with a graph resource; binding an
// undeclared (set, binding) or a reserved engine set is a structured error.
TEST(PassPipelineTest, BindResourceValidatesAgainstDeclarations) {
    class BoundPass final : public IPipelinePass {
    public:
        BoundPass(std::string name, std::uint32_t set, std::uint32_t binding, bool declare)
            : name_(std::move(name)), set_(set), binding_(binding), declare_(declare) {}

        [[nodiscard]] std::string_view GetName() const override { return name_; }

        void Setup(PassSetupContext& ctx) override {
            const auto target = ctx.CreateTransientBuffer(TransientBufferDesc{
                .name = "bound-buffer", .size = 256});
            VulkanEngine::Render::DescriptorDecl binding{};
            binding.set = set_;
            binding.binding = binding_;
            binding.kind = VulkanEngine::Render::DescriptorKind::Shared;
            binding.descriptor_type = vk::DescriptorType::eStorageBuffer;
            binding.stage_flags = vk::ShaderStageFlagBits::eCompute;
            if (declare_) {
                ctx.DeclareBindings({binding});
            }
            ctx.BindResource(set_, binding_, target);
            ctx.RequestComputePipeline(11);
        }

        void Execute(const FrameContext&, vk::CommandBuffer) override {}

    private:
        std::string name_;
        std::uint32_t set_;
        std::uint32_t binding_;
        bool declare_;
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    EXPECT_TRUE(pipeline.RegisterPass(std::make_unique<BoundPass>("bound-ok", 5, 0, true)).has_value());

    const auto undeclared = pipeline.RegisterPass(std::make_unique<BoundPass>("bound-undeclared", 5, 3, false));
    ASSERT_FALSE(undeclared.has_value());
    EXPECT_EQ(undeclared.error().code, VulkanEngine::RenderPipeline::PassErrorCode::InvalidDeclaration);

    const auto reserved = pipeline.RegisterPass(std::make_unique<BoundPass>("bound-reserved", 0, 0, true));
    ASSERT_FALSE(reserved.has_value());
    EXPECT_EQ(reserved.error().code, VulkanEngine::RenderPipeline::PassErrorCode::InvalidDeclaration);
}

// A registration rejected after Setup() must roll back the graph resources
// Setup() created, so the allocator never sees an orphaned transient.
TEST(PassPipelineTest, FailedRegistrationRollsBackCreatedTransients) {
    class FailingPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "failing-transient-pass"; }

        void Setup(PassSetupContext& ctx) override {
            const auto target = ctx.CreateTransientBuffer(
                TransientBufferDesc{.name = "orphan-buffer", .size = 128});
            // BindResource without a matching DeclareBindings is rejected after
            // Setup() has already registered "orphan-buffer".
            ctx.BindResource(5, 0, target);
            ctx.RequestComputePipeline(13);
        }

        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    const auto rejected = pipeline.RegisterPass(std::make_unique<FailingPass>());
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, VulkanEngine::RenderPipeline::PassErrorCode::InvalidDeclaration);

    pipeline.Compile();
    for (const auto& lifetime : pipeline.GetCompiledGraph().resource_lifetimes) {
        EXPECT_NE(lifetime.name, "orphan-buffer");
    }
}

// A pass on the compute queue may not touch imported (graphics-exclusive)
// resources, request a graphics pipeline, or declare render attachments.
TEST(PassPipelineTest, ComputeQueuePassRejectsImportedResources) {
    class ComputeBackbufferPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "compute-backbuffer"; }

        void Setup(PassSetupContext& ctx) override {
            const auto backbuffer = ctx.ReadBackbuffer();
            ctx.AddRead(backbuffer, VulkanEngine::RenderGraph::PipelineStageIntent::ComputeShader,
                        VulkanEngine::RenderGraph::AccessIntent::Read);
            ctx.RequestComputePipeline(1);
            ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
        }

        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    pipeline.SetAsyncComputeAvailable(true);
    pipeline.RegisterResourceResolver("swapchain-backbuffer",
        [](std::uint32_t) { return vk::Image{}; },
        [](std::uint32_t) { return vk::ImageView{}; },
        vk::Format::eR8G8B8A8Unorm);

    const auto handle = pipeline.RegisterPass(std::make_unique<ComputeBackbufferPass>());
    ASSERT_FALSE(handle.has_value());
    EXPECT_EQ(handle.error().code, VulkanEngine::RenderPipeline::PassErrorCode::InvalidDeclaration);
}

TEST(PassPipelineTest, ComputeQueuePassRejectsGraphicsPipelineAndAttachments) {
    class GraphicsPipelineOnCompute final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "graphics-on-compute"; }
        void Setup(PassSetupContext& ctx) override {
            ctx.RequestGraphicsPipeline(1, 2);
            ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
        }
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };
    class AttachmentsOnCompute final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "attachments-on-compute"; }
        void Setup(PassSetupContext& ctx) override {
            const auto target = ctx.CreateTransientImage(TransientImageDesc{
                .name = "compute-attachment", .format = vk::Format::eR8G8B8A8Unorm,
                .width = 64, .height = 64});
            ctx.AddWrite(target);
            VulkanEngine::RenderGraph::PassAttachmentSetup setup{};
            setup.auto_begin_rendering = true;
            VulkanEngine::RenderGraph::AttachmentInfo color{};
            color.resource = target;
            setup.color_attachments.push_back(color);
            ctx.SetPassAttachments(setup);
            ctx.RequestComputePipeline(1);
            ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
        }
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    pipeline.SetAsyncComputeAvailable(true);

    const auto graphics = pipeline.RegisterPass(std::make_unique<GraphicsPipelineOnCompute>());
    ASSERT_FALSE(graphics.has_value());
    EXPECT_EQ(graphics.error().code, VulkanEngine::RenderPipeline::PassErrorCode::ValidationFailed);

    const auto attachments = pipeline.RegisterPass(std::make_unique<AttachmentsOnCompute>());
    ASSERT_FALSE(attachments.has_value());
    EXPECT_EQ(attachments.error().code, VulkanEngine::RenderPipeline::PassErrorCode::ValidationFailed);
}

// Positive control: a compute-queue pass operating only on transients registers.
TEST(PassPipelineTest, ComputeQueuePassWithTransientsSucceeds) {
    class ComputeTransientPass final : public IPipelinePass {
    public:
        [[nodiscard]] std::string_view GetName() const override { return "compute-transient"; }
        void Setup(PassSetupContext& ctx) override {
            const auto target = ctx.CreateTransientBuffer(
                TransientBufferDesc{.name = "compute-only-buffer", .size = 64});
            ctx.AddWrite(target);
            ctx.RequestComputePipeline(1);
            ctx.SetQueueType(VulkanEngine::RenderGraph::QueueType::Compute);
        }
        void Execute(const FrameContext&, vk::CommandBuffer) override {}
    };

    VulkanEngine::RenderPipeline::RenderPipeline pipeline;
    pipeline.SetAsyncComputeAvailable(true);
    const auto handle = pipeline.RegisterPass(std::make_unique<ComputeTransientPass>());
    ASSERT_TRUE(handle.has_value());
    pipeline.Compile();
    EXPECT_TRUE(pipeline.IsCompiled());
}

}  // namespace
