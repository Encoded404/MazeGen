module;

#include <vulkan/vulkan.h>

#include <logging/logging_macros.hpp>

module TestSupport.TextureRenderHarness;

import std;

import logiface;

import vulkan;

import ShaderReflection;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuTexture;
import TestSupport.HeadlessVulkanBackend;

namespace TestSupport {

namespace {

// Bindless descriptor array size for the harness. Slot 0 is the sampled/albebdo
// texture, slot 1 the normal map, slot 2 the ORM map; unused slots alias the
// albedo view so no descriptor is left unwritten.
constexpr std::uint32_t kBindlessSlots = 3;
constexpr float kCameraZ = 10.0f;

std::unique_ptr<vk::raii::Buffer> CreateHostBuffer(const vk::raii::Device& device,
                                                   const vk::PhysicalDevice& physical,
                                                   vk::DeviceSize size,
                                                   std::unique_ptr<vk::raii::DeviceMemory>& out_memory) {
    const vk::BufferCreateInfo info({}, size, vk::BufferUsageFlagBits::eStorageBuffer);
    auto buffer = std::make_unique<vk::raii::Buffer>(device, info);
    const vk::MemoryRequirements req = buffer->getMemoryRequirements();
    const vk::MemoryAllocateInfo alloc(
        req.size,
        VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
            physical, req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    out_memory = std::make_unique<vk::raii::DeviceMemory>(device, alloc);
    buffer->bindMemory(*out_memory, 0);
    return buffer;
}

template <typename T>
void WriteHostBuffer(const vk::raii::DeviceMemory& memory, const T& value) {
    void* mapped = memory.mapMemory(0, sizeof(T));
    std::memcpy(mapped, &value, sizeof(T));
    memory.unmapMemory();
}

}  // namespace

HarnessShaderIds RegisterHarnessShaders(VulkanEngine::ShaderSystem::ShaderManager& shaders,
                                        std::string_view shader_dir) {
    HarnessShaderIds ids{};
    const std::string dir(shader_dir);
    ids.vertex = shaders.RegisterManual(std::format("{}/test_fullscreen_vs.spv", dir), "",
                                        ShaderStage::eVertex);
    ids.sample_fragment = shaders.RegisterManual(std::format("{}/test_sample_frag.spv", dir), "",
                                                 ShaderStage::eFragment);
    return ids;
}

TextureRenderHarness::~TextureRenderHarness() {
    Shutdown();
}

void TextureRenderHarness::Shutdown() {
    pipeline_.reset();
    pipeline_layout_.reset();
    host_buffers_.clear();
    host_memories_.clear();
    material_set5_.reset();
    scene_set4_.reset();
    bindless_set0_.reset();
    pool_.reset();
    material_set5_layout_.reset();
    scene_set4_layout_.reset();
    bindless_set0_layout_.reset();
    backend_ = nullptr;
    shaders_ = nullptr;
}

bool TextureRenderHarness::CreateDescriptorLayouts() {
    const vk::raii::Device& device = backend_->GetDevice();

    {
        const vk::DescriptorSetLayoutBinding binding(
            0, vk::DescriptorType::eCombinedImageSampler, kBindlessSlots,
            vk::ShaderStageFlagBits::eFragment, nullptr);
        bindless_set0_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            device, vk::DescriptorSetLayoutCreateInfo{{}, 1, &binding});
    }

    if (config_.mode == HarnessMode::Standard) {
        const std::array<vk::DescriptorSetLayoutBinding, 2> scene_bindings{{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eFragment, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eFragment, nullptr),
        }};
        scene_set4_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            device, vk::DescriptorSetLayoutCreateInfo{{}, 2, scene_bindings.data()});
    }
    if (config_.mode == HarnessMode::Standard || config_.mode == HarnessMode::Unlit) {
        const vk::DescriptorSetLayoutBinding material_binding(
            0, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eFragment, nullptr);
        material_set5_layout_ = std::make_unique<vk::raii::DescriptorSetLayout>(
            device, vk::DescriptorSetLayoutCreateInfo{{}, 1, &material_binding});
    }
    return bindless_set0_layout_ != nullptr;
}

bool TextureRenderHarness::CreatePipeline() {
    const vk::raii::Device& device = backend_->GetDevice();
    const vk::raii::PhysicalDevice& physical = backend_->GetPhysicalDevice();

    std::vector<vk::DescriptorSetLayout> set_layouts;
    std::vector<vk::PushConstantRange> push_ranges;
    if (config_.mode == HarnessMode::Standard) {
        // The engine shader declares its sets as 0 (bindless), 4 (scene), 5
        // (material); the reserved sets 1..3 must exist as empty layouts so the
        // scene/material layouts land at the indices the shader names.
        set_layouts = {**bindless_set0_layout_, {}, {}, {}, **scene_set4_layout_,
                       **material_set5_layout_};
        push_ranges = {vk::PushConstantRange(vk::ShaderStageFlagBits::eFragment, 0, 16)};
    } else if (config_.mode == HarnessMode::Unlit) {
        // unlit.spv names sets 0 and 5; reserve empty layouts for 1..4 so the
        // material layout lands at index 5.
        set_layouts = {**bindless_set0_layout_, {}, {}, {}, {}, **material_set5_layout_};
    } else {
        set_layouts = {**bindless_set0_layout_};
    }
    vk::PipelineLayoutCreateInfo layout_info{};
    layout_info.setSetLayouts(set_layouts);
    layout_info.setPushConstantRanges(push_ranges);
    pipeline_layout_ = std::make_unique<vk::raii::PipelineLayout>(device, layout_info);

    const vk::PipelineColorBlendAttachmentState blend(
        vk::False, {}, {}, {}, {}, {}, {},
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    VulkanEngine::ShaderSystem::GraphicsPipelineDesc desc{};
    desc.vertex_shader = config_.vertex_shader;
    desc.fragment_shader = config_.fragment_shader;
    desc.vertex_input = vk::PipelineVertexInputStateCreateInfo({}, 0, nullptr, 0, nullptr);
    desc.input_assembly = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);
    desc.viewport = vk::PipelineViewportStateCreateInfo({}, 1, nullptr, 1, nullptr);
    desc.rasterization = vk::PipelineRasterizationStateCreateInfo(
        {}, vk::False, vk::False, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
        vk::FrontFace::eCounterClockwise, vk::False, 0.0f, 0.0f, 0.0f, 1.0f);
    desc.multisample = vk::PipelineMultisampleStateCreateInfo({}, vk::SampleCountFlagBits::e1);
    desc.depth_stencil = vk::PipelineDepthStencilStateCreateInfo({}, vk::False, vk::False, vk::CompareOp::eAlways);
    desc.color_blend = vk::PipelineColorBlendStateCreateInfo({}, vk::False, vk::LogicOp::eCopy, 0, nullptr);
    desc.color_blend_attachments = {blend};
    desc.dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    desc.layout = **pipeline_layout_;
    desc.color_formats = {vk::Format::eR8G8B8A8Unorm};

    VulkanEngine::ShaderSystem::PipelineFactory factory(device, backend_->GetCapabilities(),
                                                        shaders_->GetPipelineCacheRAII());
    auto product = factory.CreateGraphics(desc, *shaders_);
    (void)physical;
    if (!product.has_value()) {
        LOGIFACE_LOG(error, "TextureRenderHarness: pipeline creation failed: " + product.error().message);
        return false;
    }

    // PipelineProduct owns its GPL library pipelines; keep the product itself so
    // a fast-linked pipeline cannot outlive its libraries.
    pipeline_ = std::make_unique<VulkanEngine::ShaderSystem::PipelineProduct>(std::move(*product));
    return true;
}

bool TextureRenderHarness::Initialize(TestSupport::HeadlessVulkanBackend& backend,
                                      VulkanEngine::ShaderSystem::ShaderManager& shaders,
                                      const TextureRenderHarnessConfig& config) {
    config_ = config;
    if (pipeline_ != nullptr) {
        return true;
    }
    backend_ = &backend;
    shaders_ = &shaders;
    if (!backend_->Initialize()) {
        return false;
    }
    if (!CreateDescriptorLayouts()) {
        return false;
    }

    const std::array<vk::DescriptorPoolSize, 2> sizes{{
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, kBindlessSlots),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3),
    }};
    pool_ = std::make_unique<vk::raii::DescriptorPool>(
        backend_->GetDevice(), vk::DescriptorPoolCreateInfo({}, 3, sizes.size(), sizes.data()));

    std::vector<vk::DescriptorSetLayout> layouts{**bindless_set0_layout_};
    if (config_.mode == HarnessMode::Standard) {
        layouts.push_back(**scene_set4_layout_);
        layouts.push_back(**material_set5_layout_);
    } else if (config_.mode == HarnessMode::Unlit) {
        layouts.push_back(**material_set5_layout_);
    }
    auto sets = backend_->GetDevice().allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{**pool_, static_cast<std::uint32_t>(layouts.size()), layouts.data()});
    if (sets.size() != layouts.size()) {
        return false;
    }
    bindless_set0_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[0]));
    if (config_.mode == HarnessMode::Standard) {
        scene_set4_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[1]));
        material_set5_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[2]));
    } else if (config_.mode == HarnessMode::Unlit) {
        material_set5_ = std::make_unique<vk::raii::DescriptorSet>(std::move(sets[1]));
    }

    // Unlit mode does not use the scene set; the pool is only sized for what
    // each mode allocates.
    return CreatePipeline();
}

void TextureRenderHarness::WriteDescriptors(const VulkanEngine::GpuResources::GpuTexture& albedo,
                                             const VulkanEngine::GpuResources::GpuTexture& normal,
                                             const VulkanEngine::GpuResources::GpuTexture& orm,
                                             const HarnessMaterialData& material,
                                             const HarnessSceneHeader& scene,
                                             const HarnessLight& light) {
    const vk::raii::Device& device = backend_->GetDevice();

    // Bindless array: albedo at 0, normal at 1, orm at 2, with every slot
    // defaulting to albedo so no descriptor is left unwritten.
    std::array<vk::DescriptorImageInfo, kBindlessSlots> images{};
    const vk::DescriptorImageInfo albedo_info(albedo.GetSampler(), albedo.GetImageView(),
                                              vk::ImageLayout::eShaderReadOnlyOptimal);
    for (auto& info : images) {
        info = albedo_info;
    }
    if (normal.IsValid()) {
        images[1] = vk::DescriptorImageInfo(normal.GetSampler(), normal.GetImageView(),
                                            vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    if (orm.IsValid()) {
        images[2] = vk::DescriptorImageInfo(orm.GetSampler(), orm.GetImageView(),
                                            vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    const vk::WriteDescriptorSet image_write(*bindless_set0_, 0, 0, kBindlessSlots,
                                             vk::DescriptorType::eCombinedImageSampler,
                                             images.data());
    device.updateDescriptorSets(image_write, {});

    if (config_.mode != HarnessMode::Standard) {
        return;
    }

    // Scene header + light + material land in host-visible storage buffers kept
    // alive by the harness until the next frame or teardown (the device is idle
    // when RenderAndReadBack returns, so reusing them per frame is safe).
    host_buffers_.clear();
    host_memories_.clear();
    std::unique_ptr<vk::raii::DeviceMemory> scene_memory;
    std::unique_ptr<vk::raii::DeviceMemory> light_memory;
    std::unique_ptr<vk::raii::DeviceMemory> material_memory;
    auto scene_buffer = CreateHostBuffer(device, backend_->GetPhysicalDevice(),
                                         std::max<vk::DeviceSize>(sizeof(HarnessSceneHeader), 64), scene_memory);
    auto light_buffer = CreateHostBuffer(device, backend_->GetPhysicalDevice(),
                                         sizeof(HarnessLight), light_memory);
    auto material_buffer = CreateHostBuffer(device, backend_->GetPhysicalDevice(),
                                            sizeof(HarnessMaterialData), material_memory);
    WriteHostBuffer(*scene_memory, scene);
    WriteHostBuffer(*light_memory, light);
    WriteHostBuffer(*material_memory, material);

    const vk::DescriptorBufferInfo scene_info(**scene_buffer, 0, vk::WholeSize);
    const vk::DescriptorBufferInfo light_info(**light_buffer, 0, vk::WholeSize);
    const vk::DescriptorBufferInfo material_info(**material_buffer, 0, vk::WholeSize);
    const std::array<vk::WriteDescriptorSet, 3> buffer_writes{{
        vk::WriteDescriptorSet(*scene_set4_, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &scene_info),
        vk::WriteDescriptorSet(*scene_set4_, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &light_info),
        vk::WriteDescriptorSet(*material_set5_, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &material_info),
    }};
    device.updateDescriptorSets(buffer_writes, {});

    host_buffers_.push_back(std::move(scene_buffer));
    host_buffers_.push_back(std::move(light_buffer));
    host_buffers_.push_back(std::move(material_buffer));
    host_memories_.push_back(std::move(scene_memory));
    host_memories_.push_back(std::move(light_memory));
    host_memories_.push_back(std::move(material_memory));
}

void TextureRenderHarness::WriteUnlitDescriptors(
    const VulkanEngine::GpuResources::GpuTexture& albedo,
    const HarnessUnlitMaterialData& material) {
    const vk::raii::Device& device = backend_->GetDevice();

    std::array<vk::DescriptorImageInfo, kBindlessSlots> images{};
    const vk::DescriptorImageInfo albedo_info(albedo.GetSampler(), albedo.GetImageView(),
                                              vk::ImageLayout::eShaderReadOnlyOptimal);
    for (auto& info : images) {
        info = albedo_info;
    }
    const vk::WriteDescriptorSet image_write(*bindless_set0_, 0, 0, kBindlessSlots,
                                             vk::DescriptorType::eCombinedImageSampler,
                                             images.data());
    device.updateDescriptorSets(image_write, {});

    host_buffers_.clear();
    host_memories_.clear();
    std::unique_ptr<vk::raii::DeviceMemory> material_memory;
    auto material_buffer = CreateHostBuffer(device, backend_->GetPhysicalDevice(),
                                            sizeof(HarnessUnlitMaterialData), material_memory);
    WriteHostBuffer(*material_memory, material);
    const vk::DescriptorBufferInfo material_info(**material_buffer, 0, vk::WholeSize);
    const vk::WriteDescriptorSet buffer_write(*material_set5_, 0, 0, 1,
                                              vk::DescriptorType::eStorageBuffer, nullptr, &material_info);
    device.updateDescriptorSets(buffer_write, {});
    host_buffers_.push_back(std::move(material_buffer));
    host_memories_.push_back(std::move(material_memory));
}

std::vector<std::uint8_t> TextureRenderHarness::RenderUnlitAndReadBack(
    const VulkanEngine::GpuResources::GpuTexture& albedo,
    const HarnessUnlitMaterialData& material_in) {
    if (!albedo.IsValid()) {
        return {};
    }
    HarnessUnlitMaterialData material = material_in;
    material.albedo_texture = 0;
    WriteUnlitDescriptors(albedo, material);
    return RenderFrame([this](vk::raii::CommandBuffer& cmd) {
        const std::array<vk::DescriptorSet, 1> set0{**bindless_set0_};
        const std::array<vk::DescriptorSet, 1> set5{**material_set5_};
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 0, set0, {});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 5, set5, {});
        cmd.draw(3, 1, 0, 0);
    });
}

std::vector<std::uint8_t> TextureRenderHarness::RenderAndReadBack(
    const VulkanEngine::GpuResources::GpuTexture& albedo,
    const VulkanEngine::GpuResources::GpuTexture& normal,
    const VulkanEngine::GpuResources::GpuTexture& orm,
    const HarnessMaterialData& material_in,
    const HarnessSceneHeader& scene,
    const HarnessLight& light) {
    if (!albedo.IsValid()) {
        return {};
    }
    HarnessMaterialData material = material_in;
    material.albedo_texture = 0;
    material.normal_texture = normal.IsValid() ? 1U : 0U;
    material.orm_texture = orm.IsValid() ? 2U : 0U;
    WriteDescriptors(albedo, normal, orm, material, scene, light);
    return RenderFrame([this](vk::raii::CommandBuffer& cmd) {
        if (config_.mode == HarnessMode::Standard) {
            // Sets 1..3 are empty/null in the layout, so bind each real set at
            // the index the shader names: 0, 4, 5.
            const std::array<vk::DescriptorSet, 1> set0{**bindless_set0_};
            const std::array<vk::DescriptorSet, 1> set4{**scene_set4_};
            const std::array<vk::DescriptorSet, 1> set5{**material_set5_};
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 0, set0, {});
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 4, set4, {});
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 5, set5, {});
            const std::array<float, 4> camera{kCameraZ, 0.0f, 0.0f, 0.0f};
            cmd.pushConstants<float>(**pipeline_layout_, vk::ShaderStageFlagBits::eFragment, 0, camera);
        } else {
            const std::array<vk::DescriptorSet, 1> sets{**bindless_set0_};
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipeline_layout_, 0, sets, {});
        }
        cmd.draw(3, 1, 0, 0);
    });
}

std::vector<std::uint8_t> TextureRenderHarness::RenderFrame(
    const std::function<void(vk::raii::CommandBuffer&)>& bind_and_draw) {
    const vk::raii::Device& device = backend_->GetDevice();
    const std::uint32_t width = config_.width;
    const std::uint32_t height = config_.height;
    const vk::DeviceSize pixel_bytes = static_cast<vk::DeviceSize>(width) * height * 4U;

    const vk::ImageCreateInfo color_info(
        {}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, vk::Extent3D{width, height, 1},
        1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive);
    vk::raii::Image color_image(device, color_info);
    const vk::MemoryRequirements color_req = color_image.getMemoryRequirements();
    vk::raii::DeviceMemory color_memory(
        device, vk::MemoryAllocateInfo(
                    color_req.size,
                    VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                        backend_->GetPhysicalDevice(), color_req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eDeviceLocal)));
    color_image.bindMemory(*color_memory, 0);
    const vk::raii::ImageView color_view(
        device, vk::ImageViewCreateInfo({}, *color_image, vk::ImageViewType::e2D,
                                        vk::Format::eR8G8B8A8Unorm,
                                        vk::ComponentMapping{},
                                        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

    const vk::raii::Buffer readback_buffer(
        device, vk::BufferCreateInfo({}, pixel_bytes, vk::BufferUsageFlagBits::eTransferDst));
    const vk::MemoryRequirements read_req = readback_buffer.getMemoryRequirements();
    vk::raii::DeviceMemory read_memory(
        device, vk::MemoryAllocateInfo(
                    read_req.size,
                    VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(
                        backend_->GetPhysicalDevice(), read_req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)));
    readback_buffer.bindMemory(*read_memory, 0);

    vk::raii::CommandBuffer& cmd = backend_->GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    const vk::ImageMemoryBarrier to_color(
        {}, vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
        *color_image, range);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                        {}, {}, {}, to_color);

    const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    vk::RenderingAttachmentInfo color_attachment{};
    color_attachment.imageView = *color_view;
    color_attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    color_attachment.resolveMode = vk::ResolveModeFlagBits::eNone;
    color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
    color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
    color_attachment.clearValue = clear;
    vk::RenderingInfo rendering{};
    rendering.renderArea = vk::Rect2D({0, 0}, {width, height});
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    cmd.beginRendering(rendering);
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->Get());
    bind_and_draw(cmd);
    cmd.endRendering();

    const vk::ImageMemoryBarrier to_src(
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
        vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *color_image, range);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {}, to_src);

    const vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                     vk::Offset3D{0, 0, 0}, vk::Extent3D{width, height, 1});
    cmd.copyImageToBuffer(*color_image, vk::ImageLayout::eTransferSrcOptimal, *readback_buffer, region);

    const vk::BufferMemoryBarrier to_host(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead, vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored, *readback_buffer, 0, pixel_bytes);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                        {}, {}, to_host, {});
    cmd.end();

    const vk::SubmitInfo submit(0, nullptr, nullptr, 1, &*cmd);
    backend_->GetGraphicsQueue().submit(submit, {});
    backend_->GetGraphicsQueue().waitIdle();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(pixel_bytes));
    const void* mapped = read_memory.mapMemory(0, pixel_bytes);
    std::memcpy(pixels.data(), mapped, pixels.size());
    read_memory.unmapMemory();
    return pixels;
}

}  // namespace TestSupport
