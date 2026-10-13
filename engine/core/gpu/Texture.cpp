module;

#include <logging/logging_macros.hpp>

module VulkanEngine.GpuTexture;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanBackend.Vulkan.VulkanDebugUtils;

import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.GpuResources.FrameRing;
import VulkanShared.DeviceScope;

namespace VulkanEngine::GpuResources {

namespace {

void CreateBufferResource(const VulkanBackend::Vulkan::IVulkanBootstrap & backend,
                          std::uint64_t size,
                          vk::BufferUsageFlags usage,
                          vk::MemoryPropertyFlags properties,
                          std::unique_ptr<vk::raii::Buffer>& out_buffer,
                          std::unique_ptr<vk::raii::DeviceMemory>& out_memory) {
    vk::BufferCreateInfo const info({}, size, usage);
    out_buffer = std::make_unique<vk::raii::Buffer>(backend.GetDevice(), info);

    const vk::DeviceBufferMemoryRequirements mem_req_info{
        &info,
        nullptr
    };
    const vk::MemoryRequirements2 requirements = backend.GetDevice().getBufferMemoryRequirements(mem_req_info);
    const auto& reqs = requirements.memoryRequirements;

    vk::MemoryAllocateInfo const alloc(reqs.size,
        VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(backend.GetPhysicalDevice(), reqs.memoryTypeBits, properties));
    out_memory = std::make_unique<vk::raii::DeviceMemory>(backend.GetDevice(), alloc);
    out_buffer->bindMemory(*out_memory, 0);
}

// Uncached sampler construction: used by the bootstrap/standalone path when no
// SamplerCache is supplied (device-free tests and the fallback upload). The
// clamp logic is shared with the cache so both produce identical samplers.
vk::raii::Sampler CreateSampler(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                const VulkanEngine::Textures::SamplerDesc& desc,
                                std::uint32_t mip_levels) {
    const auto& caps = backend.GetCapabilities();
    const bool anisotropy_supported = caps.IsFeatureEnabled(VulkanBackend::Vulkan::Feature::SamplerAnisotropy);
    const auto device_max = static_cast<std::uint32_t>(caps.GetMaxSamplerAnisotropy());
    const auto clamped =
        VulkanEngine::Textures::ClampSamplerDesc(desc, anisotropy_supported, device_max, mip_levels);
    return vk::raii::Sampler(backend.GetDevice(),
                             VulkanEngine::Textures::MakeSamplerCreateInfo(clamped));
}

// Upload barriers: uploads are consumed by fragment and possibly compute
// passes, so the final transition uses
//   dstStage = eFragmentShader | eComputeShader, dstAccess = eShaderRead.
void RecordUploadBarriers(vk::raii::CommandBuffer& cmd,
                          vk::Image image,
                          const vk::ImageSubresourceRange& range,
                          bool to_transfer_dst) {
    vk::ImageMemoryBarrier barrier{};
    barrier.image = image;
    barrier.subresourceRange = range;

    if (to_transfer_dst) {
        barrier.oldLayout = vk::ImageLayout::eUndefined;
        barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            vk::DependencyFlags{},
                            nullptr, nullptr, barrier);
    } else {
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
                            vk::DependencyFlags{},
                            nullptr, nullptr, barrier);
    }
}

} // namespace

GpuTexture::GpuTexture(GpuTexture&& other) noexcept
    : heap_(other.heap_),
      image_(other.image_),
      sampler_(other.sampler_),
      owned_sampler_(std::move(other.owned_sampler_)),
      format_(other.format_),
      width_(other.width_),
      height_(other.height_),
      mip_levels_(other.mip_levels_),
      array_layers_(other.array_layers_) {
    other.heap_ = nullptr;
    other.image_ = {};
    other.sampler_ = vk::Sampler{nullptr};
}

GpuTexture& GpuTexture::operator=(GpuTexture&& other) noexcept {
    if (this != &other) {
        if (IsValid()) {
            heap_->Free(image_);
        }
        heap_ = other.heap_;
        image_ = other.image_;
        sampler_ = other.sampler_;
        owned_sampler_ = std::move(other.owned_sampler_);
        format_ = other.format_;
        width_ = other.width_;
        height_ = other.height_;
        mip_levels_ = other.mip_levels_;
        array_layers_ = other.array_layers_;
        other.heap_ = nullptr;
        other.image_ = {};
        other.sampler_ = vk::Sampler{nullptr};
    }
    return *this;
}

GpuTexture::~GpuTexture() {
    // Shutdown-time destruction happens with the device idle (EngineBootstrap
    // waits before teardown), matching the previous raiii teardown safety;
    // mid-run releases must use Retire for frame-gated destruction.
    if (IsValid()) {
        heap_->Free(image_);
        heap_ = nullptr;
        image_ = {};
    }
}

void GpuTexture::Retire(std::uint32_t recording_frame) {
    if (!IsValid()) {
        return;
    }
    // A cache-owned sampler is shared and must live on; only an owned sampler
    // is released here.
    owned_sampler_.reset();
    sampler_ = vk::Sampler{nullptr};
    heap_->Retire(image_, recording_frame);
    image_ = {};
    heap_ = nullptr;
}

void GpuTexture::AssignSampler(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                               GpuTexture& texture,
                               const VulkanEngine::Textures::SamplerDesc& sampler_desc,
                               std::uint32_t mip_levels,
                               SamplerCache* sampler_cache) {
    if (sampler_cache != nullptr && sampler_cache->IsValid()) {
        texture.sampler_ = sampler_cache->Get(sampler_desc, mip_levels);
        return;
    }
    texture.owned_sampler_ =
        std::make_unique<vk::raii::Sampler>(CreateSampler(backend, sampler_desc, mip_levels));
    texture.sampler_ = static_cast<vk::Sampler>(**texture.owned_sampler_);
    VulkanBackend::Vulkan::SetVulkanObjectName(backend.GetDevice(), *texture.owned_sampler_,
                                               "gpu-texture-sampler");
}

vk::Image GpuTexture::GetImage() const {
    return IsValid() ? heap_->GetImage(image_) : vk::Image{nullptr};
}

vk::ImageView GpuTexture::GetImageView() const {
    return IsValid() ? heap_->GetImageView(image_) : vk::ImageView{nullptr};
}

vk::Sampler GpuTexture::GetSampler() const {
    return sampler_;
}

VulkanEngine::Textures::GpuTextureInfo GpuTexture::ToTextureInfo() const {
    VulkanEngine::Textures::GpuTextureInfo info{};
    info.width = width_;
    info.height = height_;
    info.mip_levels = mip_levels_;
    info.array_layers = array_layers_;
    info.format = static_cast<std::uint32_t>(format_);
    return info;
}

GpuTexture GpuTexture::CreatePending(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                     GpuImageHeap& heap,
                                     const VulkanEngine::Textures::TextureData& data,
                                     vk::Format resolved,
                                     const VulkanEngine::Textures::SamplerDesc& sampler_desc,
                                     SamplerCache* sampler_cache) {
    GpuTexture texture{};
    if (!heap.IsValid() || resolved == vk::Format::eUndefined || data.subresources.empty()) {
        return texture;
    }

    texture.format_ = resolved;
    texture.width_ = data.width;
    texture.height_ = data.height;
    texture.mip_levels_ = data.mip_levels;
    texture.array_layers_ = data.array_layers * data.face_count;

    // Create at the actual pixel dimensions (never padded); cube faces fold
    // into array layers: gpu_layer = layer * face_count + face.
    vk::ImageCreateInfo image_info{};
    image_info.imageType = data.depth > 1U ? vk::ImageType::e3D : vk::ImageType::e2D;
    image_info.format = resolved;
    image_info.extent = vk::Extent3D{data.width, data.height, data.depth};
    image_info.mipLevels = data.mip_levels;
    image_info.arrayLayers = texture.array_layers_;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    const bool cube = data.depth == 1U && data.face_count == 6U && data.cube_complete &&
                      texture.array_layers_ % 6U == 0U;
    if (cube) {
        image_info.flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }

    // KTXswizzle is a view property (never a sampler property).
    HeapViewDesc view_desc{};
    view_desc.components = data.swizzle;
    view_desc.aspect = vk::ImageAspectFlagBits::eColor;

    texture.heap_ = &heap;
    texture.image_ = heap.Allocate(image_info, view_desc);
    if (!texture.image_.IsValid()) {
        texture.heap_ = nullptr;
        return texture;
    }

    AssignSampler(backend, texture, sampler_desc, data.mip_levels, sampler_cache);
    return texture;
}

GpuTexture GpuTexture::CreateFromTextureData(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                             GpuImageHeap& heap,
                                             const VulkanEngine::Textures::TextureData& data,
                                             vk::Format resolved,
                                             const VulkanEngine::Textures::SamplerDesc& sampler_desc,
                                             SamplerCache* sampler_cache) {
    GpuTexture texture = CreatePending(backend, heap, data, resolved, sampler_desc, sampler_cache);
    if (!texture.IsValid()) {
        return texture;
    }

    const std::size_t total_bytes = std::ranges::fold_left(
        data.subresources, std::size_t{0},
        [](std::size_t acc, const VulkanEngine::Textures::TextureSubresource& sub) {
            return acc + static_cast<std::size_t>(sub.size);
        });

    std::unique_ptr<vk::raii::Buffer> staging_buffer;
    std::unique_ptr<vk::raii::DeviceMemory> staging_memory;
    CreateBufferResource(backend, total_bytes,
                         vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostVisible) | vk::MemoryPropertyFlagBits::eHostCoherent,
                         staging_buffer, staging_memory);
    VulkanBackend::Vulkan::SetVulkanObjectName(backend.GetDevice(), *staging_buffer, "texture-staging-buffer");

    {
        void* dst = staging_memory->mapMemory(0, total_bytes);
        auto* dst_bytes = static_cast<std::byte*>(dst);
        for (const auto& sub : data.subresources) {
            std::memcpy(dst_bytes + sub.offset, data.blob.data() + sub.offset, static_cast<std::size_t>(sub.size));
        }
        staging_memory->unmapMemory();
    }

    auto& cmd = backend.GetCommandBuffer(0);
    cmd.reset({});
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::ImageSubresourceRange full_range{vk::ImageAspectFlagBits::eColor, 0,
                                               data.mip_levels, 0, texture.array_layers_};
    RecordUploadBarriers(cmd, texture.GetImage(), full_range, /*to_transfer_dst=*/true);

    // One copy per subresource: extent = actual mip dims (edge copy at image
    // borders), tightly packed rows, offsets from the loader's subresource
    // table (block-aligned by construction).
    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(data.subresources.size());
    for (const auto& sub : data.subresources) {
        vk::BufferImageCopy region{};
        region.bufferOffset = static_cast<vk::DeviceSize>(sub.offset);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = sub.mip;
        region.imageSubresource.baseArrayLayer = sub.array_layer;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{sub.width, sub.height, std::max(sub.depth, 1U)};
        regions.push_back(region);
    }
    cmd.copyBufferToImage(*staging_buffer, texture.GetImage(),
                          vk::ImageLayout::eTransferDstOptimal, regions);

    RecordUploadBarriers(cmd, texture.GetImage(), full_range, /*to_transfer_dst=*/false);

    cmd.end();
    vk::SubmitInfo const submit(0, nullptr, nullptr, 1, &*cmd);
    backend.GetGraphicsQueue().submit(submit, nullptr);
    // Synchronous bootstrap upload path: submit and wait. A future async
    // uploader replaces this with a frame-recorded copy and no frame-0
    // command buffer.
    {
        VulkanShared::DeviceScopeGuard guard;
        backend.GetGraphicsQueue().waitIdle();
    }

    AssignSampler(backend, texture, sampler_desc, data.mip_levels, sampler_cache);

    return texture;
}

GpuTexture GpuTexture::CreateFromPixels(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                        GpuImageHeap& heap,
                                        const uint8_t* pixels,
                                        std::uint32_t width,
                                        std::uint32_t height,
                                        vk::Format format,
                                        SamplerCache* sampler_cache) {
    VulkanEngine::Textures::TextureData data{};
    data.width = width;
    data.height = height;
    data.source_format = format;
    data.source_channels = 4;
    data.source_alpha = 1;
    const std::size_t byte_count = static_cast<std::size_t>(width) * height * 4U;
    data.subresources.push_back(VulkanEngine::Textures::TextureSubresource{
        .mip = 0, .array_layer = 0, .offset = 0, .size = byte_count,
        .width = width, .height = height, .depth = 1,
    });
    data.blob.resize(byte_count);
    if (pixels != nullptr) {
        std::memcpy(data.blob.data(), pixels, byte_count);
    }
    return CreateFromTextureData(backend, heap, data, format, {}, sampler_cache);
}

GpuTexture GpuTexture::CreateStream(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                    GpuImageHeap& heap,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    vk::Format format,
                                    bool linear_filter,
                                    SamplerCache* sampler_cache) {
    GpuTexture texture{};
    if (!heap.IsValid()) {
        return texture;
    }
    texture.format_ = format;
    texture.width_ = width;
    texture.height_ = height;

    vk::ImageCreateInfo image_info{};
    image_info.imageType = vk::ImageType::e2D;
    image_info.format = format;
    image_info.extent = vk::Extent3D{width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

    texture.heap_ = &heap;
    texture.image_ = heap.Allocate(image_info, HeapViewDesc{vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}});
    if (!texture.image_.IsValid()) {
        texture.heap_ = nullptr;
        return texture;
    }

    VulkanEngine::Textures::SamplerDesc sampler_desc{};
    const auto filter = linear_filter ? vk::Filter::eLinear : vk::Filter::eNearest;
    sampler_desc.min_filter = filter;
    sampler_desc.mag_filter = filter;
    sampler_desc.mip_mode = vk::SamplerMipmapMode::eNearest;
    sampler_desc.address_u = vk::SamplerAddressMode::eClampToEdge;
    sampler_desc.address_v = vk::SamplerAddressMode::eClampToEdge;
    sampler_desc.address_w = vk::SamplerAddressMode::eClampToEdge;
    AssignSampler(backend, texture, sampler_desc, 1, sampler_cache);
    return texture;
}

GpuTexture GpuTexture::CreateColorTarget(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                         GpuImageHeap& heap,
                                         std::uint32_t width,
                                         std::uint32_t height,
                                         vk::Format format,
                                         SamplerCache* sampler_cache) {
    GpuTexture texture{};
    if (!heap.IsValid()) {
        return texture;
    }
    texture.format_ = format;
    texture.width_ = width;
    texture.height_ = height;

    vk::ImageCreateInfo image_info{};
    image_info.imageType = vk::ImageType::e2D;
    image_info.format = format;
    image_info.extent = vk::Extent3D{width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

    texture.heap_ = &heap;
    texture.image_ = heap.Allocate(image_info, HeapViewDesc{vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}});
    if (!texture.image_.IsValid()) {
        texture.heap_ = nullptr;
        return texture;
    }

    VulkanEngine::Textures::SamplerDesc sampler_desc{};
    sampler_desc.address_u = vk::SamplerAddressMode::eClampToEdge;
    sampler_desc.address_v = vk::SamplerAddressMode::eClampToEdge;
    sampler_desc.address_w = vk::SamplerAddressMode::eClampToEdge;
    AssignSampler(backend, texture, sampler_desc, 1, sampler_cache);
    return texture;
}

} // namespace VulkanEngine::GpuResources
