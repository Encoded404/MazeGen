module;

#include <logging/logging_macros.hpp>

module VulkanEngine.TextureUploader;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.GpuResources;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuTexture;
import VulkanEngine.BindlessManager;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;
import VulkanEngine.TextureUploadScheduler;
import VulkanEngine.FileLoaders.TextureLoaders;
import VulkanShared.Jobs;

namespace VulkanEngine::Textures {

namespace {

// Adapter: the pure resolver consults the backend capability snapshot through
// this interface; no device handle is exposed to the worker.
class CapabilityFormatQuery final : public FormatSupportQuery {
public:
    explicit CapabilityFormatQuery(const VulkanBackend::Vulkan::FormatCapabilities& caps) : caps_(caps) {}
    [[nodiscard]] bool sampled(vk::Format format) const override { return caps_.sampled(format); }
    [[nodiscard]] bool transfer_dst(vk::Format format) const override { return caps_.transfer_dst(format); }
    [[nodiscard]] bool linear_filter(vk::Format format) const override { return caps_.linear_filter(format); }
    [[nodiscard]] bool color_attachment(vk::Format format) const override { return caps_.color_attachment(format); }
private:
    const VulkanBackend::Vulkan::FormatCapabilities& caps_;
};

void RecordBarrier(vk::CommandBuffer cmd, vk::Image image, const vk::ImageSubresourceRange& range,
                   bool to_transfer_dst) {
    vk::ImageMemoryBarrier barrier{};
    barrier.image = image;
    barrier.subresourceRange = range;
    if (to_transfer_dst) {
        barrier.oldLayout = vk::ImageLayout::eUndefined;
        barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                            vk::DependencyFlags{}, nullptr, nullptr, barrier);
    } else {
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
                            vk::DependencyFlags{}, nullptr, nullptr, barrier);
    }
}

} // namespace

TextureUploader::~TextureUploader() {
    Shutdown();
}

bool TextureUploader::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                 GpuResources::GpuImageHeap& image_heap,
                                 GpuResources::StagingPool& staging_pool,
                                 BindlessManager::BindlessManager& bindless,
                                 const VulkanBackend::Vulkan::VulkanCapabilities& capabilities,
                                 GpuResources::SamplerCache* sampler_cache) {
    backend_ = &backend;
    image_heap_ = &image_heap;
    staging_pool_ = &staging_pool;
    bindless_ = &bindless;
    capabilities_ = &capabilities;
    sampler_cache_ = sampler_cache;
    LOGIFACE_LOG(debug, "TextureUploader initialized on the shared compute pool");
    return true;
}

void TextureUploader::Shutdown() {
    // Decode jobs already handed to the shared pool dereference the backend and
    // capability pointers below, so they must land before this drops them. The
    // wait participates in the pool rather than spinning on it.
    while (in_flight_.load(std::memory_order_acquire) != 0) {
        if (!VulkanShared::JobScheduler::Global().TryExecuteOne()) {
            std::this_thread::yield();
        }
    }

    // Resolve any promises still outstanding so callers do not block forever.
    std::lock_guard lock(ready_mutex_);
    for (auto& item : ready_) {
        if (item.promise) {
            item.promise->set_value(UploadResult{.handle = item.reservation.handle,
                                                 .status = UploadStatus::Failed,
                                                 .error = "uploader shut down"});
        }
    }
    ready_.clear();
    staged_.clear();
    backend_ = nullptr;
    image_heap_ = nullptr;
    staging_pool_ = nullptr;
    bindless_ = nullptr;
    capabilities_ = nullptr;
    sampler_cache_ = nullptr;
}

std::optional<TextureReservation> TextureUploader::Reserve(TextureSemantic semantic,
                                                           TextureNormalEncoding normal_encoding,
                                                           const SamplerDesc& sampler,
                                                           const ResourceId& id) {
    if (bindless_ == nullptr) {
        return std::nullopt;
    }
    auto handle = bindless_->ReserveSlot(id);
    if (!handle.has_value()) {
        return std::nullopt;
    }
    TextureReservation reservation{};
    reservation.handle = *handle;
    reservation.semantic = semantic;
    reservation.normal_encoding = normal_encoding;
    reservation.sampler = sampler;
    reservation.id = id;
    if (on_reserve_) {
        on_reserve_(*handle, id);
    }
    return reservation;
}

std::future<UploadResult> TextureUploader::Submit(TextureReservation reservation,
                                                  std::shared_ptr<const TextureData> data,
                                                  std::uint32_t priority) {
    auto promise = std::make_shared<std::promise<UploadResult>>();
    auto future = promise->get_future();
    if (backend_ == nullptr || data == nullptr) {
        promise->set_value(UploadResult{.handle = reservation.handle,
                                        .status = UploadStatus::Failed,
                                        .error = "uploader not initialized"});
        return future;
    }

    WorkerItem item{};
    item.reservation = reservation;
    item.source = std::move(data);
    item.priority = priority;
    item.promise = std::move(promise);
    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    VulkanShared::JobScheduler::Global().Submit(
        VulkanShared::Tier::Compute, "texture-decode",
        [this, item = std::move(item)]() mutable { ProcessItem(std::move(item)); });
    return future;
}

void TextureUploader::ProcessItem(WorkerItem item) {
    CapabilityFormatQuery query(capabilities_->GetFormatCapabilities());

    ReadyItem ready{};
    ready.reservation = item.reservation;
    ready.promise = item.promise;
    ready.data = *item.source;

    const auto resolved = ResolveUploadFormat(
        item.reservation.semantic, item.reservation.normal_encoding,
        ready.data.source_format, ready.data.needs_transcode,
        ready.data.source_channels, ready.data.source_alpha != 0,
        ready.data.is_hdr_source, query);
    if (resolved.format == vk::Format::eUndefined) {
        ready.ok = false;
        ready.error = "no legal upload format";
        ready.format = vk::Format::eUndefined;
    } else {
        ready.format = resolved.format;
        ready.ok = true;
        if (ready.data.needs_transcode) {
            if (!VulkanEngine::FileLoaders::Textures::TranscodeBasisToTarget(
                    ready.data, resolved.format, &ready.error)) {
                ready.ok = false;
            }
        } else if (resolved.decoded) {
            // Native ASTC payload without device ASTC support: CPU-decode
            // on the worker thread (libktx 4.4). The decoder picks the
            // uncompressed variant from the DFD, so the upload format is
            // whatever it actually produced, not what the resolver guessed.
            if (!VulkanEngine::FileLoaders::Textures::DecodeAstcToRgba8(
                    ready.data, &ready.error)) {
                ready.ok = false;
            } else {
                ready.format = ready.data.source_format;
            }
        } else {
            // Fill in a CPU mip chain for uncompressed sources without one.
            (void)VulkanEngine::FileLoaders::Textures::GenerateCpuMipChain(ready.data);
        }
    }

    {
        std::lock_guard lock(ready_mutex_);
        ready_.push_back(std::move(ready));
    }
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
}

void TextureUploader::BeginFrame(std::uint32_t frame_index,
                                 const std::function<bool(std::uint32_t)>& gate) {
    if (backend_ == nullptr) {
        return;
    }
    (void)frame_index;

    // 1. Collect worker-finished items into staged images.
    std::vector<ReadyItem> finished;
    {
        std::lock_guard lock(ready_mutex_);
        finished.swap(ready_);
    }
    for (auto& item : finished) {
        if (!item.ok) {
            CompleteItem(item, UploadStatus::Failed, std::move(item.error));
            continue;
        }
        GpuResources::GpuTexture texture = GpuResources::GpuTexture::CreatePending(
            *backend_, *image_heap_, item.data, item.format, item.reservation.sampler,
            sampler_cache_);
        if (!texture.IsValid()) {
            CompleteItem(item, UploadStatus::Failed, "image allocation failed");
            continue;
        }
        const std::uint64_t total = std::ranges::fold_left(
            item.data.subresources, std::uint64_t{0},
            [](std::uint64_t acc, const TextureSubresource& sub) { return acc + sub.size; });
        auto staging = staging_pool_->Allocate(total, 4);
        if (!staging.has_value()) {
            // Staging is full: keep the item for a later frame.
            std::lock_guard lock(ready_mutex_);
            ready_.push_back(std::move(item));
            continue;
        }
        auto* dst = static_cast<std::byte*>(staging->mapped_ptr);
        for (const auto& sub : item.data.subresources) {
            std::memcpy(dst + sub.offset, item.data.blob.data() + sub.offset,
                        static_cast<std::size_t>(sub.size));
        }
        StagedItem staged{};
        staged.reservation = item.reservation;
        staged.texture = std::move(texture);
        staged.staging = *staging;
        staged.data = std::move(item.data);
        staged.promise = item.promise;
        staged_.push_back(std::move(staged));
    }
    // The bindless FrameRing is drained by the caller (GameEngine::FrameRender)
    // with its own dropped-publish reporting; the uploader does not drain it.
    (void)gate;
}

void TextureUploader::CompleteItem(const ReadyItem& item, UploadStatus status, std::string error) {
    if (item.promise) {
        item.promise->set_value(UploadResult{
            .handle = item.reservation.handle, .status = status, .error = std::move(error)});
    }
}

std::uint64_t TextureUploader::RecordUploads(vk::CommandBuffer cmd, std::uint32_t frame_index,
                                             std::uint64_t budget) {
    if (backend_ == nullptr || staged_.empty()) {
        return 0;
    }

    // Plan the frame's uploads from the staged items (pure policy).
    std::vector<PendingUpload> pending;
    pending.reserve(staged_.size());
    for (std::size_t i = 0; i < staged_.size(); ++i) {
        const auto& data = staged_[i].data;
        const std::uint64_t bytes = std::ranges::fold_left(
            data.subresources, std::uint64_t{0},
            [](std::uint64_t acc, const TextureSubresource& sub) { return acc + sub.size; });
        pending.push_back(PendingUpload{.id = i, .bytes = bytes,
                                        .priority = staged_[i].priority, .sequence = i});
    }
    const auto plan = scheduler_.Plan(pending, budget, frame_index);

    std::vector<std::size_t> recorded_indices;
    recorded_indices.reserve(plan.order.size());
    std::uint64_t recorded_bytes = 0;
    for (const auto& planned : plan.order) {
        const auto index = static_cast<std::size_t>(planned.id);
        auto& item = staged_[index];
        const auto& data = item.data;
        const vk::Image image = item.texture.GetImage();
        const std::uint32_t layers = item.texture.GetArrayLayers();

        const vk::ImageSubresourceRange full_range{
            vk::ImageAspectFlagBits::eColor, 0, data.mip_levels, 0, layers};
        RecordBarrier(cmd, image, full_range, /*to_transfer_dst=*/true);

        std::vector<vk::BufferImageCopy> regions;
        regions.reserve(data.subresources.size());
        for (const auto& sub : data.subresources) {
            vk::BufferImageCopy region{};
            region.bufferOffset = 0;  // RecordBufferToImage adds alloc.offset
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
        staging_pool_->RecordBufferToImage(cmd, item.staging, image, regions);
        RecordBarrier(cmd, image, full_range, /*to_transfer_dst=*/false);

        // Bytes the image occupies, captured before ownership moves into the
        // bindless binding; residency accounts whole-texture eviction by it.
        const std::uint64_t image_bytes = item.texture.GetByteSize();

        // Hand the image to the bindless manager (published one FIF later) and
        // retire the staging range once the recording frame completes.
        bindless_->CommitSlot(item.reservation.handle, std::move(item.texture), frame_index);
        staging_pool_->Retire(item.staging, frame_index);

        if (on_resident_) {
            on_resident_(item.reservation.handle, image_bytes, frame_index);
        }

        if (item.promise) {
            item.promise->set_value(UploadResult{
                .handle = item.reservation.handle,
                .status = UploadStatus::Resident, .error = {}});
        }
        recorded_bytes += planned.bytes;
        recorded_indices.push_back(index);
    }

    // Erase recorded items (descending so indices stay valid).
    std::ranges::sort(recorded_indices, std::greater<>{});
    for (const auto index : recorded_indices) {
        staged_.erase(staged_.begin() + static_cast<std::ptrdiff_t>(index));
    }
    return recorded_bytes;
}

std::uint32_t TextureUploader::PendingCount() const {
    return in_flight_.load(std::memory_order_acquire);
}

} // namespace VulkanEngine::Textures
