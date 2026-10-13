module;

module VulkanEngine.Text.GpuTextBlobBuffer;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.FrameRing;
import VulkanEngine.Text.Blob;

namespace VulkanEngine::Text {

namespace {

// The buffer's own granularity. The allocator hands out 8-byte units, so the
// backing allocation only has to be a whole number of those.
constexpr std::uint64_t kUnit = BlobRangeAllocator::kAlignment;

[[nodiscard]] std::uint64_t RoundUp(std::uint64_t value, std::uint64_t multiple) noexcept {
    return ((value + multiple - 1U) / multiple) * multiple;
}

} // namespace

GpuTextBlobBuffer::~GpuTextBlobBuffer() = default;

bool GpuTextBlobBuffer::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                   std::uint64_t initial_bytes) {
    backend_ = &backend;
    retire_ring_.Initialize(backend.GetFramesInFlight());

    const std::uint64_t capacity = RoundUp(std::max<std::uint64_t>(initial_bytes, kUnit), kUnit);
    shadow_.assign(static_cast<std::size_t>(capacity), std::byte{0});
    buffer_ = VulkanEngine::GpuResources::GpuBuffer::Create(
        backend, capacity, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!buffer_.IsValid()) {
        shadow_.clear();
        backend_ = nullptr;
        return false;
    }
    return true;
}

void GpuTextBlobBuffer::Shutdown() {
    // Teardown: the device is idle, so the retire ring applies without gating.
    retire_ring_.Flush([](VulkanEngine::GpuResources::GpuBuffer& buffer, std::uint32_t) {
        buffer = VulkanEngine::GpuResources::GpuBuffer{};
    });
    buffer_ = VulkanEngine::GpuResources::GpuBuffer{};
    shadow_.clear();
    allocator_.Reset();
    backend_ = nullptr;
}

bool GpuTextBlobBuffer::EnsureCapacity(std::uint64_t needed, std::uint32_t recording_frame) {
    if (needed <= buffer_.GetSize()) {
        return true;
    }
    const std::uint64_t grown = RoundUp(std::max(needed, buffer_.GetSize() * 2U), kUnit);
    auto replacement = VulkanEngine::GpuResources::GpuBuffer::Create(
        *backend_, grown, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!replacement.IsValid()) {
        return false;
    }

    // Re-upload the whole CPU image, so every blob keeps the offset it was
    // allocated at and a caller that captured one before the growth still finds
    // its bytes at the same place.
    shadow_.resize(static_cast<std::size_t>(grown), std::byte{0});
    replacement.Upload(shadow_.data(), shadow_.size());

    // The old buffer may still be read by every frame that already recorded a
    // draw against it. Handing it to the frame ring destroys it one full
    // frames-in-flight cycle later, the same discipline the image heap and the
    // bindless manager use for the resources they retire mid-run.
    retire_ring_.Enqueue(std::move(buffer_), recording_frame, /*gated=*/false);
    buffer_ = std::move(replacement);
    return true;
}

std::optional<std::uint64_t> GpuTextBlobBuffer::Store(std::span<const std::byte> bytes,
                                                      std::uint32_t recording_frame) {
    if (backend_ == nullptr || !buffer_.IsValid()) {
        return std::nullopt;
    }
    // A glyph with no ink encodes to length 0 and is never drawn, so it must not
    // take a range. A length that is not a whole number of 8-byte units is not a
    // blob at all and is rejected rather than silently packed.
    if (bytes.empty() || bytes.size() % kUnit != 0) {
        return std::nullopt;
    }

    const auto offset = allocator_.Allocate(bytes.size());
    if (!offset.has_value()) {
        return std::nullopt;
    }
    if (!EnsureCapacity(allocator_.Capacity(), recording_frame)) {
        allocator_.Free(*offset);
        return std::nullopt;
    }

    std::memcpy(shadow_.data() + *offset, bytes.data(), bytes.size());
    buffer_.UploadAt(bytes.data(), bytes.size(), *offset);
    return offset;
}

void GpuTextBlobBuffer::Free(std::uint64_t byte_offset) {
    allocator_.Free(byte_offset);
}

void GpuTextBlobBuffer::Collect(std::uint32_t frame_index) {
    if (!retire_ring_.PendingCount()) {
        return;
    }
    // A retire is enqueued with the frame that is recording, and the ring applies
    // it one full frames-in-flight cycle later. The drain runs from the pass's
    // Execute, which is *after* the application queued that frame's glyphs, so an
    // op enqueued for this very frame shares the ring slot this drain just swapped
    // out. Destroying it here would free a buffer the command buffers recorded for
    // the intervening frames may still read -- validation reports exactly that as
    // vkDestroyBuffer while the buffer is in use. Hold any op whose full cycle has
    // not actually elapsed for one more drain of its ring slot.
    const std::uint32_t frames_in_flight = std::max(retire_ring_.GetFramesInFlight(), 1U);
    retire_ring_.BeginFrame(
        frame_index,
        [this, frame_index, frames_in_flight](
            VulkanEngine::GpuResources::GpuBuffer& buffer, std::uint32_t frame) {
            const std::uint32_t age = frame_index >= frame ? frame_index - frame : 0U;
            if (age >= frames_in_flight) {
                buffer = VulkanEngine::GpuResources::GpuBuffer{};
                return;
            }
            // Re-enqueueing from inside the drain lands in the now-empty ring slot
            // it came from, so it applies at that slot's next drain, one cycle on.
            retire_ring_.Enqueue(std::move(buffer), frame, /*gated=*/false);
        },
        /*gate=*/[](std::uint32_t) { return true; },
        /*on_drop=*/[](VulkanEngine::GpuResources::GpuBuffer&, std::uint32_t) {});
}

vk::Buffer GpuTextBlobBuffer::Buffer() const {
    return buffer_.IsValid() ? static_cast<vk::Buffer>(*buffer_.GetBuffer()) : vk::Buffer{};
}

std::uint64_t GpuTextBlobBuffer::Capacity() const {
    return buffer_.GetSize();
}

} // namespace VulkanEngine::Text
