module;

export module VulkanEngine.Text.GpuTextBlobBuffer;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuResources.FrameRing;
import VulkanEngine.Text.Blob;

export namespace VulkanEngine::Text {

// The GPU side of the hb-gpu blob store: one persistent storage buffer holding
// every encoded glyph blob, addressed by 8-byte unit offset from the shaders.
//
// The split mirrors GlyphAtlasGpu's: the pure part (BlobRangeAllocator, the
// encoder) decides *what* bytes go where and owns no device state, and this
// class owns the buffer they land in. Unlike the atlas there is no image, no
// bindless slot and no page: the blobs are one flat storage buffer, and a
// shader reads a glyph's blob by adding the instance's flat offsets to the
// buffer base.
//
// Growth never mutates or frees a buffer an in-flight frame could still read.
// When the packed blobs no longer fit, a new, larger GpuBuffer is created, the
// whole shadow image is re-uploaded into it, and the old buffer is enqueued into
// the engine's FrameRing retire path, which destroys it one full frame-in-flight
// cycle later -- the same ring-drain discipline GpuImageHeap and the bindless
// manager use, not a second release path. The descriptor the pass binds is the
// *current* buffer, so a frame that already recorded against the old one keeps
// reading a live buffer for its whole lifetime.
//
// The buffer is host-visible and coherent: blobs are appended a few glyphs at a
// time as text is first seen, and a mapped copy is the cheapest way to publish
// them. Nothing here rewrites bytes a frame might still be reading except
// through the growth path above, because an existing blob's offset never moves.
//
// Not thread-safe; the text system that owns one serializes submission.
class GpuTextBlobBuffer {
public:
    GpuTextBlobBuffer() = default;
    ~GpuTextBlobBuffer();

    GpuTextBlobBuffer(const GpuTextBlobBuffer&) = delete;
    GpuTextBlobBuffer& operator=(const GpuTextBlobBuffer&) = delete;
    GpuTextBlobBuffer(GpuTextBlobBuffer&&) = delete;
    GpuTextBlobBuffer& operator=(GpuTextBlobBuffer&&) = delete;

    // Creates the initial persistent buffer. `initial_bytes` is rounded up to
    // the blob unit; a growth reallocation will round up again from there.
    [[nodiscard]] bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                  std::uint64_t initial_bytes = 64U * 1024U);
    void Shutdown();

    // Copies `bytes` (a whole number of 8-byte units, a glyph blob) into the
    // buffer and returns the byte offset it was placed at. Returns nullopt for
    // an empty blob -- a space gets no range and is never drawn -- for a length
    // that is not a whole number of units, when the buffer is not initialized,
    // or when a growth reallocation fails. `recording_frame` is the frame the
    // caller is about to record: it is what a buffer retired by a growth here is
    // released against.
    [[nodiscard]] std::optional<std::uint64_t> Store(std::span<const std::byte> bytes,
                                                     std::uint32_t recording_frame);

    // Returns a range to the allocator. The bytes stay in the buffer; nothing
    // samples a freed range, and the next blob that fits reuse it in place.
    void Free(std::uint64_t byte_offset);

    // Drains the retire ring for the frame being started. Call once per frame
    // with the recording frame before submitting blobs for it.
    void Collect(std::uint32_t frame_index);

    // The buffer a shader binds. It changes only across a growth; the descriptor
    // must be rewritten whenever it does.
    [[nodiscard]] vk::Buffer Buffer() const;
    [[nodiscard]] std::uint64_t Capacity() const;
    [[nodiscard]] std::uint64_t ElementOffset(std::uint64_t byte_offset) const noexcept {
        return byte_offset / 8U;
    }
    [[nodiscard]] const BlobRangeAllocator& Allocator() const noexcept { return allocator_; }
    [[nodiscard]] bool IsValid() const noexcept { return buffer_.IsValid(); }

private:
    // Grows the buffer to hold at least `needed` bytes, retiring the old buffer
    // frame-gated. Returns false when the allocation fails; the caller then
    // releases the range it reserved.
    [[nodiscard]] bool EnsureCapacity(std::uint64_t needed, std::uint32_t recording_frame);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    // The CPU image of the buffer: the bytes needed to re-upload it whole after
    // a growth, and the reason growth does not need a device-to-device copy.
    std::vector<std::byte> shadow_{};
    VulkanEngine::GpuResources::GpuBuffer buffer_{};
    VulkanEngine::GpuResources::FrameRing<VulkanEngine::GpuResources::GpuBuffer> retire_ring_{};
    BlobRangeAllocator allocator_{};
};

} // namespace VulkanEngine::Text
