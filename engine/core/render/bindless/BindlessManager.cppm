module;

export module VulkanEngine.BindlessManager;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.GpuResources;
export import VulkanEngine.BindlessManager.TextureSlot;
import VulkanEngine.GpuResources.FrameRing;
import VulkanEngine.TextureTypes;
import VulkanEngine.ResourceSystem;

export namespace VulkanEngine::BindlessManager {

// Slot 0 is permanently reserved as the fallback descriptor. Allocation
// starts at slot 1; a failed texture load binds the caller's slot index 0 so
// every material references the checkerboard through the same slot.
inline constexpr std::uint32_t kFallbackSlot = 0;

// Capacity inputs. `other_update_after_bind_descriptors` is the descriptor
// count of every other update-after-bind pool (the engine's vertex-buffer,
// index-buffer and indirection sets, each x frames_in_flight). It is passed in
// rather than derived here to keep this module from importing the renderer.
struct BindlessCapacityConfig {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t app_capacity = 65536;
    std::uint32_t other_update_after_bind_descriptors = 0;
    // Descriptors this pool spends on bindings other than the combined-image
    // array (the GpuTextureInfo storage buffer). Subtracted from the global
    // update-after-bind budget alongside the other pools so the pool as a whole
    // stays inside it. `Initialize` raises this to at least 1, since the
    // metadata binding is always allocated; the default 0 keeps the standalone
    // ComputeBindlessCapacity arithmetic unchanged for direct callers.
    std::uint32_t own_non_image_descriptors = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct BindlessCapacityLimits {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t max_combined_image_samplers = 0;
    std::uint32_t max_update_after_bind_in_all_pools = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Pure: the single bindless capacity, clamped by the device limits minus the
// other update-after-bind pools' descriptor counts. Device-free so the clamp
// arithmetic is unit-testable.
[[nodiscard]] inline std::uint32_t ComputeBindlessCapacity(
    const BindlessCapacityConfig& config, const BindlessCapacityLimits& limits) {
    std::uint32_t capacity = config.app_capacity;
    capacity = std::min(capacity, limits.max_combined_image_samplers);
    const std::uint32_t spent = config.other_update_after_bind_descriptors +
                                config.own_non_image_descriptors;
    const std::uint32_t pool_budget =
        limits.max_update_after_bind_in_all_pools > spent
            ? limits.max_update_after_bind_in_all_pools - spent
            : 0U;
    capacity = std::min(capacity, pool_budget);
    // Slot 0 is always available for the fallback.
    return std::max(capacity, kFallbackSlot + 1U);
}

class BindlessManager {
public:
    BindlessManager() = default;
    ~BindlessManager();

    BindlessManager(const BindlessManager&) = delete;
    BindlessManager& operator=(const BindlessManager&) = delete;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    const BindlessCapacityConfig& capacity_config = {});
    void Shutdown();

    // ── Fallback (slot 0) ─────────────────────────────────────────────

    // Binds the permanent fallback descriptor at slot 0. Must be called before
    // the first ReserveSlot/AllocateTextureSlot (the fallback is always
    // sampleable). One-shot: a later call is ignored.
    void SetFallback(VulkanEngine::GpuResources::GpuTexture texture, const VulkanEngine::ResourceId& id);

    // ── Frame-gated lifecycle (async uploader path) ───────────────────

    // Reserves a slot pre-bound to the fallback (always sampleable) and returns
    // its generation-checked handle. Returns nullopt when bindless capacity is
    // exhausted. The caller commits the real binding later, after recording the
    // upload copies.
    [[nodiscard]] std::optional<TextureHandle> ReserveSlot(const VulkanEngine::ResourceId& id = {});

    // Records the real binding for a reserved handle. The descriptor write is
    // deferred: it applies at the begin-frame drain of `recording_frame + FIF`,
    // and only when that frame is in the submission record (gate). A commit for
    // a stale or released reservation is dropped.
    void CommitSlot(TextureHandle handle, VulkanEngine::GpuResources::GpuTexture binding,
                    std::uint32_t recording_frame);

    // Drops one reference. At zero users the descriptor is reset to the
    // fallback and the slot returns to the free list when the decommit applies
    // at the drain, one FIF cycle later. A stale handle is ignored.
    void ReleaseSlot(TextureHandle handle, std::uint32_t recording_frame);

    // Immediate (device-idle) reserve+commit: writes the descriptor now and
    // returns a handle. For boot-time and teardown paths where no frame is in
    // flight; not for in-frame use (the frame-gated path exists for that).
    [[nodiscard]] std::optional<TextureHandle> AllocateTextureSlot(
        VulkanEngine::GpuResources::GpuTexture texture, const VulkanEngine::ResourceId& id);

    // Immediate (device-idle) release. Returns the slot to the free list now.
    void ReleaseSlotImmediate(std::uint32_t slot);

    // Drains the deferred-op ring for `frame_index`. `gate(recording_frame)`
    // is the corrected IsFrameComplete: a Publish whose recording frame was
    // never submitted is dropped and reported through `on_drop` (the uploader
    // returns the item to a pending state); destruct-only ops ignore the gate.
    void BeginFrame(std::uint32_t frame_index,
                    const std::function<bool(std::uint32_t)>& gate,
                    const std::function<void(TextureHandle, std::uint32_t)>& on_drop = {});

    [[nodiscard]] const VulkanEngine::ResourceId* GetTextureId(std::uint32_t slot) const;
    [[nodiscard]] const VulkanEngine::GpuResources::GpuTexture* GetTexture(std::uint32_t slot) const;
    [[nodiscard]] const VulkanEngine::GpuResources::GpuTexture* GetTexture(TextureHandle handle) const;

    // Generation-checked handle for a live slot. nullopt for the reserved
    // fallback (slot 0), an out-of-range slot, or a free slot. Used by residency
    // to release a slot without holding its original handle.
    [[nodiscard]] std::optional<TextureHandle> GetHandle(std::uint32_t slot) const;
    // True when the slot currently holds a published real binding (not the
    // fallback, not a reserved-but-uncommitted reservation).
    [[nodiscard]] bool IsCommitted(std::uint32_t slot) const;

    [[nodiscard]] std::uint32_t Capacity() const { return capacity_; }
    [[nodiscard]] std::uint32_t FreeSlotCount() const { return static_cast<std::uint32_t>(free_slots_.size()); }

    [[nodiscard]] vk::DescriptorSetLayout* GetLayout();
    [[nodiscard]] vk::DescriptorSet GetDescriptorSet() const;
    // The per-slot GpuTextureInfo storage buffer (set 0, binding 1). Shaders
    // read one element per bindless slot; the host keeps it in sync with the
    // descriptor as slots are written, reserved and released.
    [[nodiscard]] vk::Buffer GetTextureInfoBuffer() const;
    [[nodiscard]] const VulkanEngine::Textures::GpuTextureInfo* GetTextureInfo(std::uint32_t slot) const;
    [[nodiscard]] bool IsValid() const { return *descriptor_set_ != nullptr; }

private:
    enum class OpKind : std::uint8_t { Publish, Decommit, Destroy };

    struct SlotOp {
        OpKind kind = OpKind::Publish;
        std::uint32_t slot = 0;
        std::uint32_t generation = 0;
        std::uint32_t recording_frame = 0;
        VulkanEngine::GpuResources::GpuTexture binding{};
    };

    struct Slot {
        std::uint32_t generation = 0;   // reservation epoch; 0 = never/live-free
        std::uint32_t users = 0;        // refcount; > 0 while referenced
        bool released = false;
        bool committed = false;         // true once a real binding is published
        VulkanEngine::ResourceId id{};
        VulkanEngine::GpuResources::GpuTexture binding{};
    };

    void UpdateSlot(std::uint32_t slot, const VulkanEngine::GpuResources::GpuTexture& texture);
    void WriteFallback(std::uint32_t slot);
    void ApplyOp(SlotOp& op);
    void DestroyBinding(VulkanEngine::GpuResources::GpuTexture& binding);

    // Writes one GpuTextureInfo element and publishes the mapped range to the
    // device. `texture` may be invalid (a released/reserved slot): the element
    // then describes the fallback that the descriptor points at.
    void WriteTextureInfo(std::uint32_t slot, const VulkanEngine::GpuResources::GpuTexture& texture);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    std::unique_ptr<vk::raii::DescriptorSetLayout> layout_{};
    std::unique_ptr<vk::raii::DescriptorPool> pool_{};
    vk::raii::DescriptorSet descriptor_set_{nullptr};

    // Host-visible per-slot metadata, mirrored to shaders at set 0 binding 1.
    std::unique_ptr<VulkanEngine::GpuResources::GpuBuffer> texture_info_buffer_{};
    VulkanEngine::Textures::GpuTextureInfo* mapped_texture_info_ = nullptr;

    std::vector<Slot> slots_{};
    std::vector<std::uint32_t> free_slots_{};   // LIFO free list, slot 0 excluded
    std::uint32_t capacity_ = 0;
    std::uint32_t next_generation_ = 1;
    bool fallback_ready_ = false;

    VulkanEngine::GpuResources::FrameRing<SlotOp> ring_{};
};

} // namespace VulkanEngine::BindlessManager
