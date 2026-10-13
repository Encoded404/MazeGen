module;

export module VulkanEngine.TextureUploader;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanBackend.Vulkan.VulkanCapabilities;
export import VulkanEngine.GpuResources;
export import VulkanEngine.GpuTexture;
export import VulkanEngine.BindlessManager;
export import VulkanEngine.BindlessManager.TextureSlot;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;
import VulkanEngine.TextureUploadScheduler;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuResources.GpuImageHeap;
import VulkanEngine.GpuResources.SamplerCache;
import VulkanEngine.ResourceSystem;

export namespace VulkanEngine::Textures {

enum class UploadStatus : std::uint8_t { Pending, Resident, Failed };

struct UploadResult {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    BindlessManager::TextureHandle handle{};
    UploadStatus status = UploadStatus::Pending;
    std::string error{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// A main-thread reservation: a bindless slot pre-bound to the fallback
// descriptor (safe to sample immediately) plus the declared semantic/encoding/
// sampler the worker resolves the upload format from. No format is resolved
// here — it depends on the source payload, known only at Submit.
struct TextureReservation {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    BindlessManager::TextureHandle handle{};
    TextureSemantic semantic = TextureSemantic::Unknown;
    TextureNormalEncoding normal_encoding = TextureNormalEncoding::Standard;
    SamplerDesc sampler{};
    ResourceId id{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Asynchronous, subresource-complete texture uploader.
//
// Reserve() is main-thread and immediate: it takes a bindless slot pre-bound to
// the fallback. Submit() hands the canonical TextureData to a dedicated worker
// pool, which resolves the device format, inflates/transcodes it, and emits CPU
// mip chains for uncompressed sources without one. BeginFrame() (main thread)
// collects finished items and stages them; RecordUploads() (record_prep) records
// the per-subresource copies into the frame command buffer and commits the
// bindless binding, which publishes one FIF cycle later. The source data is
// retained until the publish applies or is dropped.
class TextureUploader {
public:
    TextureUploader() = default;
    ~TextureUploader();

    TextureUploader(const TextureUploader&) = delete;
    TextureUploader& operator=(const TextureUploader&) = delete;

    // `sampler_cache` deduplicates the VkSampler objects the uploaded textures
    // share. It may be null (the device-free tests own their samplers), in
    // which case each texture owns a private sampler.
    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    GpuResources::GpuImageHeap& image_heap,
                    GpuResources::StagingPool& staging_pool,
                    BindlessManager::BindlessManager& bindless,
                    const VulkanBackend::Vulkan::VulkanCapabilities& capabilities,
                    GpuResources::SamplerCache* sampler_cache = nullptr,
                    std::size_t worker_count = 2);
    void Shutdown();

    // Immediate, no GPU work. Reserves a slot pre-bound to the fallback and
    // records the declared semantic/encoding/sampler. nullopt = bindless
    // capacity exhausted (caller substitutes the fallback).
    [[nodiscard]] std::optional<TextureReservation> Reserve(
        TextureSemantic semantic, TextureNormalEncoding normal_encoding,
        const SamplerDesc& sampler, const ResourceId& id);

    std::future<UploadResult> Submit(TextureReservation reservation,
                                     std::shared_ptr<const TextureData> data,
                                     std::uint32_t priority = 0);

    // Records at most `budget` bytes of ready uploads into the frame command
    // buffer; returns bytes recorded.
    std::uint64_t RecordUploads(vk::CommandBuffer cmd, std::uint32_t frame_index,
                                std::uint64_t budget);

    // Main-thread per-frame pump: resolves completed worker items into staged,
    // committable images. `gate(recording_frame)` is the corrected
    // IsFrameComplete, used to return dropped publishes to the ready queue.
    void BeginFrame(std::uint32_t frame_index,
                    const std::function<bool(std::uint32_t)>& gate);

    // Optional observer invoked on the main thread when a texture's real
    // binding is committed and recorded, with the device bytes its image
    // occupies and the recording frame. Residency uses it to move a reservation
    // to a resident, sized candidate. Safe to leave unset.
    void SetOnResident(
        std::function<void(BindlessManager::TextureHandle, std::uint64_t, std::uint32_t)> cb) {
        on_resident_ = std::move(cb);
    }
    // Optional observer invoked on the main thread at Reserve time, carrying the
    // resource id. Residency keys its resource->handle registry from it so hot
    // reload can find the slot a resource currently occupies.
    void SetOnReserve(
        std::function<void(BindlessManager::TextureHandle, const ResourceId&)> cb) {
        on_reserve_ = std::move(cb);
    }

    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }
    [[nodiscard]] std::uint32_t PendingCount() const;

    // Soft per-frame byte pacing target for RecordUploads.
    [[nodiscard]] static constexpr std::uint64_t kRecordBudget() { return 32ULL << 20; }

private:
    struct WorkerItem {
        TextureReservation reservation{};
        std::shared_ptr<const TextureData> source{};
        std::uint32_t priority = 0;
        std::uint64_t sequence = 0;
        std::shared_ptr<std::promise<UploadResult>> promise{};
    };

    // A worker-finished payload: the resolved format and the final (transcoded /
    // mip-complete) texture data, ready for the main thread to stage.
    struct ReadyItem {
        TextureReservation reservation{};
        TextureData data{};
        vk::Format format = vk::Format::eUndefined;
        bool ok = false;
        std::string error{};
        std::shared_ptr<std::promise<UploadResult>> promise{};
    };

    // A staged, committable image: bytes are in a staging range; the frame
    // command buffer still has to record the copies.
    struct StagedItem {
        TextureReservation reservation{};
        GpuResources::GpuTexture texture{};
        GpuResources::StagingAlloc staging{};
        TextureData data{};                     // copy owned here; blob outlives the copy
        std::uint32_t priority = 0;
        std::uint64_t sequence = 0;
        std::shared_ptr<std::promise<UploadResult>> promise{};
    };

    void WorkerLoop();
    void CompleteItem(const ReadyItem& item, UploadStatus status, std::string error);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    GpuResources::GpuImageHeap* image_heap_ = nullptr;
    GpuResources::StagingPool* staging_pool_ = nullptr;
    BindlessManager::BindlessManager* bindless_ = nullptr;
    const VulkanBackend::Vulkan::VulkanCapabilities* capabilities_ = nullptr;
    GpuResources::SamplerCache* sampler_cache_ = nullptr;

    // Dedicated decode/transcode worker pool (never ThreadPool::Global()).
    std::vector<std::jthread> workers_{};
    mutable std::mutex work_mutex_{};
    std::condition_variable work_cv_{};
    std::deque<WorkerItem> work_queue_{};
    std::uint64_t next_sequence_ = 1;
    bool stop_ = false;

    std::mutex ready_mutex_{};
    std::vector<ReadyItem> ready_{};

    // Main-thread-only state.
    std::function<void(BindlessManager::TextureHandle, std::uint64_t, std::uint32_t)> on_resident_{};
    std::function<void(BindlessManager::TextureHandle, const ResourceId&)> on_reserve_{};
    std::vector<StagedItem> staged_{};
    UploadScheduler scheduler_{};
    std::atomic<std::uint32_t> in_flight_{0};
};

} // namespace VulkanEngine::Textures
