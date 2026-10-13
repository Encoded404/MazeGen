module;

export module VulkanEngine.GpuResources.SamplerCache;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;

import VulkanEngine.TextureTypes;

export namespace VulkanEngine::GpuResources {

// Shared, owning cache of VkSampler objects.
//
// A texture references exactly one VkSampler per bindless slot, but the vast
// majority of textures share one of a handful of descriptions (linear/repeat
// with anisotropy, nearest/clamp for data maps, and so on). Deduplicating the
// sampler objects bounds sampler allocation by the number of *distinct*
// descriptions rather than by the number of textures, which matters against
// maxSamplerAllocationCount.
//
// Descriptions are clamped to the device limits before they are hashed, so
// equal requests collapse to one entry and the cache key is device-independent.
// When the cache reaches maxSamplerAllocationCount it does NOT fail: the
// nearest existing sampler is substituted so a draw is never lost. The cache
// owns every sampler it hands out and must outlive the textures that reference
// them.
class SamplerCache {
public:
    SamplerCache() = default;
    ~SamplerCache() = default;

    SamplerCache(const SamplerCache&) = delete;
    SamplerCache& operator=(const SamplerCache&) = delete;

    // `backend` must outlive the cache. `max_sampler_allocations` is the
    // device's maxSamplerAllocationCount (0 means unbounded, clamped only by
    // the caller's own growth).
    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    std::uint32_t max_sampler_allocations = 0);

    void Shutdown();

    // Returns the cached sampler for `desired`, creating it on first use.
    // `mip_levels` bounds max_lod to the chain that will use the sampler, so a
    // texture with no mips never advertises them. Never returns a null sampler
    // once initialized: at capacity the nearest existing sampler is returned.
    [[nodiscard]] vk::Sampler Get(const VulkanEngine::Textures::SamplerDesc& desired,
                                  std::uint32_t mip_levels = 1);

    // The clamped description that Get() would resolve `desired` to, without
    // touching the cache. Exposed for tests and diagnostics.
    [[nodiscard]] VulkanEngine::Textures::SamplerDesc Clamp(
        const VulkanEngine::Textures::SamplerDesc& desired, std::uint32_t mip_levels = 1) const;

    [[nodiscard]] std::size_t Size() const { return samplers_.size(); }
    [[nodiscard]] std::uint32_t MaxAllocations() const { return max_sampler_allocations_; }
    [[nodiscard]] std::uint64_t Substitutions() const { return substitutions_; }
    [[nodiscard]] bool IsValid() const { return backend_ != nullptr; }

private:
    struct Entry {
        VulkanEngine::Textures::SamplerDesc desc{};
        std::unique_ptr<vk::raii::Sampler> sampler{};
    };

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    std::uint32_t max_sampler_allocations_ = 0;
    std::uint64_t substitutions_ = 0;
    std::vector<Entry> samplers_{};
    std::unordered_map<VulkanEngine::Textures::SamplerDesc, std::size_t,
                       VulkanEngine::Textures::SamplerDescHash>
        index_{};
};

} // namespace VulkanEngine::GpuResources
