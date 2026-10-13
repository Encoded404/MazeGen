module;

module VulkanEngine.GpuResources.SamplerCache;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.TextureTypes;

namespace VulkanEngine::GpuResources {

bool SamplerCache::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                              std::uint32_t max_sampler_allocations) {
    backend_ = &backend;
    max_sampler_allocations_ = max_sampler_allocations;
    substitutions_ = 0;
    samplers_.clear();
    index_.clear();
    return true;
}

void SamplerCache::Shutdown() {
    samplers_.clear();
    index_.clear();
    backend_ = nullptr;
    max_sampler_allocations_ = 0;
}

VulkanEngine::Textures::SamplerDesc SamplerCache::Clamp(
    const VulkanEngine::Textures::SamplerDesc& desired, std::uint32_t mip_levels) const {
    if (backend_ == nullptr) {
        return VulkanEngine::Textures::ClampSamplerDesc(desired, false, 1, mip_levels);
    }
    const auto& caps = backend_->GetCapabilities();
    const bool anisotropy_supported =
        caps.IsFeatureEnabled(VulkanBackend::Vulkan::Feature::SamplerAnisotropy);
    const auto device_max = static_cast<std::uint32_t>(caps.GetMaxSamplerAnisotropy());
    return VulkanEngine::Textures::ClampSamplerDesc(desired, anisotropy_supported, device_max, mip_levels);
}

vk::Sampler SamplerCache::Get(const VulkanEngine::Textures::SamplerDesc& desired,
                              std::uint32_t mip_levels) {
    if (backend_ == nullptr) {
        return vk::Sampler{nullptr};
    }
    const VulkanEngine::Textures::SamplerDesc clamped = Clamp(desired, mip_levels);
    if (const auto it = index_.find(clamped); it != index_.end()) {
        return static_cast<vk::Sampler>(*samplers_[it->second].sampler);
    }

    // At capacity, substitute the nearest existing sampler rather than fail.
    if (max_sampler_allocations_ != 0 && samplers_.size() >= max_sampler_allocations_) {
        std::vector<VulkanEngine::Textures::SamplerDesc> candidates;
        candidates.reserve(samplers_.size());
        for (const auto& entry : samplers_) {
            candidates.push_back(entry.desc);
        }
        const std::size_t nearest =
            VulkanEngine::Textures::FindNearestSamplerIndex(candidates, clamped);
        ++substitutions_;
        return static_cast<vk::Sampler>(*samplers_[nearest].sampler);
    }

    const vk::SamplerCreateInfo info =
        VulkanEngine::Textures::MakeSamplerCreateInfo(clamped);
    auto sampler = std::make_unique<vk::raii::Sampler>(backend_->GetDevice(), info);
    VulkanBackend::Vulkan::SetVulkanObjectName(backend_->GetDevice(), *sampler,
                                               "sampler-cache-" + std::to_string(samplers_.size()));
    const std::size_t slot = samplers_.size();
    samplers_.push_back(Entry{.desc = clamped, .sampler = std::move(sampler)});
    index_.emplace(clamped, slot);
    return static_cast<vk::Sampler>(*samplers_.back().sampler);
}

} // namespace VulkanEngine::GpuResources
