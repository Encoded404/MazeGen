module;

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.VulkanCapabilities;

import std;

import logiface;

import vulkan;

namespace VulkanBackend::Vulkan {

// ── Module-private feature ↔ struct-bit mapping ───────────────────────

bool VulkanCapabilitiesBuilder::GetSupportedFeature(const SupportedDeviceState& supported, Feature f) noexcept {
    switch (f) {
        case Feature::PipelineStatisticsQuery:
            return supported.features2.features.pipelineStatisticsQuery == vk::True;
        case Feature::TextureCompressionBC:
            return supported.features2.features.textureCompressionBC == vk::True;
        case Feature::TextureCompressionASTC_LDR:
            return supported.features2.features.textureCompressionASTC_LDR == vk::True;
        case Feature::SamplerAnisotropy:
            return supported.features2.features.samplerAnisotropy == vk::True;
        case Feature::ShaderDrawParameters:
            return supported.vulkan11.shaderDrawParameters == vk::True;
        case Feature::HostQueryReset:
            return supported.vulkan12.hostQueryReset == vk::True;
        case Feature::DescriptorIndexing:
            return supported.vulkan12.descriptorIndexing == vk::True;
        case Feature::ShaderSampledImageArrayNonUniformIndexing:
            return supported.vulkan12.shaderSampledImageArrayNonUniformIndexing == vk::True;
        case Feature::ShaderStorageImageArrayNonUniformIndexing:
            return supported.vulkan12.shaderStorageImageArrayNonUniformIndexing == vk::True;
        case Feature::ShaderStorageBufferArrayNonUniformIndexing:
            return supported.vulkan12.shaderStorageBufferArrayNonUniformIndexing == vk::True;
        case Feature::RuntimeDescriptorArray:
            return supported.vulkan12.runtimeDescriptorArray == vk::True;
        case Feature::DescriptorBindingPartiallyBound:
            return supported.vulkan12.descriptorBindingPartiallyBound == vk::True;
        case Feature::DescriptorBindingSampledImageUpdateAfterBind:
            return supported.vulkan12.descriptorBindingSampledImageUpdateAfterBind == vk::True;
        case Feature::DescriptorBindingStorageBufferUpdateAfterBind:
            return supported.vulkan12.descriptorBindingStorageBufferUpdateAfterBind == vk::True;
        case Feature::DescriptorBindingVariableDescriptorCount:
            return supported.vulkan12.descriptorBindingVariableDescriptorCount == vk::True;
        case Feature::ScalarBlockLayout:
            return supported.vulkan12.scalarBlockLayout == vk::True;
        case Feature::DrawIndirectCount:
            return supported.vulkan12.drawIndirectCount == vk::True;
        case Feature::DrawIndirectFirstInstance:
            return supported.features2.features.drawIndirectFirstInstance == vk::True;
        case Feature::DynamicRendering:
            return supported.vulkan13.dynamicRendering == vk::True;
        case Feature::PipelineCreationCacheControl:
            return supported.vulkan13.pipelineCreationCacheControl == vk::True;
        case Feature::Synchronization2:
            return supported.vulkan13.synchronization2 == vk::True;
        case Feature::GraphicsPipelineLibrary:
            return supported.gpl.graphicsPipelineLibrary == vk::True;
        case Feature::MemoryPriority:
            return supported.memory_priority.memoryPriority == vk::True;
        case Feature::Count:
            break;
    }
    return false;
}

bool VulkanCapabilitiesBuilder::GetRequestedFeature(const VulkanCapabilities& caps, Feature f) noexcept {
    switch (f) {
        case Feature::PipelineStatisticsQuery:
            return caps.core_features2_.features.pipelineStatisticsQuery == vk::True;
        case Feature::TextureCompressionBC:
            return caps.core_features2_.features.textureCompressionBC == vk::True;
        case Feature::TextureCompressionASTC_LDR:
            return caps.core_features2_.features.textureCompressionASTC_LDR == vk::True;
        case Feature::SamplerAnisotropy:
            return caps.core_features2_.features.samplerAnisotropy == vk::True;
        case Feature::ShaderDrawParameters:
            return caps.vulkan11_features_.shaderDrawParameters == vk::True;
        case Feature::HostQueryReset:
            return caps.vulkan12_features_.hostQueryReset == vk::True;
        case Feature::DescriptorIndexing:
            return caps.vulkan12_features_.descriptorIndexing == vk::True;
        case Feature::ShaderSampledImageArrayNonUniformIndexing:
            return caps.vulkan12_features_.shaderSampledImageArrayNonUniformIndexing == vk::True;
        case Feature::ShaderStorageImageArrayNonUniformIndexing:
            return caps.vulkan12_features_.shaderStorageImageArrayNonUniformIndexing == vk::True;
        case Feature::ShaderStorageBufferArrayNonUniformIndexing:
            return caps.vulkan12_features_.shaderStorageBufferArrayNonUniformIndexing == vk::True;
        case Feature::RuntimeDescriptorArray:
            return caps.vulkan12_features_.runtimeDescriptorArray == vk::True;
        case Feature::DescriptorBindingPartiallyBound:
            return caps.vulkan12_features_.descriptorBindingPartiallyBound == vk::True;
        case Feature::DescriptorBindingSampledImageUpdateAfterBind:
            return caps.vulkan12_features_.descriptorBindingSampledImageUpdateAfterBind == vk::True;
        case Feature::DescriptorBindingStorageBufferUpdateAfterBind:
            return caps.vulkan12_features_.descriptorBindingStorageBufferUpdateAfterBind == vk::True;
        case Feature::DescriptorBindingVariableDescriptorCount:
            return caps.vulkan12_features_.descriptorBindingVariableDescriptorCount == vk::True;
        case Feature::ScalarBlockLayout:
            return caps.vulkan12_features_.scalarBlockLayout == vk::True;
        case Feature::DrawIndirectCount:
            return caps.vulkan12_features_.drawIndirectCount == vk::True;
        case Feature::DrawIndirectFirstInstance:
            return caps.core_features2_.features.drawIndirectFirstInstance == vk::True;
        case Feature::DynamicRendering:
            return caps.vulkan13_features_.dynamicRendering == vk::True;
        case Feature::PipelineCreationCacheControl:
            return caps.vulkan13_features_.pipelineCreationCacheControl == vk::True;
        case Feature::Synchronization2:
            return caps.vulkan13_features_.synchronization2 == vk::True;
        case Feature::GraphicsPipelineLibrary:
            return caps.gpl_features_.graphicsPipelineLibrary == vk::True;
        case Feature::MemoryPriority:
            return caps.memory_priority_features_.memoryPriority == vk::True;
        case Feature::Count:
            break;
    }
    return false;
}

void VulkanCapabilitiesBuilder::SetRequestedFeature(VulkanCapabilities& caps, Feature f, bool value) noexcept {
    const vk::Bool32 bit = value ? vk::True : vk::False;
    switch (f) {
        case Feature::PipelineStatisticsQuery:
            caps.core_features2_.features.pipelineStatisticsQuery = bit;
            break;
        case Feature::TextureCompressionBC:
            caps.core_features2_.features.textureCompressionBC = bit;
            break;
        case Feature::TextureCompressionASTC_LDR:
            caps.core_features2_.features.textureCompressionASTC_LDR = bit;
            break;
        case Feature::SamplerAnisotropy:
            caps.core_features2_.features.samplerAnisotropy = bit;
            break;
        case Feature::ShaderDrawParameters:
            caps.vulkan11_features_.shaderDrawParameters = bit;
            break;
        case Feature::HostQueryReset:
            caps.vulkan12_features_.hostQueryReset = bit;
            break;
        case Feature::DescriptorIndexing:
            caps.vulkan12_features_.descriptorIndexing = bit;
            break;
        case Feature::ShaderSampledImageArrayNonUniformIndexing:
            caps.vulkan12_features_.shaderSampledImageArrayNonUniformIndexing = bit;
            break;
        case Feature::ShaderStorageImageArrayNonUniformIndexing:
            caps.vulkan12_features_.shaderStorageImageArrayNonUniformIndexing = bit;
            break;
        case Feature::ShaderStorageBufferArrayNonUniformIndexing:
            caps.vulkan12_features_.shaderStorageBufferArrayNonUniformIndexing = bit;
            break;
        case Feature::RuntimeDescriptorArray:
            caps.vulkan12_features_.runtimeDescriptorArray = bit;
            break;
        case Feature::DescriptorBindingPartiallyBound:
            caps.vulkan12_features_.descriptorBindingPartiallyBound = bit;
            break;
        case Feature::DescriptorBindingSampledImageUpdateAfterBind:
            caps.vulkan12_features_.descriptorBindingSampledImageUpdateAfterBind = bit;
            break;
        case Feature::DescriptorBindingStorageBufferUpdateAfterBind:
            caps.vulkan12_features_.descriptorBindingStorageBufferUpdateAfterBind = bit;
            break;
        case Feature::DescriptorBindingVariableDescriptorCount:
            caps.vulkan12_features_.descriptorBindingVariableDescriptorCount = bit;
            break;
        case Feature::ScalarBlockLayout:
            caps.vulkan12_features_.scalarBlockLayout = bit;
            break;
        case Feature::DrawIndirectCount:
            caps.vulkan12_features_.drawIndirectCount = bit;
            break;
        case Feature::DrawIndirectFirstInstance:
            caps.core_features2_.features.drawIndirectFirstInstance = bit;
            break;
        case Feature::DynamicRendering:
            caps.vulkan13_features_.dynamicRendering = bit;
            break;
        case Feature::PipelineCreationCacheControl:
            caps.vulkan13_features_.pipelineCreationCacheControl = bit;
            break;
        case Feature::Synchronization2:
            caps.vulkan13_features_.synchronization2 = bit;
            break;
        case Feature::GraphicsPipelineLibrary:
            caps.gpl_features_.graphicsPipelineLibrary = bit;
            break;
        case Feature::MemoryPriority:
            caps.memory_priority_features_.memoryPriority = bit;
            break;
        case Feature::Count:
            break;
    }
}

bool VulkanCapabilitiesBuilder::IsFeatureSupported(Feature f) const noexcept {
    return GetSupportedFeature(supported_, f);
}

void VulkanCapabilitiesBuilder::SetFeatureRequest(Feature f, bool required) {
    if (GetSupportedFeature(supported_, f)) {
        SetRequestedFeature(caps_, f, true);
    } else if (required) {
        caps_.requirements_unmet_ = true;
        AppendError(std::string("required feature not supported: ") + std::string(kFeatureCatalog[static_cast<std::size_t>(f)].name));
    }
}

void VulkanCapabilitiesBuilder::FinalizeFeatures() {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Feature::Count); ++i) {
        caps_.features_[i] = GetRequestedFeature(caps_, static_cast<Feature>(i));
    }

    caps_.core_features2_.pNext = &caps_.vulkan11_features_;
    caps_.vulkan11_features_.pNext = &caps_.vulkan12_features_;
    caps_.vulkan12_features_.pNext = &caps_.vulkan13_features_;
    // VK_EXT_memory_priority's feature struct is chained only when its feature
    // was requested (i.e. the extension is enabled and supported), so the
    // create-time chain stays valid without the extension.
    if (caps_.features_[static_cast<std::size_t>(Feature::MemoryPriority)]) {
        caps_.vulkan13_features_.pNext = &caps_.memory_priority_features_;
        caps_.memory_priority_features_.pNext = caps_.features_[static_cast<std::size_t>(Feature::GraphicsPipelineLibrary)]
            ? static_cast<void*>(&caps_.gpl_features_)
            : nullptr;
        caps_.gpl_features_.pNext = nullptr;
    } else {
        caps_.vulkan13_features_.pNext = caps_.features_[static_cast<std::size_t>(Feature::GraphicsPipelineLibrary)]
            ? static_cast<void*>(&caps_.gpl_features_)
            : nullptr;
        caps_.gpl_features_.pNext = nullptr;
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────

std::string VulkanCapabilities::Summarize(const VulkanInstanceCapabilities& instance_caps) const {
    const std::uint32_t api = properties_.apiVersion;
    const auto api_major = (api >> 22U) & 0x7FU;
    const auto api_minor = (api >> 12U) & 0x3FFU;
    const auto api_patch = api & 0xFFFU;

    std::string result;
    result += "device=" + std::string(properties_.deviceName);
    result += " api=" + std::to_string(api_major) + "." + std::to_string(api_minor) + "." + std::to_string(api_patch);
    result += " | instance-ext=[";

    bool first = true;
    for (std::size_t i = 0; i < static_cast<std::size_t>(InstanceExtension::Count); ++i) {
        if (instance_caps.IsInstanceExtensionEnabled(static_cast<InstanceExtension>(i))) {
            if (!first) result += ", ";
            first = false;
            result += kInstanceExtensionCatalog[i].name;
        }
    }
    for (const auto& name : instance_caps.enabled_ext_names_) {
        bool in_catalog = false;
        for (const auto& spec : kInstanceExtensionCatalog) {
            if (spec.name == name) {
                in_catalog = true;
                break;
            }
        }
        if (in_catalog) continue;
        if (!first) result += ", ";
        first = false;
        result += name;
    }

    result += "] | device-ext=[";
    first = true;
    for (std::size_t i = 0; i < static_cast<std::size_t>(DeviceExtension::Count); ++i) {
        if (IsDeviceExtensionEnabled(static_cast<DeviceExtension>(i))) {
            if (!first) result += ", ";
            first = false;
            result += kDeviceExtensionCatalog[i].name;
        }
    }
    for (const auto& name : enabled_device_ext_names_) {
        bool in_catalog = false;
        for (const auto& spec : kDeviceExtensionCatalog) {
            if (spec.name == name) {
                in_catalog = true;
                break;
            }
        }
        if (in_catalog) continue;
        if (!first) result += ", ";
        first = false;
        result += name;
    }

    result += "] | features=[";
    first = true;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Feature::Count); ++i) {
        if (IsFeatureEnabled(static_cast<Feature>(i))) {
            if (!first) result += ", ";
            first = false;
            result += kFeatureCatalog[i].name;
        }
    }
    result += "]";

    return result;
}

} // namespace VulkanBackend::Vulkan
