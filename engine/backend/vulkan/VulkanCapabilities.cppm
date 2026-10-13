module;

export module VulkanBackend.Vulkan.VulkanCapabilities;

import std;

import vulkan;

export namespace VulkanBackend::Vulkan {

// ── Requirement levels ────────────────────────────────────────────────

enum class Requirement : std::uint8_t {
    Required,
    Optional,
};

// ── Instance-level extensions (loader/driver-global) ──────────────────

enum class InstanceExtension : std::uint16_t {
    DebugUtils,   // VK_EXT_debug_utils
    Count,
};

// ── Device-level extensions (per physical device) ─────────────────────

enum class DeviceExtension : std::uint16_t {
    Swapchain,              // VK_KHR_swapchain                  (Required)
    DebugUtils,             // VK_EXT_debug_utils                (Optional, linked to instance entry)
    PipelineLibrary,        // VK_KHR_pipeline_library           (Optional, auto with GPL)
    GraphicsPipelineLibrary, // VK_EXT_graphics_pipeline_library (Optional)
    MemoryBudget,           // VK_EXT_memory_budget              (Optional; budget/usage reporting)
    MemoryPriority,         // VK_EXT_memory_priority            (Optional; per-allocation priority)
    Count,
};

// ── Device features (per physical device) ─────────────────────────────

enum class Feature : std::uint16_t {
    // core 1.0
    PipelineStatisticsQuery,
    TextureCompressionBC,       // BC1..BC7 block-compressed texture support (optional)
    TextureCompressionASTC_LDR, // ASTC LDR block-compressed texture support (optional)
    SamplerAnisotropy,          // anisotropic texel filtering (optional)
    // core 1.1
    ShaderDrawParameters,
    // core 1.2
    HostQueryReset,
    DescriptorIndexing,
    ShaderSampledImageArrayNonUniformIndexing,
    ShaderStorageImageArrayNonUniformIndexing,
    ShaderStorageBufferArrayNonUniformIndexing,
    RuntimeDescriptorArray,
    DescriptorBindingPartiallyBound,
    DescriptorBindingSampledImageUpdateAfterBind,
    DescriptorBindingStorageBufferUpdateAfterBind,
    DescriptorBindingVariableDescriptorCount,
    ScalarBlockLayout,  // scalar block layout for StructuredBuffers (Slang CDataLayout/ScalarDataLayout)
    DrawIndirectCount,  // vkCmdDrawIndexedIndirectCount (MID draw mode; optional)
    DrawIndirectFirstInstance, // non-zero firstInstance in indirect commands (MID; optional)
    // core 1.3
    DynamicRendering,
    PipelineCreationCacheControl,
    Synchronization2,      // vkCmdPipelineBarrier2 and friends (render-graph barriers)
    // EXT
    GraphicsPipelineLibrary,
    MemoryPriority,       // VK_EXT_memory_priority (optional); gates the feature-chain struct
    Count,
};

// ── Catalog spec structs ──────────────────────────────────────────────

struct InstanceExtensionSpec {
    std::string_view name; // NOLINT(misc-non-private-member-variables-in-classes)
    Requirement requirement; // NOLINT(misc-non-private-member-variables-in-classes)
};

struct DeviceExtensionSpec {
    std::string_view name; // NOLINT(misc-non-private-member-variables-in-classes)
    Requirement requirement; // NOLINT(misc-non-private-member-variables-in-classes)
    std::span<const DeviceExtension> dependencies{};   // auto-enabled transitively; NOLINT(misc-non-private-member-variables-in-classes)
    std::optional<Feature> gate{};                     // feature that must also be enabled; NOLINT(misc-non-private-member-variables-in-classes)
};

struct FeatureSpec {
    std::string_view name; // NOLINT(misc-non-private-member-variables-in-classes)
    Requirement requirement; // NOLINT(misc-non-private-member-variables-in-classes)
};

// ── Catalogs ──────────────────────────────────────────────────────────

inline constexpr std::array<InstanceExtensionSpec, static_cast<std::size_t>(InstanceExtension::Count)> kInstanceExtensionCatalog = {{
    { "VK_EXT_debug_utils", Requirement::Optional },              // InstanceExtension::DebugUtils
}};

// std::span cannot be brace-initialized from a bare enum list; use a named array.
inline constexpr std::array<DeviceExtension, 1> kGplDependencies{ DeviceExtension::PipelineLibrary };

inline constexpr std::array<DeviceExtensionSpec, static_cast<std::size_t>(DeviceExtension::Count)> kDeviceExtensionCatalog = {{
    { "VK_KHR_swapchain",                 Requirement::Required, {}, {} },                                   // Swapchain
    { "VK_EXT_debug_utils",               Requirement::Optional, {}, {} },                                   // DebugUtils
    { "VK_KHR_pipeline_library",          Requirement::Optional, {}, {} },                                   // PipelineLibrary
    { "VK_EXT_graphics_pipeline_library", Requirement::Optional,
      std::span<const DeviceExtension>(kGplDependencies), Feature::GraphicsPipelineLibrary },                // GraphicsPipelineLibrary
    { "VK_EXT_memory_budget",             Requirement::Optional, {}, {} },                                   // MemoryBudget
    { "VK_EXT_memory_priority",           Requirement::Optional, {}, Feature::MemoryPriority },              // MemoryPriority
}};

inline constexpr std::array<FeatureSpec, static_cast<std::size_t>(Feature::Count)> kFeatureCatalog = {{
    { "pipelineStatisticsQuery", Requirement::Required },
    { "textureCompressionBC", Requirement::Optional },
    { "textureCompressionASTC_LDR", Requirement::Optional },
    { "samplerAnisotropy", Requirement::Optional },
    { "shaderDrawParameters", Requirement::Required },
    { "hostQueryReset", Requirement::Required },
    { "descriptorIndexing", Requirement::Required },
    { "shaderSampledImageArrayNonUniformIndexing", Requirement::Required },
    { "shaderStorageImageArrayNonUniformIndexing", Requirement::Required },
    { "shaderStorageBufferArrayNonUniformIndexing", Requirement::Required },
    { "runtimeDescriptorArray", Requirement::Required },
    { "descriptorBindingPartiallyBound", Requirement::Required },
    { "descriptorBindingSampledImageUpdateAfterBind", Requirement::Required },
    { "descriptorBindingStorageBufferUpdateAfterBind", Requirement::Required },
    { "descriptorBindingVariableDescriptorCount", Requirement::Required },
    { "scalarBlockLayout", Requirement::Required },
    { "drawIndirectCount", Requirement::Optional },
    { "drawIndirectFirstInstance", Requirement::Optional },
    { "dynamicRendering", Requirement::Required },
    { "pipelineCreationCacheControl", Requirement::Required },
    { "synchronization2", Requirement::Required },
    { "graphicsPipelineLibrary", Requirement::Optional },
    { "memoryPriority", Requirement::Optional },
}};

// ── Format support snapshot (closed candidate set) ────────────────────

// Per-format support flags for the texture-system candidate set, captured
// once at device selection from VkFormatProperties3.
struct FormatSupport {
    vk::Format format{vk::Format::eUndefined};
    vk::FormatFeatureFlags2 linear_features{};   // VkFormatProperties3::linearTilingFeatures
    vk::FormatFeatureFlags2 optimal_features{};  // VkFormatProperties3::optimalTilingFeatures
};

// The formats the resolver may choose. A fixed, ordered list so a snapshot is
// a plain vector indexed consistently across instances.
inline constexpr std::array kCandidateFormats{
    vk::Format::eR8Unorm,
    vk::Format::eR8Srgb,
    vk::Format::eR8G8Unorm,
    vk::Format::eR8G8Srgb,
    vk::Format::eR8G8B8A8Unorm,
    vk::Format::eR8G8B8A8Srgb,
    vk::Format::eB8G8R8A8Unorm,
    vk::Format::eB8G8R8A8Srgb,
    vk::Format::eR16G16B16A16Sfloat,
    vk::Format::eBc1RgbUnormBlock,
    vk::Format::eBc1RgbSrgbBlock,
    vk::Format::eBc1RgbaUnormBlock,
    vk::Format::eBc1RgbaSrgbBlock,
    vk::Format::eBc3UnormBlock,
    vk::Format::eBc3SrgbBlock,
    vk::Format::eBc4UnormBlock,
    vk::Format::eBc5UnormBlock,
    vk::Format::eBc6HUfloatBlock,
    vk::Format::eBc6HSfloatBlock,
    vk::Format::eBc7UnormBlock,
    vk::Format::eBc7SrgbBlock,
    vk::Format::eAstc4x4SrgbBlock,
};

// Value snapshot, immutable after boot. Pure table lookup over the candidate
// set: a format outside the set (or eUndefined) reports unsupported rather than
// asserting, so a native source format the resolver has no candidate for (e.g.
// BC2) fails the load instead of crashing; it is not an arbitrary query API and
// needs no device at use time.
class FormatCapabilities {
public:
    FormatCapabilities() = default;
    explicit FormatCapabilities(std::span<const FormatSupport> formats) : formats_(formats.begin(), formats.end()) {}

    [[nodiscard]] bool sampled(vk::Format format) const noexcept {
        return Has(format, vk::FormatFeatureFlagBits2::eSampledImage);
    }
    [[nodiscard]] bool transfer_dst(vk::Format format) const noexcept {
        return Has(format, vk::FormatFeatureFlagBits2::eTransferDst);
    }
    [[nodiscard]] bool linear_filter(vk::Format format) const noexcept {
        return Has(format, vk::FormatFeatureFlagBits2::eSampledImageFilterLinear);
    }
    [[nodiscard]] bool color_attachment(vk::Format format) const noexcept {
        return Has(format, vk::FormatFeatureFlagBits2::eColorAttachment);
    }
    [[nodiscard]] std::span<const FormatSupport> formats() const noexcept { return formats_; }

private:
    [[nodiscard]] const FormatSupport* Find(vk::Format format) const noexcept {
        const auto it = std::ranges::find(formats_, format, &FormatSupport::format);
        return it != formats_.end() ? &*it : nullptr;
    }
    [[nodiscard]] bool Has(vk::Format format, vk::FormatFeatureFlagBits2 flag) const noexcept {
        const FormatSupport* entry = Find(format);
        if (entry == nullptr) {
            return false;
        }
        return (entry->optimal_features & flag) == flag;
    }

    std::vector<FormatSupport> formats_{};
};

// Descriptor limits: not a format concern; derived from the descriptor-indexing
// properties at build time. A COMBINED_IMAGE_SAMPLER binding consumes both a
// sampler and a sampled-image descriptor (per set and per stage), so the
// layout cap is the minimum of the four relevant limits minus the app reserve.
struct DescriptorCapabilities {
    std::uint32_t max_bindless_combined_image_samplers{0};
    std::uint32_t max_update_after_bind_descriptors_in_all_pools{0};
};

// Shared app headroom subtracted from the device limits when computing the
// bindless combined-image-sampler capacity. Used by the capability builder and
// by BindlessManager so both clamp consistently.
inline constexpr std::uint32_t kBindlessAppSampledReserve = 64;

// ── SupportedDeviceState: raw one-time query results (device) ─────────

struct SupportedDeviceState {
    vk::PhysicalDeviceProperties properties{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceDescriptorIndexingProperties descriptor_indexing{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceDriverProperties driver_properties{}; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<vk::ExtensionProperties> extensions; // NOLINT(misc-non-private-member-variables-in-classes)

    // Format support for the texture-system candidate set, in kCandidateFormats
    // order (filled at physical-device selection).
    std::vector<FormatSupport> format_supports; // NOLINT(misc-non-private-member-variables-in-classes)

    // Features2 query chain: features2.pNext → vulkan11 → vulkan12 → vulkan13
    // → (only if VK_EXT_graphics_pipeline_library was enumerated) gpl
    vk::PhysicalDeviceFeatures2 features2{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceVulkan11Features vulkan11{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceVulkan12Features vulkan12{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceVulkan13Features vulkan13{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{}; // NOLINT(misc-non-private-member-variables-in-classes)
    vk::PhysicalDeviceMemoryPriorityFeaturesEXT memory_priority{}; // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] bool HasExtension(std::string_view name) const noexcept {
        for (const auto& ext : extensions) {
            if (std::string_view(ext.extensionName) == name) {
                return true;
            }
        }
        return false;
    }
};

// ── VulkanInstanceCapabilities snapshot (instance/loader domain) ──────

class VulkanInstanceCapabilitiesBuilder;

class VulkanInstanceCapabilities {
public:
    // ── Queries (noexcept, O(1) for IDs, immutable after instance creation) ──
    [[nodiscard]] bool IsInstanceExtensionEnabled(InstanceExtension e) const noexcept {
        return exts_[static_cast<std::size_t>(e)];
    }
    [[nodiscard]] bool IsInstanceExtensionEnabled(std::string_view name) const noexcept {
        for (const auto& enabled : enabled_ext_names_) {
            if (enabled == name) return true;
        }
        return false;
    }
    [[nodiscard]] const std::string& GetErrorMessage() const noexcept { return error_message_; }
    [[nodiscard]] bool HasUnmetRequirements() const noexcept { return requirements_unmet_; }

private:
    friend class VulkanInstanceCapabilitiesBuilder;
    friend class VulkanCapabilities;   // for Summarize()

    std::bitset<static_cast<std::size_t>(InstanceExtension::Count)> exts_{};
    std::unordered_set<std::string> enabled_ext_names_{};
    std::string error_message_{};
    bool requirements_unmet_ = false;
};

class VulkanInstanceCapabilitiesBuilder {
public:
    explicit VulkanInstanceCapabilitiesBuilder(VulkanInstanceCapabilities& caps) : caps_(caps) {}

    void SetExtensionEnabled(InstanceExtension e) {
        caps_.exts_.set(static_cast<std::size_t>(e));
    }
    void AddExtensionName(std::string_view name) {
        caps_.enabled_ext_names_.emplace(name);
    }
    void AppendError(std::string_view message) {
        if (!caps_.error_message_.empty()) caps_.error_message_ += "; ";
        caps_.error_message_ += message;
    }
    void SetRequirementsUnmet() {
        caps_.requirements_unmet_ = true;
    }

private:
    VulkanInstanceCapabilities& caps_;
};

// ── VulkanCapabilities snapshot (physical-device domain) ──────────────

class VulkanCapabilitiesBuilder;

class VulkanCapabilities {
public:
    // ── Queries (noexcept, O(1), immutable after device creation) ──
    [[nodiscard]] bool IsDeviceExtensionEnabled(DeviceExtension e) const noexcept {
        return device_exts_[static_cast<std::size_t>(e)];
    }
    [[nodiscard]] bool IsFeatureEnabled(Feature f) const noexcept {
        return features_[static_cast<std::size_t>(f)];
    }
    [[nodiscard]] bool CanUse(DeviceExtension e) const noexcept {
        if (!IsDeviceExtensionEnabled(e)) return false;
        if (const auto gate = kDeviceExtensionCatalog[static_cast<std::size_t>(e)].gate) {
            return IsFeatureEnabled(*gate);
        }
        return true;
    }
    [[nodiscard]] const vk::PhysicalDeviceProperties& GetProperties() const noexcept { return properties_; }
    [[nodiscard]] const vk::PhysicalDeviceDescriptorIndexingProperties& GetDescriptorIndexingProperties() const noexcept {
        return descriptor_indexing_;
    }
    [[nodiscard]] const FormatCapabilities& GetFormatCapabilities() const noexcept { return format_capabilities_; }
    [[nodiscard]] const DescriptorCapabilities& GetDescriptorCapabilities() const noexcept { return descriptor_capabilities_; }
    // maxSamplerAnisotropy device limit (meaningful only with SamplerAnisotropy).
    [[nodiscard]] float GetMaxSamplerAnisotropy() const noexcept { return properties_.limits.maxSamplerAnisotropy; }
    // maxSamplerAllocationCount: the number of VkSampler objects the device
    // permits. The sampler cache stays within it and degrades to the nearest
    // existing sampler when full, rather than failing a draw.
    [[nodiscard]] std::uint32_t GetMaxSamplerAllocationCount() const noexcept {
        return properties_.limits.maxSamplerAllocationCount;
    }
    [[nodiscard]] const vk::PhysicalDeviceDriverProperties& GetDriverProperties() const noexcept { return driver_properties_; }

    // ── Optional device-memory reporting (Phase 5 residency) ──
    // Both extensions are optional: the engine must run without them. When
    // VK_EXT_memory_budget is present the per-heap budget/usage snapshot is
    // populated at boot; otherwise the snapshot is empty and residency falls
    // back to a caller-supplied byte budget. VK_EXT_memory_priority is only
    // usable together with the memoryPriority core feature.
    struct MemoryHeapBudget {
        std::uint64_t budget{0};
        std::uint64_t usage{0};
    };
    [[nodiscard]] bool HasMemoryBudget() const noexcept {
        return IsDeviceExtensionEnabled(DeviceExtension::MemoryBudget);
    }
    [[nodiscard]] bool HasMemoryPriority() const noexcept {
        return CanUse(DeviceExtension::MemoryPriority);
    }
    [[nodiscard]] std::span<const MemoryHeapBudget> GetMemoryBudgetSnapshot() const noexcept {
        return memory_budget_;
    }
    // Root of the create-time feature chain (chained into VkDeviceCreateInfo).
    [[nodiscard]] const vk::PhysicalDeviceFeatures2& GetFeatureChain() const noexcept {
        return core_features2_;
    }
    [[nodiscard]] const std::string& GetErrorMessage() const noexcept { return error_message_; }
    [[nodiscard]] bool HasUnmetRequirements() const noexcept { return requirements_unmet_; }
    [[nodiscard]] std::string Summarize(const VulkanInstanceCapabilities& instance_caps) const;

private:
    friend class VulkanCapabilitiesBuilder;

    std::bitset<static_cast<std::size_t>(DeviceExtension::Count)> device_exts_{};
    std::bitset<static_cast<std::size_t>(Feature::Count)> features_{};
    std::unordered_set<std::string> enabled_device_ext_names_{};   // dynamic app device extensions

    // Feature chain structs — the EXACT structs chained into vkCreateDevice
    // (stable addresses, owned here for lifetime).
    vk::PhysicalDeviceFeatures2 core_features2_{};
    vk::PhysicalDeviceVulkan11Features vulkan11_features_{};
    vk::PhysicalDeviceVulkan12Features vulkan12_features_{};
    vk::PhysicalDeviceVulkan13Features vulkan13_features_{};
    vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features_{};
    vk::PhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features_{};

    std::vector<MemoryHeapBudget> memory_budget_{};

    vk::PhysicalDeviceProperties properties_{};
    vk::PhysicalDeviceDescriptorIndexingProperties descriptor_indexing_{};
    vk::PhysicalDeviceDriverProperties driver_properties_{};

    FormatCapabilities format_capabilities_{};
    DescriptorCapabilities descriptor_capabilities_{};

    std::string error_message_{};
    bool requirements_unmet_ = false;
};

class VulkanCapabilitiesBuilder {
public:
    VulkanCapabilitiesBuilder(VulkanCapabilities& caps, const SupportedDeviceState& supported)
        : caps_(caps), supported_(supported) {}

    void SetDeviceExtensions(std::span<const DeviceExtension> exts) {
        for (const auto e : exts) {
            caps_.device_exts_.set(static_cast<std::size_t>(e));
        }
    }
    void AddDeviceExtensionName(std::string_view name) {
        caps_.enabled_device_ext_names_.emplace(name);
    }
    void SetProperties(const vk::PhysicalDeviceProperties& properties) {
        caps_.properties_ = properties;
    }
    void SetDescriptorIndexingProperties(const vk::PhysicalDeviceDescriptorIndexingProperties& properties) {
        caps_.descriptor_indexing_ = properties;
    }
    void SetFormatSupports(std::span<const FormatSupport> formats) {
        caps_.format_capabilities_ = FormatCapabilities{formats};
    }
    // Computes the descriptor-capability snapshot from the descriptor-indexing
    // properties. `app_reserve` keeps headroom for app combined-image-sampler
    // bindings that share the layout with the bindless array.
    void FinalizeDescriptorCapabilities(std::uint32_t app_reserve) {
        const auto& p = caps_.descriptor_indexing_;
        caps_.descriptor_capabilities_.max_bindless_combined_image_samplers =
            std::min({p.maxDescriptorSetUpdateAfterBindSamplers,
                      p.maxDescriptorSetUpdateAfterBindSampledImages,
                      p.maxPerStageDescriptorUpdateAfterBindSamplers,
                      p.maxPerStageDescriptorUpdateAfterBindSampledImages}) - app_reserve;
        caps_.descriptor_capabilities_.max_update_after_bind_descriptors_in_all_pools =
            p.maxUpdateAfterBindDescriptorsInAllPools;
    }
    void SetDriverProperties(const vk::PhysicalDeviceDriverProperties& properties) {
        caps_.driver_properties_ = properties;
    }
    // Captures the optional per-heap budget/usage snapshot when the extension
    // is enabled. Empty when it is absent, so the engine runs unmodified.
    void SetMemoryBudget(std::span<const VulkanCapabilities::MemoryHeapBudget> budget) {
        caps_.memory_budget_.assign(budget.begin(), budget.end());
    }
    void AppendError(std::string_view message) {
        if (!caps_.error_message_.empty()) caps_.error_message_ += "; ";
        caps_.error_message_ += message;
    }
    void SetRequirementsUnmet() {
        caps_.requirements_unmet_ = true;
    }

    [[nodiscard]] bool IsFeatureSupported(Feature f) const noexcept;
    void SetFeatureRequest(Feature f, bool required);
    void FinalizeFeatures();

    // Root of the create-time feature chain (chained into VkDeviceCreateInfo).
    [[nodiscard]] const vk::PhysicalDeviceFeatures2& GetFeatureChain() const noexcept {
        return caps_.core_features2_;
    }

private:
    static bool GetSupportedFeature(const SupportedDeviceState& supported, Feature f) noexcept;
    static bool GetRequestedFeature(const VulkanCapabilities& caps, Feature f) noexcept;
    static void SetRequestedFeature(VulkanCapabilities& caps, Feature f, bool value) noexcept;

    VulkanCapabilities& caps_;
    const SupportedDeviceState& supported_;
};

} // namespace VulkanBackend::Vulkan
