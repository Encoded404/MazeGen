module;

export module VulkanBackend.Vulkan.VulkanDebugUtils;

import std;
import std.compat;

import vulkan;

// Global gate: set once during instance init (VulkanInstance::Initialize) to
// whether VK_EXT_debug_utils was enabled at instance AND device creation.
// All naming/label helpers early-return when disabled.
// Module-internal (non-exported) — only the setter below is exported.
inline std::atomic<bool> g_debug_utils_enabled{false};

// ── detail::GetRawVulkanHandle ────────────────────────────────────────────
// Extracts the raw Vk* C handle from any Vulkan-Hpp handle type.
//
//   vk::raii::Xxx  (RAII)        → *obj gets the non-RAII wrapper,
//                                  then static_cast<CType> invokes its
//                                  explicit operator VkXxx().
//   vk::Xxx        (non-RAII)    → static_cast<CType>(obj) invokes the
//                                  explicit operator VkXxx() directly.
//   VkXxx / VkXxx* (raw C ptrs)  → handled by the caller's is_pointer_v.
//
// The discriminant is CppType: RAII wrappers define it; non-RAII do not.
namespace VulkanBackend::Vulkan::detail {

template<typename HandleType>
decltype(auto) GetRawVulkanHandle(const HandleType& obj) {
    if constexpr (requires { typename HandleType::CppType; }) {
        // RAII type — dereference to get the non-RAII wrapper, then convert
        return static_cast<typename HandleType::CType>(*obj);
    } else {
        // Non-RAII Vulkan-Hpp wrapper — convert directly
        return static_cast<typename HandleType::CType>(obj);
    }
}

} // namespace VulkanBackend::Vulkan::detail

export namespace VulkanBackend::Vulkan {

void SetDebugUtilsEnabled(bool enabled) noexcept {
    g_debug_utils_enabled.store(enabled, std::memory_order_relaxed);
}

// RAII handles (vk::raii::*) — objectType is a static member.
template<typename HandleType>
void SetVulkanObjectName(const vk::raii::Device& dev, const HandleType& obj,
                         const std::string& name) {
    if (!g_debug_utils_enabled.load(std::memory_order_relaxed)) return;

    auto* fn = dev.getDispatcher()->vkSetDebugUtilsObjectNameEXT;
    if (!fn) return;  // device command not resolved (no device-level debug utils)

    vk::DebugUtilsObjectNameInfoEXT info{};
    info.sType = vk::StructureType::eDebugUtilsObjectNameInfoEXT;
    info.objectType = static_cast<vk::ObjectType>(HandleType::objectType);
    info.objectHandle =
        reinterpret_cast<std::uint64_t>(detail::GetRawVulkanHandle(obj));
    info.pObjectName = name.c_str();
    dev.setDebugUtilsObjectNameEXT(info);
}

// Non-RAII / raw handles — explicit type parameter required.
template<typename HandleType>
void SetVulkanObjectName(const vk::raii::Device& dev, const HandleType& obj,
                         vk::ObjectType type, const std::string& name) {
    if (!g_debug_utils_enabled.load(std::memory_order_relaxed)) return;

    auto* fn = dev.getDispatcher()->vkSetDebugUtilsObjectNameEXT;
    if (!fn) return;

    vk::DebugUtilsObjectNameInfoEXT info{};
    info.sType = vk::StructureType::eDebugUtilsObjectNameInfoEXT;
    info.objectType = type;
    if constexpr (std::is_pointer_v<HandleType>) {
        info.objectHandle = reinterpret_cast<std::uint64_t>(obj);
    } else {
        info.objectHandle =
            reinterpret_cast<std::uint64_t>(detail::GetRawVulkanHandle(obj));
    }
    info.pObjectName = name.c_str();
    dev.setDebugUtilsObjectNameEXT(info);
}

void BeginDebugUtilsLabel(vk::CommandBuffer cmd, const std::string& name) {
    if (!g_debug_utils_enabled.load(std::memory_order_relaxed)) return;

    vk::DebugUtilsLabelEXT label{};
    label.pLabelName = name.c_str();
    cmd.beginDebugUtilsLabelEXT(label);
}

void EndDebugUtilsLabel(vk::CommandBuffer cmd) {
    if (!g_debug_utils_enabled.load(std::memory_order_relaxed)) return;

    cmd.endDebugUtilsLabelEXT();
}

}
