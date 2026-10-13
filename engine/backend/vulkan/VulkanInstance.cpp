module;

#include <SDL3/SDL_vulkan.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_hpp_macros.hpp>

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.VulkanInstance;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanShared.ScopedSection;

import VulkanBackend.Vulkan.VulkanCapabilities;
import VulkanBackend.Vulkan.VulkanDebugUtils;

namespace VulkanBackend::Vulkan {

namespace {

VulkanShared::ScopedSection DebugSection(const std::string& name) {
    return VulkanShared::ScopedSection{name, [](const std::string& section, double ms) {
        LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
    }};
}

bool IsNameSupported(const std::vector<vk::ExtensionProperties>& available, std::string_view name) {
    for (const auto& ext : available) {
        if (std::string_view(ext.extensionName) == name) {
            return true;
        }
    }
    return false;
}

// Routes VK_EXT_debug_utils messages (validation layer output, loader/driver
// diagnostics) into the engine logger. Without a messenger the validation layer
// has no sink at all, so a Debug build silently swallows exactly the messages
// that explain a failed vkCreate*Pipelines.
VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT /*types*/,
        const vk::DebugUtilsMessengerCallbackDataEXT* data,
        void* /*user_data*/) {
    if (data == nullptr || data->pMessage == nullptr) return vk::False;

    std::string message = std::string("[Vulkan] ") + data->pMessage;
    // Demote the non-VUID "SPIR-V Interface" pipeline-note.
    // (vertex-stage outputs with no matching fragment input, e.g. the minimal example's checker shader) to debug level.
    // it is informational about discarded writes, not a defect, but fires on every pipeline creation otherwise.
    const bool spirv_interface_note = message.find("but there is no corresponding Input declared") != std::string::npos;
    const auto severity_bits = static_cast<VkDebugUtilsMessageSeverityFlagsEXT>(severity);
    if ((severity_bits & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u) {
        LOGIFACE_LOG(error, message);
    } else if ((severity_bits & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u) {
        if (spirv_interface_note) {
            LOGIFACE_LOG(debug, message);
        } else {
            LOGIFACE_LOG(warn, message);
        }
    } else if ((severity_bits & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0u) {
        LOGIFACE_LOG(info, message);
    } else {
        LOGIFACE_LOG(debug, message);
    }
    return vk::False;  // always continue: never abort the offending call
}

} // namespace

bool VulkanInstance::Initialize(const VulkanBootstrapConfig& config) {
    if (instance_) return true;

    window_ = config.native_window_handle;
    if (!window_) {
        LOGIFACE_LOG(error, "native window handle is null");
        return false;
    }

    loader_ = std::make_unique<vk::detail::DynamicLoader>();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*loader_);

    VulkanInstanceCapabilitiesBuilder builder{capabilities_};

    // 1. Enumerate instance extensions + layers once.
    std::vector<vk::ExtensionProperties> available_extensions;
    std::vector<vk::LayerProperties> available_layers;
    try {
        const vk::raii::Context context{};
        available_extensions = context.enumerateInstanceExtensionProperties();
        available_layers = context.enumerateInstanceLayerProperties();
    } catch (const std::exception& ex) {
        LOGIFACE_LOG(error, "instance extension enumeration failed: " + std::string(ex.what()));
        return false;
    }

    const auto is_force_disabled = [&config](std::string_view name) {
        for (const auto& disabled : config.force_disabled_extensions) {
            if (disabled == name) return true;
        }
        return false;
    };

    // 2. SDL-provided extensions (required by contract).
    std::uint32_t sdl_extension_count = 0;
    const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
    if (!sdl_extensions) {
        LOGIFACE_LOG(error, "SDL_Vulkan_GetInstanceExtensions failed");
        return false;
    }

    std::vector<std::string> requested_names;
    for (std::uint32_t i = 0; i < sdl_extension_count; ++i) {
        const std::string name{sdl_extensions[i]};
        if (!IsNameSupported(available_extensions, name)) {
            builder.SetRequirementsUnmet();
            builder.AppendError("required instance extension not supported: " + name);
        }
        requested_names.push_back(name);
    }

    // 3-4. Engine catalog + app requests (validated, deduped against SDL).
    const auto add_request = [&](const std::string& name, bool required, std::string_view source) {
        for (const auto& requested : requested_names) {
            if (requested == name) return;   // dedupe
        }
        if (!IsNameSupported(available_extensions, name)) {
            if (required) {
                builder.SetRequirementsUnmet();
                builder.AppendError("required " + std::string(source) + " instance extension not supported: " + name);
            } else {
                LOGIFACE_LOG(warn, "optional instance extension unavailable, skipping: " + name);
            }
            return;
        }
        requested_names.push_back(name);
    };

    bool app_wants_debug_utils = false;
    bool app_debug_utils_required = false;
    for (const auto& request : config.instance_extensions) {
        if (request.name == "VK_EXT_debug_utils") {
            app_wants_debug_utils = true;
            app_debug_utils_required = request.required;
        }
    }
    if (config.enable_validation || app_wants_debug_utils) {
        add_request("VK_EXT_debug_utils", app_debug_utils_required, "engine-catalog");
    }
    for (const auto& request : config.instance_extensions) {
        if (request.name == "VK_EXT_debug_utils") continue;   // handled with the catalog entry
        add_request(request.name, request.required, "app");
    }

    // 5. Validation layer (optional debug aid).
    std::vector<const char*> instance_layers;
    if (config.enable_validation) {
        bool layer_available = false;
        for (const auto& layer : available_layers) {
            if (std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
                layer_available = true;
                break;
            }
        }
        if (layer_available) {
            instance_layers.push_back("VK_LAYER_KHRONOS_validation");
        } else {
            LOGIFACE_LOG(warn, "validation layer VK_LAYER_KHRONOS_validation not available, continuing without it");
        }
    }

    // 6. Apply force-disabled extensions to the final request set.
    for (auto it = requested_names.begin(); it != requested_names.end();) {
        if (is_force_disabled(*it)) {
            LOGIFACE_LOG(warn, "force-disabled instance extension: " + *it);
            it = requested_names.erase(it);
        } else {
            ++it;
        }
    }

    // 7. Fail before creation if any required instance requirement is unmet.
    if (capabilities_.HasUnmetRequirements()) {
        LOGIFACE_LOG(error, "instance requirements unmet: " + capabilities_.GetErrorMessage());
        return false;
    }

    // 8. Create the instance.
    std::vector<const char*> instance_extension_names;
    instance_extension_names.reserve(requested_names.size());
    for (const auto& name : requested_names) {
        instance_extension_names.push_back(name.c_str());
    }

    // Debug-utils messenger, chained into the instance create info so messages
    // emitted during vkCreateInstance/teardown are captured too. Only chained
    // when VK_EXT_debug_utils actually made the final request set.
    const bool debug_utils_requested =
        std::ranges::find(requested_names, "VK_EXT_debug_utils") != requested_names.end();
    vk::DebugUtilsMessengerCreateInfoEXT messenger_ci{};
    messenger_ci.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
    messenger_ci.messageType =
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    messenger_ci.pfnUserCallback = &DebugUtilsCallback;

    constexpr vk::ApplicationInfo app_info("VulkanEngineV5", 1, "VulkanEngineV5", 1, vk::ApiVersion13);
    vk::InstanceCreateInfo instance_info({}, &app_info,
        static_cast<std::uint32_t>(instance_layers.size()), instance_layers.data(),
        static_cast<std::uint32_t>(instance_extension_names.size()), instance_extension_names.data());
    if (debug_utils_requested) {
        instance_info.setPNext(&messenger_ci);
    }

    try {
        instance_ = std::make_unique<vk::raii::Instance>(vk::raii::Context{}, instance_info);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(**instance_);
    } catch (const std::exception& ex) {
        LOGIFACE_LOG(error, "instance creation failed: " + std::string(ex.what()));
        instance_.reset();
        return false;
    }

    // 9. Record enabled names/IDs into the snapshot.
    for (const auto& name : requested_names) {
        builder.AddExtensionName(name);
        if (name == "VK_EXT_debug_utils") {
            builder.SetExtensionEnabled(InstanceExtension::DebugUtils);
        }
    }

    // 10. Debug-utils global gate (set once, before any naming calls) and the
    //     actual messenger that routes layer/driver messages into the logger.
    SetDebugUtilsEnabled(capabilities_.IsInstanceExtensionEnabled(InstanceExtension::DebugUtils));
    if (capabilities_.IsInstanceExtensionEnabled(InstanceExtension::DebugUtils)) {
        try {
            debug_messenger_ = std::make_unique<vk::raii::DebugUtilsMessengerEXT>(*instance_, messenger_ci);
            for (const auto& layer : instance_layers) {
                LOGIFACE_LOG(info, std::string("Vulkan debug messenger installed (layer: ") + layer + ")");
            }
        } catch (const std::exception& ex) {
            LOGIFACE_LOG(warn, std::string("failed to create Vulkan debug messenger: ") + ex.what());
            debug_messenger_.reset();
        }
    }

    VkSurfaceKHR raw_surface = nullptr;
    if (!SDL_Vulkan_CreateSurface(window_, static_cast<VkInstance>(**instance_), nullptr, &raw_surface)) {
        LOGIFACE_LOG(error, "SDL_Vulkan_CreateSurface failed");
        Shutdown();
        return false;
    }
    surface_ = std::make_unique<vk::raii::SurfaceKHR>(*instance_, raw_surface);

    return true;
}

void VulkanInstance::Shutdown() {
    {
        auto s = DebugSection("destroy debug messenger in VulkanInstance::Shutdown");
        debug_messenger_.reset();
    }
    {
        auto s = DebugSection("destroy surface in VulkanInstance::Shutdown");
        surface_.reset();
    }
    {
        auto s = DebugSection("destroy instance in VulkanInstance::Shutdown");
        instance_.reset();
    }
    {
        auto s = DebugSection("destroy loader in VulkanInstance::Shutdown");
        loader_.reset();
    }
    window_ = nullptr;
}

} // namespace VulkanBackend::Vulkan
