module;

export module VulkanEngine.ShaderLoader;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;

export namespace VulkanEngine::ShaderLoader {

class ShaderLoader {
public:
    [[nodiscard]] static std::expected<std::vector<std::uint32_t>, std::string> LoadSpirv(const std::filesystem::path& path);
    [[nodiscard]] static vk::raii::ShaderModule CreateModule(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                                             const std::filesystem::path& path);
    [[nodiscard]] static vk::raii::ShaderModule CreateModuleFromSpirv(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                                                      std::span<const std::uint32_t> spirv);
};

} // namespace VulkanEngine::ShaderLoader
