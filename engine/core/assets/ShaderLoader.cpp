module;

module VulkanEngine.ShaderLoader;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;

namespace VulkanEngine::ShaderLoader {

std::expected<std::vector<std::uint32_t>, std::string>
ShaderLoader::LoadSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return std::unexpected("Failed to open SPIR-V file: " + path.string());
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::unexpected("SPIR-V file is empty: " + path.string());
    }
    if ((size % static_cast<std::streamsize>(sizeof(std::uint32_t))) != 0) {
        return std::unexpected("SPIR-V file size is not word-aligned: " + path.string());
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4U);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

vk::raii::ShaderModule ShaderLoader::CreateModule(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                                   const std::filesystem::path& path) {
    auto spirv = LoadSpirv(path);
    if (!spirv) {
        throw std::runtime_error(spirv.error());
    }
    return CreateModuleFromSpirv(backend, *spirv);
}

vk::raii::ShaderModule ShaderLoader::CreateModuleFromSpirv(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                                            std::span<const std::uint32_t> spirv) {
    vk::ShaderModuleCreateInfo const info({}, spirv.size() * sizeof(std::uint32_t), spirv.data());
    return vk::raii::ShaderModule(backend.GetDevice(), info);
}

} // namespace VulkanEngine::ShaderLoader
