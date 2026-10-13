module;

module VulkanEngine.DefaultTextureFactory;

import std;
import std.compat;

import vulkan;

import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;

namespace VulkanEngine::DefaultTextureFactory {

std::shared_ptr<VulkanEngine::TextureResource> CreateCheckerboard(
    VulkanEngine::ResourceManager&  /*manager*/,
    const VulkanEngine::CheckerboardConfig& config) {
    const std::uint32_t pixel_count = config.size * config.size * 4;
    std::vector<std::byte> pixels(pixel_count);
    const std::uint32_t square_size = config.size / config.squares;

    for (std::uint32_t y = 0; y < config.size; ++y) {
        for (std::uint32_t x = 0; x < config.size; ++x) {
            const bool is_color1 = (((x / square_size) + (y / square_size)) % 2U) == 0U;
            const auto& color = is_color1 ? config.color1 : config.color2;
            const std::size_t idx = (static_cast<std::size_t>(y) * config.size + x) * 4;
            pixels[idx + 0] = static_cast<std::byte>(color[0]);
            pixels[idx + 1] = static_cast<std::byte>(color[1]);
            pixels[idx + 2] = static_cast<std::byte>(color[2]);
            pixels[idx + 3] = static_cast<std::byte>(color[3]);
        }
    }

    return std::make_shared<VulkanEngine::TextureResource>(
        VulkanEngine::ResourceId{"checkerboard_default"}, config.size, config.size,
        vk::Format::eR8G8B8A8Unorm, std::move(pixels));
}

std::shared_ptr<VulkanEngine::TextureResource> CreateWhiteTexture(
    VulkanEngine::ResourceManager&  manager) {
    VulkanEngine::CheckerboardConfig white_config{};
    white_config.color1 = {255, 255, 255, 255};
    white_config.color2 = {255, 255, 255, 255};
    return CreateCheckerboard(manager, white_config);
}

std::shared_ptr<VulkanEngine::TextureResource> CreateSolidColorTexture(
    VulkanEngine::ResourceManager&  /*manager*/,
    const std::array<uint8_t, 4>& color) {
    std::vector<std::byte> pixels(4);
    pixels[0] = static_cast<std::byte>(color[0]);
    pixels[1] = static_cast<std::byte>(color[1]);
    pixels[2] = static_cast<std::byte>(color[2]);
    pixels[3] = static_cast<std::byte>(color[3]);

    return std::make_shared<VulkanEngine::TextureResource>(
        VulkanEngine::ResourceId{"solid_color"}, 1, 1,
        vk::Format::eR8G8B8A8Unorm, std::move(pixels));
}

} // namespace VulkanEngine::DefaultTextureFactory
