module;

#include <fstream>
#include <logging/logging_macros.hpp>

module VulkanEngine.ResourceSystem.TextureResource;

import std;

import FileLoader.Types;
import logiface;

import vulkan;

import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.TextureLoaders;

namespace VulkanEngine {

TextureResource::TextureResource(ResourceId id)
    : Resource(std::move(id)) {
    Reset();
}

TextureResource::TextureResource(ResourceId id, std::uint32_t width, std::uint32_t height, vk::Format format,
                                  std::vector<std::byte> pixels)
    : Resource(std::move(id)) {
    data_.width = width;
    data_.height = height;
    data_.source_format = format;
    data_.mip_levels = 1;
    data_.array_layers = 1;
    data_.face_count = 1;
    data_.blob = std::move(pixels);
    if (!data_.blob.empty()) {
        data_.subresources.push_back(VulkanEngine::Textures::TextureSubresource{
            .mip = 0, .array_layer = 0,
            .offset = 0, .size = data_.blob.size(),
            .width = width, .height = height, .depth = 1,
        });
        data_.alpha_analysis = VulkanEngine::FileLoaders::Textures::AnalyzeAlpha(data_.blob);
    }
    loaded_ = !data_.blob.empty();
}

uint32_t TextureResource::GetWidth() const noexcept { return data_.width; }
uint32_t TextureResource::GetHeight() const noexcept { return data_.height; }
uint32_t TextureResource::GetMipLevels() const noexcept { return data_.mip_levels; }
uint32_t TextureResource::GetLayerCount() const noexcept { return data_.array_layers; }
uint32_t TextureResource::GetFaceCount() const noexcept { return data_.face_count; }
vk::Format TextureResource::GetVkFormat() const noexcept { return data_.source_format; }
const std::vector<std::byte>& TextureResource::GetPixels() const noexcept { return data_.blob; }
bool TextureResource::HasPixels() const noexcept { return !data_.blob.empty(); }
const VulkanEngine::Textures::AlphaAnalysis& TextureResource::GetAlphaAnalysis() const noexcept { return data_.alpha_analysis; }

bool TextureResource::DoLoad() {
    LOGIFACE_LOG(error, "TextureResource '" + GetId().value + "' cannot be loaded without file buffer data");
    return false;
}

bool TextureResource::DoUnload() {
    Reset();
    return true;
}

void TextureResource::Reset() noexcept {
    data_.Reset();
}

bool TextureResource::ReloadFromPath(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        LOGIFACE_LOG(warn, "TextureResource: reload failed, cannot open '" + path.string() + "'");
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        LOGIFACE_LOG(warn, "TextureResource: reload failed, empty file '" + path.string() + "'");
        return false;
    }
    file.seekg(0, std::ios::beg);
    FileLoader::ByteBuffer buffer(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        LOGIFACE_LOG(warn, "TextureResource: reload failed, short read '" + path.string() + "'");
        return false;
    }

    // Decode into a temporary first: a failed decode must not clobber the live
    // payload.
    VulkanEngine::Textures::TextureData reloaded{};
    std::string error_message{};
    if (!VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer(GetId().value, buffer,
                                                                    reloaded, &error_message)) {
        LOGIFACE_LOG(warn, "TextureResource: reload decode failed for '" + GetId().value + "'" +
                               (error_message.empty() ? std::string{} : ": " + error_message));
        return false;
    }
    data_ = std::move(reloaded);
    ++version_;
    loaded_ = true;
    return true;
}

bool TextureResource::DoLoadFromBuffer(const FileLoader::ByteBuffer& buf) {
    Reset();

    if (buf.empty()) {
        LOGIFACE_LOG(warn, "TextureResource: empty buffer for resource '" + GetId().value + "'");
        return false;
    }

    std::string error_message{};

    if (!VulkanEngine::FileLoaders::Textures::LoadTextureFromBuffer(GetId().value, buf, data_, &error_message)) {
        LOGIFACE_LOG(warn, "TextureResource: failed to load texture resource '" + GetId().value + "'" + (error_message.empty() ? std::string{} : ": " + error_message));
        return false;
    }

    ++version_;
    return true;
}

}  // namespace VulkanEngine
