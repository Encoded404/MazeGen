module;

export module VulkanEngine.ResourceSystem.TextureResource;

import std;
import std.compat;

import FileLoader.Types;

import vulkan;

import VulkanEngine.ResourceSystem;
import VulkanEngine.TextureTypes;
import VulkanEngine.FileLoaders.TextureLoaders;

export namespace VulkanEngine {

struct CheckerboardConfig {
    std::uint32_t size = 64;
    std::uint32_t squares = 8;
    std::array<std::uint8_t, 4> color1 = {255, 255, 255, 255}; // White
    std::array<std::uint8_t, 4> color2 = {255, 0, 255, 255};   // Purple
};

// Stores the canonical loader output (TextureData) plus a version for future
// hot-reload invalidation. The source format/container payload is kept as the
// file provides it; device format resolution happens at upload time.
class TextureResource final : public Resource {
public:
    explicit TextureResource(ResourceId id);

    TextureResource(ResourceId id, std::uint32_t width, std::uint32_t height, vk::Format format,
                    std::vector<std::byte> pixels);

    [[nodiscard]] std::uint32_t GetWidth() const noexcept;
    [[nodiscard]] std::uint32_t GetHeight() const noexcept;
    [[nodiscard]] std::uint32_t GetMipLevels() const noexcept;
    [[nodiscard]] std::uint32_t GetLayerCount() const noexcept;
    [[nodiscard]] std::uint32_t GetFaceCount() const noexcept;
    [[nodiscard]] vk::Format GetVkFormat() const noexcept;
    [[nodiscard]] const std::vector<std::byte>& GetPixels() const noexcept;
    [[nodiscard]] bool HasPixels() const noexcept;
    [[nodiscard]] const VulkanEngine::Textures::AlphaAnalysis& GetAlphaAnalysis() const noexcept;

    // Subresource-complete access.
    [[nodiscard]] const VulkanEngine::Textures::TextureData& GetData() const noexcept { return data_; }
    [[nodiscard]] bool HasData() const noexcept { return !data_.blob.empty(); }
    [[nodiscard]] std::uint32_t GetVersion() const noexcept { return version_; }

    // Re-reads the source file at `path` and replaces the payload in place,
    // bumping the version. Returns false and leaves the payload untouched when
    // the read or decode fails, so a bad edit never destroys a working texture.
    // Main-thread; the resource system does not track the path, so the caller
    // (the texture watcher) supplies it.
    bool ReloadFromPath(const std::filesystem::path& path);

protected:
    bool DoLoad() override;
    bool DoUnload() override;
    bool DoLoadFromBuffer(const FileLoader::ByteBuffer& buf) override;
    void Reset() noexcept;

private:
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    VulkanEngine::Textures::TextureData data_{};
    std::uint32_t version_ = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

}  // namespace VulkanEngine
