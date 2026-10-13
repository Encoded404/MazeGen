#include <gtest/gtest.h>

#include <ktx.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


import std;
import std.compat;

import vulkan;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;

namespace {

class TempTextureFile final {
public:
    explicit TempTextureFile(std::string_view extension) {
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::random_device rd;
        path_ = std::filesystem::temp_directory_path() /
                ("ve5_texture_resource_" + std::to_string(stamp) + "_" + std::to_string(rd()) + std::string(extension));
    }

    ~TempTextureFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& Path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::array<std::uint8_t, 16> Make2x2RgbaPixels() {
    return {
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 0, 255,
    };
}

void WriteTinyKtx2TextureToTempFile(const TempTextureFile& temp) {
    ktxTextureCreateInfo info{};
    info.vkFormat = static_cast<ktx_uint32_t>(vk::Format::eR8G8B8A8Unorm);
    info.baseWidth = 2;
    info.baseHeight = 2;
    info.baseDepth = 1;
    info.numDimensions = 2;
    info.numLevels = 1;
    info.numLayers = 1;
    info.numFaces = 1;
    info.isArray = KTX_FALSE;
    info.generateMipmaps = KTX_FALSE;

    ktxTexture2* texture = nullptr;
    const KTX_error_code create_result = ktxTexture2_Create(&info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
    if (create_result != KTX_SUCCESS || texture == nullptr) {
        throw std::runtime_error(std::string("ktxTexture2_Create failed: ") + ktxErrorString(create_result));
    }

    const auto pixels = Make2x2RgbaPixels();
    const KTX_error_code image_result = ktxTexture_SetImageFromMemory(
        reinterpret_cast<ktxTexture*>(texture),
        0,
        0,
        0,
        pixels.data(),
        static_cast<ktx_size_t>(pixels.size()));
    if (image_result != KTX_SUCCESS) {
        ktxTexture_Destroy(reinterpret_cast<ktxTexture*>(texture));
        throw std::runtime_error(std::string("ktxTexture_SetImageFromMemory failed: ") + ktxErrorString(image_result));
    }

    ktx_uint8_t* out_bytes = nullptr;
    ktx_size_t out_size = 0;
    const KTX_error_code write_result = ktxTexture_WriteToMemory(reinterpret_cast<ktxTexture*>(texture), &out_bytes, &out_size);
    if (write_result != KTX_SUCCESS || out_bytes == nullptr || out_size == 0) {
        ktxTexture_Destroy(reinterpret_cast<ktxTexture*>(texture));
        throw std::runtime_error(std::string("ktxTexture_WriteToMemory failed: ") + ktxErrorString(write_result));
    }

    {
        std::ofstream out(temp.Path(), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(out_bytes), static_cast<std::streamsize>(out_size));
    }

    std::free(out_bytes);
    ktxTexture_Destroy(reinterpret_cast<ktxTexture*>(texture));
}

void WriteTinyPngTextureToTempFile(const TempTextureFile& temp) {
    const auto pixels = Make2x2RgbaPixels();
    const int stride = 2 * 4;
    if (stbi_write_png(temp.Path().string().c_str(), 2, 2, 4, pixels.data(), stride) != 1) {
        throw std::runtime_error("stbi_write_png failed");
    }
}

void WriteTinyJpegTextureToTempFile(const TempTextureFile& temp) {
    const std::array<std::uint8_t, 16> pixels{
        64, 128, 224, 255,
        64, 128, 224, 255,
        64, 128, 224, 255,
        64, 128, 224, 255,
    };
    if (stbi_write_jpg(temp.Path().string().c_str(), 2, 2, 4, pixels.data(), 100) != 1) {
        throw std::runtime_error("stbi_write_jpg failed");
    }
}

void ExpectPixelsNear(const std::vector<std::byte>& actual, const std::array<std::uint8_t, 16>& expected, std::uint8_t tolerance) {
    // The blob may carry a CPU mip chain below the base level; compare the
    // base level only.
    ASSERT_GE(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto value = static_cast<int>(actual[i]);
        const auto exp = static_cast<int>(expected[i]);
        EXPECT_LE(std::abs(value - exp), static_cast<int>(tolerance)) << "mismatch at byte " << i;
    }
}

class TextureResourceTest : public ::testing::Test {};

}  // namespace

TEST_F(TextureResourceTest, LoadsTinyRgbaKtx2) {
    const TempTextureFile temp(".ktx2");
    WriteTinyKtx2TextureToTempFile(temp);

    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(),
        VulkanEngine::ResourceManager::LoadSpeed::Instant);

    ASSERT_TRUE(handle.IsValid());
    auto* texture = handle.Get();
    ASSERT_NE(texture, nullptr);

    EXPECT_EQ(texture->GetWidth(), 2u);
    EXPECT_EQ(texture->GetHeight(), 2u);
    EXPECT_EQ(texture->GetMipLevels(), 1u);
    EXPECT_EQ(texture->GetLayerCount(), 1u);
    EXPECT_EQ(texture->GetFaceCount(), 1u);
    EXPECT_EQ(texture->GetVkFormat(), vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(texture->HasPixels());

    const auto expected = Make2x2RgbaPixels();
    ExpectPixelsNear(texture->GetPixels(), expected, 0);

    const auto& alpha = texture->GetAlphaAnalysis();
    EXPECT_TRUE(alpha.hasAlphaChannel);
    EXPECT_FALSE(alpha.hasFractionalAlpha);
    EXPECT_FALSE(alpha.hasZeroAlpha);
    EXPECT_FLOAT_EQ(alpha.opaqueCoverage, 1.0f);

    manager.Release(handle.GetId());
}

// ReloadFromPath replaces the payload and bumps the version; the reloaded
// pixels match the file.
TEST_F(TextureResourceTest, ReloadFromPathUpdatesPayloadAndVersion) {
    const TempTextureFile temp(".png");
    WriteTinyPngTextureToTempFile(temp);

    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(), VulkanEngine::ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    const std::uint32_t version_before = handle->GetVersion();
    ASSERT_TRUE(handle->ReloadFromPath(temp.Path()));
    EXPECT_EQ(handle->GetVersion(), version_before + 1u);
    const auto expected = Make2x2RgbaPixels();
    ExpectPixelsNear(handle->GetData().blob, expected, 0);
    manager.Release(handle.GetId());
}

// A bad edit fails the reload and leaves the live payload untouched.
TEST_F(TextureResourceTest, ReloadFromPathRejectsBadFile) {
    const TempTextureFile temp(".png");
    WriteTinyPngTextureToTempFile(temp);
    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(), VulkanEngine::ResourceManager::LoadSpeed::Instant);
    ASSERT_TRUE(handle.IsValid());
    const std::uint32_t version_before = handle->GetVersion();
    const auto blob_before = handle->GetData().blob;

    {
        std::ofstream out(temp.Path(), std::ios::binary | std::ios::trunc);
        out << "not an image";
    }
    EXPECT_FALSE(handle->ReloadFromPath(temp.Path()));
    EXPECT_EQ(handle->GetVersion(), version_before);
    EXPECT_EQ(handle->GetData().blob, blob_before);
    manager.Release(handle.GetId());
}

TEST_F(TextureResourceTest, LoadsTinyRgbaPng) {
    const TempTextureFile temp(".png");
    WriteTinyPngTextureToTempFile(temp);

    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(),
        VulkanEngine::ResourceManager::LoadSpeed::Instant);

    ASSERT_TRUE(handle.IsValid());
    auto* texture = handle.Get();
    ASSERT_NE(texture, nullptr);

    EXPECT_EQ(texture->GetWidth(), 2u);
    EXPECT_EQ(texture->GetHeight(), 2u);
    // Uncompressed non-KTX sources now carry a full CPU mip chain:
    // 2x2 base + 1x1 tail.
    EXPECT_EQ(texture->GetMipLevels(), 2u);
    EXPECT_EQ(texture->GetVkFormat(), vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(texture->HasPixels());
    EXPECT_EQ(texture->GetPixels().size(), 20u);

    const auto expected = Make2x2RgbaPixels();
    ExpectPixelsNear(texture->GetPixels(), expected, 0);

    const auto& alpha = texture->GetAlphaAnalysis();
    EXPECT_TRUE(alpha.hasAlphaChannel);
    EXPECT_FALSE(alpha.hasFractionalAlpha);
    EXPECT_FALSE(alpha.hasZeroAlpha);
    EXPECT_FLOAT_EQ(alpha.opaqueCoverage, 1.0f);

    manager.Release(handle.GetId());
}

TEST_F(TextureResourceTest, LoadsTinyRgbaJpeg) {
    const TempTextureFile temp(".jpg");
    WriteTinyJpegTextureToTempFile(temp);

    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(),
        VulkanEngine::ResourceManager::LoadSpeed::Instant);

    ASSERT_TRUE(handle.IsValid());
    auto* texture = handle.Get();
    ASSERT_NE(texture, nullptr);

    EXPECT_EQ(texture->GetWidth(), 2u);
    EXPECT_EQ(texture->GetHeight(), 2u);
    EXPECT_EQ(texture->GetVkFormat(), vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(texture->HasPixels());
    // 2x2 base + 1x1 tail mip.
    EXPECT_EQ(texture->GetPixels().size(), 20u);
    EXPECT_EQ(texture->GetMipLevels(), 2u);

    const std::array<std::uint8_t, 16> expected{
        64, 128, 224, 255,
        64, 128, 224, 255,
        64, 128, 224, 255,
        64, 128, 224, 255,
    };
    ExpectPixelsNear(texture->GetPixels(), expected, 20);

    const auto& alpha = texture->GetAlphaAnalysis();
    EXPECT_TRUE(alpha.hasAlphaChannel);
    EXPECT_FALSE(alpha.hasZeroAlpha);
    EXPECT_FLOAT_EQ(alpha.opaqueCoverage, 1.0f);

    manager.Release(handle.GetId());
}

TEST_F(TextureResourceTest, RejectsInvalidFile) {
    const TempTextureFile temp(".png");
    {
        std::ofstream out(temp.Path(), std::ios::binary | std::ios::trunc);
        const std::array<std::uint8_t, 8> junk{1, 2, 3, 4, 5, 6, 7, 8};
        out.write(reinterpret_cast<const char*>(junk.data()), static_cast<std::streamsize>(junk.size()));
    }

    VulkanEngine::ResourceManager manager;
    auto handle = manager.LoadFromFile<VulkanEngine::TextureResource>(
        temp.Path(),
        VulkanEngine::ResourceManager::LoadSpeed::Instant);
    EXPECT_FALSE(handle.IsValid());
}





