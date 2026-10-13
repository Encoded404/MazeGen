module;

#include <ktx.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

module VulkanEngine.FileLoaders.TextureLoaders;

import FileLoader.Types;

import std;

import vulkan;

import VulkanEngine.TextureTypes;

namespace VulkanEngine::FileLoaders::Textures {

namespace {

using VulkanEngine::Textures::TextureSemantic;
using VulkanEngine::Textures::TextureSubresource;

constexpr ktx_uint32_t kGlRgb = 0x1907U;
constexpr ktx_uint32_t kGlRgba = 0x1908U;
constexpr ktx_uint32_t kGlBgra = 0x80E1U;
constexpr ktx_uint32_t kGlUnsignedByte = 0x1401U;
constexpr ktx_uint32_t kGlRgba8 = 0x8058U;
constexpr ktx_uint32_t kGlRgb8 = 0x8051U;
constexpr ktx_uint32_t kGlBgra8Ext = 0x93A1U;

struct KtxTextureDeleter {
    void operator()(ktxTexture* texture) const noexcept {
        if (texture != nullptr) {
            ktxTexture_Destroy(texture);
        }
    }
};

using KtxTexturePtr = std::unique_ptr<ktxTexture, KtxTextureDeleter>;

[[nodiscard]] std::string ToLowerCopy(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

[[nodiscard]] bool HasPrefix(const FileLoader::ByteBuffer& buffer, const std::byte* prefix, std::size_t prefix_size) {
    if (buffer.size() < prefix_size) {
        return false;
    }
    return std::equal(prefix, prefix + prefix_size, buffer.begin());
}

[[nodiscard]] std::uint32_t MipDim(std::uint32_t base, std::uint32_t level) {
    return level == 0 ? base : std::max<std::uint32_t>(1u, base >> level);
}

// KHR_DF parse: sample count, distinct data channels, alpha presence.
//
// `ktxTexture2::pDfd` points at the whole data format descriptor, whose first
// word is `dfdTotalSize`; the basic descriptor block (BDB) starts one word
// later. Within the BDB: word 1 holds the descriptor block size in bytes (upper
// 16 bits), word 2 holds the color model in the low byte, and the 16-byte
// samples start at word 6 (KHR_DF_WORD_SAMPLESTART). Reading pDfd as if it were
// the BDB yields blockSize=0 and model=0 for every texture, which silently
// zeroed source_channels (and so disabled the resolver's channel-model rules).
struct DfdInfo {
    std::uint8_t channels{0};
    bool alpha{false};
    bool hdr{false};
};

// KHR_DF color models (KHR/khr_df.h).
constexpr std::uint8_t kDfModelUASTC = 166U;
// KHR_DF_CHANNEL_RGBSDA_ALPHA.
constexpr std::uint8_t kDfChannelAlpha = 15U;

[[nodiscard]] std::uint8_t DfdChannelId(const std::uint32_t* sample) {
    return static_cast<std::uint8_t>((sample[0] >> 24U) & 0xFU);
}

// Channel count comes from libktx: the DFD channel model is format-specific
// (RGBSDA ids for uncompressed, RRR/RG/RRRG/RGBA for UASTC, RRR/GGG/AAA for
// ETC1S), and libktx already decodes it. The plan's format rules key off the
// independent-channel count (2 -> BC5, 1 data -> BC4), so this is the resolver's
// rule-5 input and must not be approximated from the sample count.
[[nodiscard]] DfdInfo ParseDfd(const ktxTexture2* texture2) {
    DfdInfo info{};
    if (texture2->pDfd == nullptr) {
        return info;
    }
    const std::uint32_t* bdb = texture2->pDfd + 1;
    const std::uint32_t block_size = bdb[1] >> 16U;
    const std::uint8_t color_model = static_cast<std::uint8_t>(bdb[2] & 0xFFU);
    if (block_size < 24U) {
        return info;
    }
    // Samples begin after the 6-word BDB preamble (KHR_DF_WORD_SAMPLESTART).
    const std::size_t sample_count = (block_size - 24U) / 16U;
    if (sample_count == 0U || sample_count > 8U) {
        return info;
    }
    const std::uint32_t* samples = texture2->pDfd + 1 + 6;

    // ETC1S encodes its second channel in the second sample, whose channel id is
    // 15 (AAA) exactly when the encoded texture carries alpha, so one scan over
    // the sample channel ids covers every model.
    bool alpha = false;
    for (std::size_t i = 0; i < sample_count; ++i) {
        if (DfdChannelId(samples + i * 4) == kDfChannelAlpha) {
            alpha = true;
            break;
        }
    }

    const ktx_uint32_t components = ktxTexture2_GetNumComponents(
        const_cast<ktxTexture2*>(texture2));
    info.channels = static_cast<std::uint8_t>(std::clamp<ktx_uint32_t>(components, 1U, 4U));
    info.alpha = alpha || info.channels == 4U;
    // UASTC HDR stores float16 samples, whose DFD bitLength is 15 (16 bits - 1).
    // A block-compressed LDR sample reports bitLength 127 instead (128-bit
    // block), so "> 8" wrongly classified every LDR UASTC texture as HDR and
    // routed it to R16G16B16A16_SFLOAT. Match the HDR width exactly.
    if (color_model == kDfModelUASTC) {
        const std::uint8_t bit_length = static_cast<std::uint8_t>((samples[0] >> 16U) & 0xFFU);
        info.hdr = bit_length == 15U;
    }
    return info;
}

// Reads a KTX metadata string from the texture's key/value list.
[[nodiscard]] std::optional<std::string> FindMetadata(ktxTexture* texture, const char* key) {
    ktx_uint32_t value_len = 0;
    void* value = nullptr;
    if (ktxHashList_FindValue(&texture->kvDataHead, key, &value_len, &value) != KTX_SUCCESS || value == nullptr) {
        return std::nullopt;
    }
    return std::string(static_cast<const char*>(value), value_len);
}

// Engine convention is top-left orientation ("rd" for 2D, "rld" for 3D);
// anything else is rejected here rather than flipped at upload.
[[nodiscard]] bool OrientationSupported(ktxTexture* texture, std::uint32_t depth, std::string* error_message) {
    const auto orientation = FindMetadata(texture, KTX_ORIENTATION_KEY);
    if (!orientation) {
        return true; // default is +r +d
    }
    const bool has_r = orientation->find('r') != std::string::npos;
    const bool has_d = orientation->find('d') != std::string::npos;
    const bool has_l = orientation->find('l') != std::string::npos;
    const bool has_u = orientation->find('u') != std::string::npos;
    const bool supported = depth > 1U ? (has_r && has_d && has_l && !has_u)
                                      : (has_r && has_d && !has_l && !has_u);
    if (!supported && error_message != nullptr) {
        *error_message = "Unsupported KTXorientation '" + *orientation +
                         "': engine requires top-left ('rd')";
    }
    return supported;
}

[[nodiscard]] vk::ComponentSwizzle SwizzleCharToComponent(char c, bool& valid) {
    switch (c) {
        case 'r': return vk::ComponentSwizzle::eR;
        case 'g': return vk::ComponentSwizzle::eG;
        case 'b': return vk::ComponentSwizzle::eB;
        case 'a': return vk::ComponentSwizzle::eA;
        case '0': return vk::ComponentSwizzle::eZero;
        case '1': return vk::ComponentSwizzle::eOne;
        default: valid = false; return vk::ComponentSwizzle::eIdentity;
    }
}

// KTXswizzle is a view property: mapped onto vk::ComponentMapping and applied
// through the image view, never through a sampler.
[[nodiscard]] vk::ComponentMapping ParseSwizzle(ktxTexture* texture) {
    const auto swizzle = FindMetadata(texture, KTX_SWIZZLE_KEY);
    if (!swizzle || swizzle->size() != 4U) {
        return vk::ComponentMapping{};
    }
    bool valid = true;
    vk::ComponentMapping mapping{};
    mapping.r = SwizzleCharToComponent((*swizzle)[0], valid);
    mapping.g = SwizzleCharToComponent((*swizzle)[1], valid);
    mapping.b = SwizzleCharToComponent((*swizzle)[2], valid);
    mapping.a = SwizzleCharToComponent((*swizzle)[3], valid);
    return valid ? mapping : vk::ComponentMapping{};
}

// Builds the per-(level, layer, face) subresource table from a KTX texture
// whose data is resident (inflated) in memory.
[[nodiscard]] bool BuildKtxSubresources(ktxTexture* texture,
                                        TextureData& out,
                                        std::size_t data_size,
                                        std::string* error_message) {
    const auto append = [&](std::uint32_t level, std::uint32_t gpu_layer, std::uint32_t w, std::uint32_t h, std::uint32_t d) -> bool {
        ktx_size_t image_size = ktxTexture_GetImageSize(texture, level);
        ktx_size_t image_offset = 0;
        if (image_size == 0) {
            if (error_message != nullptr) {
                *error_message = "KTX texture reports a zero-sized image";
            }
            return false;
        }
        if (ktxTexture_GetImageOffset(texture, level, gpu_layer / out.face_count,
                                      gpu_layer % out.face_count, &image_offset) != KTX_SUCCESS) {
            if (error_message != nullptr) {
                *error_message = "KTX texture failed to provide an image offset";
            }
            return false;
        }
        if (static_cast<std::size_t>(image_offset) + static_cast<std::size_t>(image_size) > data_size) {
            if (error_message != nullptr) {
                *error_message = "KTX2 level layout is inconsistent with the data size";
            }
            return false;
        }
        out.subresources.push_back(TextureSubresource{
            .mip = level,
            .array_layer = gpu_layer,
            .offset = image_offset,
            .size = image_size,
            .width = w,
            .height = h,
            .depth = d,
        });
        return true;
    };

    for (std::uint32_t level = 0; level < out.mip_levels; ++level) {
        const std::uint32_t w = MipDim(out.width, level);
        const std::uint32_t h = MipDim(out.height, level);
        const std::uint32_t d = std::max<std::uint32_t>(1U, MipDim(out.depth, level));
        if (d > 1U) {
            // 3D: one subresource per level.
            if (!append(level, 0, w, h, d)) {
                return false;
            }
            continue;
        }
        for (std::uint32_t layer = 0; layer < out.array_layers; ++layer) {
            for (std::uint32_t face = 0; face < out.face_count; ++face) {
                const std::uint32_t gpu_layer = layer * out.face_count + face;
                if (!append(level, gpu_layer, w, h, 1)) {
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool LoadKtx1Texture(ktxTexture* texture, TextureData& out, std::string* error_message) {
    const auto* texture1 = reinterpret_cast<ktxTexture1*>(texture);
    if (texture1->glType != kGlUnsignedByte) {
        if (error_message != nullptr) {
            *error_message = "Unsupported KTX1 pixel type";
        }
        return false;
    }

    out.width = texture->baseWidth;
    out.height = texture->baseHeight;
    out.mip_levels = 1;
    out.array_layers = 1;
    out.face_count = 1;

    bool swizzle_bgra = false;
    bool expand_rgb = false;
    if (texture1->glFormat == kGlRgba || texture1->glInternalformat == kGlRgba8) {
        out.source_format = vk::Format::eR8G8B8A8Unorm;
    } else if (texture1->glFormat == kGlBgra || texture1->glInternalformat == kGlBgra8Ext) {
        out.source_format = vk::Format::eR8G8B8A8Unorm;
        swizzle_bgra = true;
    } else if (texture1->glFormat == kGlRgb || texture1->glInternalformat == kGlRgb8) {
        out.source_format = vk::Format::eR8G8B8A8Unorm;
        expand_rgb = true;
    } else {
        if (error_message != nullptr) {
            *error_message = "Unsupported KTX1 format for RGBA upload";
        }
        return false;
    }

    const ktx_size_t image_size = ktxTexture_GetImageSize(texture, 0);
    ktx_size_t image_offset = 0;
    if (image_size == 0 ||
        ktxTexture_GetImageOffset(texture, 0, 0, 0, &image_offset) != KTX_SUCCESS) {
        if (error_message != nullptr) {
            *error_message = "KTX texture failed to provide base-level data";
        }
        return false;
    }
    const auto* data = reinterpret_cast<const std::byte*>(ktxTexture_GetData(texture));
    const auto* src = data + image_offset;
    const auto* src_bytes = reinterpret_cast<const std::uint8_t*>(src);
    const std::size_t pixel_count = static_cast<std::size_t>(out.width) * static_cast<std::size_t>(out.height);

    out.blob.resize(pixel_count * 4U);
    if (!swizzle_bgra && !expand_rgb) {
        std::memcpy(out.blob.data(), src, std::min<std::size_t>(out.blob.size(), static_cast<std::size_t>(image_size)));
    } else if (swizzle_bgra) {
        for (std::size_t i = 0; i < pixel_count; ++i) {
            out.blob[i * 4U + 0] = static_cast<std::byte>(src_bytes[i * 4U + 2]);
            out.blob[i * 4U + 1] = static_cast<std::byte>(src_bytes[i * 4U + 1]);
            out.blob[i * 4U + 2] = static_cast<std::byte>(src_bytes[i * 4U + 0]);
            out.blob[i * 4U + 3] = static_cast<std::byte>(src_bytes[i * 4U + 3]);
        }
    } else {
        for (std::size_t i = 0; i < pixel_count; ++i) {
            out.blob[i * 4U + 0] = static_cast<std::byte>(src_bytes[i * 3U + 0]);
            out.blob[i * 4U + 1] = static_cast<std::byte>(src_bytes[i * 3U + 1]);
            out.blob[i * 4U + 2] = static_cast<std::byte>(src_bytes[i * 3U + 2]);
            out.blob[i * 4U + 3] = std::byte{255};
        }
    }

    out.subresources.push_back(TextureSubresource{
        .mip = 0, .array_layer = 0,
        .offset = 0, .size = out.blob.size(),
        .width = out.width, .height = out.height, .depth = 1,
    });
    out.source_channels = 4;
    out.source_alpha = 1;
    out.alpha_analysis = AnalyzeAlpha(out.blob);
    return true;
}

// KTXcubemapIncomplete is a ktx_uint8 flag: 0 marks an incomplete cubemap,
// which must never be presented through a cube view.
[[nodiscard]] bool IsIncompleteCubemap(ktxTexture* texture) {
    ktx_uint32_t value_len = 0;
    void* value = nullptr;
    if (ktxHashList_FindValue(&texture->kvDataHead, "KTXcubemapIncomplete", &value_len, &value) != KTX_SUCCESS ||
        value == nullptr || value_len != 1U) {
        return false;
    }
    return *static_cast<const ktx_uint8_t*>(value) == 0U;
}

[[nodiscard]] bool LoadKtx2Texture(ktxTexture* texture,
                                   const FileLoader::ByteBuffer& buffer,
                                   TextureData& out,
                                   std::string* error_message) {
    auto* texture2 = reinterpret_cast<ktxTexture2*>(texture);

    out.width = texture->baseWidth;
    out.height = texture->baseHeight;
    out.depth = std::max<std::uint32_t>(texture->baseDepth, 1U);
    out.mip_levels = texture->numLevels;
    out.array_layers = std::max<std::uint32_t>(texture->numLayers, 1U);
    out.face_count = texture->numFaces;
    out.cube_complete = !IsIncompleteCubemap(texture);

    const DfdInfo dfd = ParseDfd(texture2);
    out.source_channels = dfd.channels;
    out.source_alpha = dfd.alpha ? 1U : 0U;
    out.is_hdr_source = dfd.hdr;
    out.supercompressed = texture2->supercompressionScheme != KTX_SS_NONE;

    if (ktxTexture2_NeedsTranscoding(texture2) == KTX_TRUE) {
        // Basis (UASTC/ETC1S, optionally BasisLZ-supercompressed): keep the
        // whole container payload; the transcode step re-parses it and fills
        // the subresource table.
        out.needs_transcode = true;
        out.source_format = vk::Format::eUndefined;
        out.blob.assign(buffer.begin(), buffer.end());
        return true;
    }

    // Native payload passthrough: any vkFormat, including BCn, is preserved
    // (never expanded to RGBA).
    out.source_format = static_cast<vk::Format>(texture2->vkFormat);

    const ktx_size_t data_size = ktxTexture_GetDataSize(texture);
    const auto* data = reinterpret_cast<const std::byte*>(ktxTexture_GetData(texture));
    if (data == nullptr || data_size == 0) {
        if (error_message != nullptr) {
            *error_message = "KTX texture contains no pixel data";
        }
        return false;
    }
    out.blob.assign(data, data + data_size);
    if (!BuildKtxSubresources(texture, out, static_cast<std::size_t>(data_size), error_message)) {
        return false;
    }
    // Alpha statistics for decoded-looking RGBA8 payloads (material
    // validation consumes them); compressed formats carry no pixel stats.
    const vk::Format fmt = out.source_format;
    if (fmt == vk::Format::eR8G8B8A8Unorm || fmt == vk::Format::eR8G8B8A8Srgb) {
        const auto& base = out.subresources.front();
        out.alpha_analysis = AnalyzeAlpha(std::vector<std::byte>(
            out.blob.begin() + static_cast<std::ptrdiff_t>(base.offset),
            out.blob.begin() + static_cast<std::ptrdiff_t>(base.offset + base.size)));
    }
    return true;
}

[[nodiscard]] bool LooksLikeKtx(const FileLoader::ByteBuffer& buffer) {
    static constexpr std::array<std::byte, 12> ktx1_magic = {
        std::byte{0xAB}, std::byte{'K'}, std::byte{'T'}, std::byte{'X'}, std::byte{' '}, std::byte{'1'},
        std::byte{'1'}, std::byte{0xBB}, std::byte{0x0D}, std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}
    };
    static constexpr std::array<std::byte, 12> ktx2_magic = {
        std::byte{0xAB}, std::byte{'K'}, std::byte{'T'}, std::byte{'X'}, std::byte{' '}, std::byte{'2'},
        std::byte{'0'}, std::byte{0xBB}, std::byte{0x0D}, std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}
    };

    return HasPrefix(buffer, ktx1_magic.data(), ktx1_magic.size()) ||
           HasPrefix(buffer, ktx2_magic.data(), ktx2_magic.size());
}

[[nodiscard]] bool LooksLikePng(const FileLoader::ByteBuffer& buffer) {
    static constexpr std::array<std::byte, 8> png_magic = {
        std::byte{0x89}, std::byte{'P'}, std::byte{'N'}, std::byte{'G'}, std::byte{0x0D}, std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}
    };
    return HasPrefix(buffer, png_magic.data(), png_magic.size());
}

[[nodiscard]] bool LooksLikeJpeg(const FileLoader::ByteBuffer& buffer) {
    return buffer.size() >= 3U &&
           buffer[0] == std::byte{0xFF} &&
           buffer[1] == std::byte{0xD8} &&
           buffer[2] == std::byte{0xFF};
}

[[nodiscard]] bool LoadByMagic(const std::filesystem::path& path,
                               const FileLoader::ByteBuffer& buffer,
                               TextureData& out,
                               std::string* error_message) {
    const auto lower_ext = ToLowerCopy(path.extension().string());
    if (lower_ext == ".ktx" || lower_ext == ".ktx2" || LooksLikeKtx(buffer)) {
        return LoadKtxTextureFromBuffer(path, buffer, out, error_message);
    }

    if (lower_ext == ".png" || LooksLikePng(buffer) ||
        lower_ext == ".jpg" || lower_ext == ".jpeg" || LooksLikeJpeg(buffer)) {
        return LoadStbTextureFromBuffer(path, buffer, out, error_message);
    }

    if (error_message != nullptr) {
        *error_message = "Unsupported texture format: " + path.string();
    }
    return false;
}

// 2x2 box-filter downsample of one RGBA8 image (the classic mip filter).
void BoxDownsampleRgba8(const std::byte* src, std::uint32_t src_w, std::uint32_t src_h,
                        std::byte* dst, std::uint32_t dst_w, std::uint32_t dst_h) {
    for (std::uint32_t y = 0; y < dst_h; ++y) {
        const std::uint32_t y0 = std::min(y * 2U, src_h - 1U);
        const std::uint32_t y1 = std::min(y * 2U + 1U, src_h - 1U);
        for (std::uint32_t x = 0; x < dst_w; ++x) {
            const std::uint32_t x0 = std::min(x * 2U, src_w - 1U);
            const std::uint32_t x1 = std::min(x * 2U + 1U, src_w - 1U);
            for (std::uint32_t c = 0; c < 4U; ++c) {
                const std::uint32_t sum =
                    static_cast<std::uint32_t>(src[(y0 * src_w + x0) * 4U + c]) +
                    static_cast<std::uint32_t>(src[(y0 * src_w + x1) * 4U + c]) +
                    static_cast<std::uint32_t>(src[(y1 * src_w + x0) * 4U + c]) +
                    static_cast<std::uint32_t>(src[(y1 * src_w + x1) * 4U + c]);
                dst[(y * dst_w + x) * 4U + c] = static_cast<std::byte>(sum / 4U);
            }
        }
    }
}

// Emits a full CPU mip chain below the decoded base level: each level is a
// 2x2-box-filtered downsample of the previous one, block-tight and sequential
// in `blob`. This is a simple box filter, not a production resampler; the
// algorithm can be replaced later (a better filter, stb_image_resize2, or a GPU
// blit chain) without changing this signature or its callers. This is a simple box filter, not a production resampler; the
// algorithm can be replaced later (a better filter, stb_image_resize2, or a GPU
// blit chain) without changing this signature or its callers.
void AppendCpuMipChain(std::uint32_t width, std::uint32_t height, TextureData& out) {
    const std::uint32_t max_dim = std::max({width, height});
    const std::uint32_t levels = static_cast<std::uint32_t>(std::bit_width(max_dim));
    out.mip_levels = levels;
    out.subresources.clear();
    out.subresources.reserve(levels);

    std::uint64_t offset = 0;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::uint32_t w = MipDim(width, level);
        const std::uint32_t h = MipDim(height, level);
        const std::size_t size = static_cast<std::size_t>(w) * h * 4U;
        out.subresources.push_back(TextureSubresource{
            .mip = level, .array_layer = 0,
            .offset = offset, .size = size,
            .width = w, .height = h, .depth = 1,
        });
        offset += size;
    }
    out.blob.resize(static_cast<std::size_t>(offset));
}

}  // namespace

// Emits a full CPU mip chain for an uncompressed, single-subresource RGBA8
// source that has none. Same 2x2 box filter the loader uses; exposed so the
// async upload worker can complete a chain for sources that arrive without one.
[[nodiscard]] bool GenerateCpuMipChain(TextureData& data) {
    if (data.needs_transcode || data.is_hdr_source) {
        return false;  // compressed/HDR chains come from assets, never generated
    }
    if (data.source_format != vk::Format::eR8G8B8A8Unorm &&
        data.source_format != vk::Format::eR8G8B8A8Srgb) {
        return false;
    }
    if (data.mip_levels > 1U || data.array_layers > 1U || data.face_count > 1U || data.depth > 1U) {
        return false;
    }
    if (data.width == 0 || data.height == 0) {
        return false;
    }
    const std::uint32_t levels = std::max(
        1U, static_cast<std::uint32_t>(std::bit_width(std::max(data.width, data.height))));
    if (levels <= 1U) {
        return false;
    }
    // Copy the base level out before the subresource table is rebuilt.
    const std::vector<std::byte> base = data.blob;
    AppendCpuMipChain(data.width, data.height, data);
    std::memcpy(data.blob.data(), base.data(),
                std::min(base.size(), static_cast<std::size_t>(data.width) * data.height * 4U));
    for (std::uint32_t level = 1; level < data.mip_levels; ++level) {
        const auto& src = data.subresources[level - 1];
        auto& dst = data.subresources[level];
        BoxDownsampleRgba8(data.blob.data() + src.offset, src.width, src.height,
                           data.blob.data() + dst.offset, dst.width, dst.height);
    }
    data.mip_levels = levels;
    return true;
}

AlphaAnalysis AnalyzeAlpha(const std::vector<std::byte>& pixels) {
    AlphaAnalysis result{};
    if (pixels.empty() || pixels.size() < 4) return result;

    const std::size_t pixel_count = pixels.size() / 4;
    result.hasAlphaChannel = true;

    std::size_t opaque_count = 0;

    for (std::size_t i = 0; i < pixel_count; ++i) {
        const auto alpha = static_cast<uint8_t>(pixels[i * 4 + 3]);
        if (alpha > 0 && alpha < 255) result.hasFractionalAlpha = true;
        if (alpha == 0) result.hasZeroAlpha = true;
        if (alpha >= 254) ++opaque_count;
    }

    result.opaqueCoverage = static_cast<float>(opaque_count) / static_cast<float>(pixel_count);
    return result;
}

[[nodiscard]] bool LoadKtxTextureFromBuffer(const std::filesystem::path& path,
                                             const FileLoader::ByteBuffer& buffer,
                                             TextureData& out,
                                             std::string* error_message) {
    out.Reset();

    if (buffer.empty()) {
        if (error_message != nullptr) {
            *error_message = "Empty texture buffer: " + path.string();
        }
        return false;
    }

    ktxTexture* raw_texture = nullptr;
    const auto* bytes = reinterpret_cast<const ktx_uint8_t*>(buffer.data());
    const auto size = static_cast<ktx_size_t>(buffer.size());
    const KTX_error_code create_result = ktxTexture_CreateFromMemory(bytes, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw_texture);
    if (create_result != KTX_SUCCESS || raw_texture == nullptr) {
        if (error_message != nullptr) {
            *error_message = std::string("Failed to parse KTX texture '") + path.string() + "': " + ktxErrorString(create_result);
        }
        return false;
    }

    const KtxTexturePtr texture(raw_texture);

    if (texture->classId == ktxTexture2_c) {
        if (!OrientationSupported(texture.get(), std::max<std::uint32_t>(texture->baseDepth, 1U), error_message)) {
            return false;
        }
        out.swizzle = ParseSwizzle(texture.get());
        return LoadKtx2Texture(texture.get(), buffer, out, error_message);
    }

    if (texture->classId == ktxTexture1_c) {
        if (!OrientationSupported(texture.get(), 1U, error_message)) {
            return false;
        }
        return LoadKtx1Texture(texture.get(), out, error_message);
    }

    if (error_message != nullptr) {
        *error_message = "Unknown KTX texture class";
    }
    return false;
}

[[nodiscard]] bool LoadStbTextureFromBuffer(const std::filesystem::path& path,
                                             const FileLoader::ByteBuffer& buffer,
                                             TextureData& out,
                                             std::string* error_message) {
    out.Reset();

    if (buffer.empty()) {
        if (error_message != nullptr) {
            *error_message = "Empty texture buffer: " + path.string();
        }
        return false;
    }

    int width = 0;
    int height = 0;
    int components = 0;
    auto* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(buffer.data()),
                                         static_cast<int>(buffer.size()),
                                         &width,
                                         &height,
                                         &components,
                                         4);
    if (pixels == nullptr) {
        if (error_message != nullptr) {
            const char* reason = stbi_failure_reason();
            *error_message = std::string("Failed to decode image '") + path.string() + "'" + (reason != nullptr ? ": " + std::string(reason) : std::string{});
        }
        return false;
    }

    out.width = static_cast<std::uint32_t>(width);
    out.height = static_cast<std::uint32_t>(height);
    out.source_format = vk::Format::eR8G8B8A8Unorm;
    out.source_channels = static_cast<std::uint8_t>(components > 0 ? components : 4);
    out.source_alpha = (out.source_channels == 2 || out.source_channels == 4) ? 1U : 0U;

    AppendCpuMipChain(out.width, out.height, out);
    std::memcpy(out.blob.data(), pixels,
                static_cast<std::size_t>(out.width) * out.height * 4U);
    stbi_image_free(pixels);

    // Downsample the chain level by level from the resident base level.
    for (std::uint32_t level = 1; level < out.mip_levels; ++level) {
        const auto& src = out.subresources[level - 1];
        auto& dst = out.subresources[level];
        BoxDownsampleRgba8(out.blob.data() + src.offset, src.width, src.height,
                           out.blob.data() + dst.offset, dst.width, dst.height);
    }

    out.alpha_analysis = AnalyzeAlpha(out.blob);
    return true;
}

[[nodiscard]] bool LoadTextureFromBuffer(const std::filesystem::path& path,
                                          const FileLoader::ByteBuffer& buffer,
                                          TextureData& out,
                                          std::string* error_message) {
    const auto lower_ext = ToLowerCopy(path.extension().string());

    if (lower_ext == ".ktx" || lower_ext == ".ktx2") {
        return LoadKtxTextureFromBuffer(path, buffer, out, error_message);
    }
    if (lower_ext == ".png" || lower_ext == ".jpg" || lower_ext == ".jpeg") {
        return LoadStbTextureFromBuffer(path, buffer, out, error_message);
    }

    return LoadByMagic(path, buffer, out, error_message);
}

[[nodiscard]] bool TranscodeBasisToTarget(TextureData& data,
                                           vk::Format target_format,
                                           std::string* error_message) {
    if (!data.needs_transcode || data.blob.empty()) {
        if (error_message != nullptr) {
            *error_message = "TranscodeBasisToTarget requires a Basis KTX2 payload";
        }
        return false;
    }

    ktxTexture* raw_texture = nullptr;
    const auto* bytes = reinterpret_cast<const ktx_uint8_t*>(data.blob.data());
    const auto size = static_cast<ktx_size_t>(data.blob.size());
    // LOAD_IMAGE_DATA_BIT inflates Zstd/ZLIB; BasisLZ is handled inside the
    // transcode call itself.
    const KTX_error_code create_result =
        ktxTexture_CreateFromMemory(bytes, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw_texture);
    if (create_result != KTX_SUCCESS || raw_texture == nullptr) {
        if (error_message != nullptr) {
            *error_message = std::string("Failed to re-parse KTX2 Basis payload: ") + ktxErrorString(create_result);
        }
        return false;
    }
    const KtxTexturePtr texture(raw_texture);
    auto* texture2 = reinterpret_cast<ktxTexture2*>(texture.get());

    ktx_transcode_fmt_e target = KTX_TTF_RGBA32;
    switch (target_format) {
        case vk::Format::eBc7UnormBlock:
        case vk::Format::eBc7SrgbBlock: target = KTX_TTF_BC7_RGBA; break;
        case vk::Format::eBc3UnormBlock:
        case vk::Format::eBc3SrgbBlock: target = KTX_TTF_BC3_RGBA; break;
        case vk::Format::eBc1RgbUnormBlock:
        case vk::Format::eBc1RgbSrgbBlock: target = KTX_TTF_BC1_RGB; break;
        case vk::Format::eBc1RgbaUnormBlock:
        case vk::Format::eBc1RgbaSrgbBlock: target = KTX_TTF_BC1_OR_3; break;
        case vk::Format::eBc4UnormBlock: target = KTX_TTF_BC4_R; break;
        case vk::Format::eBc5UnormBlock: target = KTX_TTF_BC5_RG; break;
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb: target = KTX_TTF_RGBA32; break;
        default:
            if (error_message != nullptr) {
                *error_message = "Unsupported Basis transcode target";
            }
            return false;
    }

    const KTX_error_code transcode_result = ktxTexture2_TranscodeBasis(texture2, target, 0);
    if (transcode_result != KTX_SUCCESS) {
        if (error_message != nullptr) {
            *error_message = std::string("Failed to transcode KTX2 texture: ") + ktxErrorString(transcode_result);
        }
        return false;
    }

    data.Reset();
    data.width = texture->baseWidth;
    data.height = texture->baseHeight;
    data.depth = std::max<std::uint32_t>(texture->baseDepth, 1U);
    data.mip_levels = texture->numLevels;
    data.array_layers = std::max<std::uint32_t>(texture->numLayers, 1U);
    data.face_count = texture->numFaces;
    data.source_format = target_format;
    data.needs_transcode = false;

    const ktx_size_t data_size = ktxTexture_GetDataSize(texture.get());
    const auto* src = reinterpret_cast<const std::byte*>(ktxTexture_GetData(texture.get()));
    if (src == nullptr || data_size == 0) {
        if (error_message != nullptr) {
            *error_message = "Transcoded KTX2 texture contains no pixel data";
        }
        return false;
    }
    data.blob.assign(src, src + data_size);
    return BuildKtxSubresources(texture.get(), data, static_cast<std::size_t>(data_size), error_message);
}

bool DecodeAstcToRgba8(TextureData& data, std::string* error_message) {
    if (data.needs_transcode || data.blob.empty()) {
        if (error_message != nullptr) {
            *error_message = "DecodeAstcToRgba8 requires a native (non-Basis) KTX2 payload";
        }
        return false;
    }

    ktxTexture* raw_texture = nullptr;
    const auto* bytes = reinterpret_cast<const ktx_uint8_t*>(data.blob.data());
    const auto size = static_cast<ktx_size_t>(data.blob.size());
    // LOAD_IMAGE_DATA_BIT inflates Zstd/ZLIB supercompression; ASTC blocks are
    // not supercompression, they are the sample format itself.
    const KTX_error_code create_result =
        ktxTexture_CreateFromMemory(bytes, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw_texture);
    if (create_result != KTX_SUCCESS || raw_texture == nullptr) {
        if (error_message != nullptr) {
            *error_message = std::string("Failed to re-parse KTX2 ASTC payload: ") + ktxErrorString(create_result);
        }
        return false;
    }
    const KtxTexturePtr texture(raw_texture);
    auto* texture2 = reinterpret_cast<ktxTexture2*>(texture.get());

    const KTX_error_code decode_result = ktxTexture2_DecodeAstc(texture2);
    if (decode_result != KTX_SUCCESS) {
        if (error_message != nullptr) {
            *error_message = std::string("Failed to CPU-decode KTX2 ASTC payload: ") + ktxErrorString(decode_result);
        }
        return false;
    }

    data.Reset();
    data.width = texture->baseWidth;
    data.height = texture->baseHeight;
    data.depth = std::max<std::uint32_t>(texture->baseDepth, 1U);
    data.mip_levels = texture->numLevels;
    data.array_layers = std::max<std::uint32_t>(texture->numLayers, 1U);
    data.face_count = texture->numFaces;
    // The decoder picked the uncompressed variant from the DFD (sRGB transfer
    // -> R8G8B8A8_SRGB, linear -> UNORM, float -> RGBA32F); report it verbatim.
    data.source_format = static_cast<vk::Format>(texture2->vkFormat);
    data.needs_transcode = false;

    const ktx_size_t data_size = ktxTexture_GetDataSize(texture.get());
    const auto* src = reinterpret_cast<const std::byte*>(ktxTexture_GetData(texture.get()));
    if (src == nullptr || data_size == 0) {
        if (error_message != nullptr) {
            *error_message = "Decoded KTX2 texture contains no pixel data";
        }
        return false;
    }
    data.blob.assign(src, src + data_size);
    if (!BuildKtxSubresources(texture.get(), data, static_cast<std::size_t>(data_size), error_message)) {
        return false;
    }
    // The payload is now uncompressed RGBA8: capture the same alpha statistics
    // the loader records for native RGBA8 payloads (material validation).
    const vk::Format fmt = data.source_format;
    if (fmt == vk::Format::eR8G8B8A8Unorm || fmt == vk::Format::eR8G8B8A8Srgb) {
        const auto& base = data.subresources.front();
        data.alpha_analysis = AnalyzeAlpha(std::vector<std::byte>(
            data.blob.begin() + static_cast<std::ptrdiff_t>(base.offset),
            data.blob.begin() + static_cast<std::ptrdiff_t>(base.offset + base.size)));
    }
    return true;
}

}  // namespace VulkanEngine::FileLoaders::Textures
