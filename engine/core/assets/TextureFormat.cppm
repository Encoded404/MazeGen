module;

export module VulkanEngine.TextureFormat;

import std;

import vulkan;

import VulkanEngine.TextureTypes;

export namespace VulkanEngine::Textures {

// Pure capability predicate the resolver consults; implemented by the backend
// FormatCapabilities snapshot so the resolver holds no device handle.
class FormatSupportQuery {
public:
    virtual ~FormatSupportQuery() = default;
    [[nodiscard]] virtual bool sampled(vk::Format format) const = 0;
    [[nodiscard]] virtual bool transfer_dst(vk::Format format) const = 0;
    [[nodiscard]] virtual bool linear_filter(vk::Format format) const = 0;
    [[nodiscard]] virtual bool color_attachment(vk::Format format) const = 0;

protected:
    FormatSupportQuery() = default;
    FormatSupportQuery(const FormatSupportQuery&) = default;
    FormatSupportQuery& operator=(const FormatSupportQuery&) = default;
};

struct ResolvedFormat {
    vk::Format format{vk::Format::eUndefined}; // eUndefined = resolution failed
    bool reinterpreted{false};                 // same-bit sRGB/UNORM variant switch
    bool transcoded{false};                    // Basis -> BCn/RGBA8 transcode
    bool decoded{false};                       // CPU ASTC -> RGBA8/float32 decode
};

// Color semantics: BaseColor, Emissive, UI, HDR. Data semantics: Normal, ORM,
// Mask. Unknown is a data semantic with a one-time warning at the caller.
[[nodiscard]] bool IsColorSemantic(TextureSemantic semantic) noexcept;

// Resolution rules:
//  1. Color semantic + bit-compatible _SRGB variant -> reinterpret.
//  2. _SRGB source on a data semantic -> reinterpret to _UNORM.
//  3. Color semantic with no _SRGB variant (e.g. A2B10G10R10) -> FAIL (never
//     sample linearly; caller substitutes the fallback).
//  4. Uncompressed source -> bit-depth-preserving map (8-bit RGBA8 family,
//     HDR -> RGBA16F, 1/2-channel data -> R8/R8G8).
//  5. Basis (needs_transcode) -> first device-supported BCn target from the
//     channel model; RGBA8/float16 when no BCn.
//  6. Native BCn source without textureCompressionBC -> FAIL (no runtime
//     decode).
//  6b. Native ASTC without device ASTC support -> CPU-decode via libktx 4.4
//     (ktxTexture2_DecodeAstc), targeting what that decoder produces from the
//     DFD: float32 for HDR sources; R8G8B8A8 sRGB/UNORM mirroring the payload's
//     transfer for color semantics and linear data semantics (decoded=true in
//     the result, uploader calls DecodeAstcToRgba8). sRGB-transfer data
//     semantics FAIL (the decoded bytes are gamma-encoded and no bit-compatible
//     linear variant exists to reinterpret into).
//  7. Never re-encode a native format; only same-bit sRGB/UNORM pairs.
//  8. Normal encoding constrains the format: Standard allows BC5/2-channel;
//     Full/Bent require >= 3 channels (2-channel transcode sources are forced
//     to an RGB target, native 2-channel sources FAIL).
[[nodiscard]] ResolvedFormat ResolveUploadFormat(TextureSemantic semantic,
                                                 TextureNormalEncoding normal_encoding,
                                                 vk::Format source,
                                                 bool needs_transcode,
                                                 std::uint8_t source_channels,
                                                 bool source_alpha,
                                                 bool hdr,
                                                 const FormatSupportQuery& caps);

} // namespace VulkanEngine::Textures
