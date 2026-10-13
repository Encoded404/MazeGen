module;

export module VulkanEngine.FileLoaders.TextureLoaders;

import std;
import std.compat;

import FileLoader.Types;

import vulkan;

import VulkanEngine.TextureTypes;

export namespace VulkanEngine::FileLoaders::Textures {

using VulkanEngine::Textures::AlphaAnalysis;
using VulkanEngine::Textures::TextureData;

// Alpha statistics over packed RGBA8 pixels (base level).
[[nodiscard]] AlphaAnalysis AnalyzeAlpha(const std::vector<std::byte>& pixels);

// Full-chain PNG/JPG: decodes to RGBA8 and emits a complete CPU mip chain
// (compressed chains come from assets; uncompressed non-KTX sources get CPU
// chains from the loader until an async decode worker owns them).
[[nodiscard]] bool LoadKtxTextureFromBuffer(const std::filesystem::path& path,
                                             const ::FileLoader::ByteBuffer& buffer,
                                             TextureData& out,
                                             std::string* error_message = nullptr);

[[nodiscard]] bool LoadStbTextureFromBuffer(const std::filesystem::path& path,
                                             const ::FileLoader::ByteBuffer& buffer,
                                             TextureData& out,
                                             std::string* error_message = nullptr);

[[nodiscard]] bool LoadTextureFromBuffer(const std::filesystem::path& path,
                                          const ::FileLoader::ByteBuffer& buffer,
                                          TextureData& out,
                                          std::string* error_message = nullptr);

// Transcodes a Basis KTX2 payload (loaded by LoadKtxTextureFromBuffer with
// needs_transcode set) to `target_format` on the caller's thread. Synchronous
// load-path step; an async decode worker owns this once it exists. The payload
// is re-parsed from data.blob, transcoded via libktx, and `data` is rewritten
// with the transcoded subresource table (needs_transcode cleared).
[[nodiscard]] bool TranscodeBasisToTarget(TextureData& data,
                                           vk::Format target_format,
                                           std::string* error_message = nullptr);

// Emits a full CPU mip chain (2x2 box filter) below the base level for an
// uncompressed, single-subresource, RGBA8 source that has none. Returns false
// when the source already has a chain or is not eligible (compressed/HDR/
// multi-layer/3D); callers treat false as "keep as-is", never as an error.
[[nodiscard]] bool GenerateCpuMipChain(TextureData& data);

}  // namespace VulkanEngine::FileLoaders::Textures
