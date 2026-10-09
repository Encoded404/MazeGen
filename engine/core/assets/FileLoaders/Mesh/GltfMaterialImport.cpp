module;

module VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;

import std;

import vulkan;
import fastgltf;

import VulkanEngine.TextureTypes;
import VulkanEngine.MaterialManager.MaterialId;

namespace VulkanEngine::FileLoaders::Mesh {

namespace {

[[nodiscard]] GltfWrap ToWrap(const fastgltf::Wrap wrap) noexcept {
    switch (wrap) {
        case fastgltf::Wrap::ClampToEdge: return GltfWrap::ClampToEdge;
        case fastgltf::Wrap::MirroredRepeat: return GltfWrap::MirroredRepeat;
        case fastgltf::Wrap::Repeat: return GltfWrap::Repeat;
    }
    return GltfWrap::Repeat;
}

[[nodiscard]] GltfFilter ToFilter(const fastgltf::Filter filter) noexcept {
    switch (filter) {
        case fastgltf::Filter::Nearest: return GltfFilter::Nearest;
        case fastgltf::Filter::Linear: return GltfFilter::Linear;
        case fastgltf::Filter::NearestMipMapNearest: return GltfFilter::NearestMipmapNearest;
        case fastgltf::Filter::LinearMipMapNearest: return GltfFilter::LinearMipmapNearest;
        case fastgltf::Filter::NearestMipMapLinear: return GltfFilter::NearestMipmapLinear;
        case fastgltf::Filter::LinearMipMapLinear: return GltfFilter::LinearMipmapLinear;
    }
    return GltfFilter::Linear;
}

[[nodiscard]] GltfAlphaMode ToAlphaMode(const fastgltf::AlphaMode mode) noexcept {
    switch (mode) {
        case fastgltf::AlphaMode::Opaque: return GltfAlphaMode::Opaque;
        case fastgltf::AlphaMode::Mask: return GltfAlphaMode::Mask;
        case fastgltf::AlphaMode::Blend: return GltfAlphaMode::Blend;
    }
    return GltfAlphaMode::Opaque;
}

// Converts one glTF textureInfo into the engine's slot. The texture index is
// left as the glTF index; GltfImportResult::textures resolves it to an image.
[[nodiscard]] GltfTextureSlot ToSlot(const fastgltf::TextureInfo& info) {
    GltfTextureSlot slot{};
    slot.present = true;
    slot.texture_index = info.textureIndex;
    slot.tex_coord = static_cast<std::uint32_t>(info.texCoordIndex);
    if (info.transform) {
        slot.transform.present = true;
        slot.transform.offset_u = info.transform->uvOffset.x();
        slot.transform.offset_v = info.transform->uvOffset.y();
        slot.transform.scale_u = info.transform->uvScale.x();
        slot.transform.scale_v = info.transform->uvScale.y();
        slot.transform.rotation = info.transform->rotation;
        if (info.transform->texCoordIndex.has_value()) {
            slot.tex_coord = static_cast<std::uint32_t>(*info.transform->texCoordIndex);
        }
    }
    return slot;
}

template <typename OptionalTextureInfo>
[[nodiscard]] GltfTextureSlot ToNormalOrOcclusionSlot(const OptionalTextureInfo& info) {
    if (!info.has_value()) {
        return {};
    }
    return ToSlot(static_cast<const fastgltf::TextureInfo&>(*info));
}

[[nodiscard]] const char* MimeName(const fastgltf::MimeType mime) noexcept {
    switch (mime) {
        case fastgltf::MimeType::JPEG: return "image/jpeg";
        case fastgltf::MimeType::PNG: return "image/png";
        case fastgltf::MimeType::KTX2: return "image/ktx2";
        case fastgltf::MimeType::DDS: return "image/vnd-ms.dds";
        case fastgltf::MimeType::WEBP: return "image/webp";
        default: return "";
    }
}

[[nodiscard]] GltfImageRef ToImageRef(const fastgltf::Image& image) {
    GltfImageRef ref{};
    if (const auto* uri = std::get_if<fastgltf::sources::URI>(&image.data)) {
        // An external (or, rarely, un-inlined data) URI: the caller resolves it
        // against the asset directory.
        ref.source = GltfImageRef::Source::Uri;
        ref.uri = std::string(uri->uri.string());
        ref.mime_type = MimeName(uri->mimeType);
    } else if (const auto* array = std::get_if<fastgltf::sources::Array>(&image.data)) {
        // A GLB-embedded image, a `data:` URI (fastgltf decodes those itself),
        // or an image backed by a buffer view: the payload is already in memory.
        ref.source = GltfImageRef::Source::Embedded;
        ref.mime_type = MimeName(array->mimeType);
        ref.embedded_bytes.assign(array->bytes.begin(), array->bytes.end());
    } else if (std::get_if<fastgltf::sources::BufferView>(&image.data) != nullptr) {
        // fastgltf normally resolves a buffer-view image to sources::Array (the
        // GLB buffer is loaded). If it does not, the caller cannot reach the
        // bytes through this interface, so report None rather than a bad index.
        ref.source = GltfImageRef::Source::None;
    } else {
        ref.source = GltfImageRef::Source::None;
    }
    return ref;
}

} // namespace

MaterialId MapGltfMaterialIndex(const std::size_t material_index,
                                const std::vector<MaterialId>* bindings) noexcept {
    if (bindings != nullptr && material_index < bindings->size()) {
        return (*bindings)[material_index];
    }
    return MaterialId{0};
}

std::uint32_t ComputeGltfUvSets(const GltfMaterialInfo& material) noexcept {
    // Slot order matches the technique's DefaultMeshPerMaterialData::uv_sets:
    // 0=baseColor, 1=normal, 2=ORM, 3=emissive. glTF has no occlusion texCoord
    // distinct from metallicRoughness when they share a texture, so the ORM
    // slot prefers metallicRoughness and falls back to occlusion.
    const auto clamp_set = [](const std::uint32_t set) { return std::min(set, 3u); };
    std::uint32_t uv_sets = 0;
    if (material.base_color_texture.present) {
        uv_sets |= clamp_set(material.base_color_texture.tex_coord) << 0u;
    }
    if (material.normal_texture.present) {
        uv_sets |= clamp_set(material.normal_texture.tex_coord) << 2u;
    }
    if (material.metallic_roughness_texture.present) {
        uv_sets |= clamp_set(material.metallic_roughness_texture.tex_coord) << 4u;
    } else if (material.occlusion_texture.present) {
        uv_sets |= clamp_set(material.occlusion_texture.tex_coord) << 4u;
    }
    if (material.emissive_texture.present) {
        uv_sets |= clamp_set(material.emissive_texture.tex_coord) << 6u;
    }
    return uv_sets;
}

bool GltfMaterialHasTransform(const GltfMaterialInfo& material) noexcept {
    const auto has = [](const GltfTextureSlot& slot) {
        return slot.present && slot.transform.present;
    };
    return has(material.base_color_texture) || has(material.normal_texture) ||
           has(material.metallic_roughness_texture) || has(material.occlusion_texture) ||
           has(material.emissive_texture);
}

SamplerDesc ToSamplerDesc(const GltfSamplerInfo& sampler) noexcept {
    SamplerDesc desc{};
    if (!sampler.specified) {
        // glTF default sampler: linear/linear, linear mip, repeat wrap.
        desc.min_filter = vk::Filter::eLinear;
        desc.mag_filter = vk::Filter::eLinear;
        desc.mip_mode = vk::SamplerMipmapMode::eLinear;
        desc.address_u = vk::SamplerAddressMode::eRepeat;
        desc.address_v = vk::SamplerAddressMode::eRepeat;
        desc.address_w = vk::SamplerAddressMode::eRepeat;
        return desc;
    }

    const auto to_vk_filter = [](const GltfFilter filter) -> vk::Filter {
        switch (filter) {
            case GltfFilter::Nearest:
            case GltfFilter::NearestMipmapNearest:
            case GltfFilter::NearestMipmapLinear:
                return vk::Filter::eNearest;
            default:
                return vk::Filter::eLinear;
        }
    };
    const auto to_vk_mip = [](const GltfFilter filter) -> vk::SamplerMipmapMode {
        switch (filter) {
            case GltfFilter::NearestMipmapNearest:
            case GltfFilter::LinearMipmapNearest:
                return vk::SamplerMipmapMode::eNearest;
            default:
                return vk::SamplerMipmapMode::eLinear;
        }
    };
    const auto to_vk_wrap = [](const GltfWrap wrap) -> vk::SamplerAddressMode {
        switch (wrap) {
            case GltfWrap::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
            case GltfWrap::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
            case GltfWrap::Repeat: return vk::SamplerAddressMode::eRepeat;
        }
        return vk::SamplerAddressMode::eRepeat;
    };

    desc.mag_filter = to_vk_filter(sampler.mag_filter);
    desc.min_filter = to_vk_filter(sampler.min_filter);
    desc.mip_mode = to_vk_mip(sampler.min_filter);
    desc.address_u = to_vk_wrap(sampler.wrap_s);
    desc.address_v = to_vk_wrap(sampler.wrap_t);
    desc.address_w = to_vk_wrap(sampler.wrap_s);
    return desc;
}

GltfImportResult ImportGltfMaterials(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("ImportGltfMaterials: failed to open " + path.string());
    }
    const auto size = static_cast<std::size_t>(file.tellg());
    if (size == 0) {
        throw std::runtime_error("ImportGltfMaterials: empty file " + path.string());
    }
    std::vector<std::byte> bytes(size);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("ImportGltfMaterials: failed to read " + path.string());
    }

    fastgltf::Parser parser(fastgltf::Extensions::KHR_texture_transform |
                            fastgltf::Extensions::KHR_texture_basisu |
                            fastgltf::Extensions::KHR_materials_emissive_strength |
                            fastgltf::Extensions::KHR_materials_unlit);
    auto data = fastgltf::GltfDataBuffer::FromBytes(bytes.data(), bytes.size());
    if (!data) {
        throw std::runtime_error("ImportGltfMaterials: could not create GLTF data buffer");
    }

    // loadGltf auto-detects .gltf JSON vs .glb; external buffers/images are
    // resolved relative to the file's directory.
    auto asset = parser.loadGltf(data.get(), path.parent_path(),
                                 fastgltf::Options::LoadExternalBuffers |
                                 fastgltf::Options::LoadExternalImages);
    if (asset.error() != fastgltf::Error::None) {
        throw std::runtime_error("ImportGltfMaterials: parse failed for " + path.string() +
                                 ": " + std::string(fastgltf::getErrorMessage(asset.error())));
    }

    GltfImportResult result;
    result.samplers.reserve(asset->samplers.size());
    for (const auto& sampler : asset->samplers) {
        GltfSamplerInfo info{};
        info.specified = true;
        info.min_filter = sampler.minFilter.has_value() ? ToFilter(*sampler.minFilter) : GltfFilter::Linear;
        info.mag_filter = sampler.magFilter.has_value() ? ToFilter(*sampler.magFilter) : GltfFilter::Linear;
        info.wrap_s = ToWrap(sampler.wrapS);
        info.wrap_t = ToWrap(sampler.wrapT);
        result.samplers.push_back(info);
    }

    result.images.reserve(asset->images.size());
    for (const auto& image : asset->images) {
        result.images.push_back(ToImageRef(image));
    }

    result.textures.reserve(asset->textures.size());
    for (const auto& texture : asset->textures) {
        GltfTextureRef ref{};
        // KHR_texture_basisu supersedes imageIndex for a KTX2 source.
        if (texture.basisuImageIndex.has_value()) {
            ref.image_index = *texture.basisuImageIndex;
            ref.basisu = true;
            ref.valid = true;
        } else if (texture.imageIndex.has_value()) {
            ref.image_index = *texture.imageIndex;
            ref.valid = true;
        }
        if (texture.samplerIndex.has_value()) {
            ref.sampler_index = *texture.samplerIndex;
        } else {
            ref.sampler_index = kNoGltfSampler;
        }
        result.textures.push_back(ref);
    }

    result.materials.reserve(asset->materials.size());
    for (std::size_t i = 0; i < asset->materials.size(); ++i) {
        const auto& material = asset->materials[i];
        GltfMaterialInfo info{};
        info.index = i;
        info.name = std::string(material.name);
        info.unlit = material.unlit;
        info.alpha_mode = ToAlphaMode(material.alphaMode);
        info.alpha_cutoff = material.alphaCutoff;
        info.double_sided = material.doubleSided;
        info.base_color_factor[0] = material.pbrData.baseColorFactor.x();
        info.base_color_factor[1] = material.pbrData.baseColorFactor.y();
        info.base_color_factor[2] = material.pbrData.baseColorFactor.z();
        info.base_color_factor[3] = material.pbrData.baseColorFactor.w();
        info.metallic_factor = material.pbrData.metallicFactor;
        info.roughness_factor = material.pbrData.roughnessFactor;
        info.emissive_factor[0] = material.emissiveFactor.x();
        info.emissive_factor[1] = material.emissiveFactor.y();
        info.emissive_factor[2] = material.emissiveFactor.z();
        info.emissive_strength = material.emissiveStrength;
        info.normal_scale = material.normalTexture.has_value() ? material.normalTexture->scale : 1.0f;
        info.occlusion_strength =
            material.occlusionTexture.has_value() ? material.occlusionTexture->strength : 1.0f;

        if (material.pbrData.baseColorTexture.has_value()) {
            info.base_color_texture = ToSlot(*material.pbrData.baseColorTexture);
        }
        info.normal_texture = ToNormalOrOcclusionSlot(material.normalTexture);
        if (material.pbrData.metallicRoughnessTexture.has_value()) {
            info.metallic_roughness_texture = ToSlot(*material.pbrData.metallicRoughnessTexture);
        }
        info.occlusion_texture = ToNormalOrOcclusionSlot(material.occlusionTexture);
        if (material.emissiveTexture.has_value()) {
            info.emissive_texture = ToSlot(*material.emissiveTexture);
        }

        // A glTF normal map is authored RGB, so the default declared encoding is
        // Full; content that packs RG declares Standard itself at binding time.
        info.normal_encoding = info.normal_texture.present ? TextureNormalEncoding::Full
                                                           : TextureNormalEncoding::Standard;

        result.materials.push_back(info);
    }

    return result;
}

} // namespace VulkanEngine::FileLoaders::Mesh
