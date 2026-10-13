module;

#include <filesystem>
#include <logging/logging_macros.hpp>

module VulkanEngine.SceneLoader;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanEngine.Components.Transform;
import VulkanEngine.Mesh.MeshTypes;
import VulkanEngine.Mesh.NormalEncoding;
import VulkanEngine.GpuResources;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.StandardMeshPipeline;

namespace VulkanEngine::SceneLoader {

namespace {
    // Local alias for the NormalEncoding helpers' 3D vector type.
    using float3 = MeshVertexVec3;

    [[nodiscard]] std::filesystem::path FindFirstFileWithExtension(const std::filesystem::path& dir,
                                                                    const std::vector<std::string>& extensions) {
        if (!std::filesystem::exists(dir)) {
            LOGIFACE_LOG(debug, "Directory does not exist: " + dir.string());
            return {};
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string ext = entry.path().extension().string();
            if (std::ranges::find(extensions, ext) != extensions.end()) {
                return entry.path();
            }
        }
        return {};
    }

    [[nodiscard]] std::vector<std::filesystem::path> FindAllFilesWithExtensions(
        const std::filesystem::path& dir, const std::vector<std::string>& extensions) {
        std::vector<std::filesystem::path> result;
        if (!std::filesystem::exists(dir)) return result;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string ext = entry.path().extension().string();
            if (std::ranges::find(extensions, ext) != extensions.end()) {
                result.push_back(entry.path());
            }
        }
        return result;
    }

}

LoadedMeshData CreateFallbackQuad() {
    LOGIFACE_LOG(trace, "CreateFallbackQuad: 4 vertices, 6 indices, 1 submesh");
    LoadedMeshData data{
        .positions = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f },
        .normals = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f },
        .uvs = { 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f },
        .tangents = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f },
        .tangent_handedness = { 1.0f, 1.0f, 1.0f, 1.0f },
        .indices = { 0, 1, 2, 2, 3, 0 },
    };
    data.submeshes.push_back(SubMesh{
        .index_start = 0,
        .index_count = 6,
        .material_id = MaterialId{0},
        .sphere = BoundingSphere{{0.0f, 0.0f, 0.0f}, 1.41421356f},  // unit quad: half-diagonal = sqrt(2)
        .obb = BoundingOBB{
            {0.0f, 0.0f, 0.0f}, 0.0f,
            {1.0f, 0.0f, 0.0f}, 1.0f,
            {0.0f, 1.0f, 0.0f}, 1.0f,
            {0.0f, 0.0f, 1.0f}, 0.0f,  // flat quad: zero z extent
        },
    });
    return data;
}

LoadedMeshData LoadMeshFromFile(const std::filesystem::path& models_dir,
                                              const std::vector<MaterialId>* material_bindings) {
    const auto mesh_path = FindFirstFileWithExtension(models_dir, VulkanEngine::FileLoaders::Mesh::KnownMeshExtensions());
    if (!mesh_path.empty()) {
        try {
            auto mesh = VulkanEngine::FileLoaders::Mesh::LoadMeshFromFile(mesh_path, material_bindings).get();
            if (mesh) {
                LOGIFACE_LOG(trace, "Loaded mesh from dir: " + mesh_path.string() + " -> " +
                             std::to_string(mesh->vertices.size()) + " verts, " +
                             std::to_string(mesh->indices.size()) + " indices, " +
                             std::to_string(mesh->subMeshes.size()) + " submeshes");
                LoadedMeshData data{};
                for (const auto& v : mesh->vertices) { data.positions.push_back(v.x); data.positions.push_back(v.y); data.positions.push_back(v.z); }
                for (const auto& n : mesh->normals) { data.normals.push_back(n.x); data.normals.push_back(n.y); data.normals.push_back(n.z); }
                for (const auto& uv : mesh->uvs) { data.uvs.push_back(uv.u); data.uvs.push_back(1.0f - uv.v); }
                for (const auto& t : mesh->tangents) { data.tangents.push_back(t.x); data.tangents.push_back(t.y); data.tangents.push_back(t.z); }
                data.tangent_handedness = mesh->tangent_handedness;
                data.indices = mesh->indices;
                data.submeshes = mesh->subMeshes;
                return data;
            }
        } catch (const std::exception& e) {
            LOGIFACE_LOG(error, "got error in function LoadMeshFromFile: " + std::string(e.what()));
        }
    }
    LOGIFACE_LOG(warn, "No mesh files found in dir: " + models_dir.string() + " - using fallback quad");
    return CreateFallbackQuad();
}

LoadedMeshData LoadMeshFromFilePath(const std::filesystem::path& file_path,
                                                    const std::vector<MaterialId>* material_bindings) {
    try {
        auto mesh = VulkanEngine::FileLoaders::Mesh::LoadMeshFromFile(file_path, material_bindings).get();
        if (mesh) {
            LOGIFACE_LOG(debug, "Loaded mesh: " + file_path.string() + " -> " +
                         std::to_string(mesh->vertices.size()) + " verts, " +
                         std::to_string(mesh->indices.size()) + " indices, " +
                         std::to_string(mesh->subMeshes.size()) + " submeshes");
            LoadedMeshData data{};
            for (const auto& v : mesh->vertices) { data.positions.push_back(v.x); data.positions.push_back(v.y); data.positions.push_back(v.z); }
            for (const auto& n : mesh->normals) { data.normals.push_back(n.x); data.normals.push_back(n.y); data.normals.push_back(n.z); }
            for (const auto& uv : mesh->uvs) { data.uvs.push_back(uv.u); data.uvs.push_back(1.0f - uv.v); }
            for (const auto& t : mesh->tangents) { data.tangents.push_back(t.x); data.tangents.push_back(t.y); data.tangents.push_back(t.z); }
            data.tangent_handedness = mesh->tangent_handedness;
            data.indices = mesh->indices;
            data.submeshes = mesh->subMeshes;
            return data;
        }
    } catch (const std::exception& e) {
        LOGIFACE_LOG(error, "Failed to load mesh: " + file_path.string() + " - " + std::string(e.what()));
    }
    LOGIFACE_LOG(warn, "Mesh loading failed for: " + file_path.string() + " - using fallback quad");
    return CreateFallbackQuad();
}

VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>
LoadTextureFromPath(VulkanEngine::ResourceManager& resource_manager,
                                  const std::filesystem::path& texture_path,
                                  const VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>& fallback) {
    if (texture_path.empty()) return fallback;
    auto handle = resource_manager.LoadFromFile<VulkanEngine::TextureResource>(texture_path);
    return (handle.IsValid() && handle->IsLoaded()) ? handle : fallback;
}

bool LoadAllMeshes(const std::filesystem::path& models_dir,
                                  std::vector<LoadedMeshData>& out_meshes,
                                  std::vector<std::string>& out_names,
                                  const std::vector<MaterialId>* material_bindings) {
    const auto mesh_files = FindAllFilesWithExtensions(models_dir, VulkanEngine::FileLoaders::Mesh::KnownMeshExtensions());
    if (mesh_files.empty()) {
        LOGIFACE_LOG(warn, "No supported mesh files found in: " + models_dir.string());
        return false;
    }

    for (const auto& path : mesh_files) {
        try {
            auto mesh = VulkanEngine::FileLoaders::Mesh::LoadMeshFromFile(path, material_bindings).get();
            if (mesh) {
                LoadedMeshData data{};
                for (const auto& v : mesh->vertices) { data.positions.push_back(v.x); data.positions.push_back(v.y); data.positions.push_back(v.z); }
                for (const auto& n : mesh->normals) { data.normals.push_back(n.x); data.normals.push_back(n.y); data.normals.push_back(n.z); }
                for (const auto& uv : mesh->uvs) { data.uvs.push_back(uv.u); data.uvs.push_back(1.0f - uv.v); }
                for (const auto& t : mesh->tangents) { data.tangents.push_back(t.x); data.tangents.push_back(t.y); data.tangents.push_back(t.z); }
                data.tangent_handedness = mesh->tangent_handedness;
                data.indices = mesh->indices;
                data.submeshes = mesh->subMeshes;
                out_meshes.push_back(std::move(data));
                out_names.push_back(path.stem().string());
            }
        } catch (const std::exception& e) {
            LOGIFACE_LOG(error, "Failed to load mesh: " + path.string() + " - " + std::string(e.what()));
        }
    }
    return !out_meshes.empty();
}

VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>
LoadTexture(VulkanEngine::ResourceManager& resource_manager,
                          const std::filesystem::path& textures_dir,
                          const VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>& fallback) {
    const auto texture_path = FindFirstFileWithExtension(textures_dir, {".ktx2", ".ktx", ".png", ".jpg", ".jpeg"});
    if (texture_path.empty()) return fallback;
    auto handle = resource_manager.LoadFromFile<VulkanEngine::TextureResource>(texture_path);
    return (handle.IsValid() && handle->IsLoaded()) ? handle : fallback;
}

std::vector<VulkanEngine::StandardMeshPipeline::Vertex>
ConvertToVertices(const LoadedMeshData& mesh) {
    std::vector<VulkanEngine::StandardMeshPipeline::Vertex> vertices{};
    const std::size_t vertex_count = mesh.positions.size() / 3U;
    if (vertex_count == 0U) return vertices;
    vertices.resize(vertex_count);
    const bool has_tangents = mesh.tangents.size() >= mesh.positions.size()
                           && mesh.tangent_handedness.size() >= vertex_count;
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const std::size_t p = i * 3U, t = i * 2U;
        vertices[i].px = mesh.positions[p + 0];
        vertices[i].py = mesh.positions[p + 1];
        vertices[i].pz = mesh.positions[p + 2];
        float3 normal{0.0f, 0.0f, 0.0f};
        if (mesh.normals.size() > p + 2) {
            normal.x = mesh.normals[p + 0];
            normal.y = mesh.normals[p + 1];
            normal.z = mesh.normals[p + 2];
        } else {
            // Degenerate geometry without normals: give the packer a unit
            // vector so the octahedral projection stays well-defined.
            normal.y = 1.0f;
        }
        // Zero tangent = no tangent space; PackTBN still encodes a valid
        // frame (basis tangent fallback) which the shader-side normal-map
        // gating (normal_texture == 0) simply never uses.
        float3 tangent{0.0f, 0.0f, 0.0f};
        float handedness = 1.0f;
        if (has_tangents) {
            tangent.x = mesh.tangents[p + 0];
            tangent.y = mesh.tangents[p + 1];
            tangent.z = mesh.tangents[p + 2];
            handedness = mesh.tangent_handedness[i];
        }
        vertices[i].packedTBN = VulkanEngine::NormalEncoding::PackTBN(normal, tangent, handedness);
        vertices[i].u = (t + 1 < mesh.uvs.size()) ? mesh.uvs[t + 0] : 0.0f;
        vertices[i].v = (t + 1 < mesh.uvs.size()) ? mesh.uvs[t + 1] : 0.0f;
    }
    return vertices;
}

CombinedScene UploadCombined(
    VulkanBackend::Vulkan::VulkanBootstrap& /*bootstrap*/,
    VulkanEngine::GpuResources::StagingPool& staging_mgr,
    VulkanEngine::GpuResources::DeviceBufferHeap& vertex_heap,
    VulkanEngine::GpuResources::DeviceBufferHeap& index_heap,
    const std::vector<LoadedMeshData>& meshes) {
    CombinedScene scene;

    // Convert all meshes to vertices and combine
    std::vector<VulkanEngine::StandardMeshPipeline::Vertex> all_vertices;
    std::vector<std::uint32_t> all_indices;
    std::uint32_t vertex_offset = 0;
    std::uint32_t index_offset = 0;

    std::vector<VulkanEngine::SubMesh> all_submeshes;

    for (std::size_t i = 0; i < meshes.size(); ++i) {
        auto verts = ConvertToVertices(meshes[i]);

        MeshInfo info{};
        info.name = "mesh_" + std::to_string(i);
        info.vertex_offset = vertex_offset;
        info.vertex_count = static_cast<std::uint32_t>(verts.size());
        info.index_offset = index_offset;
        info.index_count = static_cast<std::uint32_t>(meshes[i].indices.size());

        LOGIFACE_LOG(trace, "UploadCombined mesh[" + std::to_string(i) + "]: name=" + info.name +
                     " vertices=" + std::to_string(info.vertex_count) +
                     " indices=" + std::to_string(info.index_count) +
                     " submeshes=" + std::to_string(meshes[i].submeshes.size()));

        info.first_submesh_index = static_cast<std::uint32_t>(all_submeshes.size());
        info.submesh_count = static_cast<std::uint32_t>(meshes[i].submeshes.size());
        if (info.submesh_count == 0) {
            VulkanEngine::SubMesh default_sm{};
            default_sm.index_start = index_offset;
            default_sm.index_count = info.index_count;
            all_submeshes.push_back(default_sm);
            info.submesh_count = 1;
        } else {
            for (const auto& sm : meshes[i].submeshes) {
                VulkanEngine::SubMesh adjusted = sm;
                adjusted.index_start += index_offset;
                all_submeshes.push_back(adjusted);
            }
        }

        scene.meshes.push_back(info);

        all_vertices.insert(all_vertices.end(), verts.begin(), verts.end());

        for (const std::uint32_t idx : meshes[i].indices) {
            all_indices.push_back(idx + vertex_offset);
        }

        vertex_offset += static_cast<std::uint32_t>(verts.size());
        index_offset += static_cast<std::uint32_t>(meshes[i].indices.size());
    }

    scene.submeshes = std::move(all_submeshes);

    LOGIFACE_LOG(trace, "UploadCombined: total submeshes=" + std::to_string(scene.submeshes.size()) +
                 " total_indices=" + std::to_string(all_indices.size()) +
                 " total_vertices=" + std::to_string(all_vertices.size()));
    for (std::size_t i = 0; i < scene.submeshes.size(); ++i) {
        const auto& sm = scene.submeshes[i];
        LOGIFACE_LOG(trace, "  submesh[" + std::to_string(i) + "]: index_start=" +
                     std::to_string(sm.index_start) + " index_count=" +
                     std::to_string(sm.index_count) + " material_id=" +
                     std::to_string(sm.material_id.value));
    }

    if (all_vertices.empty()) return scene;

    const std::uint64_t vertex_data_size = all_vertices.size() * sizeof(VulkanEngine::StandardMeshPipeline::Vertex);
    const std::uint64_t index_data_size = all_indices.size() * sizeof(std::uint32_t);

    // Allocate from heaps
    // Vertex allocations must be aligned to a multiple of sizeof(Vertex) (24):
    // baseVertex is computed as offset / sizeof(Vertex) with integer division
    // (expand.slang), so a non-multiple offset truncates and every vertex of
    // the mesh reads shifted bytes. 240 = LCM(24, 16).
    constexpr std::uint64_t VERTEX_ALIGNMENT = 240;
    scene.vertex_allocation = vertex_heap.Allocate(vertex_data_size, VERTEX_ALIGNMENT);
    if (scene.vertex_allocation.buffer_index == UINT32_MAX) {
        LOGIFACE_LOG(error, "UploadCombined: vertex heap allocation failed");
        return scene;
    }

    // Pack indices with buffer index into the top 8 bits
    std::vector<std::uint32_t> packed_indices;
    packed_indices.reserve(all_indices.size());
    for (const std::uint32_t idx : all_indices) {
        const std::uint32_t packed = (scene.vertex_allocation.buffer_index << 24) | idx;
        packed_indices.push_back(packed);
    }

    scene.index_allocation = index_heap.Allocate(index_data_size, 4);
    if (scene.index_allocation.buffer_index == UINT32_MAX) {
        LOGIFACE_LOG(error, "UploadCombined: index heap allocation failed");
        return scene;
    }

    LOGIFACE_LOG(debug, "UploadCombined: " + std::to_string(scene.meshes.size()) + " meshes, " +
                 std::to_string(all_vertices.size()) + " vertices, " +
                 std::to_string(all_indices.size()) + " indices");

    // Upload vertex data via staging
    {
        auto slice = staging_mgr.Allocate(vertex_data_size);
        if (!slice.has_value()) {
            LOGIFACE_LOG(error, "UploadCombined: staging allocation failed for vertex data");
            return scene;
        }
        std::memcpy(slice->mapped_ptr, all_vertices.data(), vertex_data_size);

        staging_mgr.RecordBufferCopy(*slice,
            vertex_heap.GetBuffer(scene.vertex_allocation.buffer_index),
            scene.vertex_allocation.offset);
    }

    // Upload index data via staging
    {
        auto slice = staging_mgr.Allocate(index_data_size);
        if (!slice.has_value()) {
            LOGIFACE_LOG(error, "UploadCombined: staging allocation failed for index data");
            return scene;
        }
        std::memcpy(slice->mapped_ptr, packed_indices.data(), index_data_size);

        staging_mgr.RecordBufferCopy(*slice,
            index_heap.GetBuffer(scene.index_allocation.buffer_index),
            scene.index_allocation.offset);
    }

    // Flush staging and wait for completion
    staging_mgr.FlushImmediate();

    return scene;
}

VulkanEngine::GpuResources::MeshData LoadMeshData(
    const std::filesystem::path& file_path,
    const std::vector<MaterialId>* material_bindings) {
    auto loaded = LoadMeshFromFilePath(file_path, material_bindings);
    return ToMeshData(loaded);
}

VulkanEngine::GpuResources::MeshData ToMeshData(
    const LoadedMeshData& loaded) {
    VulkanEngine::GpuResources::MeshData result{};
    result.vertices = ConvertToVertices(loaded);
    result.indices = loaded.indices;
    result.sub_meshes = loaded.submeshes;
    // UV1 is parallel to the vertices (one float2 per vertex), flipped the same
    // way as UV0 so both sets share the importer's orientation.
    if (loaded.uv1.size() >= loaded.positions.size() / 3U * 2U &&
        !loaded.uv1.empty()) {
        const std::size_t n = loaded.positions.size() / 3U;
        auto& dst = result.uv_extra[0];
        dst.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i].u = loaded.uv1[i * 2U + 0U];
            dst[i].v = 1.0f - loaded.uv1[i * 2U + 1U];
        }
    }
    return result;
}

} // namespace VulkanEngine::SceneLoader
