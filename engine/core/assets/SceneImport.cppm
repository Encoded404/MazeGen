module;

export module VulkanEngine.SceneLoader;

import std;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanEngine.ResourceSystem;
export import VulkanEngine.ResourceSystem.TextureResource;
export import VulkanEngine.FileLoaders.Mesh.MeshMagicLoader;
export import VulkanEngine.GpuResources;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.Mesh.MeshTypes;
export import VulkanEngine.MaterialManager.MaterialId;

import VulkanEngine.GpuResources.MeshData;

#ifndef UINT32_MAX
constexpr std::uint32_t UINT32_MAX =
    std::numeric_limits<std::uint32_t>::max();
#endif

export namespace VulkanEngine::SceneLoader {

using MaterialManager::MaterialId;

struct LoadedMeshData {
    std::vector<float> positions;
    std::vector<float> normals;
    // UV coordinates in the engine's orientation: the importer's V is already
    // flipped. Every set shares that convention — `uvs` as well as `uv1` — so
    // consumers copy them verbatim rather than flipping per set.
    std::vector<float> uvs;
    // Optional out-of-line UV1, interleaved (u,v) like `uvs`, parallel to
    // `positions`, and flipped the same way. Empty when the source has no UV1.
    std::vector<float> uv1;
    // MikkTSpace tangents (xyz, unnormalized ok) + handedness (+1/-1),
    // per-vertex, parallel to `normals`. May be empty when the mesh has no
    // UVs — packedTBN then encodes a zero tangent (no tangent-space normal
    // mapping for that mesh).
    std::vector<float> tangents;
    std::vector<float> tangent_handedness;
    std::vector<std::uint32_t> indices;
    // Per-submesh placement and, when the source declared one, its physics hint
    // (`SubMesh::physics`). The loader copies whole `SubMesh` values, so a
    // consumer of this result — and of `ToMeshData` below — sees the
    // `KHR_physics_rigid_bodies` collider data unchanged and can turn it into
    // `Physics::RigidBody`/`Collider` components when it builds entities.
    // Nothing on this path creates a body.
    std::vector<VulkanEngine::SubMesh> submeshes;
};

struct MeshInfo {
    std::string name;
    std::uint32_t vertex_offset = 0;
    std::uint32_t vertex_count = 0;
    std::uint32_t index_offset = 0;
    std::uint32_t index_count = 0;
    std::uint32_t first_submesh_index = 0;
    std::uint32_t submesh_count = 0;
};

struct CombinedScene {
    VulkanEngine::GpuResources::HeapAllocation vertex_allocation{};
    VulkanEngine::GpuResources::HeapAllocation index_allocation{};
    std::vector<MeshInfo> meshes{};
    std::vector<VulkanEngine::SubMesh> submeshes{};
};

[[nodiscard]] LoadedMeshData CreateFallbackQuad();

[[nodiscard]] LoadedMeshData LoadMeshFromFile(
    const std::filesystem::path& models_dir,
    const std::vector<MaterialId>* material_bindings = nullptr);

[[nodiscard]] LoadedMeshData LoadMeshFromFilePath(
    const std::filesystem::path& file_path,
    const std::vector<MaterialId>* material_bindings = nullptr);

[[nodiscard]] std::vector<VulkanEngine::StandardMeshPipeline::Vertex>
ConvertToVertices(const LoadedMeshData& mesh);

[[nodiscard]] VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>
LoadTexture(VulkanEngine::ResourceManager& resource_manager,
            const std::filesystem::path& textures_dir,
            const VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>& fallback);

[[nodiscard]] VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>
LoadTextureFromPath(VulkanEngine::ResourceManager& resource_manager,
                    const std::filesystem::path& texture_path,
                    const VulkanEngine::ResourceHandle<VulkanEngine::TextureResource>& fallback);

[[nodiscard]] bool LoadAllMeshes(const std::filesystem::path& models_dir,
                                 std::vector<LoadedMeshData>& out_meshes,
                                 std::vector<std::string>& out_names,
                                 const std::vector<MaterialId>* material_bindings = nullptr);

[[nodiscard]] CombinedScene UploadCombined(
    VulkanBackend::Vulkan::VulkanBootstrap& bootstrap,
    VulkanEngine::GpuResources::StagingPool& staging_mgr,
    VulkanEngine::GpuResources::DeviceBufferHeap& vertex_heap,
    VulkanEngine::GpuResources::DeviceBufferHeap& index_heap,
    const std::vector<LoadedMeshData>& meshes);

[[nodiscard]] VulkanEngine::GpuResources::MeshData LoadMeshData(
    const std::filesystem::path& file_path,
    const std::vector<MaterialId>* material_bindings = nullptr);

[[nodiscard]] VulkanEngine::GpuResources::MeshData ToMeshData(
    const LoadedMeshData& loaded);

}
