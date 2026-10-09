module;

export module VulkanEngine.FileLoaders.Mesh.GltfMeshAssembler;

import std;

import FileLoader;
import VulkanEngine.Mesh.MeshTypes;
import VulkanEngine.FileLoaders.Mesh.MeshLoaderBase;
import VulkanEngine.MaterialManager.MaterialId;

// glTF mesh geometry import — the geometry half of the glTF path; materials,
// textures and samplers live in GltfMaterialImport.
//
// fastgltf is confined to the module implementation unit (GltfMeshAssembler.cpp),
// so a consumer — and a device-free test — imports this interface without naming
// or linking the parser.
//
// The assembler flattens the asset's scene into one engine Mesh, following the
// engine's one-file-one-Mesh contract: every mesh primitive reachable from the
// chosen scene becomes one submesh, carrying its material id and the world
// transform accumulated down the node hierarchy. That per-submesh transform is
// what lets a multi-node asset place correctly without duplicating geometry.

export namespace VulkanEngine::FileLoaders::Mesh {

class GltfMeshAssembler : public FileLoader::IAssembler<VulkanEngine::Mesh, FileLoader::AssemblyMode::FullBuffer>
{
public:
    // Asset material index -> engine MaterialId, matching the OBJ/BIN path.
    // The importer never names a technique; the app supplies the table.
    void SetMaterialBindings(const std::vector<VulkanEngine::MaterialId>* bindings) {
        material_bindings_ = bindings;
    }

    // The asset path. Required: external buffers and images resolve relative to
    // the asset's own directory, and a .gltf has no other way to find them.
    void SetSourcePath(std::filesystem::path path) {
        source_path_ = std::move(path);
    }

    std::future<std::shared_ptr<VulkanEngine::Mesh>> AssembleFromFullBuffer(std::shared_ptr<FileLoader::ByteBuffer> buffer) override;

private:
    const std::vector<VulkanEngine::MaterialId>* material_bindings_ = nullptr;
    std::filesystem::path source_path_{};
};

class GltfMeshLoader : public IMeshLoader {
public:
    GltfMeshLoader() = default;

protected:
    std::shared_ptr<VulkanEngine::Mesh> DoLoad(const std::filesystem::path& path) override;

private:
    static std::vector<std::byte> ReadEntireFile(const std::filesystem::path& file_path);
};

} // namespace VulkanEngine::FileLoaders::Mesh
