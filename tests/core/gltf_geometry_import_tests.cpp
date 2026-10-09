#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

import std;

import FileLoader.Types;

import VulkanEngine.Mesh.MeshTypes;
import VulkanEngine.MaterialManager.MaterialId;
import VulkanEngine.FileLoaders.Mesh.GltfMeshAssembler;

namespace {

using VulkanEngine::Mesh;
using VulkanEngine::MeshVertexVec2;
using VulkanEngine::MeshVertexVec3;
using VulkanEngine::SubMesh;
using VulkanEngine::FileLoaders::Mesh::GltfMeshAssembler;
using VulkanEngine::MaterialManager::MaterialId;

constexpr float kEps = 1e-5f;

// ── GLB container ──────────────────────────────────────────────────────────

void AppendU32(std::vector<std::byte>& out, const std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFFu));
}

void AppendFloat(std::vector<std::byte>& out, const float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    AppendU32(out, bits);
}

void AppendU16(std::vector<std::byte>& out, const std::uint16_t v) {
    out.push_back(static_cast<std::byte>(v & 0xFFu));
    out.push_back(static_cast<std::byte>((v >> 8) & 0xFFu));
}

[[nodiscard]] std::size_t Pad4(const std::size_t n) { return (4u - (n % 4u)) % 4u; }

// A minimal GLB: 12-byte header, one JSON chunk, one BIN chunk. The JSON is
// padded with spaces and the BIN with zeros, as the spec requires.
[[nodiscard]] std::vector<std::byte> MakeGlb(const std::string& json,
                                             const std::vector<std::byte>& bin) {
    std::vector<std::byte> json_chunk;
    json_chunk.reserve(json.size() + 4);
    for (const char c : json) json_chunk.push_back(static_cast<std::byte>(c));
    json_chunk.insert(json_chunk.end(), Pad4(json_chunk.size()), std::byte{0x20});

    std::vector<std::byte> bin_chunk = bin;
    bin_chunk.insert(bin_chunk.end(), Pad4(bin_chunk.size()), std::byte{0x00});

    std::vector<std::byte> out;
    out.push_back(std::byte{'g'});
    out.push_back(std::byte{'l'});
    out.push_back(std::byte{'T'});
    out.push_back(std::byte{'F'});
    AppendU32(out, 2u);
    AppendU32(out, static_cast<std::uint32_t>(
                       12u + 8u + json_chunk.size() + 8u + bin_chunk.size()));
    AppendU32(out, static_cast<std::uint32_t>(json_chunk.size()));
    AppendU32(out, 0x4E4F534Au); // 'JSON'
    out.insert(out.end(), json_chunk.begin(), json_chunk.end());
    AppendU32(out, static_cast<std::uint32_t>(bin_chunk.size()));
    AppendU32(out, 0x004E4942u); // 'BIN\0'
    out.insert(out.end(), bin_chunk.begin(), bin_chunk.end());
    return out;
}

// ── fixture plumbing ───────────────────────────────────────────────────────

class GltfGeometryImportTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "vkengine_gltf_geometry_tests";
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path WriteFixture(const std::string& name,
                                       const std::span<const std::byte> bytes) {
        const auto path = dir_ / name;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        file.close();
        return path;
    }

    // Mirrors MeshMagicLoader: the bytes go in, the asset's own path comes along
    // so external buffers and images resolve.
    static std::shared_ptr<Mesh> Assemble(const std::filesystem::path& path,
                                          const std::vector<MaterialId>* bindings = nullptr) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        EXPECT_TRUE(file.is_open());
        const auto size = static_cast<std::size_t>(file.tellg());
        std::vector<std::byte> bytes(size);
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        file.close();

        GltfMeshAssembler assembler;
        assembler.SetSourcePath(path);
        assembler.SetMaterialBindings(bindings);
        auto buffer = std::make_shared<FileLoader::ByteBuffer>(bytes.begin(), bytes.end());
        return assembler.AssembleFromFullBuffer(std::move(buffer)).get();
    }

    std::filesystem::path dir_{};
};

void ExpectVec3(const MeshVertexVec3& v, const float x, const float y, const float z) {
    EXPECT_NEAR(v.x, x, kEps);
    EXPECT_NEAR(v.y, y, kEps);
    EXPECT_NEAR(v.z, z, kEps);
}

// ── a two-node, two-material, two-mesh GLB ─────────────────────────────────

// BIN: mesh0 positions (36 B) | mesh0 indices (6 B) | pad | mesh1 positions
// (36 B) | mesh1 indices (6 B).
[[nodiscard]] std::vector<std::byte> TwoMeshBin() {
    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
    while (bin.size() % 4 != 0) bin.push_back(std::byte{0});
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 2.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 2.0f); AppendFloat(bin, 0.0f);
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
    return bin;
}

[[nodiscard]] std::string TwoMeshJson(const std::string& buffer_entry) {
    return std::string(R"json({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0, 1]}],
  "nodes": [
    {"mesh": 0, "translation": [1.0, 0.0, 0.0]},
    {"mesh": 1, "translation": [0.0, 2.0, 0.0]}
  ],
  "meshes": [
    {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0, "mode": 4}]},
    {"primitives": [{"attributes": {"POSITION": 2}, "indices": 3, "material": 1, "mode": 4}]}
  ],
  "materials": [{"name": "A"}, {"name": "B"}],
  "buffers": [)json") + buffer_entry + R"json(],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
    {"buffer": 0, "byteOffset": 44, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 80, "byteLength": 6, "target": 34963}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ]
})json";
}

// One primitive, one node, arbitrary attributes. `attributes_json` holds only
// the attributes object's body; `indices_accessor` is the primitive-level index
// accessor (nullopt for a non-indexed primitive).
struct SinglePrimitiveSpec {
    std::string attributes_json;                        // e.g. R"("POSITION": 0)"
    std::string accessors_json;                         // the accessors array body
    std::string buffer_views_json;                      // the bufferViews array body
    std::optional<std::uint32_t> indices_accessor{1u};
    std::string node_json = R"({"mesh": 0})";
    std::uint32_t mode = 4;
    std::uint32_t bin_length = 0;
};

[[nodiscard]] std::string SinglePrimitiveJson(const SinglePrimitiveSpec& spec) {
    const std::string indices_field = spec.indices_accessor
        ? R"(, "indices": )" + std::to_string(*spec.indices_accessor)
        : std::string{};
    return std::string(R"json({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [)json") + spec.node_json + R"json(],
  "meshes": [{"primitives": [{"attributes": {)json" + spec.attributes_json +
        R"json(}, "mode": )json" + std::to_string(spec.mode) + indices_field + R"json(}]}],
  "buffers": [{"byteLength": )json" + std::to_string(spec.bin_length) + R"json(}],
  "bufferViews": [)json" + spec.buffer_views_json + R"json(],
  "accessors": [)json" + spec.accessors_json + R"json(]
})json";
}

// ── tests ──────────────────────────────────────────────────────────────────

TEST_F(GltfGeometryImportTest, TwoNodesTwoMaterialsAndTheirTransforms) {
    const auto path = WriteFixture("two_meshes.glb",
                                   MakeGlb(TwoMeshJson(R"({"byteLength": 86})"), TwoMeshBin()));

    const std::vector<MaterialId> bindings{MaterialId{7}, MaterialId{9}};
    const std::shared_ptr<Mesh> mesh = Assemble(path, &bindings);
    ASSERT_NE(mesh, nullptr);

    EXPECT_EQ(mesh->vertices.size(), 6u);
    EXPECT_EQ(mesh->indices.size(), 6u);
    ASSERT_EQ(mesh->subMeshes.size(), 2u);

    // One submesh per (node, primitive), each resolving its own material.
    EXPECT_EQ(mesh->subMeshes[0].index_start, 0u);
    EXPECT_EQ(mesh->subMeshes[0].index_count, 3u);
    EXPECT_EQ(mesh->subMeshes[0].material_id.value, 7u);
    EXPECT_EQ(mesh->subMeshes[1].index_start, 3u);
    EXPECT_EQ(mesh->subMeshes[1].index_count, 3u);
    EXPECT_EQ(mesh->subMeshes[1].material_id.value, 9u);

    // The node's world transform is baked per submesh, which is what lets the
    // two meshes keep their own placement inside one flattened Mesh.
    ExpectVec3(mesh->subMeshes[0].transform.position, 1.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->subMeshes[1].transform.position, 0.0f, 2.0f, 0.0f);

    // No UV domain, so no tangent frame: the mesh must not carry stale tangents.
    EXPECT_TRUE(mesh->tangents.empty());
    EXPECT_TRUE(mesh->tangent_handedness.empty());
}

TEST_F(GltfGeometryImportTest, ExternalGltfBufferResolvesAgainstTheAssetDirectory) {
    const auto bin = TwoMeshBin();
    WriteFixture("external.bin", bin);

    const std::string json = TwoMeshJson(R"({"uri": "external.bin", "byteLength": 86})");
    const auto path = WriteFixture(
        "external.gltf",
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(json.data()), json.size()));

    // A .gltf is JSON, so this also pins that the loader no longer insists on a
    // GLB container.
    const std::vector<MaterialId> bindings{MaterialId{1}, MaterialId{2}};
    const std::shared_ptr<Mesh> mesh = Assemble(path, &bindings);
    ASSERT_NE(mesh, nullptr);
    EXPECT_EQ(mesh->vertices.size(), 6u);
    ASSERT_EQ(mesh->subMeshes.size(), 2u);
    EXPECT_EQ(mesh->subMeshes[1].material_id.value, 2u);
}

TEST_F(GltfGeometryImportTest, NonIndexedPrimitiveGeneratesIndices) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0)";
    spec.indices_accessor = std::nullopt;
    spec.accessors_json = R"({"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"})";
    spec.buffer_views_json = R"({"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("non_indexed.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);
    ASSERT_EQ(mesh->indices.size(), 3u);
    EXPECT_EQ(mesh->indices[0], 0u);
    EXPECT_EQ(mesh->indices[1], 1u);
    EXPECT_EQ(mesh->indices[2], 2u);
}

TEST_F(GltfGeometryImportTest, TriangleStripIsTriangulated) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0)";
    spec.mode = 5; // TRIANGLE_STRIP
    spec.accessors_json = R"(
    {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR"})";
    spec.buffer_views_json = R"(
    {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
    {"buffer": 0, "byteOffset": 48, "byteLength": 8, "target": 34963})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2); AppendU16(bin, 3);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("strip.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);

    // An even triangle keeps the source winding; an odd one swaps its first two
    // corners, so a strip [0,1,2,3] becomes (p0,p1,p2) and (p2,p1,p3). Without
    // normals the face normal is per-triangle, so corners are not shared and the
    // indices are sequential: the winding shows up in the emitted vertex order.
    ASSERT_EQ(mesh->indices.size(), 6u);
    ASSERT_EQ(mesh->vertices.size(), 6u);
    ExpectVec3(mesh->vertices[0], 0.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->vertices[1], 1.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->vertices[2], 0.0f, 1.0f, 0.0f);
    ExpectVec3(mesh->vertices[3], 0.0f, 1.0f, 0.0f);
    ExpectVec3(mesh->vertices[4], 1.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->vertices[5], 1.0f, 1.0f, 0.0f);
}

TEST_F(GltfGeometryImportTest, MissingNormalsGetFlatFaceNormals) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0)";
    spec.accessors_json = R"(
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"})";
    spec.buffer_views_json = R"(
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("flat_normals.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);

    ASSERT_EQ(mesh->normals.size(), 3u);
    // Counter-clockwise in the XY plane faces +Z, and all three corners share it.
    ExpectVec3(mesh->normals[0], 0.0f, 0.0f, 1.0f);
    ExpectVec3(mesh->normals[1], 0.0f, 0.0f, 1.0f);
    ExpectVec3(mesh->normals[2], 0.0f, 0.0f, 1.0f);
}

TEST_F(GltfGeometryImportTest, NormalizedTexcoordIsDecoded) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0, "TEXCOORD_0": 1)";
    spec.indices_accessor = 2u;
    spec.accessors_json = R"(
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "normalized": true, "count": 3, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"})";
    spec.buffer_views_json = R"(
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34962},
    {"buffer": 0, "byteOffset": 42, "byteLength": 6, "target": 34963})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    // (0, 255), (255, 0), (255, 255) as normalized bytes -> (0,1), (1,0), (1,1).
    const std::uint8_t uv[6] = {0, 255, 255, 0, 255, 255};
    for (const std::uint8_t b : uv) bin.push_back(static_cast<std::byte>(b));
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("normalized_uv.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);

    ASSERT_EQ(mesh->uvs.size(), 3u);
    EXPECT_NEAR(mesh->uvs[0].u, 0.0f, kEps);
    EXPECT_NEAR(mesh->uvs[0].v, 1.0f, kEps);
    EXPECT_NEAR(mesh->uvs[1].u, 1.0f, kEps);
    EXPECT_NEAR(mesh->uvs[1].v, 0.0f, kEps);
}

TEST_F(GltfGeometryImportTest, SecondUvSetAndAuthoredTangentsSurvive) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2, "TANGENT": 3)";
    spec.indices_accessor = 4u;
    spec.accessors_json = R"(
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"})";
    spec.buffer_views_json = R"(
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 60, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 84, "byteLength": 48, "target": 34962},
    {"buffer": 0, "byteOffset": 132, "byteLength": 6, "target": 34963})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    for (int i = 0; i < 3; ++i) { AppendFloat(bin, 0.1f * static_cast<float>(i)); AppendFloat(bin, 0.2f); }
    for (int i = 0; i < 3; ++i) { AppendFloat(bin, 0.3f); AppendFloat(bin, 0.4f * static_cast<float>(i)); }
    for (int i = 0; i < 3; ++i) {
        AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, -1.0f);
    }
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("uv1_tangents.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);

    // UV1 is parallel to the vertices and carries the asset's values; the V flip
    // happens later, on the way into MeshData.
    ASSERT_EQ(mesh->uv_extra[0].size(), 3u);
    EXPECT_NEAR(mesh->uv_extra[0][1].u, 0.3f, kEps);
    EXPECT_NEAR(mesh->uv_extra[0][2].v, 0.8f, kEps);

    // Authored tangents win over MikkTSpace, handedness included.
    ASSERT_EQ(mesh->tangents.size(), 3u);
    ASSERT_EQ(mesh->tangent_handedness.size(), 3u);
    ExpectVec3(mesh->tangents[0], 1.0f, 0.0f, 0.0f);
    EXPECT_NEAR(mesh->tangent_handedness[0], -1.0f, kEps);
}

TEST_F(GltfGeometryImportTest, UnmappedMaterialFallsBackToTheDefault) {
    const auto path = WriteFixture("two_meshes.glb",
                                   MakeGlb(TwoMeshJson(R"({"byteLength": 86})"), TwoMeshBin()));

    // No binding table at all: every submesh must land on MaterialId{0} rather
    // than an unvalidated index.
    const std::shared_ptr<Mesh> mesh = Assemble(path, nullptr);
    ASSERT_NE(mesh, nullptr);
    ASSERT_EQ(mesh->subMeshes.size(), 2u);
    EXPECT_EQ(mesh->subMeshes[0].material_id.value, 0u);
    EXPECT_EQ(mesh->subMeshes[1].material_id.value, 0u);
}

TEST_F(GltfGeometryImportTest, PointsPrimitiveIsNotRenderableGeometry) {
    SinglePrimitiveSpec spec{};
    spec.attributes_json = R"("POSITION": 0)";
    spec.mode = 0; // POINTS: the engine has no pipeline for it
    spec.indices_accessor = std::nullopt;
    spec.accessors_json = R"({"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"})";
    spec.buffer_views_json = R"({"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962})";

    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    spec.bin_length = static_cast<std::uint32_t>(bin.size());

    const auto path = WriteFixture("points.glb", MakeGlb(SinglePrimitiveJson(spec), bin));
    // A parse that yields nothing must fail loudly, not return an empty mesh that
    // renders as nothing.
    EXPECT_THROW((void)Assemble(path), std::runtime_error);
}

TEST_F(GltfGeometryImportTest, SharedMeshGeometryIsEmittedOnceAndPlacedPerNode) {
    // glTF instances a mesh by referencing it from several nodes. One triangle
    // referenced by three nodes must cost one copy of the geometry and three
    // placements, not three copies.
    const std::string json = R"json({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0, 1, 2]}],
  "nodes": [
    {"mesh": 0},
    {"mesh": 0, "translation": [1.0, 0.0, 0.0]},
    {"mesh": 0, "translation": [0.0, 2.0, 0.0]}
  ],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
  "buffers": [{"byteLength": 42}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ]
})json";
    std::vector<std::byte> bin;
    AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
    AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
    AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);

    const auto path = WriteFixture("shared_mesh.glb", MakeGlb(json, bin));
    const std::shared_ptr<Mesh> mesh = Assemble(path);
    ASSERT_NE(mesh, nullptr);

    EXPECT_EQ(mesh->vertices.size(), 3u);
    EXPECT_EQ(mesh->indices.size(), 3u);
    ASSERT_EQ(mesh->subMeshes.size(), 3u);

    // Every placement draws the same index range, over the same vertices...
    for (const SubMesh& sm : mesh->subMeshes) {
        EXPECT_EQ(sm.index_start, 0u);
        EXPECT_EQ(sm.index_count, 3u);
    }
    // ...and carries only its own node transform.
    ExpectVec3(mesh->subMeshes[0].transform.position, 0.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->subMeshes[1].transform.position, 1.0f, 0.0f, 0.0f);
    ExpectVec3(mesh->subMeshes[2].transform.position, 0.0f, 2.0f, 0.0f);
}

} // namespace
