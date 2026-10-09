#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

import std;

import VulkanEngine.SceneLoader;

namespace {

constexpr float kEps = 1e-5f;

void AppendFloat(std::vector<std::byte>& out, const float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xFFu));
}

void AppendU16(std::vector<std::byte>& out, const std::uint16_t v) {
    out.push_back(static_cast<std::byte>(v & 0xFFu));
    out.push_back(static_cast<std::byte>((v >> 8) & 0xFFu));
}

// One indexed triangle with two UV sets, written as a .gltf + external .bin so
// the whole SceneImport path (not just the glTF assembler) is exercised.
[[nodiscard]] std::string TwoUvSetJson(const std::string& bin_uri, const std::size_t bin_length) {
    return std::string(R"json({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{
    "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 3},
    "indices": 4, "mode": 4}]}],
  "buffers": [{"uri": ")json") + bin_uri + R"json(", "byteLength": )json" +
        std::to_string(bin_length) + R"json(}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 72, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 96, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 120, "byteLength": 6, "target": 34963}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ]
})json";
}

class SceneImportUvTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "vkengine_scene_import_uv_tests";
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    // Writes a triangle whose TEXCOORD_0 and TEXCOORD_1 hold the given raw
    // (u,v) pairs, flattened. Returns the .gltf path.
    std::filesystem::path WriteFixture(const std::string& name,
                                       const std::vector<float>& uv0,
                                       const std::vector<float>& uv1) {
        std::vector<std::byte> bin;
        AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
        AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f);
        AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); AppendFloat(bin, 0.0f);
        for (int i = 0; i < 3; ++i) { AppendFloat(bin, 0.0f); AppendFloat(bin, 0.0f); AppendFloat(bin, 1.0f); }
        for (const float v : uv0) AppendFloat(bin, v);
        for (const float v : uv1) AppendFloat(bin, v);
        AppendU16(bin, 0); AppendU16(bin, 1); AppendU16(bin, 2);
        while (bin.size() % 4 != 0) bin.push_back(std::byte{0});

        const std::string bin_name = name + ".bin";
        {
            std::ofstream file(dir_ / bin_name, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(bin.data()),
                       static_cast<std::streamsize>(bin.size()));
        }
        const auto path = dir_ / (name + ".gltf");
        {
            std::ofstream file(path, std::ios::trunc);
            file << TwoUvSetJson(bin_name, bin.size());
        }
        return path;
    }

    std::filesystem::path dir_{};
};

// The invariant this pins down: every UV set leaves LoadedMeshData in the same
// orientation, so UV0 (copied verbatim into the vertex) and UV1 (copied
// verbatim into uv_extra) agree, and neither is flipped a second time.
TEST_F(SceneImportUvTest, EveryUvSetLeavesImportInTheSameOrientation) {
    // Raw file values. UV0's V runs 0,0,1; UV1's runs 0,0,0.25, so the two sets
    // differ and a flip applied to only one of them is visible.
    const std::vector<float> uv0{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> uv1{0.0f, 0.0f, 0.25f, 0.0f, 0.0f, 0.25f};
    const auto path = WriteFixture("two_uv_sets", uv0, uv1);

    const auto loaded = VulkanEngine::SceneLoader::LoadMeshFromFilePath(path, nullptr);
    ASSERT_EQ(loaded.positions.size(), 9u) << "fixture fell back to the fallback quad";
    ASSERT_EQ(loaded.uvs.size(), 6u);
    ASSERT_EQ(loaded.uv1.size(), 6u);

    // Both sets arrive already flipped: v' = 1 - v.
    EXPECT_NEAR(loaded.uvs[1], 1.0f, kEps);
    EXPECT_NEAR(loaded.uvs[3], 1.0f, kEps);
    EXPECT_NEAR(loaded.uvs[5], 0.0f, kEps);
    EXPECT_NEAR(loaded.uv1[1], 1.0f, kEps);
    EXPECT_NEAR(loaded.uv1[3], 1.0f, kEps);
    EXPECT_NEAR(loaded.uv1[5], 0.75f, kEps);

    // The U component is never touched.
    EXPECT_NEAR(loaded.uv1[2], 0.25f, kEps);

    const auto data = VulkanEngine::SceneLoader::ToMeshData(loaded);
    ASSERT_EQ(data.vertices.size(), 3u);
    ASSERT_EQ(data.uv_extra[0].size(), 3u);

    // UV0 reaches the vertex unflipped by ConvertToVertices...
    EXPECT_NEAR(data.vertices[0].v, 1.0f, kEps);
    EXPECT_NEAR(data.vertices[1].v, 1.0f, kEps);
    EXPECT_NEAR(data.vertices[2].v, 0.0f, kEps);
    // ...and UV1 reaches uv_extra the same way, so the net values are unchanged
    // from when the flip lived in ToMeshData.
    EXPECT_NEAR(data.uv_extra[0][0].v, 1.0f, kEps);
    EXPECT_NEAR(data.uv_extra[0][1].v, 1.0f, kEps);
    EXPECT_NEAR(data.uv_extra[0][2].v, 0.75f, kEps);
}

// Identical source values must survive as identical sets: one convention, no
// per-set special casing.
TEST_F(SceneImportUvTest, UvSetsWithEqualSourceValuesStayEqual) {
    const std::vector<float> uv{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const auto path = WriteFixture("equal_uv_sets", uv, uv);

    const auto loaded = VulkanEngine::SceneLoader::LoadMeshFromFilePath(path, nullptr);
    ASSERT_EQ(loaded.positions.size(), 9u) << "fixture fell back to the fallback quad";
    ASSERT_EQ(loaded.uv1.size(), loaded.uvs.size());
    for (std::size_t i = 0; i < loaded.uvs.size(); ++i) {
        EXPECT_NEAR(loaded.uvs[i], loaded.uv1[i], kEps) << "component " << i;
    }

    const auto data = VulkanEngine::SceneLoader::ToMeshData(loaded);
    ASSERT_EQ(data.uv_extra[0].size(), data.vertices.size());
    for (std::size_t i = 0; i < data.vertices.size(); ++i) {
        EXPECT_NEAR(data.vertices[i].v, data.uv_extra[0][i].v, kEps) << "vertex " << i;
    }
}

} // namespace
