module;

#include <logging/logging_macros.hpp>

module VulkanEngine.FileLoaders.Mesh.GltfMeshAssembler;

import std;

import logiface;
import fastgltf;

import VulkanEngine.Mesh.MeshTypes;
import VulkanEngine.FileLoaders.Mesh.MeshLoaderBase;
import VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;
import VulkanEngine.MaterialManager.MaterialId;
import VulkanEngine.Mesh.TangentGenerator;

namespace VulkanEngine::FileLoaders::Mesh {

namespace {

// ── fastgltf values -> engine values ────────────────────────────────────────

[[nodiscard]] MeshVertexVec3 ToVec3(const fastgltf::math::fvec3& v) {
    return MeshVertexVec3{v.x(), v.y(), v.z()};
}

[[nodiscard]] MeshVertexVec2 ToVec2(const fastgltf::math::fvec2& v) {
    return MeshVertexVec2{v.x(), v.y()};
}

[[nodiscard]] SubMeshTransform ToSubMeshTransform(const fastgltf::TRS& trs) {
    SubMeshTransform out{};
    out.position = ToVec3(trs.translation);
    out.rotation[0] = trs.rotation.x();
    out.rotation[1] = trs.rotation.y();
    out.rotation[2] = trs.rotation.z();
    out.rotation[3] = trs.rotation.w();
    out.scale = ToVec3(trs.scale);
    return out;
}

// A node's local transform as the engine's TRS. A matrix node is decomposed,
// which is exact for the shear-free matrices the glTF spec requires; anything
// else is reported by HasShear on the accumulated world matrix.
[[nodiscard]] SubMeshTransform LocalTrs(const fastgltf::Node& node) {
    if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
        return ToSubMeshTransform(*trs);
    }
    const auto& matrix = std::get<fastgltf::math::fmat4x4>(node.transform);
    fastgltf::math::fvec3 scale, translation;
    fastgltf::math::fquat rotation;
    fastgltf::math::decomposeTransformMatrix(matrix, scale, rotation, translation);

    SubMeshTransform out{};
    out.position = ToVec3(translation);
    out.rotation[0] = rotation.x();
    out.rotation[1] = rotation.y();
    out.rotation[2] = rotation.z();
    out.rotation[3] = rotation.w();
    out.scale = ToVec3(scale);
    return out;
}

// The same local transform as a matrix, for the accumulated world matrix the
// shear test needs. Translation, then rotation, then scale: T * R * S.
[[nodiscard]] fastgltf::math::fmat4x4 LocalMatrix(const fastgltf::Node& node) {
    if (const auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
        return *matrix;
    }
    const auto& trs = std::get<fastgltf::TRS>(node.transform);
    fastgltf::math::fmat4x4 m;
    m = fastgltf::math::translate(m, trs.translation);
    m = fastgltf::math::rotate(m, trs.rotation);
    m = fastgltf::math::scale(m, trs.scale);
    return m;
}

// Shear test for an accumulated world matrix. The engine's TRS placement (and
// the expand pass's closed-form normal matrix) is exact only for R*S; a sheared
// matrix shows up as non-orthogonal axes, which this checks directly. Testing
// orthogonality rather than round-tripping decomposeTransformMatrix matters:
// that decomposition reconstructs positive scales, so it would report a false
// positive on a mirrored node, whose axes are still orthogonal.
[[nodiscard]] bool HasShear(const fastgltf::math::fmat4x4& m, const float epsilon = 1e-4f) {
    std::array<std::array<float, 3>, 3> axis{};
    for (std::size_t c = 0; c < 3; ++c) {
        float length = 0.0f;
        for (std::size_t r = 0; r < 3; ++r) length += m[c][r] * m[c][r];
        length = std::sqrt(length);
        if (length <= 1e-8f) return true; // a degenerate axis cannot be represented
        for (std::size_t r = 0; r < 3; ++r) axis[c][r] = m[c][r] / length;
    }
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = a + 1; b < 3; ++b) {
            float dot = 0.0f;
            for (std::size_t r = 0; r < 3; ++r) dot += axis[a][r] * axis[b][r];
            if (std::abs(dot) > epsilon) return true;
        }
    }
    return false;
}

// Reads one accessor into a plain vector. iterateAccessor resolves the accessor's
// component type, normalization and sparseness, so normalized integers,
// KHR_mesh_quantization and sparse accessors all arrive as floats here.
template <typename T>
[[nodiscard]] std::vector<T> ReadAccessor(const fastgltf::Asset& asset, const std::size_t accessor_index) {
    std::vector<T> values;
    const auto& accessor = asset.accessors[accessor_index];
    values.reserve(accessor.count);
    fastgltf::iterateAccessor<T>(asset, accessor, [&values](const T value) { values.push_back(value); });
    return values;
}

// ── vertex emission ─────────────────────────────────────────────────────────

// The dedup key: every attribute that is actually consumed. Two vertices merge
// only when they agree on all of them, which is what keeps UV or tangent seams
// split instead of smearing one corner's values across the seam.
struct VertexKey {
    std::array<float, 3> position{};
    std::array<float, 3> normal{};
    std::array<float, 2> uv0{};
    std::array<float, 2> uv1{};
    std::array<float, 4> tangent{};
    bool has_tangent = false;

    bool operator==(const VertexKey& o) const {
        return position == o.position && normal == o.normal && uv0 == o.uv0 &&
               uv1 == o.uv1 && tangent == o.tangent && has_tangent == o.has_tangent;
    }
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const {
        std::size_t h = 1469598103934665603ull;
        const auto mix = [&h](const float value) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;
        };
        for (const float v : key.position) mix(v);
        for (const float v : key.normal) mix(v);
        for (const float v : key.uv0) mix(v);
        for (const float v : key.uv1) mix(v);
        for (const float v : key.tangent) mix(v);
        mix(key.has_tangent ? 1.0f : 0.0f);
        return h;
    }
};

[[nodiscard]] VertexKey MakeKey(const MeshVertexVec3& position,
                                const MeshVertexVec3& normal,
                                const MeshVertexVec2& uv0,
                                const MeshVertexVec2& uv1,
                                const float tangent[4],
                                const bool has_tangent) {
    VertexKey key{};
    key.position = {position.x, position.y, position.z};
    key.normal = {normal.x, normal.y, normal.z};
    key.uv0 = {uv0.u, uv0.v};
    key.uv1 = {uv1.u, uv1.v};
    key.tangent = {tangent[0], tangent[1], tangent[2], tangent[3]};
    key.has_tangent = has_tangent;
    return key;
}

struct ConversionState {
    const fastgltf::Asset& asset;
    const std::vector<VulkanEngine::MaterialId>* bindings = nullptr;
    VulkanEngine::Mesh& mesh;
    // Parallel to mesh.subMeshes: whether that submesh's primitive authored
    // TANGENT, so MikkTSpace only fills the ones that did not.
    std::vector<bool> submesh_authored_tangents;
    bool any_uv_authored = false;
    bool any_uv1 = false;
    bool warned_unsupported_mode = false;
    bool warned_extra_uv_set = false;
    bool warned_material = false;
    bool warned_skinning = false;
};

// Appends one vertex and returns its mesh-local index. Every parallel array is
// kept parallel: a set that appears in any primitive is populated for all
// vertices (zeros where the primitive does not author it), because downstream
// consumers index uvs/tangents by vertex.
[[nodiscard]] std::uint32_t EmitVertex(ConversionState& state,
                                       const MeshVertexVec3& position,
                                       const MeshVertexVec3& normal,
                                       const MeshVertexVec2& uv0,
                                       const MeshVertexVec2& uv1,
                                       const MeshVertexVec3& tangent,
                                       const float handedness) {
    auto& mesh = state.mesh;
    const auto index = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(position);
    mesh.normals.push_back(normal);
    mesh.uvs.push_back(uv0);
    if (state.any_uv1) {
        mesh.uv_extra[0].push_back(uv1);
    }
    mesh.tangents.push_back(tangent);
    mesh.tangent_handedness.push_back(handedness);
    return index;
}

// Expands a primitive's index list (or its implicit one) into triangles. The
// engine draws triangles only, so strips and fans are triangulated here.
[[nodiscard]] bool Triangulate(const fastgltf::PrimitiveType mode,
                               std::vector<std::uint32_t> source,
                               std::vector<std::uint32_t>& out) {
    const auto push = [&out](const std::uint32_t a, const std::uint32_t b, const std::uint32_t c) {
        out.push_back(a);
        out.push_back(b);
        out.push_back(c);
    };

    switch (mode) {
        case fastgltf::PrimitiveType::Triangles:
            out = std::move(source);
            return true;
        case fastgltf::PrimitiveType::TriangleStrip:
            for (std::size_t i = 0; i + 2 < source.size(); ++i) {
                if (i % 2 == 0) {
                    push(source[i], source[i + 1], source[i + 2]);
                } else {
                    push(source[i + 1], source[i], source[i + 2]);
                }
            }
            return true;
        case fastgltf::PrimitiveType::TriangleFan:
            for (std::size_t i = 1; i + 1 < source.size(); ++i) {
                push(source[0], source[i], source[i + 1]);
            }
            return true;
        default:
            return false;
    }
}

[[nodiscard]] MeshVertexVec3 FaceNormal(const MeshVertexVec3& a,
                                        const MeshVertexVec3& b,
                                        const MeshVertexVec3& c) {
    const MeshVertexVec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
    const MeshVertexVec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
    MeshVertexVec3 n{
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x,
    };
    const float length = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (length > 1e-8f) {
        n.x /= length;
        n.y /= length;
        n.z /= length;
    } else {
        n = MeshVertexVec3{0.0f, 1.0f, 0.0f}; // degenerate triangle: a valid unit normal
    }
    return n;
}

void ConvertPrimitive(ConversionState& state,
                      const fastgltf::Primitive& primitive,
                      const SubMeshTransform& transform) {
    auto& mesh = state.mesh;
    const auto& asset = state.asset;

    if (primitive.dracoCompression) {
        throw std::runtime_error(
            "GltfMeshAssembler: the asset uses KHR_draco_mesh_compression, which is not supported");
    }

    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) {
        return; // a primitive with no positions has nothing to draw
    }

    const std::vector<fastgltf::math::fvec3> positions =
        ReadAccessor<fastgltf::math::fvec3>(asset, position_it->accessorIndex);
    if (positions.empty()) return;

    const auto normal_it = primitive.findAttribute("NORMAL");
    const std::vector<fastgltf::math::fvec3> normals = normal_it != primitive.attributes.end()
        ? ReadAccessor<fastgltf::math::fvec3>(asset, normal_it->accessorIndex)
        : std::vector<fastgltf::math::fvec3>{};

    const auto uv0_it = primitive.findAttribute("TEXCOORD_0");
    const std::vector<fastgltf::math::fvec2> uv0 = uv0_it != primitive.attributes.end()
        ? ReadAccessor<fastgltf::math::fvec2>(asset, uv0_it->accessorIndex)
        : std::vector<fastgltf::math::fvec2>{};

    const auto uv1_it = primitive.findAttribute("TEXCOORD_1");
    const std::vector<fastgltf::math::fvec2> uv1 = uv1_it != primitive.attributes.end()
        ? ReadAccessor<fastgltf::math::fvec2>(asset, uv1_it->accessorIndex)
        : std::vector<fastgltf::math::fvec2>{};

    const auto tangent_it = primitive.findAttribute("TANGENT");
    const std::vector<fastgltf::math::fvec4> tangents = tangent_it != primitive.attributes.end()
        ? ReadAccessor<fastgltf::math::fvec4>(asset, tangent_it->accessorIndex)
        : std::vector<fastgltf::math::fvec4>{};

    // Sets 2 and 3 have no home while MAX_UV_SETS is 2. The shader resolves a
    // material's absent set to UV0, so dropping them degrades instead of breaks.
    if (!state.warned_extra_uv_set &&
        (primitive.findAttribute("TEXCOORD_2") != primitive.attributes.end() ||
         primitive.findAttribute("TEXCOORD_3") != primitive.attributes.end())) {
        state.warned_extra_uv_set = true;
        LOGIFACE_LOG(warn, "GltfMeshAssembler: TEXCOORD_2/3 present but MAX_UV_SETS is 2; the extra UV set(s) are ignored");
    }

    std::vector<std::uint32_t> source_indices;
    if (primitive.indicesAccessor) {
        source_indices = ReadAccessor<std::uint32_t>(asset, *primitive.indicesAccessor);
    } else {
        source_indices.resize(positions.size());
        std::iota(source_indices.begin(), source_indices.end(), 0u);
    }

    std::vector<std::uint32_t> triangles;
    if (!Triangulate(primitive.type, std::move(source_indices), triangles)) {
        if (!state.warned_unsupported_mode) {
            state.warned_unsupported_mode = true;
            LOGIFACE_LOG(warn, "GltfMeshAssembler: skipping a non-triangle primitive "
                               "(only TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN are supported)");
        }
        return;
    }
    triangles.resize(triangles.size() - (triangles.size() % 3U));

    VulkanEngine::MaterialId material_id{0};
    if (primitive.materialIndex) {
        material_id = MapGltfMaterialIndex(*primitive.materialIndex, state.bindings);
        if (!state.warned_material &&
            (state.bindings == nullptr || *primitive.materialIndex >= state.bindings->size())) {
            state.warned_material = true;
            LOGIFACE_LOG(warn, "GltfMeshAssembler: material index " + std::to_string(*primitive.materialIndex) +
                               " has no binding; the default material is used");
        }
    }

    const bool has_normals = normals.size() >= positions.size();
    const bool has_tangents = tangents.size() >= positions.size();
    const bool has_uv0 = uv0.size() >= positions.size();
    const bool has_uv1 = uv1.size() >= positions.size();
    if (has_uv0) state.any_uv_authored = true;

    // A UV set that appears in any primitive must be parallel for every vertex.
    if (has_uv1 && !state.any_uv1) {
        state.any_uv1 = true;
        mesh.uv_extra[0].resize(mesh.vertices.size(), MeshVertexVec2{0.0f, 0.0f});
    }

    const std::uint32_t index_start = static_cast<std::uint32_t>(mesh.indices.size());

    if (has_normals) {
        std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> dedup;
        dedup.reserve(triangles.size());
        for (const std::uint32_t index : triangles) {
            if (index >= positions.size()) continue; // malformed asset: drop the corner
            const MeshVertexVec3 position = ToVec3(positions[index]);
            const MeshVertexVec3 normal = ToVec3(normals[index]);
            const MeshVertexVec2 tex0 = has_uv0 ? ToVec2(uv0[index]) : MeshVertexVec2{0.0f, 0.0f};
            const MeshVertexVec2 tex1 = has_uv1 ? ToVec2(uv1[index]) : MeshVertexVec2{0.0f, 0.0f};
            MeshVertexVec3 tangent{0.0f, 0.0f, 0.0f};
            float handedness = 1.0f;
            if (has_tangents) {
                tangent = MeshVertexVec3{tangents[index].x(), tangents[index].y(), tangents[index].z()};
                handedness = tangents[index].w();
            }
            const float tangent4[4] = {tangent.x, tangent.y, tangent.z, handedness};

            const auto [it, inserted] = dedup.try_emplace(
                MakeKey(position, normal, tex0, tex1, tangent4, has_tangents), 0u);
            if (inserted) {
                it->second = EmitVertex(state, position, normal, tex0, tex1, tangent, handedness);
            }
            mesh.indices.push_back(it->second);
        }
    } else {
        // No NORMAL: generate flat per-triangle normals, which the spec allows a
        // consumer to do and which keeps the octahedral packing well defined.
        // The corners are deliberately not deduplicated: adjacent triangles
        // share an edge but not a normal.
        for (std::size_t t = 0; t + 2 < triangles.size(); t += 3) {
            const std::uint32_t i0 = triangles[t];
            const std::uint32_t i1 = triangles[t + 1];
            const std::uint32_t i2 = triangles[t + 2];
            if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) continue;
            const MeshVertexVec3 p0 = ToVec3(positions[i0]);
            const MeshVertexVec3 p1 = ToVec3(positions[i1]);
            const MeshVertexVec3 p2 = ToVec3(positions[i2]);
            const MeshVertexVec3 normal = FaceNormal(p0, p1, p2);
            const std::uint32_t corners[3] = {i0, i1, i2};
            for (const std::uint32_t corner : corners) {
                const MeshVertexVec2 tex0 = has_uv0 ? ToVec2(uv0[corner]) : MeshVertexVec2{0.0f, 0.0f};
                const MeshVertexVec2 tex1 = has_uv1 ? ToVec2(uv1[corner]) : MeshVertexVec2{0.0f, 0.0f};
                // An authored tangent is honored here too; only the normal is
                // synthesized, so the file's tangent space is not discarded.
                MeshVertexVec3 tangent{0.0f, 0.0f, 0.0f};
                float handedness = 1.0f;
                if (has_tangents) {
                    tangent = MeshVertexVec3{tangents[corner].x(), tangents[corner].y(), tangents[corner].z()};
                    handedness = tangents[corner].w();
                }
                mesh.indices.push_back(
                    EmitVertex(state, ToVec3(positions[corner]), normal, tex0, tex1, tangent, handedness));
            }
        }
    }

    const std::uint32_t index_count = static_cast<std::uint32_t>(mesh.indices.size()) - index_start;
    if (index_count == 0) return;

    SubMesh submesh{};
    submesh.index_start = index_start;
    submesh.index_count = index_count;
    submesh.material_id = material_id;
    submesh.transform = transform;
    mesh.subMeshes.push_back(submesh);
    state.submesh_authored_tangents.push_back(has_tangents);
}

// Fills the tangent frame: the file's baked TANGENT where a primitive authored
// one, MikkTSpace for the rest. Runs after every submesh exists, because
// MikkTSpace accumulates across shared vertices and must see the deduplicated
// geometry. A mesh with no UV domain has no tangent space and gets none.
void FinalizeTangents(ConversionState& state) {
    auto& mesh = state.mesh;
    if (mesh.vertices.empty()) return;

    if (!state.any_uv_authored) {
        mesh.tangents.clear();
        mesh.tangent_handedness.clear();
        return;
    }

    bool all_generated = true;
    for (std::size_t s = 0; s < mesh.subMeshes.size(); ++s) {
        if (s < state.submesh_authored_tangents.size() && state.submesh_authored_tangents[s]) continue;
        const SubMesh& submesh = mesh.subMeshes[s];
        all_generated &= VulkanEngine::TangentGen::GenerateSubmeshTangents(
            mesh.vertices, mesh.normals, mesh.uvs, mesh.indices,
            submesh.index_start, submesh.index_count,
            mesh.tangents, mesh.tangent_handedness);
    }
    if (!all_generated) {
        LOGIFACE_LOG(warn, "GltfMeshAssembler: MikkTSpace failed for submesh(es); tangents discarded");
        mesh.tangents.clear();
        mesh.tangent_handedness.clear();
    }
}

void ConvertAsset(const fastgltf::Asset& asset,
                  const std::vector<VulkanEngine::MaterialId>* bindings,
                  VulkanEngine::Mesh& mesh) {
    ConversionState state{asset, bindings, mesh};

    // Which nodes are reachable as roots: the default scene when it names one,
    // else scene 0, else every node that is nobody's child. An asset with no
    // scenes (legal) still imports.
    std::vector<bool> is_child(asset.nodes.size(), false);
    for (const auto& node : asset.nodes) {
        for (const std::size_t child : node.children) {
            if (child < is_child.size()) is_child[child] = true;
        }
    }

    std::vector<std::size_t> roots;
    const auto append_scene_roots = [&roots, &asset](const std::size_t scene_index) {
        if (scene_index >= asset.scenes.size()) return;
        for (const std::size_t node : asset.scenes[scene_index].nodeIndices) roots.push_back(node);
    };
    if (asset.defaultScene.has_value()) append_scene_roots(*asset.defaultScene);
    if (roots.empty() && !asset.scenes.empty()) append_scene_roots(0);
    if (roots.empty()) {
        for (std::size_t i = 0; i < asset.nodes.size(); ++i) {
            if (!is_child[i]) roots.push_back(i);
        }
    }

    // Each stack entry carries the transform of the node's *parent*; the node's
    // own local transform is composed when it is visited, so a root's own
    // transform is applied exactly once.
    struct Pending {
        std::size_t node = 0;
        SubMeshTransform parent_trs{};
        fastgltf::math::fmat4x4 parent_matrix{};
    };

    std::vector<Pending> stack;
    stack.reserve(roots.size());
    for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
        stack.push_back(Pending{*it, SubMeshTransform{}, fastgltf::math::fmat4x4{}});
    }

    std::unordered_set<std::size_t> shear_warned;
    while (!stack.empty()) {
        const Pending parent = stack.back();
        stack.pop_back();
        if (parent.node >= asset.nodes.size()) continue;
        const fastgltf::Node& node = asset.nodes[parent.node];

        const SubMeshTransform node_trs = ComposeSubmeshTransform(parent.parent_trs, LocalTrs(node));
        const fastgltf::math::fmat4x4 node_matrix = parent.parent_matrix * LocalMatrix(node);

        if (HasShear(node_matrix) && shear_warned.insert(parent.node).second) {
            LOGIFACE_LOG(warn, "GltfMeshAssembler: node " + std::to_string(parent.node) +
                               " accumulates a sheared transform; the engine's TRS placement cannot "
                               "represent it, so this node is placed approximately");
        }

        if (node.skinIndex.has_value() && !state.warned_skinning) {
            state.warned_skinning = true;
            LOGIFACE_LOG(warn, "GltfMeshAssembler: the asset contains a skinned node; skinning is not imported");
        }

        if (node.meshIndex.has_value() && *node.meshIndex < asset.meshes.size()) {
            for (const fastgltf::Primitive& primitive : asset.meshes[*node.meshIndex].primitives) {
                ConvertPrimitive(state, primitive, node_trs);
            }
        }

        for (const std::size_t child : node.children) {
            if (child >= asset.nodes.size()) continue;
            stack.push_back(Pending{child, node_trs, node_matrix});
        }
    }

    FinalizeTangents(state);
}

} // namespace

std::future<std::shared_ptr<VulkanEngine::Mesh>>
GltfMeshAssembler::AssembleFromFullBuffer(std::shared_ptr<FileLoader::ByteBuffer> buffer) {
    auto prom = std::make_shared<std::promise<std::shared_ptr<VulkanEngine::Mesh>>>();
    try {
        if (source_path_.empty()) {
            throw std::runtime_error(
                "GltfMeshAssembler: SetSourcePath must be called before loading: external buffers and "
                "images resolve against the asset's own directory");
        }
        if (buffer == nullptr || buffer->empty()) {
            throw std::runtime_error("GltfMeshAssembler: empty buffer for '" + source_path_.string() + "'");
        }

        // Extensions the asset may use. Quantization is a geometry extension, so
        // it belongs here rather than only in the material importer.
        fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization |
                                fastgltf::Extensions::KHR_texture_transform |
                                fastgltf::Extensions::KHR_texture_basisu |
                                fastgltf::Extensions::KHR_materials_emissive_strength |
                                fastgltf::Extensions::KHR_materials_unlit);

        auto data = fastgltf::GltfDataBuffer::FromBytes(
            reinterpret_cast<const std::byte*>(buffer->data()), buffer->size());
        if (!data) {
            throw std::runtime_error("GltfMeshAssembler: could not create a GLTF data buffer");
        }

        // loadGltf detects JSON vs GLB, so .gltf and .glb share one path;
        // external buffers and images resolve against the asset's directory.
        auto asset = parser.loadGltf(data.get(), source_path_.parent_path(),
                                     fastgltf::Options::LoadExternalBuffers |
                                     fastgltf::Options::LoadExternalImages);
        if (asset.error() != fastgltf::Error::None) {
            throw std::runtime_error("GltfMeshAssembler: failed to load '" + source_path_.string() +
                                     "': " + std::string(fastgltf::getErrorMessage(asset.error())));
        }

        auto mesh = std::make_shared<VulkanEngine::Mesh>();
        ConvertAsset(asset.get(), material_bindings_, *mesh);

        if (mesh->vertices.empty()) {
            // A "successful" parse that yields nothing is worse than a failure:
            // the caller would draw an empty mesh with no indication why.
            throw std::runtime_error("GltfMeshAssembler: no renderable geometry in '" + source_path_.string() + "'");
        }

        prom->set_value(mesh);
    } catch (...) {
        prom->set_exception(std::current_exception());
    }
    return prom->get_future();
}

std::shared_ptr<VulkanEngine::Mesh> GltfMeshLoader::DoLoad(const std::filesystem::path& path) {
    auto buf = ReadEntireFile(path);
    auto buf_ptr = std::make_shared<FileLoader::ByteBuffer>(buf.begin(), buf.end());
    GltfMeshAssembler assembler;
    assembler.SetMaterialBindings(material_bindings_);
    assembler.SetSourcePath(path);
    return assembler.AssembleFromFullBuffer(std::move(buf_ptr)).get();
}

std::vector<std::byte> GltfMeshLoader::ReadEntireFile(const std::filesystem::path& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("GltfMeshLoader: Failed to open file: " + file_path.string());
    }
    const auto size = static_cast<std::size_t>(file.tellg());
    if (size == 0) {
        throw std::runtime_error("GltfMeshLoader: File is empty: " + file_path.string());
    }
    std::vector<std::byte> buf(size);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("GltfMeshLoader: Failed to read file: " + file_path.string());
    }
    return buf;
}

} // namespace VulkanEngine::FileLoaders::Mesh
