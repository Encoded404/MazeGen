module;

export module VulkanEngine.ShaderRegistration;

import std;
import ShaderReflection;

import VulkanEngine.ShaderManager;

import Shaders.Engine.ExpandComp;
import Shaders.Engine.OcclusionCullComp;
import Shaders.Engine.OccluderSelectComp;
import Shaders.Engine.PreCullComp;
import Shaders.Engine.HizGenComp;
import Shaders.Engine.CollectCountCompactComp;
import Shaders.Engine.CollectWriteComp;
import Shaders.Engine.MainIndirVert;
import Shaders.Engine.MainIndirVertUV1;
import Shaders.Engine.DepthIndirVert;
import Shaders.Engine.StandardMeshFrag;
import Shaders.Engine.StandardMeshFragUV1;
import Shaders.Engine.StandardMeshFragMask;
import Shaders.Engine.StandardMeshFragMaskUV1;
import Shaders.Engine.DepthPrepassFrag;
import Shaders.Engine.UnlitFrag;
import Shaders.Engine.UiTextVert;
import Shaders.Engine.UiTextFrag;
import Shaders.Engine.WorldTextVert;
import Shaders.Engine.WorldTextFrag;
import Shaders.Engine.SlugTextVert;
import Shaders.Engine.SlugTextFrag;

#ifdef VKENGINE_PHYSICAL_CAMERA
import Shaders.Engine.PhysicalCameraCompositeVert;
import Shaders.Engine.PhysicalCameraCompositeFrag;
#endif

export namespace VulkanEngine {

struct EngineShaderIds {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    ShaderSystem::ShaderId expand_comp;
    ShaderSystem::ShaderId occlusion_cull_comp;
    ShaderSystem::ShaderId occluder_select_comp;
    ShaderSystem::ShaderId pre_cull_comp;
    ShaderSystem::ShaderId hiz_gen_comp;
    ShaderSystem::ShaderId collect_count_compact_comp;
    ShaderSystem::ShaderId collect_write_comp;
    ShaderSystem::ShaderId main_indir_vert;
    // Out-of-line UV1 interface variant rows (named entry point main_uv1).
    ShaderSystem::ShaderId main_indir_vert_uv1;
    ShaderSystem::ShaderId depth_indir_vert;
    ShaderSystem::ShaderId standard_mesh_frag;
    ShaderSystem::ShaderId standard_mesh_frag_uv1;
    // MASK interface-variant rows (named entry points main_mask /
    // main_mask_uv1). Paired with the existing vertex shaders by the default
    // mesh technique's variant table.
    ShaderSystem::ShaderId standard_mesh_frag_mask;
    ShaderSystem::ShaderId standard_mesh_frag_mask_uv1;
    ShaderSystem::ShaderId depth_prepass_frag;
    ShaderSystem::ShaderId unlit_frag;
    // Screen-space text overlay: the vertex pulls the quad from SV_VertexID and
    // reads per-glyph instances; the fragment samples the R8 atlas page.
    ShaderSystem::ShaderId ui_text_vert;
    ShaderSystem::ShaderId ui_text_frag;
    // Depth-tested world-space text: the vertex pulls the quad from
    // SV_VertexID and the world-space basis from a per-glyph instance; the
    // fragment reconstructs coverage from the RGBA8 MSDF page.
    ShaderSystem::ShaderId world_text_vert;
    ShaderSystem::ShaderId world_text_frag;
    // The Slug (hb-gpu) world-text backend: the vertex pulls the quad from
    // SV_VertexID and the em-space box plus a flat blob offset from a per-glyph
    // instance; the fragment reconstructs coverage analytically from the blob.
    ShaderSystem::ShaderId slug_text_vert;
    ShaderSystem::ShaderId slug_text_frag;
#ifdef VKENGINE_PHYSICAL_CAMERA
    ShaderSystem::ShaderId physical_camera_composite_vert;
    ShaderSystem::ShaderId physical_camera_composite_frag;
#endif
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    void RegisterAll(ShaderSystem::ShaderManager& mgr, std::string_view data_dir) {
        expand_comp              = Shaders::Engine::ExpandComp::Register(mgr, data_dir);
        occlusion_cull_comp      = Shaders::Engine::OcclusionCullComp::Register(mgr, data_dir);
        occluder_select_comp     = Shaders::Engine::OccluderSelectComp::Register(mgr, data_dir);
        pre_cull_comp            = Shaders::Engine::PreCullComp::Register(mgr, data_dir);
        hiz_gen_comp             = Shaders::Engine::HizGenComp::Register(mgr, data_dir);
        collect_count_compact_comp = Shaders::Engine::CollectCountCompactComp::Register(mgr, data_dir);
        collect_write_comp       = Shaders::Engine::CollectWriteComp::Register(mgr, data_dir);
        main_indir_vert          = Shaders::Engine::MainIndirVert::Register(mgr, data_dir);
        main_indir_vert_uv1      = Shaders::Engine::MainIndirVertUV1::Register(mgr, data_dir);
        depth_indir_vert         = Shaders::Engine::DepthIndirVert::Register(mgr, data_dir);
        standard_mesh_frag       = Shaders::Engine::StandardMeshFrag::Register(mgr, data_dir);
        standard_mesh_frag_uv1   = Shaders::Engine::StandardMeshFragUV1::Register(mgr, data_dir);
        standard_mesh_frag_mask  = Shaders::Engine::StandardMeshFragMask::Register(mgr, data_dir);
        standard_mesh_frag_mask_uv1 =
            Shaders::Engine::StandardMeshFragMaskUV1::Register(mgr, data_dir);
        depth_prepass_frag       = Shaders::Engine::DepthPrepassFrag::Register(mgr, data_dir);
        unlit_frag               = Shaders::Engine::UnlitFrag::Register(mgr, data_dir);
        ui_text_vert             = Shaders::Engine::UiTextVert::Register(mgr, data_dir);
        ui_text_frag             = Shaders::Engine::UiTextFrag::Register(mgr, data_dir);
        world_text_vert          = Shaders::Engine::WorldTextVert::Register(mgr, data_dir);
        world_text_frag          = Shaders::Engine::WorldTextFrag::Register(mgr, data_dir);
        slug_text_vert           = Shaders::Engine::SlugTextVert::Register(mgr, data_dir);
        slug_text_frag           = Shaders::Engine::SlugTextFrag::Register(mgr, data_dir);
#ifdef VKENGINE_PHYSICAL_CAMERA
        physical_camera_composite_vert = Shaders::Engine::PhysicalCameraCompositeVert::Register(mgr, data_dir);
        physical_camera_composite_frag = Shaders::Engine::PhysicalCameraCompositeFrag::Register(mgr, data_dir);
#endif
    }
};

} // namespace VulkanEngine
