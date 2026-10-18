module;
#include <glm/glm.hpp>

#include <glm/vec4.hpp> // NOLINT(misc-include-cleaner)
#include <glm/gtc/quaternion.hpp> // NOLINT(misc-include-cleaner)

#include <SDL3/SDL_keycode.h>
#include <imgui.h>

#include <logging/logging_macros.hpp>

module Examples.BasicScene.Game;

import std;

import vulkan;

import logiface;

import VulkanEngine.GameEngine;
import VulkanEngine.TextureTypes;
import VulkanEngine.GpuResources.MeshData;
import VulkanEngine.GplPolicy;
import Examples.BasicScene.Components.SimpleControllerComponent;
import Examples.BasicScene.Components.TransformControlComponent;
import Shaders.Engine.StandardMeshFrag;
import Shaders.BasicScene.NormalsFrag;
import Shaders.BasicScene.SolidFrag;
import VulkanEngine.ShaderManager;

namespace Examples::BasicScene::Game {

DemoGame::DemoGame(const RenderMode render_mode, const std::filesystem::path& executable_path,
                   std::filesystem::path model_path,
                   std::filesystem::path texture_path,
                   VulkanEngine::ShaderSystem::GplPolicy gpl_policy,
                   VulkanEngine::ShaderSystem::GplStructurePolicy gpl_structure)
    : render_mode_(render_mode)
    , exe_dir_(executable_path.parent_path())
    , model_path_(std::move(model_path))
    , texture_path_(std::move(texture_path))
    , gpl_policy_(gpl_policy)
    , gpl_structure_(gpl_structure) {
    setup_token_ = hooks_.on_setup.Register([this](VulkanEngine::Application::ApplicationContext& ctx) -> bool {
        return OnSetup(ctx);
    });
    pre_input_token_ = hooks_.on_pre_input.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnPreInput(ctx);
    });
    hooks_.should_filter_mouse_input = [this]() -> bool {
        return ShouldFilterMouseInput();
    };
    hooks_.should_filter_keyboard_input = [this]() -> bool {
        return ShouldFilterKeyboardInput();
    };
    fixed_update_token_ = hooks_.on_fixed_update.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFixedUpdate(ctx);
    });
    frame_update_token_ = hooks_.on_frame_update.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFrameUpdate(ctx);
    });
    frame_render_token_ = hooks_.on_frame_render.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnFrameRender(ctx);
    });
    shutdown_token_ = hooks_.on_shutdown.Register([this](VulkanEngine::Application::ApplicationContext& ctx) {
        OnShutdown(ctx);
    });
}

DemoGame::~DemoGame() = default;

bool DemoGame::OnSetup(VulkanEngine::Application::ApplicationContext& ctx) {
    // 1. Configure and init engine subsystems
    VulkanEngine::GameConfig config{};
    config.enable_imgui = true;
    config.renderer_config.clear_color = {0.1f, 0.1f, 0.1f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();
    config.gpl_policy = gpl_policy_;
    config.gpl_structure = gpl_structure_;

    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    // 2. Register app shaders and select fragment shader
    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    auto standard_frag_id = engine_game_.GetContext().GetShaderIds().standard_mesh_frag;

    auto normals_frag_id = Shaders::BasicScene::NormalsFrag::Register(shader_mgr, config.shader_data_dir);
    auto solid_frag_id = Shaders::BasicScene::SolidFrag::Register(shader_mgr, config.shader_data_dir);

    VulkanEngine::ShaderSystem::ShaderId frag_id;
    switch (render_mode_) {
        case RenderMode::Normals:
            frag_id = normals_frag_id;
            break;
        case RenderMode::NoTextures:
            frag_id = solid_frag_id;
            break;
        default:
            frag_id = standard_frag_id;
            break;
    }

    auto vert_id = engine_game_.GetContext().GetShaderIds().main_indir_vert;
    if (!engine_game_.InitRenderer(ctx, vert_id, frag_id, &shader_mgr)) {
        return false;
    }

    // 3. Create a custom material for the viking room
    const std::uint32_t tex_slot = engine_game_.LoadTexture(
        ctx, exe_dir_ / "textures" / "viking_room.png",
        VulkanEngine::Textures::TextureSemantic::BaseColor,
        VulkanEngine::Textures::TextureNormalEncoding::Standard,
        VulkanEngine::Textures::SamplerDesc{});
    constexpr auto viking_blend = VulkanEngine::MaterialManager::BlendMode::Transparent;

    constexpr auto pbr_roughness = 0.6f;
    constexpr auto pbr_metallic = 0.0f;
    auto viking_handle = engine_game_.GetContext().GetMaterialManager().Register<VulkanEngine::TechniqueManager::DefaultMeshTechnique>(
        viking_blend,
        VulkanEngine::TechniqueManager::DefaultMeshPerMaterialData{
            .albedo_texture = tex_slot,
            .roughness_factor = pbr_roughness,
            .metallic_factor = pbr_metallic,
            .ao_factor = 1.0f
        });
    const auto viking_mat_id = VulkanEngine::MaterialManager::MaterialId{viking_handle.Id()};

    // 4. Load meshes and register with MeshRegistry
    const std::vector<VulkanEngine::SceneLoader::MaterialId> viking_bindings = {viking_mat_id};

    auto viking_mesh = VulkanEngine::SceneLoader::LoadMeshData(
        exe_dir_ / "models" / "viking_room.obj", &viking_bindings);
    auto monkey_mesh = VulkanEngine::SceneLoader::LoadMeshData(
        exe_dir_ / "models" / "simple-monkey.bin", nullptr);

    auto& mesh_registry = engine_game_.GetMeshRegistry();
    const std::uint32_t viking_id = mesh_registry.Register(viking_mesh);
    const std::uint32_t monkey_id = mesh_registry.Register(monkey_mesh);

    // Mark scene as valid so rendering begins
    engine_game_.MarkSceneValid();

    // 5. Create camera
    auto& component_registry = engine_game_.GetContext().GetComponentRegistry();
    engine_game_.CreateCamera(component_registry);

#ifdef VKENGINE_PHYSICAL_CAMERA
    // 5b. PhysicalCamera (webcam): open device 0, composite into a target
    // texture, and expose the target's bindless slot for materials/shaders.
    {
        auto* phys = engine_game_.GetPhysicalCameraSystem();
        if (phys && phys->IsInitialized()) {
            const auto devices = phys->EnumerateDevices();
            if (!devices.empty()) {
                cam_handle_ = phys->Open(devices[0].index);
                if (cam_handle_.IsValid()) {
                    cam_target_ = phys->CreateTarget(640, 360);
                    if (cam_target_.IsValid()) {
                        VulkanEngine::PhysicalCamera::PhysicalCameraBindingConfig bcfg{};
                        bcfg.fit = VulkanEngine::PhysicalCamera::FitMode::Contain;
                        bcfg.clear_before = true;
                        cam_binding_ = phys->Bind(cam_handle_, cam_target_, bcfg);
                        cam_target_slot_ = phys->GetTargetTextureSlot(cam_target_);
                        const auto native_slots = phys->GetNativeTextureSlots(cam_handle_);
                        if (!native_slots.empty()) cam_native_slot_ = native_slots[0];
                        LOGIFACE_LOG(info,
                            "webcam bound: target bindless slot " + std::to_string(cam_target_slot_) +
                            ", native slot " + std::to_string(cam_native_slot_));

                        // Live preview quad: register an unlit material pointing at
                        // the camera target's bindless slot, and a unit quad mesh
                        // bound to it. The entity is created in step 6 below.
                        if (cam_target_slot_ != 0) {
                            auto webcam_handle = engine_game_.GetContext().GetMaterialManager().Register<
                                VulkanEngine::TechniqueManager::UnlitTextureTechnique>(
                                VulkanEngine::MaterialManager::BlendMode::Opaque,
                                VulkanEngine::TechniqueManager::UnlitPerMaterialData{
                                    .albedo_texture = cam_target_slot_
                                });
                            const auto webcam_mat_id = VulkanEngine::MaterialManager::MaterialId{webcam_handle.Id()};
                            auto quad = VulkanEngine::SceneLoader::CreateFallbackQuad();
                            quad.submeshes[0].material_id = webcam_mat_id;
                            webcam_mesh_id_ = engine_game_.GetMeshRegistry().Register(
                                VulkanEngine::SceneLoader::ToMeshData(quad));
                        }
                    }
                }
            }
        }
    }
#endif

    // 6. Create game entities with simplified MeshReference
    {
        auto& entity = component_registry.CreateEntity();
        component_registry.AddComponent<VulkanEngine::Components::Transform>(entity);
        auto& mesh_ref = component_registry.AddComponent<VulkanEngine::Components::MeshReference>(entity);
        mesh_ref.loaded_mesh_id = viking_id;

        auto& debug_comp = component_registry.AddComponent<Examples::BasicScene::Components::TransformControlComponent>(entity);
        debug_comp.position = glm::vec3{0.0f, 0.0f, 0.0f};
        controllable_objects_.push_back(ControllableObject{"Viking house", &debug_comp});
    }
    {
        auto& entity = component_registry.CreateEntity();
        component_registry.AddComponent<VulkanEngine::Components::Transform>(entity);
        auto& mesh_ref = component_registry.AddComponent<VulkanEngine::Components::MeshReference>(entity);
        mesh_ref.loaded_mesh_id = monkey_id;
        // No service wiring: the component receives the granted ComponentContext
        // in Update (rung 3).
        component_registry.AddComponent<Examples::BasicScene::Components::SimpleControllerComponent>(entity);
    }
#ifdef VKENGINE_PHYSICAL_CAMERA
    if (webcam_mesh_id_ != 0) {
        auto& entity = component_registry.CreateEntity();
        component_registry.AddComponent<VulkanEngine::Components::Transform>(entity);
        auto& mesh_ref = component_registry.AddComponent<VulkanEngine::Components::MeshReference>(entity);
        mesh_ref.loaded_mesh_id = webcam_mesh_id_;

        auto& debug_comp = component_registry.AddComponent<Examples::BasicScene::Components::TransformControlComponent>(entity);
        debug_comp.position = glm::vec3{2.0f, 0.0f, 0.0f};
        controllable_objects_.push_back(ControllableObject{"Camera quad", &debug_comp});
    }
#endif

    // 7. Register ImGui debug UI
    auto* imgui = engine_game_.GetImGuiSystem();
    if (imgui) {
        imgui_draw_handle_ = imgui->draw_callbacks.Register([this]() {
            if (controllable_objects_.empty()) return;
            ImGui::Begin("Transform Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            // Object selector: pick which controllable object the controls drive.
            std::vector<const char*> names;
            names.reserve(controllable_objects_.size());
            for (const auto& obj : controllable_objects_) names.push_back(obj.name.c_str());
            ImGui::SetNextItemWidth(160.0f);
            ImGui::Combo("Object", &selected_object_, names.data(),
                         static_cast<int>(names.size()));
            selected_object_ = std::clamp(selected_object_, 0,
                                          static_cast<int>(controllable_objects_.size()) - 1);

            auto* dc = controllable_objects_[static_cast<std::size_t>(selected_object_)].component;
            if (dc == nullptr) {
                ImGui::End();
                return;
            }

            ImGui::DragFloat3("Position", &dc->position.x, 0.1f);

            ImGui::SeparatorText("Texture");
            ImGui::DragInt("Slot", &dc->texture_slot, 1, 0, 255);

            ImGui::SeparatorText("Rotation");
            constexpr const char* modes[] = {"Euler (vec3)", "Quaternion (vec4)"}; // NOLINT(modernize-avoid-c-arrays)
            int mode = static_cast<int>(dc->rotation_mode);
            if (ImGui::Combo("Mode", &mode, modes, 2)) {
                dc->rotation_mode = static_cast<Examples::BasicScene::Components::RotationMode>(mode);
            }
            if (dc->rotation_mode == Examples::BasicScene::Components::RotationMode::Euler) {
                ImGui::DragFloat3("Euler (deg)", &dc->rotation_euler.x, 1.0f);
            } else {
                ImGui::DragFloat4("Quaternion", &dc->rotation_quat.x, 0.01f);
            }

            ImGui::End();
        });

#ifdef VKENGINE_PHYSICAL_CAMERA
        imgui_camera_draw_handle_ = imgui->draw_callbacks.Register([this]() {
            ImGui::Begin("Physical Camera");
            if (engine_game_.GetPhysicalCameraSystem() == nullptr ||
                !engine_game_.GetPhysicalCameraSystem()->IsInitialized()) {
                ImGui::TextUnformatted("disabled");
            } else if (!cam_handle_.IsValid()) {
                ImGui::TextUnformatted("no camera opened");
            } else {
                auto* phys = engine_game_.GetPhysicalCameraSystem();
                const auto state = phys->GetState(cam_handle_);
                ImGui::Text("state: %u", static_cast<std::uint32_t>(state));
                ImGui::Text("streams active: %zu", phys->GetActiveStreamCount());
                ImGui::Separator();
                if (const auto* fmt = phys->GetFormat(cam_handle_); fmt != nullptr) {
                    ImGui::Text("format: %u", static_cast<std::uint32_t>(fmt->format));
                    ImGui::Text("resolution: %ux%u", fmt->width, fmt->height);
                    if (fmt->fps_denominator != 0) {
                        ImGui::Text("fps: %u", fmt->fps_numerator / fmt->fps_denominator);
                    }
                }
                ImGui::Text("target bindless slot: %u", cam_target_slot_);
                ImGui::Text("native bindless slot: %u", cam_native_slot_);
            }
            ImGui::End();
        });
#endif
    }

    // 8. Bind quit action
    ctx.quit_action_handle = ctx.input_system->BindAction("quit",
        VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));

    return true;
}

void DemoGame::OnPreInput(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
}

bool DemoGame::ShouldFilterMouseInput() {
    return ImGui::GetIO().WantCaptureMouse;
}

bool DemoGame::ShouldFilterKeyboardInput() {
    return ImGui::GetIO().WantCaptureKeyboard;
}

void DemoGame::OnFixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FixedUpdate(ctx);
}

void DemoGame::OnFrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameUpdate(ctx);
}

void DemoGame::OnFrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    engine_game_.FrameRender(ctx);
}

void DemoGame::OnShutdown(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
#ifdef VKENGINE_PHYSICAL_CAMERA
    imgui_camera_draw_handle_ = {};
#endif
    imgui_draw_handle_ = {};
    engine_game_.Shutdown();
}

} // namespace Examples::BasicScene::Game
