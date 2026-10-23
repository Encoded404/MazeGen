module;

export module VulkanEngine.GameEngine;

import std;

import vulkan;

export import VulkanEngine.Renderer;
export import VulkanEngine.SceneLoader;
export import VulkanEngine.FileLoaders.Mesh.GltfMaterialImport;
export import VulkanEngine.TechniqueManager;
export import VulkanEngine.MaterialManager;
export import VulkanEngine.BindlessManager;
export import VulkanEngine.SceneRenderer;
export import VulkanEngine.Components.Camera;
export import VulkanEngine.Components.Transform;
export import VulkanEngine.Components.MeshReference;
export import VulkanEngine.Components.Text;
export import VulkanEngine.StandardMeshPipeline;
export import VulkanEngine.ImGui;
export import VulkanEngine.ResourceSystem;
export import VulkanEngine.ResourceSystem.TextureResource;
export import VulkanEngine.DefaultTextureFactory;
export import VulkanEngine.ShaderLoader;
export import VulkanEngine.GpuResources;
export import VulkanEngine.Application;
export import VulkanEngine.Input;
export import VulkanEngine.EngineContext;
export import VulkanEngine.ECS.SystemSchedule;
export import VulkanEngine.ECS.FrameSchedule;
export import VulkanEngine.Modules.EngineModules;
export import VulkanEngine.Text.TextSystem;
export import VulkanEngine.Text.Font;

#ifdef VKENGINE_PHYSICAL_CAMERA
export import VulkanEngine.PhysicalCameraSystem;
#endif

#ifdef VKENGINE_PHYSICS
export import VulkanEngine.Physics.World;
export import VulkanEngine.Physics.Components;
export import VulkanEngine.Physics.BodyBridge;
export import VulkanEngine.Physics.Character;
#endif

import VulkanBackend.Platform.SdlPlatform;
import VulkanShared.CallbackList;
import VulkanEngine.MeshManager;
import VulkanEngine.RenderPipeline;
import VulkanEngine.MeshRegistry;
import VulkanEngine.MeshRenderSystem;
import VulkanEngine.EngineBootstrap;
import VulkanEngine.ShaderManager;
import VulkanEngine.ShaderWatcher;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;

export namespace VulkanEngine {
using VulkanEngine::Textures::TextureSemantic;
using VulkanEngine::Textures::TextureNormalEncoding;
using VulkanEngine::Textures::SamplerDesc;

class GameEngine {
public:
    GameEngine();
    ~GameEngine();
    GameEngine(const GameEngine&) = delete;
    GameEngine& operator=(const GameEngine&) = delete;

    bool Setup(VulkanEngine::Application::ApplicationContext& ctx, const GameConfig& config);

    // ── Engine modules ──
    // The lifecycle seam: modules registered here are configured/started by
    // Setup (Stage::Startup) and after the renderer comes up
    // (Stage::Scene), and torn down in the exact reverse of the start order
    // by Shutdown. The engine's default preset — device bootstrap, physics,
    // resource hooks, physical camera, the frame-phase seeds — is registered
    // through the same public surface a composed game uses, so a game adds
    // its own engines modules with identity-based ordering and never edits
    // the engine.
    //
    // Register app modules before Setup; the registry is filled by Setup's
    // default-preset registration (and cleared first).
    Modules::EngineModules& Modules() { return modules_; }
    [[nodiscard]] const Modules::EngineModules& Modules() const { return modules_; }

    // Replaces the engine's default technique set (the built-in DefaultMesh/
    // Unlit registration inside init) with a game-owned registration. Call
    // before InitRenderer; the callback receives the engine and context and
    // must return false to abort renderer init. A game's own techniques,
    // entry-point variant matrices and passes register here — module content,
    // not choreography edits.
    using TechniqueRegistrarFn =
        std::function<bool(GameEngine&, VulkanEngine::Application::ApplicationContext&,
                           VulkanEngine::ShaderSystem::ShaderManager&, VulkanEngine::ShaderSystem::ShaderId vert_id,
                           VulkanEngine::ShaderSystem::ShaderId frag_id)>;
    void SetTechniqueRegistrar(TechniqueRegistrarFn registrar) { technique_registrar_ = std::move(registrar); }

    // Replaces the engine's default single-sun lighting upload (from
    // GameConfig::lighting) at runtime. The scene renders with this from the
    // next frame; defaults stay configuration, not init constants.
    void UploadLighting(const SceneRenderer::SceneHeader& header,
                        std::span<const SceneRenderer::Light> lights);

    // Loads and uploads a texture. The semantic and normal encoding are
    // declared content (no default; Unknown resolves to _UNORM with a one-time
    // warning), and the sampler is clamped to device limits at upload.
    std::uint32_t LoadTexture(VulkanEngine::Application::ApplicationContext& ctx,
                              const std::filesystem::path& path,
                              TextureSemantic semantic,
                              TextureNormalEncoding normal_encoding,
                              const SamplerDesc& sampler);

    // Loads and uploads a texture from a glTF image reference, so a GLB's
    // embedded images (and a data: URI or a buffer-view image) reach the
    // bindless set without the app having to know which form the file used.
    // `asset_dir` resolves a relative URI; an absolute URI is used as-is.
    // Returns the fallback slot when the image cannot be resolved or decoded.
    // Hot reload applies to a URI image only: embedded bytes have no path to
    // watch.
    std::uint32_t LoadTextureFromGltfImage(VulkanEngine::Application::ApplicationContext& ctx,
                                           const FileLoaders::Mesh::GltfImageRef& image,
                                           TextureSemantic semantic,
                                           TextureNormalEncoding normal_encoding,
                                           const SamplerDesc& sampler,
                                           const std::filesystem::path& asset_dir = {});
    bool InitRenderer(VulkanEngine::Application::ApplicationContext& ctx,
                      ShaderSystem::ShaderId vert_id = 0,
                      ShaderSystem::ShaderId frag_id = 0,
                      ShaderSystem::ShaderManager* shader_mgr = nullptr);

    // Loads a font file into the text system's registry and, when hot reload is
    // built in, registers the source file with the font watcher so an edit to it
    // is picked up on the next frame. Mirrors LoadTexture; returns null when the
    // file is not a readable font.
    std::shared_ptr<const Text::FontFace> LoadFont(const std::filesystem::path& path,
                                                   std::uint32_t face_index = 0);

    // Switches the indexed-drawing mode at runtime. Resolves the request against
    // device capabilities, re-creates the mode-dependent SceneRenderer buffers
    // and compaction pipelines, and re-specializes the main-pass technique
    // pipelines (which SceneRenderer does not own). No-op when the resolved mode
    // already matches. Returns false if a pipeline rebuild failed.
    bool SetDrawMode(SceneRenderer::DrawMode requested);

    struct UploadedMesh {
        std::uint32_t first_submesh = 0;
        std::uint32_t submesh_count = 0;
        std::uint8_t vertex_buffer_index = 0;
        std::uint8_t index_buffer_index = 0;
    };

    std::vector<UploadedMesh> UploadScene(
        VulkanEngine::Application::ApplicationContext& ctx,
        const std::vector<VulkanEngine::GpuResources::MeshData>& meshes);
    std::vector<UploadedMesh> UploadSceneFromFiles(
        VulkanEngine::Application::ApplicationContext& ctx,
        const std::vector<std::filesystem::path>& file_paths,
        const std::vector<SceneLoader::MaterialId>* material_bindings = nullptr);

    // Creates the camera entity and remembers its id. Camera is a data
    // component, so callers re-fetch it through GetCamera() rather than caching
    // a pointer (the storage can reallocate on any structural change).
    EntityId CreateCamera(ComponentRegistry& registry);
    [[nodiscard]] Components::Camera* GetCamera() {
        Entity* entity = ctx_.component_registry.TryGetEntity(camera_entity_);
        return entity != nullptr ? entity->GetComponent<Components::Camera>() : nullptr;
    }

    // Simulation. Runs, in order, on the on_fixed_update hook (which may fire
    // several times per rendered frame): apply queued structural changes, run
    // data-component systems in registration order, dispatch object-component
    // Update, then apply structural changes queued during those phases.
    //
    // Where data-component systems get an ordered home. Object components keep
    // driving themselves through Component::Update.
    //
    // Constraints are by identity, never by name: a system the frame contract
    // names registers against a SystemSlot enumerator (so a rename is a compile
    // error), and anything registered for itself gets an opaque SystemId back to
    // constrain against. `label` is diagnostics only and must be a string
    // literal. See SystemSchedule for the ordering contract.
    using SystemFn = VulkanEngine::SystemSchedule::SystemFn;
    SystemId AddSystem(SystemFn fn, std::string_view label = {}) {
        return systems_.Add(std::move(fn), label);
    }
    SystemId AddSystem(SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return systems_.Add(slot, std::move(fn), label);
    }
    SystemId AddSystemAfter(SystemRef after, SystemFn fn, std::string_view label = {}) {
        return systems_.AddAfter(after, std::move(fn), label);
    }
    SystemId AddSystemAfter(SystemRef after, SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return systems_.AddAfter(after, slot, std::move(fn), label);
    }
    SystemId AddSystemBefore(SystemRef before, SystemFn fn, std::string_view label = {}) {
        return systems_.AddBefore(before, std::move(fn), label);
    }
    SystemId AddSystemBefore(SystemRef before, SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return systems_.AddBefore(before, slot, std::move(fn), label);
    }
    void FixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx);

    // ── Frame phases ──
    // The per-rendered-frame pipeline is a named-phase schedule (FrameSchedule),
    // the same seam the fixed step uses: every hardcoded frame call is an
    // engine-registered system, and a game adds its own per-frame work with
    // identity-based constraints — never an edit to the pipeline. With nothing
    // extra registered, this emits exactly the calls the direct path made.
    //
    // Constraints are by identity (FrameSlot enumerators for the engine's own
    // systems, opaque FrameId handles otherwise); labels are diagnostics only
    // and must be string literals. A new system joins the phase of the system
    // it constrains against, or the phase passed to AddFrameSystem.
    using FrameFn = FrameSchedule::FrameFn;
    FrameId AddFrameSystem(FramePhase phase, FrameFn fn,
                           std::string_view label = {}) {
        return frames_.Add(phase, std::move(fn), label);
    }
    FrameId AddFrameSystem(FramePhase phase, FrameSlot slot, FrameFn fn,
                           std::string_view label = {}) {
        return frames_.Add(phase, slot, std::move(fn), label);
    }
    FrameId AddFrameSystemAfter(FrameRef after, FrameFn fn, std::string_view label = {}) {
        return frames_.AddAfter(after, std::move(fn), label);
    }
    FrameId AddFrameSystemAfter(FrameRef after, FrameSlot slot, FrameFn fn,
                                std::string_view label = {}) {
        return frames_.AddAfter(after, slot, std::move(fn), label);
    }
    FrameId AddFrameSystemBefore(FrameRef before, FrameFn fn, std::string_view label = {}) {
        return frames_.AddBefore(before, std::move(fn), label);
    }
    FrameId AddFrameSystemBefore(FrameRef before, FrameSlot slot, FrameFn fn,
                                 std::string_view label = {}) {
        return frames_.AddBefore(before, slot, std::move(fn), label);
    }
    // Direct access for a composed game that wants to inspect or seed a phase
    // (e.g. enumerate the engine's registered frame systems).
    FrameSchedule& FrameSystems() { return frames_; }

    // Everything that must happen exactly once per rendered frame: the text
    // display scale and the per-frame mesh/render processing (which indexes the
    // in-flight rings by frame counter, so it must not run per simulation step).
    // Runs the FramePhase::Bridge schedule.
    void FrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx);
    void FrameRender(const VulkanEngine::Application::ApplicationContext& ctx);
    void Shutdown();

    // Subsystem accessors delegate to EngineContext
    ResourceManager& GetResourceManager() { return ctx_.resource_manager; }
    // The text pipeline's single submission entry point and font registry. The
    // render pass it queues into is attached during InitRenderer, so submit
    // draws from the frame that calls it onward.
    Text::TextSystem& GetTextSystem() { return *ctx_.text_system; }
    BindlessManager::BindlessManager& GetBindlessManager() { return *ctx_.bindless_mgr; }
    SceneRenderer::SceneRenderer& GetSceneRenderer() { return *ctx_.scene_renderer; }
    TechniqueManager::TechniqueManager& GetTechniqueManager() { return *ctx_.technique_mgr; }
    Renderer::Renderer& GetRenderer() { return *ctx_.renderer; }
    // Engine-managed render pass registration/reconfiguration surface.
    VulkanEngine::RenderPipeline::RenderPipeline& GetRenderPipeline();
    ImGui::ImGuiSystem* GetImGuiSystem() { return ctx_.imgui_system.get(); }
    GpuResources::DeviceBufferHeap& GetVertexHeap() { return ctx_.vertex_heap; }
    GpuResources::DeviceBufferHeap& GetIndexHeap() { return ctx_.index_heap; }
    GpuResources::StagingPool& GetStagingPool() { return ctx_.staging_pool; }
    std::vector<GpuResources::DeviceBufferHeap>& GetDynamicVertexHeaps() { return ctx_.dynamic_vertex_heaps; }
    std::vector<GpuResources::DeviceBufferHeap>& GetDynamicIndexHeaps() { return ctx_.dynamic_index_heaps; }
    MeshManager& GetMeshManager() { return *ctx_.mesh_manager; }
    MeshRegistry& GetMeshRegistry() { return ctx_.mesh_registry; }
    MeshRenderSystem& GetMeshRenderSystem() { return ctx_.mesh_render_system; }
#ifdef VKENGINE_PHYSICAL_CAMERA
    PhysicalCamera::PhysicalCameraSystem* GetPhysicalCameraSystem() { return ctx_.physical_camera.get(); }
#endif
#ifdef VKENGINE_PHYSICS
    // The physics world, stepped once per fixed step in the
    // SystemSlot::PhysicsSimulate system. Null when physics is not compiled in.
    Physics::World* GetPhysicsWorld() { return ctx_.physics_world.get(); }
#endif
    // Id of the main lit technique. Materials of that technique resolve to a
    // draw group derived from their render state; the technique's base group
    // (no render-state variant) has the same value as the technique id.
    std::uint16_t GetMainTechniqueIdRaw() const { return main_technique_id_; }
    VulkanEngine::TechniqueManager::TechniqueId GetMainTechniqueId() const { return VulkanEngine::TechniqueManager::TechniqueId{main_technique_id_}; }
    std::uint16_t GetMainDrawGroup() const { return main_draw_group_; }
    // The service view granted to object components (see ComponentContext),
    // built once during Setup from the app-provided input system and the
    // running engine subsystems. Object components receive it in Update and
    // store no engine pointer of their own.
    [[nodiscard]] const ComponentContext& GetComponentContext() const { return component_context_; }
    bool IsInitialized() const { return initialized_; }
    void MarkSceneValid() {
        scene_valid_ = true;
        scene_invalid_reported_ = false;
    }
    std::uint32_t UploadTextureToBindless(VulkanEngine::Application::ApplicationContext& ctx,
                                          TextureResource* tex,
                                          TextureSemantic semantic,
                                          TextureNormalEncoding normal_encoding,
                                          const SamplerDesc& sampler);

    // ── App shader hot-reload registration ──
    // The watcher is started during Setup() with the engine shader directories.
    // After registering application shaders, call one of these so their
    // directories are watched too (register-before-Start is also supported).
    void RefreshShaderWatcher();
    bool AddShaderDirectory(const std::string& directory);

    EngineContext& GetContext() { return ctx_; }
    const EngineContext& GetContext() const { return ctx_; }

private:
    // The rendered-frame pipeline, seeded by the engine: every call the direct
    // path made, registered into a FrameSchedule phase (see FrameSchedule).
    void RegisterFramePipeline();

    // A scene-less frame is a normal startup/menu state, not a per-frame
    // warning. Report the transition into it once; MarkSceneValid re-arms it.
    void ReportSceneInvalidOnce();

    // The engine's default module preset: device bootstrap, physics, resource
    // hooks, physical camera and the frame-phase seeds — registered through
    // the same public EngineModules surface a composed game uses. Called by
    // the constructor (no runtime state is needed to register), so a composed
    // game composes against the preset slots any time before Setup.
    void RegisterEngineDefaultModules();

    // Wipes the registry and re-registers the preset: the teardown of a
    // Shutdown or a failed startup leaves a fresh engine ready for Setup.
    void RestoreDefaultModules();

    // The engine's built-in technique set (DefaultMesh + its uv/mask variant
    // matrix, plus Unlit): the default TechniqueRegistrar body.
    bool RegisterEngineTechniques(VulkanEngine::Application::ApplicationContext& ctx,
                                  VulkanEngine::ShaderSystem::ShaderManager& shaders,
                                  VulkanEngine::ShaderSystem::ShaderId vert_id,
                                  VulkanEngine::ShaderSystem::ShaderId frag_id);

    // Dispatches to SetTechniqueRegistrar's override, else the built-in set.
    bool RegisterTechniques(VulkanEngine::Application::ApplicationContext& ctx,
                            VulkanEngine::ShaderSystem::ShaderManager* shader_mgr,
                            VulkanEngine::ShaderSystem::ShaderId vert_id,
                            VulkanEngine::ShaderSystem::ShaderId frag_id);

    EngineContext ctx_;
    EngineBootstrap bootstrap_;

    VulkanShared::ScopedHandle<void(void*)> imgui_event_token_{};

    VulkanBackend::Vulkan::VulkanBootstrap* vk_backend_ = nullptr;
    GameConfig config_{};

    Modules::EngineModules modules_{};
    TechniqueRegistrarFn technique_registrar_{};

    EntityId camera_entity_{};

    // Data-component systems, in registration order (see AddSystem).
    VulkanEngine::SystemSchedule systems_{};

    // Rendered-frame phase systems, in phase + registration order
    // (see AddFrameSystem).
    FrameSchedule frames_{};

    // The granted service view for object components (see GetComponentContext).
    ComponentContext component_context_{};

    std::uint16_t main_technique_id_ = 0;
    std::uint16_t main_draw_group_ = 0;
    bool scene_valid_ = false;
    bool scene_invalid_reported_ = false;
    bool initialized_ = false;
};

} // namespace VulkanEngine
