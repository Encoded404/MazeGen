module;

#include <logging/logging_macros.hpp>
#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

module VulkanEngine.GameEngine;

import std;
import std.compat;
import logiface;

import vulkan;

import VulkanEngine.MeshManager;
import VulkanEngine.Components.DynamicMesh;
import VulkanEngine.EngineBootstrap;
import VulkanEngine.ShaderWatcher;
import VulkanShared.ScopedSection;
import VulkanShared.Storage;
import VulkanShared.UserPaths;
import VulkanEngine.TextureTypes;
import VulkanEngine.TextureFormat;
import VulkanEngine.TextureUploader;
import VulkanEngine.FileLoaders.TextureLoaders;
import FileLoader.Types;
import VulkanEngine.Text.FontReloader;
import VulkanBackend.Vulkan.VulkanBootstrap;

#ifdef VKENGINE_PHYSICS
import VulkanEngine.Physics.World;
import VulkanEngine.Physics.BodyBridge;
import VulkanEngine.Physics.Character;
#endif

namespace VulkanEngine {

namespace {

// Same pattern as EngineBootstrap.cpp: an RAII timing scope that logs its wall
// time at debug level. Kept local so GameEngine does not depend on the
// bootstrap-internal helper.
VulkanShared::ScopedSection DebugSection(const std::string& name) {
    return VulkanShared::ScopedSection{name, [](const std::string& section, double ms) {
        LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
    }};
}

// Content address for an image that arrived as bytes: identical embedded images
// share one upload, and two different ones can never collide on a resource id.
[[nodiscard]] std::uint64_t HashImageBytes(const std::span<const std::byte> bytes) {
    std::uint64_t h = 1469598103934665603ull;
    for (const std::byte b : bytes) {
        h ^= static_cast<std::uint8_t>(b);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

GameEngine::~GameEngine() {
    if (initialized_) {
        Shutdown();
    }
}

bool GameEngine::Setup(VulkanEngine::Application::ApplicationContext& ctx, const GameConfig& config) {
    vk_backend_ = ctx.bootstrap;
    config_ = config;

    // A draw-mode override from the command line replaces the game's choice.
    // Unset ("auto") leaves GameConfig::draw_mode untouched. Applied here,
    // before InitRenderer, so SceneRenderer::Initialize sees the final request
    // and its capability fallback is the single resolution point.
    if (ctx.draw_mode.has_value()) {
        config_.draw_mode = *ctx.draw_mode;
    }

    // An empty cache directory means "wherever the resolved per-user cache
    // root is". It used to default to "data/cache", which is relative to the
    // process working directory: the same build cached into a different place
    // depending on how it was launched, and a user-data reset could not tell
    // the cache apart from settings. Without storage (an embedded or test
    // harness with its own bootstrap) fall back to the OS temp directory rather
    // than to anything CWD-relative.
    if (config_.shader_cache_dir.empty()) {
        config_.shader_cache_dir =
            ctx.storage != nullptr
                ? VulkanShared::UserPaths::ToUtf8(ctx.storage->CacheDir())
                : VulkanShared::UserPaths::ToUtf8(std::filesystem::temp_directory_path() /
                                                  "vulkanengine_v5" / "cache");
    }

    if (!bootstrap_.Initialize(ctx_, config_, *ctx.bootstrap)) {
        return false;
    }

#ifdef VKENGINE_PHYSICS
    // Physics is constructed here rather than in EngineBootstrap because its
    // frame contract is expressed against the ECS schedule and the registry's
    // add/remove hooks, which are GameEngine state. The world itself touches no
    // Vulkan object, so its lifetime only has to bracket the systems that call
    // it.
    ctx_.physics_world = std::make_unique<Physics::World>();
    {
        // The disk half of the cooked-shape cache is off unless the host names a
        // directory: an empty `physics_shape_cache_dir` is the "cook everything"
        // default, so an engine that was never told where caches live never
        // writes one. The blobs are keyed by Jolt version and shape hash, so a
        // stale or corrupt file is a miss that re-cooks, never a load failure.
        Physics::PhysicsWorldConfig physics_config{};
        physics_config.shape_cache_directory = config_.physics_shape_cache_dir;
        if (!ctx_.physics_world->Initialize(physics_config)) {
            LOGIFACE_LOG(error, "physics: world initialization failed");
            ctx_.physics_world.reset();
            return false;
        }
    }

    // The frame contract's three physics slots, stated as constraints rather than
    // left to registration order. A gameplay system that declares itself After
    // PhysicsSimulate reads post-step state; Before PhysicsInput reads pre-step
    // state. The simulate body itself is one blocking Jolt update per fixed step,
    // on the calling thread, with Jolt fanning out onto the shared compute pool.
    systems_.Add(
        SystemSlot::PhysicsInput,
        [this](ComponentRegistry& registry, float delta_time) {
            if (ctx_.physics_world) {
                Physics::ApplyInputs(*ctx_.physics_world, registry, delta_time);
                // Characters are swept, not solved, so they run here: before the
                // step, and after kinematic platforms have been pushed to their
                // new pose, which is what lets a platform carry a character.
                Physics::StepCharacters(*ctx_.physics_world, registry, delta_time);
            }
        },
        "physics.input");
    systems_.AddAfter(
        SystemSlot::PhysicsInput, SystemSlot::PhysicsSimulate,
        [this](ComponentRegistry& /*registry*/, float /*delta_time*/) {
            if (ctx_.physics_world) ctx_.physics_world->Step();
        },
        "physics.simulate");
    systems_.AddAfter(
        SystemSlot::PhysicsSimulate, SystemSlot::PhysicsWriteBack,
        [this](ComponentRegistry& registry, float delta_time) {
            if (ctx_.physics_world) {
                Physics::WriteBack(*ctx_.physics_world, registry, delta_time);
            }
        },
        "physics.writeback");

    // Bodies are owned by the world but created from components, so the
    // registry's add/remove hooks are the attachment seam (the same pattern
    // DynamicMesh uses for its GPU allocation). The hook body receives the
    // component by reference: the registry is still inside the add when the hook
    // fires, so the component's mask bit is not set yet and a lookup by type
    // would find nothing.
    ctx_.component_registry.OnAdd<Physics::RigidBody>(
        [this](Entity& entity, Physics::RigidBody& body) {
            if (ctx_.physics_world) Physics::AttachBody(*ctx_.physics_world, entity, body);
        });
    ctx_.component_registry.OnRemove<Physics::RigidBody>(
        [this](Entity& /*entity*/, Physics::RigidBody& body) {
            if (ctx_.physics_world) Physics::DetachBody(*ctx_.physics_world, body);
        });

    ctx_.component_registry.OnAdd<Physics::Character>(
        [this](Entity& entity, Physics::Character& character) {
            if (ctx_.physics_world) Physics::AttachCharacter(*ctx_.physics_world, entity, character);
        });
    ctx_.component_registry.OnRemove<Physics::Character>(
        [this](Entity& /*entity*/, Physics::Character& character) {
            if (ctx_.physics_world) Physics::DetachCharacter(*ctx_.physics_world, character);
        });

    // A collider change means a new shape set, and Jolt shapes are immutable once
    // a body owns them, so the body is rebuilt at its live pose. This makes the
    // add order irrelevant: adding the children before or after the RigidBody
    // produces the same compound body.
    ctx_.component_registry.OnAdd<Physics::Collider>(
        [this](Entity& entity, Physics::Collider& collider) {
            if (!ctx_.physics_world) return;
            if (auto* body = entity.GetComponent<Physics::RigidBody>()) {
                Physics::RebuildBody(*ctx_.physics_world, entity, *body, collider.children);
            }
        });
    // On removal the children are gone by definition, so the body goes back to
    // its single shape. The component still reads correctly here (the hook runs
    // before the mask bit is cleared), so its size is what tells us it was empty.
    ctx_.component_registry.OnRemove<Physics::Collider>(
        [this](Entity& entity, Physics::Collider& /*collider*/) {
            if (!ctx_.physics_world) return;
            if (auto* body = entity.GetComponent<Physics::RigidBody>()) {
                Physics::RebuildBody(*ctx_.physics_world, entity, *body, {});
            }
        });
#endif

    // Release a DynamicMesh entity's streamed GPU allocation when the component
    // is removed or its entity is destroyed. Data components carry no destructor
    // hook, so engine-owned resources are released through registry callbacks.
    if (ctx_.mesh_manager) {
        ctx_.component_registry.OnRemove<Components::DynamicMesh>(
            [this](Entity& /*entity*/, Components::DynamicMesh& dynamic_mesh) {
                if (ctx_.mesh_manager && dynamic_mesh.gpu_handle.IsValid()) {
                    ctx_.mesh_manager->Remove(dynamic_mesh.gpu_handle);
                }
            });
    }

#ifdef VKENGINE_PHYSICAL_CAMERA
    if (ctx_.physical_camera) {
        physical_camera_sdl_token_ = ctx.platform->GetBackend().GetSdlEventProcessors().Register(
            [this](void* sdl_event) {
                if (ctx_.physical_camera) {
                    ctx_.physical_camera->ProcessSdlEvent(sdl_event);
                }
            });
    }
#endif

    initialized_ = true;
    return true;
}

uint32_t GameEngine::UploadTextureToBindless(VulkanEngine::Application::ApplicationContext& /*ctx*/,
                                             TextureResource* tex,
                                             TextureSemantic semantic,
                                             TextureNormalEncoding normal_encoding,
                                             const SamplerDesc& sampler) {
    // 0 is the permanently-bound fallback slot; a failure maps to it.
    if (!tex || !tex->HasData() || tex->GetId().value == ctx_.fallback_handle.GetId().value) {
        return BindlessManager::kFallbackSlot;
    }

    // Asynchronous upload: reserve a slot pre-bound to the fallback (safe to
    // sample immediately), hand the canonical data to the decode/transcode
    // worker pool, and return the slot now. No wait, no frame-0 command buffer.
    // The binding publishes one FIF cycle after the frame that records its
    // copies; an exhausted capacity maps to the fallback.
    if (ctx_.texture_uploader) {
        auto reservation = ctx_.texture_uploader->Reserve(semantic, normal_encoding, sampler, tex->GetId());
        if (!reservation.has_value()) {
            LOGIFACE_LOG(warn, "Texture '" + tex->GetId().value +
                                   "': bindless capacity exhausted; using fallback");
            return BindlessManager::kFallbackSlot;
        }
        auto data = std::make_shared<VulkanEngine::Textures::TextureData>(tex->GetData());
        (void)ctx_.texture_uploader->Submit(*reservation, std::move(data));
        return reservation->handle.slot;
    }

    LOGIFACE_LOG(debug, "Failed to create GPU texture for: " + tex->GetId().value + ", using fallback");
    return BindlessManager::kFallbackSlot;
}

uint32_t GameEngine::LoadTexture(VulkanEngine::Application::ApplicationContext& ctx,
                                 const std::filesystem::path& path,
                                 TextureSemantic semantic,
                                 TextureNormalEncoding normal_encoding,
                                 const SamplerDesc& sampler) {
    auto tex_handle = SceneLoader::LoadTextureFromPath(
        ctx_.resource_manager, path, ctx_.fallback_handle);
    // The fallback resource resolves to (is) the registered checkerboard;
    // it is already resident at kFallbackSlot, so short-circuit instead of
    // re-uploading it per failed load.
    if (tex_handle.IsValid() && tex_handle->HasData()
        && tex_handle.GetId().value != ctx_.fallback_handle.GetId().value) {
        const std::uint32_t slot =
            UploadTextureToBindless(ctx, tex_handle.Get(), semantic, normal_encoding, sampler);
#ifdef VKENGINE_HOT_RELOAD
        if (ctx_.texture_reloader && slot != BindlessManager::kFallbackSlot) {
            ctx_.texture_reloader->WatchResource(tex_handle.GetId(), path, semantic,
                                                 normal_encoding, sampler);
        }
#endif
        return slot;
    }
    LOGIFACE_LOG(debug, "Failed to load texture from path: " + path.string() + ", using fallback");
    return BindlessManager::kFallbackSlot;
}

uint32_t GameEngine::LoadTextureFromGltfImage(VulkanEngine::Application::ApplicationContext& ctx,
                                              const FileLoaders::Mesh::GltfImageRef& image,
                                              TextureSemantic semantic,
                                              TextureNormalEncoding normal_encoding,
                                              const SamplerDesc& sampler,
                                              const std::filesystem::path& asset_dir) {
    using FileLoaders::Mesh::GltfImageRef;

    if (image.source == GltfImageRef::Source::None) {
        return BindlessManager::kFallbackSlot;
    }

    std::vector<std::byte> bytes;
    std::filesystem::path source_path;

    if (image.source == GltfImageRef::Source::Embedded) {
        // A GLB image, a data: URI inlined by the importer, or a buffer view:
        // the payload is already in memory.
        bytes = image.embedded_bytes;
    } else {
        source_path = asset_dir.empty() || std::filesystem::path(image.uri).is_absolute()
            ? std::filesystem::path(image.uri)
            : asset_dir / image.uri;
        std::ifstream file(source_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOGIFACE_LOG(warn, "LoadTextureFromGltfImage: cannot open '" + source_path.string() +
                               "', using fallback");
            return BindlessManager::kFallbackSlot;
        }
        const auto size = static_cast<std::size_t>(file.tellg());
        bytes.resize(size);
        file.seekg(0, std::ios::beg);
        if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()),
                                   static_cast<std::streamsize>(size))) {
            LOGIFACE_LOG(warn, "LoadTextureFromGltfImage: cannot read '" + source_path.string() +
                               "', using fallback");
            return BindlessManager::kFallbackSlot;
        }
    }

    if (bytes.empty()) {
        LOGIFACE_LOG(warn, "LoadTextureFromGltfImage: empty image payload, using fallback");
        return BindlessManager::kFallbackSlot;
    }

    const std::string resource_name = !image.uri.empty()
        ? image.uri
        : "gltf_embedded_" + std::to_string(bytes.size()) + "_" +
              std::to_string(HashImageBytes(bytes));

    // The resource decodes by content (PNG/JPG/KTX2) through the same loader the
    // path-based LoadTexture uses; only the source of the bytes differs.
    auto resource = std::make_shared<TextureResource>(ResourceId{resource_name});
    const FileLoader::ByteBuffer buffer(bytes.begin(), bytes.end());
    if (!resource->Load(buffer) || !resource->HasData()) {
        LOGIFACE_LOG(warn, "LoadTextureFromGltfImage: failed to decode '" + resource_name +
                           "', using fallback");
        return BindlessManager::kFallbackSlot;
    }

    auto handle = ctx_.resource_manager.Register(resource);
    if (!handle.IsValid()) {
        LOGIFACE_LOG(warn, "LoadTextureFromGltfImage: failed to register '" + resource_name +
                           "', using fallback");
        return BindlessManager::kFallbackSlot;
    }

    const std::uint32_t slot =
        UploadTextureToBindless(ctx, handle.Get(), semantic, normal_encoding, sampler);
#ifdef VKENGINE_HOT_RELOAD
    // Only a URI image has a file to watch; embedded bytes are re-imported with
    // the asset, so there is nothing to register with the path-keyed watcher.
    if (ctx_.texture_reloader && slot != BindlessManager::kFallbackSlot && !source_path.empty()) {
        ctx_.texture_reloader->WatchResource(handle.GetId(), source_path, semantic,
                                            normal_encoding, sampler);
    }
#endif
    return slot;
}

std::shared_ptr<const VulkanEngine::Text::FontFace> GameEngine::LoadFont(
    const std::filesystem::path& path, std::uint32_t face_index) {
    if (!ctx_.text_system) {
        LOGIFACE_LOG(warn, "GameEngine::LoadFont: text system is not initialized");
        return nullptr;
    }
    const std::shared_ptr<const VulkanEngine::Text::FontFace> face =
        ctx_.text_system->LoadFontFromPath(path, face_index);
    if (face == nullptr) {
        LOGIFACE_LOG(warn, "GameEngine::LoadFont: failed to load '" + path.string() + "'");
        return nullptr;
    }
#ifdef VKENGINE_HOT_RELOAD
    if (ctx_.font_reloader) {
        // The registry id a path load assigns is the path itself, so that is the
        // id the reloader reports back when the file changes.
        ctx_.font_reloader->WatchResource(ResourceId{path.string()}, path, face_index);
    }
#endif
    return face;
}

bool GameEngine::InitRenderer(VulkanEngine::Application::ApplicationContext& ctx,
                              ShaderSystem::ShaderId vert_id,
                              ShaderSystem::ShaderId frag_id,
                              ShaderSystem::ShaderManager* shader_mgr) {
    auto& backend = ctx.bootstrap->GetBackend();

    // Initial indexed-drawing capacity. The gather pass grows these from the
    // real frame totals (EnsureSceneCapacity), so this is only a starting hint.
    const SceneRenderer::SceneCapacity initial_capacity{
        .index_count = 1u << 20,
        .vertex_span = 1u << 19,
        .submesh_count = 1u << 14,
    };
    ctx_.scene_renderer = std::make_unique<SceneRenderer::SceneRenderer>();
    {
        auto s = DebugSection("engineinit.scene_renderer");
        if (!ctx_.scene_renderer->Initialize(backend, ctx_.vertex_heap, initial_capacity,
                                             ctx_.GetShaderManager(), ctx_.GetPipelineFactory(),
                                             ctx_.GetShaderIds(), backend.GetFramesInFlight(),
                                             config_.draw_mode)) {
            LOGIFACE_LOG(error, "SceneRenderer::Initialize failed");
            return false;
        }
    }

    ctx_.technique_mgr = std::make_unique<TechniqueManager::TechniqueManager>();
    // The draw mode is resolved during Initialize (capability fallback), so the
    // technique pipelines must specialize on the resolved mode, not the
    // requested config_.draw_mode.
    config_.pipeline_config.draw_mode =
        static_cast<std::uint32_t>(ctx_.scene_renderer->GetDrawMode());
    {
        auto s = DebugSection("engineinit.technique_manager");
        auto mesh_tech = std::make_unique<TechniqueManager::DefaultMeshTechnique>();
        // Interface-variant rows: slot 0 is the base pipeline; slot 1 is the
        // out-of-line UV1 entry-point pair. Both stages must be named main_uv1
        // to link on the shared descriptor layout.
        const std::array<TechniqueManager::VariantShaderPair, 2> rows = {{
            {vert_id, frag_id},
            {ctx_.shader_ids.main_indir_vert_uv1, ctx_.shader_ids.standard_mesh_frag_uv1},
        }};
        mesh_tech->SetVariantShaderRows(rows);
        if (!mesh_tech->CompileDefaultMesh(
                *ctx.bootstrap, *shader_mgr, *ctx_.pipeline_factory,
                vert_id, frag_id, config_.pipeline_config,
                *ctx_.bindless_mgr->GetLayout(),
                *ctx_.scene_renderer->GetSubmeshVertexEntriesLayout(),
                *ctx_.scene_renderer->GetVertexBuffersLayout(),
                *ctx_.scene_renderer->GetIndirectionLayout(),
                ctx_.scene_renderer->GetSceneUniformLayout())) {
            LOGIFACE_LOG(error, "GameEngine::InitRenderer: DefaultMeshTechnique pipeline creation failed");
            return false;
        }
        auto tech_id = ctx_.technique_mgr->Register(std::move(mesh_tech));
        main_technique_id_ = tech_id.value;
        // The technique's base group (no render-state variant) is seeded at
        // registration and carries the technique id.
        main_draw_group_ = ctx_.technique_mgr->InternDrawGroup(tech_id.value, 0, 0);
        // Unlit technique: same engine sets + PerMaterial layout as the main
        // technique, but paired with the unlit fragment shader (no lighting).
        auto unlit_tech = std::make_unique<TechniqueManager::UnlitTextureTechnique>();
        if (!unlit_tech->CompileUnlit(
                *ctx.bootstrap, *shader_mgr, *ctx_.pipeline_factory,
                vert_id, ctx_.shader_ids.unlit_frag, config_.pipeline_config,
                *ctx_.bindless_mgr->GetLayout(),
                *ctx_.scene_renderer->GetSubmeshVertexEntriesLayout(),
                *ctx_.scene_renderer->GetVertexBuffersLayout(),
                *ctx_.scene_renderer->GetIndirectionLayout(),
                ctx_.scene_renderer->GetSceneUniformLayout())) {
            LOGIFACE_LOG(error, "GameEngine::InitRenderer: UnlitTextureTechnique pipeline creation failed");
            return false;
        }
        ctx_.technique_mgr->Register(std::move(unlit_tech));
    }

    {
        auto s = DebugSection("engineinit.material_manager");
        ctx_.material_mgr.Initialize(&ctx_.staging_pool);
        ctx_.material_mgr.SetTechniqueManager(ctx_.technique_mgr.get());

        // Upload initial lighting data via staging
        {
            SceneRenderer::SceneHeader header{};
            header.ambient_color[0] = 0.03f; header.ambient_color[1] = 0.03f;
            header.ambient_color[2] = 0.03f; header.ambient_color[3] = 1.0f;
            header.sun_direction[0] = 0.5f; header.sun_direction[1] = -0.707f;
            header.sun_direction[2] = 0.5f;
            header.sun_color[0] = 1.0f; header.sun_color[1] = 0.95f;
            header.sun_color[2] = 0.9f; header.sun_color[3] = 2.0f;

            SceneRenderer::Light sun_light{};
            sun_light.direction[0] = 0.5f; sun_light.direction[1] = -0.707f;
            sun_light.direction[2] = 0.5f;
            sun_light.color[0] = 1.0f; sun_light.color[1] = 0.95f;
            sun_light.color[2] = 0.9f; sun_light.color[3] = 2.0f;
            sun_light.position[3] = 0.0f; // type = directional

            std::array<SceneRenderer::Light, 1> lights = {sun_light};
            header.light_count = 1;
            ctx_.scene_renderer->UploadLighting(header, lights, ctx_.staging_pool);
        }

        // Register fallback material (ID 0): main technique, bindless checkerboard.
        // The checkerboard uploads through the dedicated fallback path landing at
        // bindless slot 0 (kFallbackSlot), never by allocation order, and its slot
        // is never released.
        ctx_.bindless_mgr->SetFallback(
            GpuResources::GpuTexture::CreateFromTextureData(
                backend, ctx_.image_heap, ctx_.missing_texture->GetData(),
                vk::Format::eR8G8B8A8Unorm, SamplerDesc{}, &ctx_.sampler_cache),
            ctx_.missing_texture->GetId());
        const std::uint32_t fallback_slot = BindlessManager::kFallbackSlot;
        [[maybe_unused]] auto fallback_handle =
            ctx_.material_mgr.Register<TechniqueManager::DefaultMeshTechnique>(
                MaterialManager::BlendMode::Opaque,
                TechniqueManager::DefaultMeshPerMaterialData{
                    .albedo_texture = fallback_slot,
                    .roughness_factor = 1.0f,
                    .metallic_factor = 0.0f,
                    .ao_factor = 1.0f
                });
    }

    {
        auto s = DebugSection("engineinit.renderer");
        ctx_.renderer = std::make_unique<Renderer::Renderer>();
        ctx_.renderer->Initialize(*ctx.bootstrap, config_.renderer_config, *ctx_.scene_renderer,
                                  &ctx_.GetShaderManager(), &ctx_.GetPipelineFactory(),
                                  &ctx_.GetShaderIds());

        // The text system's single submission entry point queues into the renderer's
        // built-in text pass. Attaching it here -- after the renderer owns the pass,
        // before the first frame records -- is the one seam that keeps submission
        // one call without the text layer owning render-graph state. Until this
        // runs, SubmitScreenText is deliberately a no-op.
        if (ctx_.text_system && ctx_.renderer) {
            ctx_.text_system->SetTextPass(ctx_.renderer->GetTextPass());
        }

        // Engine-standard descriptor set layouts let the renderer build custom-pass
        // pipelines and layouts.
        ctx_.renderer->SetEngineDescriptorSetLayouts(std::array<vk::DescriptorSetLayout, 5>{
            *ctx_.bindless_mgr->GetLayout(),
            *ctx_.scene_renderer->GetSubmeshVertexEntriesLayout(),
            *ctx_.scene_renderer->GetVertexBuffersLayout(),
            *ctx_.scene_renderer->GetIndirectionLayout(),
            ctx_.scene_renderer->GetSceneUniformLayout(),
        });
    }

    if (config_.enable_imgui) {
        auto s = DebugSection("engineinit.imgui_system");
        {
            auto s_backend = DebugSection("engineinit.imgui_backend");
            ctx_.imgui_backend = VulkanBackend::ImGui::CreateImGuiBackend();
        }
        ctx_.imgui_system = std::make_unique<ImGui::ImGuiSystem>(ctx_.imgui_backend);

        const auto surface_format = backend.GetSurfaceFormat();
        VulkanBackend::ImGui::ImGuiBackendConfig imgui_backend_config{};
        imgui_backend_config.image_count = ctx.bootstrap->GetSnapshot().swapchain_image_count;
        imgui_backend_config.swapchain_format = static_cast<vk::Format>(surface_format.format);

        ImGui::ImGuiSystemInitInfo imgui_init_info{};
        imgui_init_info.sdl_window = ctx.window;
        imgui_init_info.backend_config = imgui_backend_config;
        imgui_init_info.instance = backend.GetInstance();
        imgui_init_info.physical_device = backend.GetPhysicalDevice();
        imgui_init_info.device = backend.GetDevice();
        imgui_init_info.queue_family = backend.GetGraphicsQueueFamily();
        imgui_init_info.queue = backend.GetGraphicsQueue();
        imgui_init_info.api_version = vk::ApiVersion13;

        [[maybe_unused]] const bool imgui_ok = ctx_.imgui_system->Initialize(imgui_init_info);

        auto& platform_backend = ctx.platform->GetBackend();
        imgui_event_token_ = platform_backend.GetSdlEventProcessors().Register(
            [this](void* sdl_event) {
                if (ctx_.imgui_system && ctx_.imgui_system->IsInitialized()) {
                    ctx_.imgui_system->ProcessSDLEvent(sdl_event);
                }
            });
    }

    // ── Flush pipeline cache (warm data persists across runs) ──
    {
        auto s = DebugSection("engineinit.shader_cache_flush");
        ctx_.shader_manager->FlushCache();
    }

    // ── Start shader file watching (watches all registered slang directories) ──
    // Directories registered after this (e.g. application shaders) are picked up
    // by RefreshShaderWatcher()/AddShaderDirectory().
    {
        auto s = DebugSection("engineinit.shader_watcher");
        ctx_.shader_watcher = std::make_unique<ShaderSystem::ShaderWatcher>(*ctx_.shader_manager);
        ctx_.shader_watcher->Start();
    }

    return true;
}

bool GameEngine::SetDrawMode(SceneRenderer::DrawMode requested) {
    if (!initialized_ || !ctx_.scene_renderer || !ctx_.technique_mgr) {
        LOGIFACE_LOG(warn, "GameEngine::SetDrawMode: engine is not initialized");
        return false;
    }
    const SceneRenderer::DrawMode resolved =
        ctx_.scene_renderer->ResolveDrawMode(requested);
    if (resolved != requested) {
        LOGIFACE_LOG(warn, "GameEngine::SetDrawMode: requested draw mode unsupported; "
                           "falling back to CID");
    }
    if (resolved == ctx_.scene_renderer->GetDrawMode()) return true;

    // Reinitialize device-idles, then re-creates the mode-dependent buffers and
    // rebuilds the compaction pipelines (expand, cull, collect, depth).
    ctx_.scene_renderer->Reinitialize(resolved);

    // The main-pass technique pipelines are owned by TechniqueManager, not
    // SceneRenderer, so they must be re-specialized here. Retire-ring index 0 is
    // safe: Reinitialize already idled the device.
    if (!ctx_.technique_mgr->RebuildForDrawMode(*ctx_.shader_manager,
                                                *ctx_.pipeline_factory,
                                                static_cast<std::uint32_t>(resolved), 0)) {
        LOGIFACE_LOG(error, "GameEngine::SetDrawMode: technique pipeline rebuild failed");
        return false;
    }

    // Keep the stored config in sync so later recompiles (hot reload) use the
    // resolved mode.
    config_.pipeline_config.draw_mode = static_cast<std::uint32_t>(resolved);
    LOGIFACE_LOG(info, std::string("GameEngine: draw mode set to ") +
        (resolved == SceneRenderer::DrawMode::MID ? "MID" : "CID"));
    return true;
}

VulkanEngine::RenderPipeline::RenderPipeline& GameEngine::GetRenderPipeline() {
    return ctx_.renderer->GetRenderPipeline();
}

void GameEngine::RefreshShaderWatcher() {    if (ctx_.shader_watcher) {
        ctx_.shader_watcher->Refresh();
    }
}

bool GameEngine::AddShaderDirectory(const std::string& directory) {
    return ctx_.shader_watcher ? ctx_.shader_watcher->AddDirectory(directory) : false;
}

std::vector<GameEngine::UploadedMesh> GameEngine::UploadScene(
    VulkanEngine::Application::ApplicationContext& /*ctx*/,
    const std::vector<VulkanEngine::GpuResources::MeshData>& meshes) {
    std::vector<UploadedMesh> result;

    if (!ctx_.mesh_manager) return result;

    std::vector<VulkanEngine::SubMesh> all_submeshes;
    std::unordered_set<std::uint32_t> vertex_buffers_updated;
    std::unordered_set<std::uint32_t> index_buffers_updated;
    std::unordered_set<std::uint32_t> uv_buffers_updated;

    for (const auto& mesh_data : meshes) {
        auto handle = ctx_.mesh_manager->UploadPersistent(mesh_data);
        if (!handle.IsValid()) {
            LOGIFACE_LOG(error, "UploadScene: failed to upload mesh");
            continue;
        }

        const auto* info = ctx_.mesh_manager->GetMeshInfo(handle);
        if (!info) continue;

        const std::uint32_t index_offset = static_cast<std::uint32_t>(
            info->index_allocation.offset / sizeof(std::uint32_t));

        UploadedMesh uploaded{};
        uploaded.first_submesh = static_cast<std::uint32_t>(all_submeshes.size());
        uploaded.submesh_count = static_cast<std::uint32_t>(info->sub_meshes.size());
        uploaded.vertex_buffer_index = info->vertex_allocation.buffer_index;
        uploaded.index_buffer_index = info->index_allocation.buffer_index;

        if (info->sub_meshes.empty()) {
            const std::uint32_t total_indices = static_cast<std::uint32_t>(
                info->index_allocation.size / sizeof(std::uint32_t));
            SubMesh default_sm{};
            default_sm.index_start = index_offset;
            default_sm.index_count = total_indices;
            all_submeshes.push_back(default_sm);
            uploaded.submesh_count = 1;
        } else {
            for (const auto& sm : info->sub_meshes) {
                auto adjusted = sm;
                adjusted.index_start += index_offset;
                all_submeshes.push_back(adjusted);
            }
        }

        result.push_back(uploaded);

        // Update SceneRenderer descriptors for new heap blocks (all frames)
        if (vertex_buffers_updated.insert(info->vertex_allocation.buffer_index).second) {
            if (info->vertex_allocation.buffer_index < ctx_.vertex_heap.GetBufferCount()) {
                ctx_.scene_renderer->UpdateAllFrameVertexBufferArrayElements(
                    info->vertex_allocation.buffer_index,
                    ctx_.vertex_heap.GetBuffer(info->vertex_allocation.buffer_index),
                    ctx_.vertex_heap.GetConfig().block_size);
            }
        }
        if (index_buffers_updated.insert(info->index_allocation.buffer_index).second) {
            if (info->index_allocation.buffer_index < ctx_.index_heap.GetBufferCount()) {
                ctx_.scene_renderer->UpdateAllFrameIndexBufferArrayElements(
                    info->index_allocation.buffer_index,
                    ctx_.index_heap.GetBuffer(info->index_allocation.buffer_index),
                    ctx_.index_heap.GetConfig().block_size);
            }
        }
        if (info->has_uv1 &&
            uv_buffers_updated.insert(info->uv_allocation.buffer_index).second) {
            if (info->uv_allocation.buffer_index < ctx_.uv_heap.GetBufferCount()) {
                ctx_.scene_renderer->UpdateAllFrameUvBufferArrayElements(
                    info->uv_allocation.buffer_index,
                    ctx_.uv_heap.GetBuffer(info->uv_allocation.buffer_index),
                    ctx_.uv_heap.GetConfig().block_size);
            }
        }
    }

    ctx_.scene_renderer->SetSubmeshes(all_submeshes);
    scene_valid_ = true;

    return result;
}

std::vector<GameEngine::UploadedMesh> GameEngine::UploadSceneFromFiles(
    VulkanEngine::Application::ApplicationContext& ctx,
    const std::vector<std::filesystem::path>& file_paths,
    const std::vector<SceneLoader::MaterialId>* material_bindings) {
    std::vector<VulkanEngine::GpuResources::MeshData> mesh_data_list;
    mesh_data_list.reserve(file_paths.size());

    for (const auto& path : file_paths) {
        auto md = SceneLoader::LoadMeshData(path, material_bindings);
        mesh_data_list.push_back(std::move(md));
    }

    return UploadScene(ctx, mesh_data_list);
}

EntityId GameEngine::CreateCamera(ComponentRegistry& registry) {
    Entity& entity = registry.CreateEntity();
    registry.AddComponent<Components::Camera>(entity);
    camera_entity_ = entity.GetId();
    return camera_entity_;
}

void GameEngine::FixedUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    // Simulation phase order (docs/ecs-design.md):
    //   1. apply structural changes queued during the previous step,
    //   2. run data-component systems in registration order,
    //   3. dispatch object-component Update (parallel over stable pointers),
    //   4. apply structural changes queued during 2-3.
    // The fixed timestep equals the wall-clock delta when fixed stepping is
    // disabled. This runs on the on_fixed_update hook, so it may execute more
    // than once per rendered frame; nothing here may touch per-frame render state.
    ctx_.component_registry.ApplyStructuralChanges();
    systems_.Run(ctx_.component_registry, ctx.frame.fixed_delta_time);
    ctx_.component_registry.UpdateAllComponentsAsync(ctx.frame.fixed_delta_time);
    ctx_.component_registry.ApplyStructuralChanges();
}

void GameEngine::FrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    // Screen text is authored in logical points while the overlay pass and the glyph
    // rasterizer are in physical pixels, so the ratio between them is pushed here: once
    // per frame, before any hook that submits text, and deliberately before the early
    // returns below, because a frame with no valid scene still draws its UI. The platform
    // owns the value; the text system does not read a window it does not own.
    if (ctx_.text_system != nullptr && ctx.platform_state != nullptr) {
        ctx_.text_system->SetDisplayScale(ctx.platform_state->content_scale);
    }

    if (!ctx_.mesh_manager) {
        LOGIFACE_LOG(warn, "Game::FrameUpdate: mesh_manager is null, skipping ProcessFrame");
        return;
    }
    if (!scene_valid_) {
        LOGIFACE_LOG(warn, "Game::FrameUpdate: scene_valid_ is false, skipping ProcessFrame");
        return;
    }

    {
        // Index every engine FIF ring (dynamic mesh uploads, per-frame descriptors,
        // frame block arrays) with the monotonic frame counter modulo the pipeline
        // depth. Image indices must never drive FIF slot selection.
        const std::uint32_t fif = ctx.bootstrap->GetBackend().GetFramesInFlight();
        const std::uint32_t frame_index = ctx.frame.frame_counter % fif;
        ctx_.mesh_render_system.ProcessFrame(
            ctx_.component_registry,
            ctx_.mesh_registry,
            *ctx_.mesh_manager,
            *ctx_.scene_renderer,
            ctx_.vertex_heap,
            ctx_.index_heap,
            ctx_.material_mgr,
            frame_index);
    }
}

void GameEngine::FrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    if (!ctx_.renderer) {
        LOGIFACE_LOG(warn, "Game::FrameRender: renderer is null, skipping");
        return;
    }
    Components::Camera* const camera = GetCamera();
    if (camera == nullptr) {
        LOGIFACE_LOG(warn, "Game::FrameRender: camera is null, skipping");
        return;
    }
    if (!scene_valid_) {
        LOGIFACE_LOG(debug, "Game::FrameRender: scene_valid_ is false, skipping");
        return;
    }

    if (ctx_.mesh_manager) {
        const std::uint32_t fif = ctx.bootstrap->GetBackend().GetFramesInFlight();
        ctx_.mesh_manager->EndFrame(ctx.frame.frame_counter % fif);
    }

    // Pump finished async uploads into staged images before the frame records.
    if (ctx_.texture_uploader) {
        ctx_.texture_uploader->BeginFrame(
            ctx.frame.frame_counter,
            [&ctx](std::uint32_t recording_frame) {
                return ctx.bootstrap->IsFrameComplete(recording_frame);
            });
    }

    // Drain deferred bindless descriptor ops (uploads published, released slots
    // returned to the free list, retired bindings destroyed). This runs after
    // AcquireNextImage has waited the ring's in-flight fence for this slot, so
    // any op from `frame_counter - FIF` is GPU-safe. A gated publish applies
    // only when its recording frame was actually submitted.
    ctx_.bindless_mgr->BeginFrame(
        ctx.frame.frame_counter,
        [&ctx](std::uint32_t recording_frame) {
            return ctx.bootstrap->IsFrameComplete(recording_frame);
        },
        [](BindlessManager::TextureHandle handle, std::uint32_t /*frame*/) {
            LOGIFACE_LOG(warn, "BindlessManager: dropped a publish for slot " +
                                   std::to_string(handle.slot) + " (recording frame not submitted)");
        });

#ifdef VKENGINE_HOT_RELOAD
    // Texture hot reload: apply completed slot swaps and reload changed files.
    // Runs after the bindless drain so a published new binding is visible, and
    // before materials flush so the rewritten slots reach the GPU this frame.
    if (ctx_.texture_reloader) {
        (void)ctx_.texture_reloader->Pump(ctx.frame.frame_counter);
    }
    // Font hot reload is pumped exactly alongside it: the same point in the
    // frame, after the bindless drain so a page retired by a reload is applied
    // at the right ring, and before the pass records so the reload is visible to
    // the frame being built.
    if (ctx_.font_reloader) {
        (void)ctx_.font_reloader->Pump(ctx.frame.frame_counter);
    }
#endif

    // Whole-texture residency: evict least-recently-used textures when the
    // resident set exceeds the device memory budget. Inert when no budget is
    // available. Eviction goes through the deferred bindless release, so a
    // slot's descriptor is reset to the fallback at the ring drain and no
    // in-flight frame samples a destroyed image.
    if (ctx_.texture_residency && ctx_.texture_residency->IsEnabled()) {
        const std::uint32_t residency_fif = ctx.bootstrap->GetBackend().GetFramesInFlight();
        (void)ctx_.texture_residency->Collect(ctx.frame.frame_counter,
                                              /*min_residency_frames=*/residency_fif);
    }

    // Flush dirty material data to GPU before rendering
    ctx_.material_mgr.FlushDirtyMaterials();

    // Rebuild technique pipelines whose shaders changed (hot reload)
    const auto frame_index = ctx.bootstrap->GetSnapshot().frame_index;
    ctx_.technique_mgr->PollShaders(*ctx_.shader_manager, *ctx_.pipeline_factory,
                                    frame_index);

#ifdef VKENGINE_PHYSICAL_CAMERA
    if (ctx_.physical_camera) {
        ctx_.physical_camera->PollShaders(frame_index);
    }
#endif

    ctx_.renderer->RenderFrame(*ctx.bootstrap,
                               ctx_.component_registry,
                               *camera,
                               *ctx_.technique_mgr,
                               *ctx_.bindless_mgr,
                               *ctx_.scene_renderer,
                               ctx_.imgui_system.get(),
                               ctx.frame.image_index,
                               ctx_.texture_uploader.get()
#ifdef VKENGINE_PHYSICAL_CAMERA
                               , ctx_.physical_camera.get()
#endif
                               );
}

void GameEngine::Shutdown() {
    if (!initialized_) return;

#ifdef VKENGINE_PHYSICS
    // Shut the physics world down before the device teardown so no simulation
    // step can run against a half-destroyed engine. The world owns no Vulkan
    // object, but the systems that call it live in this engine's schedule.
    if (ctx_.physics_world) {
        LOGIFACE_LOG(debug, std::format("physics: {} fixed steps, {} bodies",
                                        ctx_.physics_world->StepCount(),
                                        ctx_.physics_world->BodyCount()));
        ctx_.physics_world->Shutdown();
        ctx_.physics_world.reset();
    }
#endif

    if (vk_backend_) {
        bootstrap_.Shutdown(ctx_, *vk_backend_);
    }

    imgui_event_token_ = {};
#ifdef VKENGINE_PHYSICAL_CAMERA
    physical_camera_sdl_token_ = {};
#endif

    if (scene_valid_) {
        scene_valid_ = false;
    }

    initialized_ = false;
    vk_backend_ = nullptr;
}

} // namespace VulkanEngine
