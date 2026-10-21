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

// ── The engine's default frame-phase systems ─────────────────────────────────
//
// Rung 1 of the modularization: each call the hardcoded per-frame pipeline used
// to make is registered here, into the FrameSchedule phase it belongs to, in
// exactly the order the old choreography ran it. A game that registers nothing
// observes this order — the same calls, the same sequence — and every check the
// old choreography performed per block stays in the block that owns it.
void GameEngine::RegisterFramePipeline() {
    // Screen text is authored in logical points while the overlay pass and the
    // glyph rasterizer are in physical pixels, so the ratio between them is
    // pushed here: once per frame, before any hook that submits text, and
    // deliberately not gated on the scene being valid, because a frame with no
    // valid scene still draws its UI. The platform owns the value; the text
    // system does not read a window it does not own.
    frames_.Add(
        FramePhase::Bridge, FrameSlot::TextDisplayScale,
        [](FramePhaseContext& fctx) {
            EngineContext& ctx = fctx.Engine();
            Application::ApplicationContext& app = fctx.App();
            if (ctx.text_system != nullptr && app.platform_state != nullptr) {
                ctx.text_system->SetDisplayScale(app.platform_state->content_scale);
            }
        },
        "frame.text_display_scale");

    {
        // Index every engine FIF ring (dynamic mesh uploads, per-frame
        // descriptors, frame block arrays) with the monotonic frame counter
        // modulo the pipeline depth. Image indices must never drive FIF slot
        // selection.
        frames_.Add(
            FramePhase::Bridge, FrameSlot::MeshProcessFrame,
            [this](FramePhaseContext& fctx) {
                if (!ctx_.mesh_manager) {
                    LOGIFACE_LOG(warn, "Game::FrameUpdate: mesh_manager is null, skipping ProcessFrame");
                    return;
                }
                if (!scene_valid_) {
                    LOGIFACE_LOG(warn, "Game::FrameUpdate: scene_valid_ is false, skipping ProcessFrame");
                    return;
                }
                Application::ApplicationContext& app = fctx.App();
                const std::uint32_t fif = app.bootstrap->GetBackend().GetFramesInFlight();
                const std::uint32_t frame_index = app.frame.frame_counter % fif;
                ctx_.mesh_render_system.ProcessFrame(
                    ctx_.component_registry,
                    ctx_.mesh_registry,
                    *ctx_.mesh_manager,
                    *ctx_.scene_renderer,
                    ctx_.vertex_heap,
                    ctx_.index_heap,
                    ctx_.material_mgr,
                    frame_index);
            },
            "frame.mesh_process");
    }

    // The mesh manager's per-frame retire rides the same FIF slot as the
    // uploads, so it runs before the frame records.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::MeshEndFrame,
        [this](FramePhaseContext& fctx) {
            if (ctx_.mesh_manager) {
                Application::ApplicationContext& app = fctx.App();
                const std::uint32_t fif = app.bootstrap->GetBackend().GetFramesInFlight();
                ctx_.mesh_manager->EndFrame(app.frame.frame_counter % fif);
            }
        },
        "frame.mesh_end");

    // Pump finished async uploads into staged images before the frame records.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::TextureUploadPump,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.texture_uploader) return;
            Application::ApplicationContext& app = fctx.App();
            ctx_.texture_uploader->BeginFrame(
                app.frame.frame_counter,
                [&app](std::uint32_t recording_frame) {
                    return app.bootstrap->IsFrameComplete(recording_frame);
                });
        },
        "frame.texture_upload_pump");

    // Drain deferred bindless descriptor ops (uploads published, released slots
    // returned to the free list, retired bindings destroyed). This runs after
    // AcquireNextImage has waited the ring's in-flight fence for this slot, so
    // any op from `frame_counter - FIF` is GPU-safe. A gated publish applies
    // only when its recording frame was actually submitted.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::BindlessDrain,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.bindless_mgr) return;
            Application::ApplicationContext& app = fctx.App();
            ctx_.bindless_mgr->BeginFrame(
                app.frame.frame_counter,
                [&app](std::uint32_t recording_frame) {
                    return app.bootstrap->IsFrameComplete(recording_frame);
                },
                [](BindlessManager::TextureHandle handle, std::uint32_t /*frame*/) {
                    LOGIFACE_LOG(warn, "BindlessManager: dropped a publish for slot " +
                                           std::to_string(handle.slot) + " (recording frame not submitted)");
                });
        },
        "frame.bindless_drain");

#ifdef VKENGINE_HOT_RELOAD
    // Texture hot reload: apply completed slot swaps and reload changed files.
    // Runs after the bindless drain so a published new binding is visible, and
    // before materials flush so the rewritten slots reach the GPU this frame.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::TextureReloadPump,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.texture_reloader) return;
            (void)ctx_.texture_reloader->Pump(fctx.App().frame.frame_counter);
        },
        "frame.texture_reload_pump");
    // Font hot reload is pumped exactly alongside it: the same point in the
    // frame, after the bindless drain so a page retired by a reload is applied
    // at the right ring, and before the pass records so the reload is visible
    // to the frame being built.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::FontReloadPump,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.font_reloader) return;
            (void)ctx_.font_reloader->Pump(fctx.App().frame.frame_counter);
        },
        "frame.font_reload_pump");
#endif

    // Whole-texture residency: evict least-recently-used textures when the
    // resident set exceeds the device memory budget. Inert when no budget is
    // available. Eviction goes through the deferred bindless release, so a
    // slot's descriptor is reset to the fallback at the ring drain and no
    // in-flight frame samples a destroyed image.
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::ResidencyCollect,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.texture_residency || !ctx_.texture_residency->IsEnabled()) return;
            Application::ApplicationContext& app = fctx.App();
            const std::uint32_t residency_fif = app.bootstrap->GetBackend().GetFramesInFlight();
            (void)ctx_.texture_residency->Collect(app.frame.frame_counter,
                                                  /*min_residency_frames=*/residency_fif);
        },
        "frame.residency_collect");

    // Flush dirty material data to GPU before rendering
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::MaterialFlush,
        [this](FramePhaseContext& /*fctx*/) {
            ctx_.material_mgr.FlushDirtyMaterials();
        },
        "frame.material_flush");

    // Rebuild technique pipelines whose shaders changed (hot reload)
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::TechniquePoll,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.technique_mgr) return;
            const auto frame_index = fctx.App().bootstrap->GetSnapshot().frame_index;
            ctx_.technique_mgr->PollShaders(*ctx_.shader_manager, *ctx_.pipeline_factory,
                                            frame_index);
        },
        "frame.technique_poll");

#ifdef VKENGINE_PHYSICAL_CAMERA
    frames_.Add(
        FramePhase::PreRecord, FrameSlot::PhysicalCameraPoll,
        [this](FramePhaseContext& fctx) {
            if (!ctx_.physical_camera) return;
            const auto frame_index = fctx.App().bootstrap->GetSnapshot().frame_index;
            ctx_.physical_camera->PollShaders(frame_index);
        },
        "frame.physical_camera_poll");
#endif

    // Render submission is a Record-phase participant, not the phase itself.
    frames_.Add(
        FramePhase::Record, FrameSlot::RenderSubmit,
        [this](FramePhaseContext& fctx) {
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
            ctx_.renderer->RenderFrame(*fctx.App().bootstrap,
                                       ctx_.component_registry,
                                       *camera,
                                       *ctx_.technique_mgr,
                                       *ctx_.bindless_mgr,
                                       *ctx_.scene_renderer,
                                       ctx_.imgui_system.get(),
                                       fctx.App().frame.image_index,
                                       ctx_.texture_uploader.get()
#ifdef VKENGINE_PHYSICAL_CAMERA
                                       , ctx_.physical_camera.get()
#endif
                                       );
        },
        "frame.render_submit");
}

GameEngine::GameEngine() {
    // The default preset registers at construction, with no runtime state
    // needed: a composed game therefore composes against the preset's identity
    // slots (engine.device, engine.physics, ...) any time before Setup.
    RegisterEngineDefaultModules();
}

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

    // Modules drive the rest of setup. The default preset registered at
    // construction — through the same public EngineModules surface a composed
    // game uses — now runs its configures and startups in resolved order.
    Modules::ModuleContext mctx{};
    mctx.app = &ctx;
    mctx.engine = &ctx_;
    mctx.registry = &ctx_.component_registry;
    mctx.fixed_systems = &systems_;
    mctx.frame_systems = &frames_;
    mctx.config = &config_;

    modules_.ConfigureAll(config_);
    if (!modules_.StartupAll(Modules::Stage::Startup, mctx)) {
        // Unwind in exact reverse of the start order; nothing half-initialized
        // outlives the failed Setup.
        modules_.ShutdownAll(mctx);
        RestoreDefaultModules();
        return false;
    }

    // Grant the object-component service view (see ComponentContext): the
    // app's input system and whatever engine subsystems just started. Built
    // once here; Update receives the view instead of every component storing
    // an engine pointer.
    component_context_.input = ctx.input_system;
#ifdef VKENGINE_PHYSICS
    component_context_.physics_world = ctx_.physics_world.get();
#endif

    initialized_ = true;
    return true;
}

// ── The engine's default module preset ───────────────────────────────────────
//
// Rung 2 of the modularization: the bootstrap's job shrinks from constructing
// everything in a fixed private order to running registered modules in a
// resolved order. The preset below is registered through the same public
// EngineModules surface a composed game uses — a game orders its own modules
// against ModuleSlot identities (engine.device, engine.physics, ...) without
// editing this engine. Physics's old five-editor wiring (a Setup block, a
// Shutdown block, a context member, an accessor, export imports) is the
// reference migration and lives entirely in its module now; the world pointer
// stays in EngineContext because it is the granted service location that
// GetPhysicsWorld and the object-component context read.
void GameEngine::RestoreDefaultModules() {
    modules_.Clear();
    RegisterEngineDefaultModules();
}

void GameEngine::RegisterEngineDefaultModules() {
    // No Clear() here on purpose: a composed game registers its own modules
    // before Setup, and they must survive alongside the preset. ShutdownAll()
    // + Clear() in Shutdown()/the failure paths keep a re-Setup from hitting
    // duplicate preset slots.
    using Modules::EngineModuleDesc;
    using Modules::ModuleContext;
    using Modules::ModuleSlot;
    using Modules::Stage;

    // 1. Device bootstrap. Owns the whole GPU resource choreography
    //    (EngineBootstrap::Initialize) and its parallel teardown graph
    //    (EngineBootstrap::Shutdown) as a tested artifact.
    modules_.Add(EngineModuleDesc{
        .stage = Stage::Startup,
        .label = "engine.device",
        .slot = ModuleSlot::Device,
        .startup = [this](ModuleContext& mctx) {
            return bootstrap_.Initialize(mctx.Ctx(), mctx.Config(), *mctx.App().bootstrap);
        },
        .shutdown = [this](ModuleContext& /*mctx*/) {
            if (vk_backend_ != nullptr) {
                bootstrap_.Shutdown(ctx_, *vk_backend_);
            }
        },
    });

    // 2. Physics lifecycle: the world's construction is bracketed by the
    //    schedule, hooks and shutdown that own its frame contract.
#ifdef VKENGINE_PHYSICS
    {
        // The disk half of the cooked-shape cache is off unless a host names a
        // directory: an empty directory is the "cook everything" default, so an
        // engine that was never told where caches live never writes one. The
        // blobs are keyed by Jolt version and shape hash, so a stale or corrupt
        // file is a miss that re-cooks, never a load failure. Configured by
        // passing a custom PhysicsWorldConfig when registering this module.
        modules_.AddAfter(
            ModuleSlot::Device,
            EngineModuleDesc{
                .stage = Stage::Startup,
                .label = "engine.physics",
                .slot = ModuleSlot::Physics,
                .startup = [](ModuleContext& mctx) {
                    // Physics is constructed here rather than in EngineBootstrap
                    // because its frame contract is expressed against the ECS
                    // schedule and the registry's add/remove hooks. The world
                    // itself touches no Vulkan object, so its lifetime only has
                    // to bracket the systems that call it.
                    auto& ctx = mctx.Ctx();
                    Physics::PhysicsWorldConfig physics_config{};
                    ctx.physics_world = std::make_unique<Physics::World>();
                    if (!ctx.physics_world->Initialize(physics_config)) {
                        LOGIFACE_LOG(error, "physics: world initialization failed");
                        ctx.physics_world.reset();
                        return false;
                    }

                    // The frame contract's three physics slots, stated as
                    // constraints rather than left to registration order. A
                    // gameplay system that declares itself After PhysicsSimulate
                    // reads post-step state; Before PhysicsInput reads pre-step
                    // state. The simulate body itself is one blocking Jolt update
                    // per fixed step, on the calling thread, with Jolt fanning out
                    // onto the shared compute pool.
                    mctx.FixedSystems().Add(
                        SystemSlot::PhysicsInput,
                        [&ctx](ComponentRegistry& registry, float delta_time) {
                            if (ctx.physics_world) {
                                Physics::ApplyInputs(*ctx.physics_world, registry, delta_time);
                                // Characters are swept, not solved, so they run
                                // here: before the step, and after kinematic
                                // platforms have been pushed to their new pose,
                                // which is what lets a platform carry a
                                // character.
                                Physics::StepCharacters(*ctx.physics_world, registry, delta_time);
                            }
                        },
                        "physics.input");
                    mctx.FixedSystems().AddAfter(
                        SystemSlot::PhysicsInput, SystemSlot::PhysicsSimulate,
                        [&ctx](ComponentRegistry& /*registry*/, float /*delta_time*/) {
                            if (ctx.physics_world) ctx.physics_world->Step();
                        },
                        "physics.simulate");
                    mctx.FixedSystems().AddAfter(
                        SystemSlot::PhysicsSimulate, SystemSlot::PhysicsWriteBack,
                        [&ctx](ComponentRegistry& registry, float delta_time) {
                            if (ctx.physics_world) {
                                Physics::WriteBack(*ctx.physics_world, registry, delta_time);
                            }
                        },
                        "physics.writeback");

                    // Bodies are owned by the world but created from components,
                    // so the registry's add/remove hooks are the attachment seam
                    // (the same pattern DynamicMesh uses for its GPU
                    // allocation). The hook body receives the component by
                    // reference: the registry is still inside the add when the
                    // hook fires, so the component's mask bit is not set yet and
                    // a lookup by type would find nothing.
                    auto& registry = mctx.Registry();
                    registry.OnAdd<Physics::RigidBody>(
                        [&ctx](Entity& entity, Physics::RigidBody& body) {
                            if (ctx.physics_world) Physics::AttachBody(*ctx.physics_world, entity, body);
                        });
                    registry.OnRemove<Physics::RigidBody>(
                        [&ctx](Entity& /*entity*/, Physics::RigidBody& body) {
                            if (ctx.physics_world) Physics::DetachBody(*ctx.physics_world, body);
                        });

                    registry.OnAdd<Physics::Character>(
                        [&ctx](Entity& entity, Physics::Character& character) {
                            if (ctx.physics_world) Physics::AttachCharacter(*ctx.physics_world, entity, character);
                        });
                    registry.OnRemove<Physics::Character>(
                        [&ctx](Entity& /*entity*/, Physics::Character& character) {
                            if (ctx.physics_world) Physics::DetachCharacter(*ctx.physics_world, character);
                        });

                    // A collider change means a new shape set, and Jolt shapes
                    // are immutable once a body owns them, so the body is
                    // rebuilt at its live pose. This makes the add order
                    // irrelevant: adding the children before or after the
                    // RigidBody produces the same compound body.
                    registry.OnAdd<Physics::Collider>(
                        [&ctx](Entity& entity, Physics::Collider& collider) {
                            if (!ctx.physics_world) return;
                            if (auto* body = entity.GetComponent<Physics::RigidBody>()) {
                                Physics::RebuildBody(*ctx.physics_world, entity, *body, collider.children);
                            }
                        });
                    // On removal the children are gone by definition, so the
                    // body goes back to its single shape. The component still
                    // reads correctly here (the hook runs before the mask bit is
                    // cleared), so its size is what tells us it was empty.
                    registry.OnRemove<Physics::Collider>(
                        [&ctx](Entity& entity, Physics::Collider& /*collider*/) {
                            if (!ctx.physics_world) return;
                            if (auto* body = entity.GetComponent<Physics::RigidBody>()) {
                                Physics::RebuildBody(*ctx.physics_world, entity, *body, {});
                            }
                        });
                    return true;
                },
                .shutdown = [this](ModuleContext& /*mctx*/) {
                    // Shut the physics world down before the device teardown so
                    // no simulation step can run against a half-destroyed
                    // engine. The world owns no Vulkan object, but the systems
                    // that call it live in this engine's schedule.
                    if (ctx_.physics_world) {
                        LOGIFACE_LOG(debug, std::format("physics: {} fixed steps, {} bodies",
                                                        ctx_.physics_world->StepCount(),
                                                        ctx_.physics_world->BodyCount()));
                        ctx_.physics_world->Shutdown();
                        ctx_.physics_world.reset();
                    }
                },
            });
    }
#endif

    // 3. Engine-owned resource release hooks. DynamicMesh allocations are
    //    streamed GPU resources, and data components carry no destructor hook,
    //    so release goes through registry callbacks.
    modules_.Add(EngineModuleDesc{
        .stage = Stage::Startup,
        .label = "engine.resource_hooks",
        .slot = ModuleSlot::ResourceHooks,
        .startup = [this](ModuleContext& mctx) {
            mctx.Registry().OnRemove<Components::DynamicMesh>(
                [this](Entity& /*entity*/, Components::DynamicMesh& dynamic_mesh) {
                    if (ctx_.mesh_manager && dynamic_mesh.gpu_handle.IsValid()) {
                        ctx_.mesh_manager->Remove(dynamic_mesh.gpu_handle);
                    }
                });
            return true;
        },
    });

#ifdef VKENGINE_PHYSICAL_CAMERA
    // 4. Physical camera: the optional webcam subsystem's SDL event routing.
    //    The module owns its registration token; teardown releases it.
    {
        auto sdl_token = std::make_shared<VulkanShared::ScopedHandle<void(void*)>>();
        modules_.AddAfter(
            ModuleSlot::Device,
            EngineModuleDesc{
                .stage = Stage::Startup,
                .label = "engine.physical_camera",
                .slot = ModuleSlot::PhysicalCamera,
                .startup = [this, sdl_token](ModuleContext& mctx) {
                    if (!mctx.Ctx().physical_camera) return true;
                    *sdl_token = mctx.App().platform->GetBackend().GetSdlEventProcessors().Register(
                        [this](void* sdl_event) {
                            if (ctx_.physical_camera) {
                                ctx_.physical_camera->ProcessSdlEvent(sdl_event);
                            }
                        });
                    return true;
                },
                .shutdown = [sdl_token](ModuleContext& /*mctx*/) {
                    *sdl_token = {};
                },
            });
    }
#endif

    // 5. The rendered-frame phase seeds: every call the old direct path made,
    //    now a registrable FrameSchedule surface the game extends with
    //    AddFrameSystemAfter/Before.
    modules_.Add(EngineModuleDesc{
        .stage = Stage::Startup,
        .label = "engine.frame_pipeline",
        .slot = ModuleSlot::FramePipeline,
        .startup = [this](ModuleContext& /*mctx*/) {
            RegisterFramePipeline();
            return true;
        },
    });
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
    if (!RegisterTechniques(ctx, shader_mgr, vert_id, frag_id)) {
        return false;
    }

    {
        auto s = DebugSection("engineinit.material_manager");
        ctx_.material_mgr.Initialize(&ctx_.staging_pool);
        ctx_.material_mgr.SetTechniqueManager(ctx_.technique_mgr.get());

        // Upload the configured default lighting (see GameConfig::lighting; a
        // game overrides the values there, or at runtime with UploadLighting)
        // via staging. These defaults are the old hardcoded init constants.
        {
            const LightingDefaults& lighting = config_.lighting;
            SceneRenderer::SceneHeader header{};
            std::ranges::copy(lighting.ambient_color, std::begin(header.ambient_color));
            std::ranges::copy(lighting.sun_direction, std::begin(header.sun_direction));
            std::ranges::copy(lighting.sun_color, std::begin(header.sun_color));

            SceneRenderer::Light sun_light{};
            std::ranges::copy(lighting.sun_direction, std::begin(sun_light.direction));
            std::ranges::copy(lighting.sun_color, std::begin(sun_light.color));
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

    // Renderer init is done, so scene-stage modules (scene building, resource
    // loads, cameras) start here through the module registry the game filled
    // before Setup.
    Modules::ModuleContext mctx{};
    mctx.app = &ctx;
    mctx.engine = &ctx_;
    mctx.registry = &ctx_.component_registry;
    mctx.fixed_systems = &systems_;
    mctx.frame_systems = &frames_;
    mctx.config = &config_;
    if (!modules_.StartupAll(Modules::Stage::Scene, mctx)) {
        modules_.ShutdownAll(mctx);
        RestoreDefaultModules();
        return false;
    }

    return true;
}


/// The engine's built-in technique set, exactly as init hardcoded it. Served
/// through GameEngine::SetTechniqueRegistrar when a game replaces it with its
/// own technique/entry-point-matrix registration.
bool GameEngine::RegisterEngineTechniques(VulkanEngine::Application::ApplicationContext& ctx,
                                          VulkanEngine::ShaderSystem::ShaderManager& shader_mgr,
                                          VulkanEngine::ShaderSystem::ShaderId vert_id,
                                          VulkanEngine::ShaderSystem::ShaderId frag_id) {
    // The draw mode is resolved during SceneRenderer::Initialize (capability
    // fallback), so the technique pipelines must specialize on the resolved
    // mode, not the requested config_.draw_mode.
    config_.pipeline_config.draw_mode =
        static_cast<std::uint32_t>(ctx_.scene_renderer->GetDrawMode());
    {
        auto s = DebugSection("engineinit.technique_manager");
        auto mesh_tech = std::make_unique<TechniqueManager::DefaultMeshTechnique>();
        // Interface-variant rows: slot = uv bit (0 = inline UV0, 1 =
        // out-of-line UV1) + mask bit (1 = the alpha-masked entry-point pair).
        //   0 main  1 main_uv1  2 main_mask  3 main_mask_uv1
        // Both stages of the uv1 pair must be named main_uv1 to link on the
        // shared descriptor layout; the mask pairs reuse the same vertex
        // shaders, since the mask is a fragment-only concern.
        const std::array<TechniqueManager::VariantShaderPair, 4> rows = {{
            {vert_id, frag_id},
            {ctx_.shader_ids.main_indir_vert_uv1, ctx_.shader_ids.standard_mesh_frag_uv1},
            {vert_id, ctx_.shader_ids.standard_mesh_frag_mask},
            {ctx_.shader_ids.main_indir_vert_uv1, ctx_.shader_ids.standard_mesh_frag_mask_uv1},
        }};
        mesh_tech->SetVariantShaderRows(rows);
        if (!mesh_tech->CompileDefaultMesh(
                *ctx.bootstrap, shader_mgr, *ctx_.pipeline_factory,
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
                *ctx.bootstrap, shader_mgr, *ctx_.pipeline_factory,
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
    return true;
}

/// The default technique registrar, or the game's override when one is set.
bool GameEngine::RegisterTechniques(VulkanEngine::Application::ApplicationContext& ctx,
                                    VulkanEngine::ShaderSystem::ShaderManager* shader_mgr,
                                    VulkanEngine::ShaderSystem::ShaderId vert_id,
                                    VulkanEngine::ShaderSystem::ShaderId frag_id) {
    if (technique_registrar_) {
        return technique_registrar_(*this, ctx, *shader_mgr, vert_id, frag_id);
    }
    return RegisterEngineTechniques(ctx, *shader_mgr, vert_id, frag_id);
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
    systems_.Run(ctx_.component_registry, ctx.frame.fixed_delta_time, ctx.system_profile_sink);
    ctx_.component_registry.UpdateAllComponentsAsync(ctx.frame.fixed_delta_time,
                                                     component_context_);
    ctx_.component_registry.ApplyStructuralChanges();
}

void GameEngine::FrameUpdate(const VulkanEngine::Application::ApplicationContext& ctx) {
    // The bridge phase, once per rendered frame: the display-scale push (which
    // runs even when no valid scene exists, so a scene-less frame still draws
    // its UI) and the per-frame mesh/render processing. The systems own their
    // own gating; see RegisterFramePipeline for each block's checks.
    //
    // The frame's systems mutate application-side frame state (the hook surface
    // hands the same non-const context), so the const-ref public signature is
    // unwrapped here; this is the one place the frame schedule meets a loop.
    FramePhaseContext fctx{&const_cast<Application::ApplicationContext&>(ctx), &ctx_};
    frames_.Run(FramePhase::Bridge, fctx, ctx.system_profile_sink);
}

void GameEngine::FrameRender(const VulkanEngine::Application::ApplicationContext& ctx) {
    if (!ctx_.renderer) {
        LOGIFACE_LOG(warn, "Game::FrameRender: renderer is null, skipping");
        return;
    }
    if (GetCamera() == nullptr) {
        LOGIFACE_LOG(warn, "Game::FrameRender: camera is null, skipping");
        return;
    }
    if (!scene_valid_) {
        LOGIFACE_LOG(debug, "Game::FrameRender: scene_valid_ is false, skipping");
        return;
    }

    // Pre-record drains/flushes, then render submission, then post-present
    // work — the phases in their fixed order, each running its resolved
    // member sequence.
    FramePhaseContext fctx{&const_cast<Application::ApplicationContext&>(ctx), &ctx_};
    frames_.Run(FramePhase::PreRecord, fctx, ctx.system_profile_sink);
    frames_.Run(FramePhase::Record, fctx, ctx.system_profile_sink);
    frames_.Run(FramePhase::PostPresent, fctx, ctx.system_profile_sink);
}

void GameEngine::Shutdown() {
    if (!initialized_) return;

    // Registered modules tear down in the exact reverse of their start order
    // across all stages: camera event routing, resource hooks, the physics
    // world, then the device teardown graph. The shutdown context carries no
    // application loop (Shutdown has no ApplicationContext), so module
    // shutdowns must not read it.
    Modules::ModuleContext mctx{};
    mctx.engine = &ctx_;
    mctx.registry = &ctx_.component_registry;
    mctx.fixed_systems = &systems_;
    mctx.frame_systems = &frames_;
    mctx.config = &config_;
    modules_.ShutdownAll(mctx);
    RestoreDefaultModules(); // a fresh Setup re-registers the preset cleanly

    imgui_event_token_ = {};

    scene_valid_ = false;

    initialized_ = false;
    vk_backend_ = nullptr;
}

void GameEngine::UploadLighting(const SceneRenderer::SceneHeader& header,
                                std::span<const SceneRenderer::Light> lights) {
    if (!ctx_.scene_renderer) {
        LOGIFACE_LOG(warn, "GameEngine::UploadLighting: engine is not initialized");
        return;
    }
    ctx_.scene_renderer->UploadLighting(header, lights, ctx_.staging_pool);
}

} // namespace VulkanEngine
