module;

#include <logging/logging_macros.hpp>

module VulkanEngine.EngineBootstrap;

import std;
import std.compat;
import logiface;
import vulkan;

import VulkanShared.ScopedSection;
import VulkanShared.Teardown;

import VulkanEngine.BindlessManager;
import VulkanEngine.TextureUploader;
import VulkanEngine.TextureResidency;
import VulkanEngine.TextureReloader;
import VulkanEngine.TextureWatcher;
import VulkanEngine.Text.TextSystem;
import VulkanEngine.Text.FontWatcher;
import VulkanEngine.Text.FontReloader;
import VulkanEngine.GpuResources;
import VulkanEngine.DefaultTextureFactory;
import VulkanEngine.MeshManager;
import VulkanEngine.MeshRegistry;
import VulkanEngine.MeshRenderSystem;
import VulkanEngine.MaterialManager;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.TextureResource;
import VulkanEngine.EngineContext;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderWatcher;
import VulkanEngine.Text.FontLibrary;

#ifdef VKENGINE_PHYSICAL_CAMERA
import VulkanEngine.PhysicalCameraSystem;
#endif

namespace VulkanEngine {

namespace {

VulkanShared::ScopedSection DebugSection(const std::string& name) {
    return VulkanShared::ScopedSection{name, [](const std::string& section, double ms) {
        LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
    }};
}

} // anonymous namespace

bool EngineBootstrap::Initialize(EngineContext& ctx,
                                  const GameConfig& config,
                                  VulkanBackend::Vulkan::VulkanBootstrap& backend) {
    auto& vk_backend = backend.GetBackend();

    // Record the shaping/rasterization library identity once at startup. These
    // two dependencies are the usual explanation for a glyph or layout
    // difference in a bug report, and the linked version is not otherwise
    // observable from a shipped build.
    LOGIFACE_LOG(info, "text stack: " + Text::FormatLibraryVersions());

    ctx.missing_texture = DefaultTextureFactory::CreateCheckerboard(ctx.resource_manager);
    // The checkerboard must be registered so ctx.fallback_handle resolves
    // (ResourceHandle::IsValid is a manager lookup, not a truthiness check).
    ctx.fallback_handle = ctx.resource_manager.Register(ctx.missing_texture);

    ctx.bindless_mgr = std::make_unique<BindlessManager::BindlessManager>();
    {
        // Subtract every other update-after-bind pool's descriptor count so the
        // bindless pool cannot exceed the global update-after-bind budget. The
        // engine's vertex-buffer, index-buffer and indirection sets are
        // allocated per frame in flight (see SceneRenderer::Initialize).
        const std::uint32_t fif = vk_backend.GetFramesInFlight();
        constexpr std::uint32_t kVertexBuffers = 64;
        constexpr std::uint32_t kIndexBuffers = 64;
        BindlessManager::BindlessCapacityConfig capacity_config{};
        capacity_config.app_capacity = config.bindless_capacity;
        capacity_config.other_update_after_bind_descriptors =
            fif * (kVertexBuffers + kIndexBuffers + 1U);
        if (!ctx.bindless_mgr->Initialize(vk_backend, capacity_config)) {
            return false;
        }
    }

    GpuResources::HeapConfig heap_config{};
    heap_config.block_size = config.geometry_buffer_size_mb << 20;
    if (!ctx.vertex_heap.Initialize(vk_backend, heap_config, "vertex")) return false;
    if (!ctx.index_heap.Initialize(vk_backend, heap_config, "index")) return false;
    if (!ctx.uv_heap.Initialize(vk_backend, heap_config, "uv")) return false;
    if (!ctx.image_heap.Initialize(vk_backend, {}, "image")) return false;
    if (!ctx.staging_pool.Initialize(vk_backend)) return false;
    if (!ctx.sampler_cache.Initialize(vk_backend,
                                      vk_backend.GetCapabilities().GetMaxSamplerAllocationCount())) {
        return false;
    }

    // Async texture uploader: needs the image heap, staging pool, bindless
    // manager, sampler cache and the device capability snapshot, so it is
    // created last.
    ctx.texture_uploader = std::make_unique<Textures::TextureUploader>();
    if (!ctx.texture_uploader->Initialize(vk_backend, ctx.image_heap, ctx.staging_pool,
                                          *ctx.bindless_mgr, vk_backend.GetCapabilities(),
                                          &ctx.sampler_cache)) {
        return false;
    }

    // Whole-texture residency: active only with VK_EXT_memory_budget or an
    // explicit budget override, otherwise inert so the engine is unchanged.
    ctx.texture_residency = std::make_unique<Textures::TextureResidency>();
    ctx.texture_residency->Initialize(*ctx.bindless_mgr, ctx.material_mgr,
                                      vk_backend.GetCapabilities(),
                                      config.texture_memory_budget_bytes);
    ctx.texture_uploader->SetOnReserve(
        [&ctx](BindlessManager::TextureHandle handle, const ResourceId& id) {
            if (ctx.texture_residency) {
                ctx.texture_residency->RegisterResourceHandle(id, handle, 0);
            }
        });
    ctx.texture_uploader->SetOnResident(
        [&ctx](BindlessManager::TextureHandle handle, std::uint64_t bytes, std::uint32_t frame) {
            if (ctx.texture_residency) {
                ctx.texture_residency->MarkResident(handle, bytes, frame);
            }
        });

#ifdef VKENGINE_HOT_RELOAD
    // Texture hot reload. The watcher starts empty; GameEngine::LoadTexture
    // registers each source file it uploads. Without hot reload the members stay
    // null and the per-frame pump is skipped.
    ctx.texture_watcher = std::make_unique<TextureSystem::TextureWatcher>();
    ctx.texture_reloader = std::make_unique<Textures::TextureReloader>();
    ctx.texture_reloader->Initialize(*ctx.texture_watcher, ctx.resource_manager,
                                     *ctx.bindless_mgr, *ctx.texture_uploader,
                                     *ctx.texture_residency, ctx.material_mgr);
    ctx.texture_watcher->Start();
#endif

    // Text pipeline. The registry, shaping, layout, rasterization and the pure
    // atlas are device-free; the GPU glyph-atlas layer needs the bindless
    // manager, image heap, sampler cache and staging pool, all of which exist by
    // here. Text is optional, so a device whose glyph-atlas layer cannot come up
    // still boots -- it simply never draws text.
    ctx.text_system = std::make_unique<Text::TextSystem>();
    ctx.text_system->SetResourceManager(ctx.resource_manager);
    if (!ctx.text_system->InitializeGpu(vk_backend, ctx.image_heap, ctx.staging_pool,
                                        *ctx.bindless_mgr, &ctx.sampler_cache)) {
        LOGIFACE_LOG(warn, "EngineBootstrap: glyph atlas GPU layer unavailable; text stays CPU-only");
    }

#ifdef VKENGINE_HOT_RELOAD
    // Font hot reload mirrors the texture pair: the watcher starts empty and
    // GameEngine registers each font file it loads through the text system.
    ctx.font_watcher = std::make_unique<Text::FontWatcher>();
    ctx.font_reloader = std::make_unique<Text::FontReloader>();
    ctx.font_reloader->Initialize(*ctx.font_watcher, *ctx.text_system);
    ctx.font_watcher->Start();
#endif

    {
        GpuResources::HeapConfig dynamic_heap_config{};
        dynamic_heap_config.block_size = 32ULL << 20;
        dynamic_heap_config.memory_flags =
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent;

        const std::uint32_t fif = backend.GetBackend().GetFramesInFlight();
        ctx.dynamic_vertex_heaps.resize(fif);
        ctx.dynamic_index_heaps.resize(fif);
        for (std::uint32_t i = 0; i < fif; ++i) {
            if (!ctx.dynamic_vertex_heaps[i].Initialize(vk_backend, dynamic_heap_config,
                "dynamic_vertex_fifo" + std::to_string(i))) return false;
            if (!ctx.dynamic_index_heaps[i].Initialize(vk_backend, dynamic_heap_config,
                "dynamic_index_fifo" + std::to_string(i))) return false;
        }
    }

    ctx.mesh_manager = std::make_unique<MeshManager>();
    if (!ctx.mesh_manager->Initialize(vk_backend, &ctx.vertex_heap, &ctx.index_heap,
                                       &ctx.uv_heap,
                                       &ctx.staging_pool,
                                       ctx.dynamic_vertex_heaps.data(),
                                       ctx.dynamic_index_heaps.data(),
                                       static_cast<std::uint32_t>(ctx.dynamic_vertex_heaps.size()))) {
        return false;
    }

    ctx.shader_manager = std::make_unique<ShaderSystem::ShaderManager>(
        vk_backend.GetDevice(), vk_backend.GetCapabilities(),
        config.shader_cache_dir);
    ctx.pipeline_factory = std::make_unique<ShaderSystem::PipelineFactory>(
        vk_backend.GetDevice(), vk_backend.GetCapabilities(),
        ctx.shader_manager->GetPipelineCacheRAII(), config.gpl_policy, config.gpl_structure);

    if (!config.shader_data_dir.empty()) {
        ctx.shader_ids.RegisterAll(*ctx.shader_manager, config.shader_data_dir);
    }

#ifdef VKENGINE_PHYSICAL_CAMERA
    if (config.enable_physical_camera) {
        ctx.physical_camera = std::make_unique<PhysicalCamera::PhysicalCameraSystem>();
        if (!ctx.physical_camera->Initialize(vk_backend, *ctx.bindless_mgr, ctx.image_heap,
                                              ctx.sampler_cache,
                                              *ctx.shader_manager, *ctx.pipeline_factory,
                                              ctx.shader_ids.physical_camera_composite_vert,
                                              ctx.shader_ids.physical_camera_composite_frag)) {
            ctx.physical_camera.reset();
        }
    }
#endif

    return true;
}

void EngineBootstrap::Shutdown(EngineContext& ctx,
                                VulkanBackend::Vulkan::VulkanBootstrap& backend) {
    // Device idle is the first teardown task, not a serial step before the
    // scheduler: it is the single point where the device goes quiescent for
    // teardown, and every GPU-owning task depends on it. Tasks that reach a
    // device-scope call (several destructors, PhysicalCameraSystem) are also
    // serialized by the device scope inside WaitDeviceIdle, so parallel
    // destruction of disjoint state cannot race the VkQueue.
    //
    // Everything below owns GPU resources (safe to destroy once the device is
    // idle). Subsystems are torn down in parallel on a small worker pool, but
    // only where their state is genuinely disjoint.
    //
    // Dependency edges (a task waits for everything it names):
    //   - imgui_backend     -> imgui_system: both touch the same shared backend object.
    //   - shader_manager    -> shader_watcher: the watcher's efsw thread may be inside
    //     OnFileChanged(); StopDetached() quiesces it before the manager is destroyed.
    //   - texture_reloader  -> texture_watcher: the listener must stop enqueuing first.
    //   - texture_residency -> texture_uploader, texture_reloader.
    //   - font_reloader     -> font_watcher: same listener rule for font files.
    //   - text_system       -> font_reloader (it pumps into the system), renderer
    //     (the system borrows the pass the renderer owns), and it runs before the
    //     bindless manager / image heap / sampler cache / staging pool it holds
    //     descriptors, images and upload ranges in.
    //   - texture_uploader  -> scene_renderer, physical_camera: no upload may be submitted
    //     after the worker pool stops.
    //   - mesh_manager      -> mesh_registry; material_manager -> its writers/readers;
    //     technique_manager -> material_manager (its block arrays back material entries).
    //   - staging/dynamic/static heaps -> everything that owns a range in them.
    //   - sampler_cache     -> every sampler owner, and runs before the device is destroyed.
    // The ids held above are exactly the ones referenced by a later edge.
    const std::size_t worker_count =
        std::min<std::size_t>(4, std::max<std::size_t>(2, std::thread::hardware_concurrency()));
    VulkanShared::TeardownScheduler teardown{worker_count};

    // Idle barrier. Every task below that may touch the device or queue declares
    // this as a dependency, so the idle completes before any destruction starts.
    const VulkanShared::TeardownId idle_id = teardown.Add("engineshutdown.wait_idle", [&backend] {
        auto s = DebugSection("engineshutdown.wait_idle");
        backend.GetBackend().WaitDeviceIdle();
    });

    // Owner tasks referenced by dependency edges further down. Keeping the ids
    // lets a resource be destroyed only once every subsystem that reaches into
    // it has stopped, instead of racing it on the worker pool.
    std::optional<VulkanShared::TeardownId> texture_uploader_id;
    std::optional<VulkanShared::TeardownId> texture_residency_id;
    std::optional<VulkanShared::TeardownId> texture_watcher_id;
    std::optional<VulkanShared::TeardownId> texture_reloader_id;
    std::optional<VulkanShared::TeardownId> renderer_id;
    std::optional<VulkanShared::TeardownId> font_watcher_id;
    std::optional<VulkanShared::TeardownId> font_reloader_id;
    std::optional<VulkanShared::TeardownId> text_system_id;
    std::optional<VulkanShared::TeardownId> material_manager_id;
    std::optional<VulkanShared::TeardownId> scene_renderer_id;
    std::optional<VulkanShared::TeardownId> mesh_registry_id;
    std::optional<VulkanShared::TeardownId> mesh_manager_id;
#ifdef VKENGINE_PHYSICAL_CAMERA
    std::optional<VulkanShared::TeardownId> physical_camera_id;
#endif

    std::optional<VulkanShared::TeardownId> shader_watcher_id;
    if (ctx.shader_watcher) {
        shader_watcher_id = teardown.Add("engineshutdown.shader_watcher", [&ctx] {
            auto s = DebugSection("engineshutdown.shader_watcher");
            // The efsw join inside ~ShaderWatcher can block for efsw's poll
            // interval; StopDetached() quiesces inline and hands the whole
            // object to a detached thread so the join stays off this path.
            ShaderSystem::ShaderWatcher::StopDetached(std::move(ctx.shader_watcher));
        });
    }

    if (ctx.pipeline_factory) {
        teardown.Add("engineshutdown.pipeline_factory", [&ctx] {
            auto s = DebugSection("engineshutdown.pipeline_factory");
            ctx.pipeline_factory.reset();
        });
    }

    if (ctx.renderer) {
        renderer_id = teardown.Add("engineshutdown.renderer", [&ctx] {
            auto s = DebugSection("engineshutdown.renderer");
            ctx.renderer->Shutdown();
            ctx.renderer.reset();
        }, {idle_id});
    }

    std::optional<VulkanShared::TeardownId> imgui_system_id;
    if (ctx.imgui_system) {
        imgui_system_id = teardown.Add("engineshutdown.imgui_system", [&ctx] {
            auto s = DebugSection("engineshutdown.imgui_system");
            ctx.imgui_system->Shutdown();
            ctx.imgui_system.reset();
        }, {idle_id});
    }
    if (ctx.imgui_backend) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (imgui_system_id) {
            deps.push_back(*imgui_system_id);
        }
        teardown.Add("engineshutdown.imgui_backend",
                     [&ctx] {
                         auto s = DebugSection("engineshutdown.imgui_backend");
                         ctx.imgui_backend.reset();
                     },
                     std::move(deps));
    }

    if (ctx.scene_renderer) {
        scene_renderer_id = teardown.Add("engineshutdown.scene_renderer", [&ctx] {
            auto s = DebugSection("engineshutdown.scene_renderer");
            ctx.scene_renderer->Shutdown();
            ctx.scene_renderer.reset();
        }, {idle_id});
    }

#ifdef VKENGINE_PHYSICAL_CAMERA
    if (ctx.physical_camera) {
        physical_camera_id = teardown.Add("engineshutdown.physical_camera", [&ctx] {
            auto s = DebugSection("engineshutdown.physical_camera");
            ctx.physical_camera->Shutdown();
            ctx.physical_camera.reset();
        }, {idle_id});
    }
#endif

    // The uploader owns worker threads and staged images; stop it (and return
    // any staging ranges) before the staging pool and image heap are torn down.
    // It must also outlive every subsystem that can still submit an upload
    // (scene renderer, physical camera), so those stop first.
    if (ctx.texture_uploader) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        texture_uploader_id = teardown.Add("engineshutdown.texture_uploader", [&ctx] {
            auto s = DebugSection("engineshutdown.texture_uploader");
            ctx.texture_uploader->Shutdown();
            ctx.texture_uploader.reset();
        }, std::move(deps));
    }

    // The reloader references the watcher, uploader and residency; stop the
    // watcher first so no listener thread can enqueue a change mid-teardown,
    // then the reloader, and only then tear down the residency it reports into.
    if (ctx.texture_watcher) {
        texture_watcher_id = teardown.Add("engineshutdown.texture_watcher", [&ctx] {
            auto s = DebugSection("engineshutdown.texture_watcher");
            // StopDetached() quiesces inline and hands the object to a detached
            // thread: the efsw join can block for efsw's poll interval and must
            // not gate texture_reloader/residency below.
            TextureSystem::TextureWatcher::StopDetached(std::move(ctx.texture_watcher));
        });
    }
    if (ctx.texture_reloader) {
        std::vector<VulkanShared::TeardownId> deps;
        if (texture_watcher_id) deps.push_back(*texture_watcher_id);
        texture_reloader_id = teardown.Add("engineshutdown.texture_reloader", [&ctx] {
            auto s = DebugSection("engineshutdown.texture_reloader");
            ctx.texture_reloader->Shutdown();
            ctx.texture_reloader.reset();
        }, std::move(deps));
    }

    if (ctx.texture_residency) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (texture_uploader_id) deps.push_back(*texture_uploader_id);
        if (texture_reloader_id) deps.push_back(*texture_reloader_id);
        texture_residency_id = teardown.Add("engineshutdown.texture_residency", [&ctx] {
            auto s = DebugSection("engineshutdown.texture_residency");
            ctx.texture_residency->Shutdown();
            ctx.texture_residency.reset();
        }, std::move(deps));
    }

    // Font hot reload mirrors the texture pair: stop the watcher first so no
    // listener thread can enqueue a change mid-teardown, then the reloader that
    // pumps into the text system, and only then the text system itself.
    if (ctx.font_watcher) {
        font_watcher_id = teardown.Add("engineshutdown.font_watcher", [&ctx] {
            auto s = DebugSection("engineshutdown.font_watcher");
            // StopDetached() quiesces inline and hands the object to a detached
            // thread: the efsw join can block for efsw's poll interval and must
            // not gate font_reloader/text_system below.
            Text::FontWatcher::StopDetached(std::move(ctx.font_watcher));
        });
    }
    if (ctx.font_reloader) {
        std::vector<VulkanShared::TeardownId> deps;
        if (font_watcher_id) deps.push_back(*font_watcher_id);
        font_reloader_id = teardown.Add("engineshutdown.font_reloader", [&ctx] {
            auto s = DebugSection("engineshutdown.font_reloader");
            ctx.font_reloader->Shutdown();
            ctx.font_reloader.reset();
        }, std::move(deps));
    }

    // The text system holds glyph-atlas bindless slots whose images live in the
    // image heap, samples through the sampler cache, uploads through the staging
    // pool, and borrows the render pass the renderer owns. It therefore stops
    // after the reloader that pumps into it and after the renderer, and before
    // the bindless manager and heaps whose descriptors and images it holds.
    if (ctx.text_system) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (font_reloader_id) deps.push_back(*font_reloader_id);
        if (renderer_id) deps.push_back(*renderer_id);
        text_system_id = teardown.Add("engineshutdown.text_system", [&ctx] {
            auto s = DebugSection("engineshutdown.text_system");
            ctx.text_system->Shutdown();
            ctx.text_system.reset();
        }, std::move(deps));
    }

    if (ctx.shader_manager) {
        std::vector<VulkanShared::TeardownId> deps;
        if (shader_watcher_id) {
            deps.push_back(*shader_watcher_id);
        }
        teardown.Add("engineshutdown.shader_manager",
                     [&ctx] {
                         auto s = DebugSection("engineshutdown.shader_manager");
                         ctx.shader_manager.reset();
                     },
                     std::move(deps));
    }

    // The registry owns GPU allocations; stop every subsystem that reads it
    // (the scene renderer uploads meshes into it) before it is freed.
    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
        mesh_registry_id = teardown.Add("engineshutdown.mesh_registry", [&ctx] {
            auto s = DebugSection("engineshutdown.mesh_registry");
            ctx.mesh_registry.Shutdown();
        }, std::move(deps));
    }

    if (ctx.mesh_manager) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (mesh_registry_id) deps.push_back(*mesh_registry_id);
        mesh_manager_id = teardown.Add("engineshutdown.mesh_manager", [&ctx] {
            auto s = DebugSection("engineshutdown.mesh_manager");
            ctx.mesh_manager->Shutdown();
            ctx.mesh_manager.reset();
        }, std::move(deps));
    }

    // The per-frame dynamic rings, static heaps and the staging pool are all
    // written by the scene renderer / physical camera / uploader; those own the
    // ranges and must stop before the backing heaps are torn down.
    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        if (mesh_registry_id) deps.push_back(*mesh_registry_id);
        teardown.Add("engineshutdown.dynamic_heaps", [&ctx] {
            auto s = DebugSection("engineshutdown.dynamic_heaps");
            for (auto& heap : ctx.dynamic_vertex_heaps) heap.Shutdown();
            for (auto& heap : ctx.dynamic_index_heaps) heap.Shutdown();
        }, std::move(deps));
    }

    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        if (texture_uploader_id) deps.push_back(*texture_uploader_id);
        if (mesh_manager_id) deps.push_back(*mesh_manager_id);
        if (text_system_id) deps.push_back(*text_system_id);
        teardown.Add("engineshutdown.staging_manager", [&ctx] {
            auto s = DebugSection("engineshutdown.staging_manager");
            ctx.staging_pool.Shutdown();
        }, std::move(deps));
    }

    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        if (mesh_registry_id) deps.push_back(*mesh_registry_id);
        teardown.Add("engineshutdown.static_heaps", [&ctx] {
            auto s = DebugSection("engineshutdown.static_heaps");
            ctx.vertex_heap.Shutdown();
            ctx.index_heap.Shutdown();
            ctx.uv_heap.Shutdown();
        }, std::move(deps));
    }

    std::optional<VulkanShared::TeardownId> bindless_id;
    if (ctx.bindless_mgr) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        if (text_system_id) deps.push_back(*text_system_id);
        bindless_id = teardown.Add("engineshutdown.bindless_manager", [&ctx] {
            auto s = DebugSection("engineshutdown.bindless_manager");
            ctx.bindless_mgr->Shutdown();
            ctx.bindless_mgr.reset();
        }, std::move(deps));
    }

    // The image heap must outlive the bindless manager (bindless slots own
    // heap-backed textures whose destructors free heap records) and the uploader
    // (staged images free on destruction).
    {
        std::vector<VulkanShared::TeardownId> image_heap_deps{idle_id};
        if (bindless_id) image_heap_deps.push_back(*bindless_id);
        if (texture_uploader_id) image_heap_deps.push_back(*texture_uploader_id);
        if (scene_renderer_id) image_heap_deps.push_back(*scene_renderer_id);
        if (text_system_id) image_heap_deps.push_back(*text_system_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) image_heap_deps.push_back(*physical_camera_id);
#endif
        teardown.Add("engineshutdown.image_heap", [&ctx] {
            auto s = DebugSection("engineshutdown.image_heap");
            ctx.image_heap.Shutdown();
        }, std::move(image_heap_deps));
    }

    // Material entries alias technique arrays and staging-range-backed buffers,
    // and are rewritten by the texture reloader/residency. Stop every writer and
    // reader of those structures before the manager drops them.
    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
        if (texture_uploader_id) deps.push_back(*texture_uploader_id);
        if (texture_reloader_id) deps.push_back(*texture_reloader_id);
        if (texture_residency_id) deps.push_back(*texture_residency_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        material_manager_id = teardown.Add("engineshutdown.material_manager", [&ctx] {
            auto s = DebugSection("engineshutdown.material_manager");
            ctx.material_mgr.Shutdown();
        }, std::move(deps));
    }

    // Techniques own the per-material descriptor block arrays that material
    // entries point into, so they must outlive the material manager.
    if (ctx.technique_mgr) {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (material_manager_id) deps.push_back(*material_manager_id);
        teardown.Add("engineshutdown.technique_manager", [&ctx] {
            auto s = DebugSection("engineshutdown.technique_manager");
            ctx.technique_mgr->Shutdown();
            ctx.technique_mgr.reset();
        }, std::move(deps));
    }

    // The sampler cache owns every VkSampler handed to a texture, and each
    // vk::raii::Sampler holds a dispatcher pointer into the logical device. It
    // must therefore be released here, before the device is destroyed (the
    // device outlives the engine context), and after every subsystem that can
    // request a sampler has stopped.
    {
        std::vector<VulkanShared::TeardownId> deps{idle_id};
        if (bindless_id) deps.push_back(*bindless_id);
        if (texture_uploader_id) deps.push_back(*texture_uploader_id);
        if (scene_renderer_id) deps.push_back(*scene_renderer_id);
        if (text_system_id) deps.push_back(*text_system_id);
#ifdef VKENGINE_PHYSICAL_CAMERA
        if (physical_camera_id) deps.push_back(*physical_camera_id);
#endif
        teardown.Add("engineshutdown.sampler_cache", [&ctx] {
            auto s = DebugSection("engineshutdown.sampler_cache");
            ctx.sampler_cache.Shutdown();
        }, std::move(deps));
    }

    try {
        teardown.Run();
    } catch (const std::exception& err) {
        LOGIFACE_LOG(warn, std::string("Exception during parallel engine shutdown: ") + err.what());
    }
}

} // namespace VulkanEngine
