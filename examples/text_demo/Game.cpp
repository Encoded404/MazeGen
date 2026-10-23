module;

#include <glm/glm.hpp> // NOLINT(misc-include-cleaner)

#include <SDL3/SDL_keycode.h>

#include <logging/logging_macros.hpp>

module Examples.TextDemo.Game;

import std;

import vulkan;

import logiface;

import VulkanEngine.GameEngine;
import VulkanEngine.ShaderManager;
import VulkanEngine.Components.Text;
import VulkanEngine.Components.Transform;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Blob;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.GlyphAtlasGpu;
import VulkanEngine.Text.Layout;
import VulkanEngine.Text.Msdf;
import VulkanEngine.Text.Shaping;

namespace Examples::TextDemo {

namespace {

// The font is the vendored Lato test font, deployed from tests/assets/fonts by
// this example's CMakeLists (with tests/assets/fonts/OFL.txt alongside it).
// Nothing is committed per-example: the 656 KB binary has one source of truth.
constexpr std::string_view kFontDir = "fonts";
constexpr std::string_view kFontFile = "Lato-Regular.ttf";

// Screen-space strings. The body is deliberately long enough to wrap and is
// centred, which is what exercises the layout stage of the submission seam.
constexpr std::string_view kScreenTitle = "MazeGenerator text";
constexpr std::string_view kScreenBody =
    "HarfBuzz shapes each string once, in design units, so a run is reusable at "
    "every size. The screen path rasterizes it hinted into an A8 page; the world "
    "path is resolution independent, either as an MSDF field or as an hb-gpu "
    "outline blob the Slug shader decodes.";

// One world-space text entity the example spawns. The string, size and
// placement are the only things that vary; the backend is a property of the
// pass, not of the entity.
struct WorldTextSpec {
    // Owned: one spec's string is built by concatenation, so a view would
    // outlive the temporary it points into.
    std::string content;
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    float world_height = 0.25f;
    std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f};
    VulkanEngine::Components::TextAlign align = VulkanEngine::Components::TextAlign::Left;
    // Wrapping width in local units; 0 means "do not wrap".
    float wrap_width = 0.0f;
};

} // namespace

TextDemoGame::TextDemoGame(VulkanEngine::SceneRenderer::TextBackend backend,
                           const std::filesystem::path& executable_path)
    : backend_(backend), exe_dir_(executable_path.parent_path()) {
    setup_token_ = hooks_.on_setup.Register([this](VulkanEngine::Application::ApplicationContext& ctx) -> bool {
        return OnSetup(ctx);
    });
    frame_render_token_ = hooks_.on_frame_render.Register(
        [this](const VulkanEngine::Application::ApplicationContext& ctx) { OnFrameRender(ctx); });
    shutdown_token_ = hooks_.on_shutdown.Register(
        [this](VulkanEngine::Application::ApplicationContext& ctx) { OnShutdown(ctx); });
}

TextDemoGame::~TextDemoGame() = default;

bool TextDemoGame::OnSetup(VulkanEngine::Application::ApplicationContext& ctx) {
    VulkanEngine::GameConfig config{};
    config.enable_imgui = false;
    // The world-text backend is chosen here, on RendererConfig, because the
    // engine's world-text built-in is built with one graphics pipeline; see
    // TextDemoConfig.
    config.renderer_config = MakeRendererConfig(backend_);
    config.renderer_config.clear_color = {0.08f, 0.09f, 0.11f, 1.0f};
    config.shader_data_dir = (exe_dir_ / "shaders").string();

    if (!engine_game_.Setup(ctx, config)) {
        return false;
    }

    // The example ships no shaders of its own: the engine's standard PBR mesh
    // shader draws the occluder and the built-in text pipelines draw the text.
    auto& shader_mgr = engine_game_.GetContext().GetShaderManager();
    const auto& shader_ids = engine_game_.GetContext().GetShaderIds();
    if (!engine_game_.InitRenderer(ctx, shader_ids.main_indir_vert, shader_ids.standard_mesh_frag,
                                   &shader_mgr)) {
        return false;
    }

    face_ = engine_game_.LoadFont(exe_dir_ / std::string(kFontDir) / std::string(kFontFile));
    if (face_ == nullptr) {
        LOGIFACE_LOG(error, "text_demo: could not load the vendored test font");
        return false;
    }

    auto& engine_ctx = engine_game_.GetContext();
    auto& registry = engine_ctx.GetComponentRegistry();
    auto* world_pass = engine_game_.GetRenderer().GetWorldTextPass();

    // ── world backend setup ────────────────────────────────────────────────
    // The Slug backend needs nothing here: its encoder is device-free and the
    // pass owns the blob storage buffer. The MSDF backend needs a generator and
    // an RGBA8 atlas of its own, because the text system's page is the screen
    // path's hinted A8 coverage.
    if (backend_ == VulkanEngine::SceneRenderer::TextBackend::Msdf) {
        if (!world_atlas_.Initialize(ctx.bootstrap->GetBackend(), engine_ctx.GetImageHeap(),
                                     engine_ctx.GetStagingPool(), engine_ctx.GetBindlessManager(),
                                     &engine_ctx.GetSamplerCache())) {
            LOGIFACE_LOG(error, "text_demo: MSDF world atlas unavailable");
            return false;
        }
        world_atlas_.SetPageFormat(VulkanEngine::Text::AtlasPageFormat::Rgba8);
        world_atlas_.SetByteSource(
            [this](std::uint64_t key) { return msdf_.BitmapForAtlasKey(key); });
        if (world_pass != nullptr) {
            // Same hook the text system installs on the screen-space pass: upload
            // the dirty field pages in the frame's command buffer, ordered before
            // the draw that samples them.
            world_pass->SetPreRecordHook(
                [this](vk::CommandBuffer cmd, std::uint32_t frame_index) {
                    (void)world_atlas_.UploadDirty(msdf_.MutableAtlas(), cmd, frame_index);
                });
        }
    }

    // ── scene geometry for the depth test ──────────────────────────────────
    // An opaque quad at z = 0, in front of the first world-text entity (which
    // sits at a negative z). The quad covers the right half of that text, so the
    // depth test is visible rather than inferred: the covered glyphs lose, the
    // rest of the line draws.
    const std::uint32_t quad_id = engine_game_.GetMeshRegistry().Register(
        VulkanEngine::SceneLoader::ToMeshData(VulkanEngine::SceneLoader::CreateFallbackQuad()));
    {
        auto& entity = registry.CreateEntity();
        auto& transform = registry.AddComponent<VulkanEngine::Components::Transform>(entity);
        transform.position = glm::vec3{0.85f, 0.0f, 0.0f};
        transform.scale = glm::vec3{0.75f, 0.6f, 1.0f};
        auto& mesh_ref =
            registry.AddComponent<VulkanEngine::Components::MeshReference>(entity);
        mesh_ref.loaded_mesh_id = quad_id;
    }

    engine_game_.MarkSceneValid();
    engine_game_.CreateCamera(registry);

    // ── world-text entities ────────────────────────────────────────────────
    const std::string backend_label{TextBackendName(backend_)};
    const std::array<WorldTextSpec, 2> specs{{
        // Behind the occluder, wrapped and centred: exercises layout in the
        // world path too, and half of it is depth-tested away.
        WorldTextSpec{
            .content = "world text, wrapped at its box width and centred",
            .position = glm::vec3{0.0f, -0.1f, -0.4f},
            .world_height = 0.24f,
            .color = {0.85f, 0.9f, 1.0f, 1.0f},
            .align = VulkanEngine::Components::TextAlign::Center,
            .wrap_width = 2.2f,
        },
        // In front of the occluder and names the backend that drew it, so a run
        // is self-identifying in a screenshot.
        WorldTextSpec{
            .content = "world backend: " + backend_label,
            .position = glm::vec3{0.0f, 0.75f, 0.3f},
            .world_height = 0.18f,
            .color = {1.0f, 0.85f, 0.4f, 1.0f},
        },
    }};
    for (const WorldTextSpec& spec : specs) {
        auto& entity = registry.CreateEntity();
        auto& transform = registry.AddComponent<VulkanEngine::Components::Transform>(entity);
        transform.position = spec.position;
        auto& text = registry.AddComponent<VulkanEngine::Components::Text>(entity);
        // The component names a font; this example already holds the only face
        // it has, so the id is a nonzero marker rather than a registry lookup.
        text.font_id = 1;
        text.content = spec.content;
        text.world_height = spec.world_height;
        text.color = spec.color;
        text.align = spec.align;
        text.wrap = spec.wrap_width > 0.0f ? VulkanEngine::Components::TextWrap::Word
                                           : VulkanEngine::Components::TextWrap::None;
        text.max_width = spec.wrap_width;
        world_text_entities_.push_back(WorldTextEntity{entity.GetId()});
    }

    LOGIFACE_LOG(info, "text_demo: world text backend '" + backend_label +
                           "', world-text pass " +
                           (world_pass != nullptr ? "present" : "MISSING") + ", accepts slug runs " +
                           (world_pass != nullptr && world_pass->AcceptsSlugRuns() ? "yes" : "no"));
    LOGIFACE_LOG(info, "text_demo: font upem " +
                           std::to_string(face_->Metrics().units_per_em) + ", " +
                           std::to_string(world_text_entities_.size()) + " world-space entities");

    ctx.quit_action_handle = ctx.input_system->BindAction(
        "quit", VulkanEngine::Input::InputBinding::Key(SDLK_ESCAPE));
    return true;
}

void TextDemoGame::OnFrameRender(
    const VulkanEngine::Application::ApplicationContext& ctx) {
    auto& text_system = engine_game_.GetTextSystem();

    // Screen-space path: this single call shapes, wraps, aligns, rasterizes
    // hinted and queues into the overlay pass. The wrapped body is what proves
    // the layout stage runs through the same seam.
    text_system.SubmitScreenText(*face_, kScreenTitle, 28.0f, 24.0f, 48.0f,
                                 std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f});
    text_system.SubmitScreenText(*face_, kScreenBody, 18.0f, 24.0f, 110.0f,
                                 std::array<float, 4>{0.75f, 0.82f, 0.92f, 1.0f},
                                 VulkanEngine::Text::LayoutOptions{
                                     .max_width = 460.0f,
                                     .align = VulkanEngine::Text::TextAlign::Center,
                                     .line_height_scale = 1.15f});

    // World-space path: the configured backend's pass draws the entities'
    // components against the scene depth buffer. The pass routes on its own
    // Backend(), so the example asks for the queue method that matches the
    // backend it selected at startup; the other method is a deliberate no-op.
    auto* world_pass = engine_game_.GetRenderer().GetWorldTextPass();
    if (world_pass != nullptr) {
        // The frame the caller is about to record. This is the monotonic counter
        // the pass's Execute() also drains the blob buffer's retire ring with,
        // not the frames-in-flight ring slot: the ring is keyed by recording
        // frame so a buffer retired by a growth is released exactly one full
        // frames-in-flight cycle later.
        const std::uint32_t recording_frame = ctx.frame.frame_counter;
        auto& registry = engine_game_.GetContext().GetComponentRegistry();
        for (const WorldTextEntity& spawned : world_text_entities_) {
            auto* spawned_entity = registry.TryGetEntity(spawned.entity);
            if (spawned_entity == nullptr) {
                continue;
            }
            auto* text = spawned_entity->GetComponent<VulkanEngine::Components::Text>();
            auto* transform = spawned_entity->GetComponent<VulkanEngine::Components::Transform>();
            if (text == nullptr || transform == nullptr) {
                continue;
            }
            const std::shared_ptr<const VulkanEngine::Text::ShapedRun> run =
                text_system.GetShapingCache().Shape(*face_, text->content);
            if (run == nullptr || run->Empty()) {
                continue;
            }
            if (world_pass->AcceptsSlugRuns()) {
                // One design-unit blob per glyph, whatever the world size; the
                // pass places it in its blob buffer and the shader reconstructs
                // coverage analytically.
                world_pass->QueueSlugRun(slug_encoder_, *text, *transform, *face_,
                                         *run, recording_frame);
            } else if (world_pass->AcceptsMsdfRuns()) {
                // Resolve before queueing, exactly as SubmitScreenText does: the
                // generator packs the CPU atlas, EnsurePages creates the GPU
                // images, and only then can an instance capture a real bindless
                // page slot instead of the fallback.
                for (const VulkanEngine::Text::ShapedGlyph& glyph : run->glyphs) {
                    (void)msdf_.Generate(*face_, glyph.glyph_id, msdf_config_);
                }
                (void)world_atlas_.EnsurePages(msdf_.Atlas());
                world_pass->QueueRun(msdf_, world_atlas_, *text, *transform, *face_,
                                     *run, msdf_config_);
            }
        }
        if (!reported_queue_) {
            reported_queue_ = true;
            LOGIFACE_LOG(info, "text_demo: queued " +
                                   std::to_string(world_pass->QueuedInstanceCount()) +
                                   " MSDF glyph quads and " +
                                   std::to_string(world_pass->QueuedSlugInstanceCount()) +
                                   " Slug glyph quads for frame " +
                                   std::to_string(recording_frame));
        }
    }

    engine_game_.FrameRender(ctx);
}

void TextDemoGame::OnShutdown(VulkanEngine::Application::ApplicationContext& /*ctx*/) {
    // The atlas borrows the engine's image heap, staging pool and bindless
    // manager, so it is released before the engine tears those down. The
    // pre-record hook dies with the pass inside engine_game_.Shutdown().
    world_atlas_.Shutdown();
    engine_game_.Shutdown();
    slug_encoder_.Clear();
}

} // namespace Examples::TextDemo
