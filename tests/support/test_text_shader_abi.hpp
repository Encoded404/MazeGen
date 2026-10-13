#pragma once

// One place declaring the text-shader pipeline ABI the GPU text tests mirror.
//
// The shipped text shaders carry a fixed resource contract: the engine bindless
// array at set 0, the pass's own instance storage buffer at set 5 binding 0
// (plus, for the Slug backend, the blob storage buffer at set 5 binding 1), and
// an 80-byte push-constant block for both world-space backends or an 8-byte
// screen-size push constant for the screen-space one. The gpu text tests
// hand-build pipelines that have to place every one of those resources exactly
// where the engine pass does, and the numbers previously lived inline in each
// test; they are named here once, with the engine declaration each mirrors.
//
// Blend/depth expectations (not numeric constants, so listed as comments):
//   - UI text   (ui_text.slang / TextPass::Setup): straight-alpha source-over
//     blend (srcAlpha / oneMinusSrcAlpha), no depth test -- the pass composites
//     over the scene's backbuffer with a plain colour load.
//   - World text (world_text.slang + slug_text.slang / WorldTextPass::Setup):
//     the same straight-alpha source-over blend, but depth test on with
//     less-or-equal (the engine's depth 0..1 convention) and depth write off.
//
// This is a plain header: the gpu_tests target already puts ${TEST_SUPPORT_DIR}
// on its include path (tests/gpu/CMakeLists.txt), so it is reachable by a plain
// #include with no target registration.

#include <cstdint>

namespace TestSupport::TextShaderAbi {

// Set 0: the engine bindless texture array. Mirrors `[[vk::binding(0, 0)]]
// allTextures[]` in engine/core/shaders/ui_text.slang, world_text.slang and
// slug_text.slang, which is the array every engine pass layout prefixes.
inline constexpr std::uint32_t kBindlessDescriptorSet = 0;

// Set 5: the first app-defined descriptor set. Mirrors
// `VulkanEngine::Render::kFirstAppDescriptorSet` (engine/core/render/
// DescriptorDecl.cppm), `[[vk::binding(0, 5)]] *Instances` in all three text
// shaders, and the `decl.set` the passes declare in TextPass::Setup /
// WorldTextPass::Setup. Sets 1-4 are reserved prefixes with empty layouts.
inline constexpr std::uint32_t kFirstAppDescriptorSet = 5;

// Binding 0 of the app set: the per-frame glyph instance storage buffer, one
// buffer for the whole draw. Mirrors `[[vk::binding(0, 5)]]` instances in
// ui_text.slang / world_text.slang / slug_text.slang and the binding-0
// DescriptorDecl the passes declare (vertex stage, one element).
inline constexpr std::uint32_t kUiTextInstanceBinding = 0;
inline constexpr std::uint32_t kWorldTextInstanceBinding = 0;
inline constexpr std::uint32_t kSlugTextInstanceBinding = 0;

// Binding 1 of the app set: the Slug backend's persistent glyph blob storage
// buffer the fragment reads the instance's flat offset into. Mirrors the
// Slug-only `blob_decl` in WorldTextPass::Setup (fragment stage only) and the
// Slug fragment's blob accessor in slug_gpu.slang.
inline constexpr std::uint32_t kSlugTextBlobBinding = 1;

// Push-constant size of the screen-space text pass: `std::array<float, 2>`
// (shader-side `float2 screen_size`), declared vertex-only. Mirrors
// TextPass::Setup's DeclarePushConstants + the `[[vk::push_constant]] float2
// screen_size` in ui_text.slang.
inline constexpr std::uint32_t kUiTextPushConstantBytes = 2U * sizeof(float);

// Push-constant size of the world-space text passes. Mirrors
// `sizeof(WorldTextConstants)` / `sizeof(SlugTextConstants)` and their
// static_asserts in engine/core/render/passes/WorldTextPass.cppm (64-byte
// view_proj + 8-byte atlas_size/viewport + padding), declared vertex+fragment.
inline constexpr std::uint32_t kWorldTextPushConstantBytes = 80;
inline constexpr std::uint32_t kSlugTextPushConstantBytes = 80;

// Stride of the per-glyph instance storage, kept here so a test binding a
// whole instance buffer or asserting a batch's size pins the engine layout.
// Mirrors the `sizeof(...) == N` static_asserts next to TextInstance
// (engine/core/render/passes/TextPass.cppm) and WorldTextInstance /
// SlugTextInstance (engine/core/render/passes/WorldTextPass.cppm); the C
// layouts are ui_text.slang TextInstance (64), world_text.slang
// WorldTextInstance (80) and slug_text.slang SlugTextInstance (76).
inline constexpr std::uint32_t kUiTextInstanceBytes = 64;
inline constexpr std::uint32_t kWorldTextInstanceBytes = 80;
inline constexpr std::uint32_t kSlugTextInstanceBytes = 76;

}  // namespace TestSupport::TextShaderAbi