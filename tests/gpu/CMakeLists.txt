add_executable(gpu_tests
  transient_allocator_gpu_tests.cpp
  shader_watcher_tests.cpp
  texture_subresource_gpu_tests.cpp
  texture_render_gpu_tests.cpp
  normal_mapping_gpu_tests.cpp
  unlit_render_gpu_tests.cpp
  uv_variant_gpu_tests.cpp
  mask_variant_gpu_tests.cpp
  bindless_slot_gpu_tests.cpp
  staging_pool_gpu_tests.cpp
  texture_uploader_gpu_tests.cpp
  sampler_cache_gpu_tests.cpp
  texture_info_gpu_tests.cpp
  texture_residency_gpu_tests.cpp
  texture_hot_reload_gpu_tests.cpp
  glyph_atlas_gpu_tests.cpp
  msdf_atlas_gpu_tests.cpp
  text_render_gpu_tests.cpp
  text_system_gpu_tests.cpp
  world_text_gpu_tests.cpp
  slug_blob_gpu_tests.cpp
  gpu_pass_timer_gpu_tests.cpp
)
find_package(Ktx CONFIG REQUIRED)
target_link_libraries(gpu_tests PRIVATE
  ${ENGINE_TARGET}
  ${BACKEND_TARGET}
  KTX::ktx
)
target_include_directories(gpu_tests PRIVATE ${TEST_SUPPORT_DIR})

enable_test_support_modules(gpu_tests)

# Headless real-device backend + offscreen graphics harness for the texture
# subresource and render tests (gpu label).
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_sources(gpu_tests PRIVATE
    FILE_SET CXX_MODULES
    BASE_DIRS ${TEST_SUPPORT_DIR}
    FILES
      ${TEST_SUPPORT_DIR}/test_vulkan_backend.cppm
      ${TEST_SUPPORT_DIR}/test_texture_render_harness.cppm
  )
endif()
target_sources(gpu_tests PRIVATE
  ${TEST_SUPPORT_DIR}/test_vulkan_backend.cpp
  ${TEST_SUPPORT_DIR}/test_texture_render_harness.cpp
)

# Test-only shader: a full-screen triangle supplying the VS->FS interface the
# real standard_mesh fragment shader declares.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/SlangSpriVCompilerHelper/cmake")
include(SlangSpirVCompiler)
set(_gpu_test_shader_dir "${CMAKE_BINARY_DIR}/generated/shaders/tests/gpu")
set(_gpu_test_shader_opt "$<$<CONFIG:Debug>:0>$<$<CONFIG:RelWithDebInfo>:2>$<$<CONFIG:Release>:3>")
add_slang_shaders(
  TARGET gpu_test_shaders
  OUTPUT_DIR ${_gpu_test_shader_dir}
  NAMESPACE Tests.Gpu
  SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  COMPILER ${VKENGINE_SLANG_SPIRV_COMPILER}
  COMPILER_DEPENDS ${VKENGINE_SLANG_COMPILER_DEPENDS}
  OPT_LEVEL ${_gpu_test_shader_opt}
  SHADERS
    test_fullscreen_vs test_fullscreen_vs.slang vertex main
    test_fullscreen_vs_uv1 test_fullscreen_vs.slang vertex main_uv1
    test_sample_frag   test_sample_frag.slang   fragment main
    test_texture_info  test_texture_info.cs.slang compute main
    test_read_page     test_read_page.cs.slang  compute main
    test_read_page_rgba test_read_page_rgba.cs.slang compute main
    test_slug_blob_probe test_slug_blob_probe.cs.slang compute main
)
add_dependencies(gpu_tests gpu_test_shaders)

# The harness loads SPIR-V straight from the build tree, exactly as the engine
# loads its own shaders from config.shader_data_dir.
get_property(_engine_shader_dir GLOBAL PROPERTY VKENGINE_ENGINE_SHADER_DIR)
target_compile_definitions(gpu_tests PRIVATE
  VKENGINE_TEST_SHADER_DIR="${_gpu_test_shader_dir}"
  VKENGINE_ENGINE_SHADER_DIR="${_engine_shader_dir}"
  VKENGINE_TEST_FONT_DIR="${CMAKE_SOURCE_DIR}/tests/assets/fonts"
)

# `gpu` label: excluded by the default test presets, opt in with the *-gpu
# test presets or `ctest --preset debug --label-regex gpu`.
#
# Leak detection is disabled by the *-gpu test presets (see CMakePresets.json):
# creating a Vulkan instance loads the loader and an ICD whose allocations are
# not all released before exit (observed inside libvulkan/ICD frames, not in
# engine code). The device-free suites still run full leak detection.
setup_test_target(gpu_tests LABELS gpu)
