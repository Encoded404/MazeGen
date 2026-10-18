add_executable(engine_tests
  input_system_tests.cpp
  resource_system_texture_tests.cpp
  texture_format_tests.cpp
  texture_upload_scheduler_tests.cpp
  bindless_capacity_tests.cpp
  sampler_cache_tests.cpp
  texture_mip_chain_tests.cpp
  texture_loader_subresource_tests.cpp
  normal_encoding_material_tests.cpp
  unlit_material_tests.cpp
  gltf_material_import_tests.cpp
  gltf_geometry_import_tests.cpp
  scene_import_uv_tests.cpp
  mesh_transform_compose_tests.cpp
  transform_override_tests.cpp
  draw_key_bound_tests.cpp
  draw_group_tests.cpp
  variant_pipeline_state_tests.cpp
  uv_variant_tests.cpp
  frame_ring_tests.cpp
  residency_policy_tests.cpp
  material_slot_rewrite_tests.cpp
  texture_watcher_tests.cpp
  gpl_resolution_tests.cpp
  pipeline_template_tests.cpp
  pipeline_compiler_tests.cpp
  tlsf_allocator_tests.cpp
  transient_allocator_tests.cpp
  render_pipeline_resolution_tests.cpp
  render_pass_context_tests.cpp
  descriptor_layout_tests.cpp
  pass_pipeline_tests.cpp
  render_pipeline_registration_tests.cpp
  builtin_pass_declarations_tests.cpp
  render_resize_tests.cpp
  gpu_stats_tests.cpp
  font_library_tests.cpp
  font_loader_tests.cpp
  font_resource_tests.cpp
  font_face_tests.cpp
  text_shaping_tests.cpp
  text_layout_tests.cpp
  text_atlas_tests.cpp
  text_glyph_raster_tests.cpp
  text_msdf_tests.cpp
  text_blob_tests.cpp
  slug_text_tests.cpp
  slug_shader_provenance_tests.cpp
  text_pass_tests.cpp
  world_text_tests.cpp
  font_watcher_tests.cpp
  text_system_tests.cpp
  frame_schedule_tests.cpp
)
target_link_libraries(engine_tests PRIVATE ${ENGINE_TARGET} ${BACKEND_TARGET})

# The text-stack gate test includes and calls the third-party headers directly
# (hb-gpu.h, msdfgen.h) rather than going through an engine module, so it proves
# the libraries are installed, exported, and linkable on its own. The engine
# links them PRIVATE on purpose: the headers stay out of every other consumer.
find_package(harfbuzz CONFIG REQUIRED)
find_package(msdfgen CONFIG REQUIRED)
target_link_libraries(engine_tests PRIVATE
  harfbuzz::harfbuzz
  harfbuzz::harfbuzz-gpu
  msdfgen::msdfgen
)

target_include_directories(engine_tests PRIVATE ${TEST_SUPPORT_DIR})

# The text fixtures live in the source tree rather than being copied next to the
# test binary: they are read-only inputs, and pointing at the source keeps the
# test that reads them honest about what it is reading. Same shape as the
# GPU tests' VKENGINE_TEST_SHADER_DIR.
target_compile_definitions(engine_tests PRIVATE
  VKENGINE_TEST_FONT_DIR="${CMAKE_SOURCE_DIR}/tests/assets/fonts"
  # The Slug provenance test reads the committed port to assert it still carries
  # the two adaptations; pointing at the source keeps it honest about the file.
  VKENGINE_ENGINE_SHADER_SOURCE_DIR="${CMAKE_SOURCE_DIR}/engine/core/shaders"
)
setup_test_target(engine_tests)
