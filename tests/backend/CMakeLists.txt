add_executable(backend_tests
  vulkan_bootstrap_tests.cpp
  platform_shell_tests.cpp
  render_graph_tests.cpp
  render_graph_diagnostics_tests.cpp
  render_graph_barrier_plan_tests.cpp
  runtime_shell_tests.cpp
  event_system_tests.cpp
  component_system_tests.cpp
  system_schedule_tests.cpp
  scene_limits_tests.cpp
  block_array_limit_tests.cpp
  vulkan_capabilities_tests.cpp
  format_utils_tests.cpp
)
target_link_libraries(backend_tests PRIVATE ${BACKEND_TARGET} ${ENGINE_TARGET})

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_sources(backend_tests PRIVATE
    FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_SOURCE_DIR}/tests/runtime
    FILES
      ${CMAKE_SOURCE_DIR}/tests/runtime/test_logging.cppm
  )
endif()

enable_test_support_modules(backend_tests)

setup_test_target(backend_tests)
