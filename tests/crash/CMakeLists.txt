# Crash-handler tests.
#
# crash_probe links Crash.cpp directly (not the engine) and is deliberately
# built without sanitizers: the tests must exercise the real signal /
# structured-exception handlers, which AddressSanitizer would otherwise own in
# Debug builds. Each test then spawns the probe with a crash type and inspects
# the crash report and session log it left behind.

add_executable(crash_probe
  ${CMAKE_SOURCE_DIR}/engine/core/bootstrap/Crash.cpp
  crash_probe.cpp
)
target_include_directories(crash_probe PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(crash_probe PRIVATE cxx_std_23)
set_target_properties(crash_probe PROPERTIES CXX_MODULE_STD ON)
target_compile_options(crash_probe PRIVATE ${TEST_COMPILE_OPTS})
if(CMAKE_CONFIGURATION_TYPES)
  target_compile_options(crash_probe PRIVATE ${TEST_COMPILE_OPTS_DEBUG})
endif()
target_link_options(crash_probe PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--gc-sections>
)
if(TARGET project_libcpp_static)
  target_link_libraries(crash_probe PRIVATE project_libcpp_static)
endif()
if(TARGET project_libcpp_sanitized)
  add_dependencies(crash_probe project_libcpp_sanitized)
endif()

add_executable(crash_tests
  crash_handler_tests.cpp
)
target_compile_features(crash_tests PRIVATE cxx_std_23)
set_target_properties(crash_tests PROPERTIES CXX_MODULE_STD ON)
target_compile_definitions(crash_tests PRIVATE
  CRASH_PROBE_PATH="$<TARGET_FILE:crash_probe>"
)
add_dependencies(crash_tests crash_probe)

setup_test_target(crash_tests)
