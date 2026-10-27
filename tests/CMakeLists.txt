find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# Directory holding the shared test modules (fakes, device probe). Exposed as a
# normal variable so the helper below works from every test subdirectory.
set(TEST_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/support)

# Shared options and helper shared by all test subdirectories
set(TEST_COMPILE_OPTS
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
  $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)
set(TEST_COMPILE_OPTS_DEBUG
  "$<$<CONFIG:Debug>:-g>"
  "$<$<CONFIG:Debug>:-O0>"
  "$<$<CONFIG:RelWithDebInfo>:-g>"
  "$<$<CONFIG:RelWithDebInfo>:-O2>"
  "$<$<CONFIG:Release>:-O3>"
)

# Adds the shared test-support modules to a test target's CXX_MODULES file set.
# Compiled per target (rather than linked as a library) to avoid BMI/link
# ordering issues across test executables.
function(enable_test_support_modules TARGET)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(${TARGET} PRIVATE
      FILE_SET CXX_MODULES
      BASE_DIRS ${TEST_SUPPORT_DIR}
      FILES
        ${TEST_SUPPORT_DIR}/test_vulkan_fakes.cppm
        ${TEST_SUPPORT_DIR}/test_gpu.cppm
    )
  endif()
endfunction()

function(setup_test_target TARGET)
  cmake_parse_arguments(ARG "" "" "LABELS" ${ARGN})
  target_link_libraries(${TARGET} PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(${TARGET} PRIVATE LOGIFACE_ENABLE_LOGGING=1)
  target_compile_options(${TARGET} PRIVATE ${TEST_COMPILE_OPTS})
  if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(${TARGET} PRIVATE ${TEST_COMPILE_OPTS_DEBUG})
  endif()
  target_compile_options(${TARGET} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffunction-sections>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fdata-sections>
  )
  target_link_options(${TARGET} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--gc-sections>
  )
  # Same debug-info compression as the shipped binaries; test executables are
  # not stripped, so a failing test is always symbolizable.
  if(TARGET project_size)
    target_link_libraries(${TARGET} PRIVATE project_size)
  endif()
  # Statically linked libc++/libc++abi for the slang-free configurations (see
  # project_libcpp_static); also keeps Release test runs exercising the same
  # runtime linkage a shipped configuration uses.
  if(TARGET project_libcpp_static)
    target_link_libraries(${TARGET} PRIVATE project_libcpp_static)
  endif()
  # Build-order: sanitized libc++ archives exist before the link consumes
  # them (see libcpp-sanitize in the top-level CMakeLists).
  if(TARGET project_libcpp_sanitized)
    add_dependencies(${TARGET} project_libcpp_sanitized)
  endif()
  if(ARG_LABELS)
    set(TEST_LABELS "${ARG_LABELS}")
  else()
    set(TEST_LABELS "")
  endif()
  set(TEST_ENV "TEST_LOG_DIR=${CMAKE_BINARY_DIR}/Testing/logs")
  gtest_discover_tests(${TARGET}
    PROPERTIES
      ENVIRONMENT "${TEST_ENV}"
      LABELS "${TEST_LABELS}"
  )
endfunction()

add_subdirectory(backend)
add_subdirectory(crash)
add_subdirectory(core)
add_subdirectory(gpu)
add_subdirectory(physics)
add_subdirectory(runtime)
add_subdirectory(shared)
add_subdirectory(security)
# The example tests link the example libraries, so they can only be built when
# the examples themselves are enabled.
if(VKENGINE_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()
add_subdirectory(app)
