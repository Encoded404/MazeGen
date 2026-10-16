# Build-time secrets sealer. Standalone so it can be configured for the host
# when the main project cross-compiles (see the top-level CMakeLists).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(secrets-gen LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

get_filename_component(VULKAN_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Same trick as MikkTSpace: build the C source as C++ to keep the project
# CXX-only. Monocypher's headers carry extern "C" guards.
set_source_files_properties(
    "${VULKAN_ENGINE_ROOT}/external/Monocypher/src/monocypher.c"
    PROPERTIES LANGUAGE CXX
)

add_executable(secrets-gen
    main.cpp
    "${VULKAN_ENGINE_ROOT}/external/Monocypher/src/monocypher.c"
)

target_compile_features(secrets-gen PRIVATE cxx_std_20)
target_include_directories(secrets-gen PRIVATE
    "${VULKAN_ENGINE_ROOT}"
    "${VULKAN_ENGINE_ROOT}/external/Monocypher/src"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_target_properties(secrets-gen PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Statically linked libc++/libc++abi for the slang-free configurations (see
# project_libcpp_static in the top-level CMakeLists; the target only exists
# there, not in standalone cross-builds, which keep the dynamic link).
if(TARGET project_libcpp_static)
    target_link_libraries(secrets-gen PRIVATE project_libcpp_static)
endif()
# Build-order: sanitized libc++ archives exist before the link consumes them.
if(TARGET project_libcpp_sanitized)
    add_dependencies(secrets-gen project_libcpp_sanitized)
endif()
