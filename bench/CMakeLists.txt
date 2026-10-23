# Headless micro-benchmark for the packed TiledMaze storage and the density
# builder. It links only the app's standalone maze modules - no engine, no
# Vulkan, no window - so it runs anywhere and in any configuration.
add_executable(maze_bench main.cpp)

# Compile the app modules into the benchmark target, the same way app_tests
# does. They are standalone modules with no engine dependencies.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(maze_bench PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_SOURCE_DIR}/app
        FILES
            ${CMAKE_SOURCE_DIR}/app/TiledMaze.cppm
            ${CMAKE_SOURCE_DIR}/app/MazeDensity.cppm
    )
endif()

find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(maze_bench PRIVATE CLI11::CLI11)

target_compile_features(maze_bench PRIVATE cxx_std_23)

target_compile_options(maze_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(maze_bench PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

if(OUTPUT_PATH)
    set_target_properties(maze_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_PATH}/bench/$<CONFIG>")
endif()

if(TARGET project_sanitizers)
    target_link_libraries(maze_bench PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(maze_bench PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(maze_bench PRIVATE project_size)
endif()
# Statically linked libc++/libc++abi for the slang-free configurations (see
# project_libcpp_static). Kept last on purpose, like add_app and
# setup_test_target do.
if(TARGET project_libcpp_static)
    target_link_libraries(maze_bench PRIVATE project_libcpp_static)
endif()
if(TARGET project_libcpp_sanitized)
    add_dependencies(maze_bench project_libcpp_sanitized)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(maze_bench)
endif()
