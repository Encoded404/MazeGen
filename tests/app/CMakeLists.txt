add_executable(app_tests
    maze_density_tests.cpp
    solver_tests.cpp
    debug_grid_tests.cpp
    generation_tests.cpp
    tiled_maze_tests.cpp
    packed_cell_grid_tests.cpp
)

# Compile the app's standalone modules into the test target so the density
# builder, generators, solvers, and debug grid can be exercised without a
# Vulkan device.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(app_tests PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_SOURCE_DIR}/app
        FILES
            ${CMAKE_SOURCE_DIR}/app/TiledMaze.cppm
            ${CMAKE_SOURCE_DIR}/app/PackedCellGrid.cppm
            ${CMAKE_SOURCE_DIR}/app/MazeDensity.cppm
            ${CMAKE_SOURCE_DIR}/app/Generation.cppm
            ${CMAKE_SOURCE_DIR}/app/generators/Random.cppm
            ${CMAKE_SOURCE_DIR}/app/generators/CustomGeneratorV1.cppm
            ${CMAKE_SOURCE_DIR}/app/Generators.cppm
            ${CMAKE_SOURCE_DIR}/app/Solver.cppm
            ${CMAKE_SOURCE_DIR}/app/solvers/AStar.cppm
            ${CMAKE_SOURCE_DIR}/app/Solvers.cppm
            ${CMAKE_SOURCE_DIR}/app/DebugGrid.cppm
    )
endif()

# The app modules compiled here are shared with MazeGeneratorApp, which links
# logiface::logiface. Some of them (e.g. generators/CustomGeneratorV1.cppm) use
# the logging interface, so the test target needs the same dependency to get the
# logging include directory, LOGIFACE_PROJECT_ROOT, and the `logiface` module.
target_link_libraries(app_tests PRIVATE logiface::logiface)

setup_test_target(app_tests)
