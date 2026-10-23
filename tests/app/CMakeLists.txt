add_executable(app_tests
    maze_density_tests.cpp
    solver_tests.cpp
    debug_grid_tests.cpp
    generation_tests.cpp
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
            ${CMAKE_SOURCE_DIR}/app/MazeDensity.cppm
            ${CMAKE_SOURCE_DIR}/app/Generation.cppm
            ${CMAKE_SOURCE_DIR}/app/RandomGenerator.cppm
            ${CMAKE_SOURCE_DIR}/app/Generators.cppm
            ${CMAKE_SOURCE_DIR}/app/Solver.cppm
            ${CMAKE_SOURCE_DIR}/app/AStarSolver.cppm
            ${CMAKE_SOURCE_DIR}/app/Solvers.cppm
            ${CMAKE_SOURCE_DIR}/app/DebugGrid.cppm
    )
endif()

setup_test_target(app_tests)
