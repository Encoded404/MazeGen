add_executable(app_tests
    maze_density_tests.cpp
)

# Compile the app's standalone modules into the test target so the density
# builder can be exercised without a Vulkan device.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(app_tests PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_SOURCE_DIR}/app
        FILES
            ${CMAKE_SOURCE_DIR}/app/TiledMaze.cppm
            ${CMAKE_SOURCE_DIR}/app/MazeDensity.cppm
    )
endif()

setup_test_target(app_tests)
