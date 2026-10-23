# ── Shared simulation core ──────────────────────────────────────────────
# BalanceConfig (ruleset + runtime fingerprint) and the pure geometry helpers
# (Sweep, Wall) live in their own static library so the example, the tests and
# future leaderboard tooling can import one definition of the ruleset.
add_library(infinite_runner_core STATIC
    Balance.cpp
)
target_compile_features(infinite_runner_core PUBLIC cxx_std_23)

find_package(glm CONFIG REQUIRED)
target_link_libraries(infinite_runner_core PUBLIC glm::glm)

target_include_directories(infinite_runner_core PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_compile_options(infinite_runner_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(infinite_runner_core PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(infinite_runner_core PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            Balance.cppm
            Sweep.cppm
            Wall.cppm
    )
endif()

if(TARGET project_sanitizers)
    target_link_libraries(infinite_runner_core PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(infinite_runner_core PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(infinite_runner_core PRIVATE project_size)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(infinite_runner_core)
endif()

# ── Sealed per-example secrets ──────────────────────────────────────────
# A static library owns the generated module so the game client and the
# leaderboard server both import the same sealed values.
add_library(infinite_runner_secrets STATIC)
target_compile_features(infinite_runner_secrets PUBLIC cxx_std_23)
target_link_libraries(infinite_runner_secrets PUBLIC VulkanSecurity)

embed_example_secrets(infinite_runner_secrets
    SECRETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/secrets"
    MODULE      Examples.InfiniteRunner.Secrets
    NAMESPACE   Examples::InfiniteRunner::Secrets
    OUT_DIR     "${CMAKE_BINARY_DIR}/generated/secrets/infinite_runner"
)

# ── Leaderboard library + headless server ───────────────────────────────
add_subdirectory(leaderboard)

# ── Client-side profiles/settings (needs the per-user Storage layer) ─────
add_subdirectory(account)

# ── The example executable ──────────────────────────────────────────────
# Infinite runner: a cube weaves through gaps in oncoming walls.
# Geometry and colours are generated at runtime (procedural cube + solid
# colour textures), so no asset files are needed. It uses the engine's built-in
# standard PBR mesh shader (no per-example SHADERS).
add_app(infinite_runner
    APP_TITLE        "MazeGenerator Infinite Runner"
    MODULE_PREFIX    Examples.InfiniteRunner
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
)
target_link_libraries(infinite_runner PRIVATE
    infinite_runner_core
    infinite_runner_leaderboard
    infinite_runner_account
)
