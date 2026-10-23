# ── Shared configuration surface ───────────────────────────────────────
# The backend-name parser and the RendererConfig mapping live in their own
# static library so the example and the device-free test import one definition
# of the choice, the way examples/custom_pass shares its pass graph.
add_library(text_demo_core STATIC
    TextDemoConfig.cpp
)
target_compile_features(text_demo_core PUBLIC cxx_std_23)
target_link_libraries(text_demo_core PUBLIC ${ENGINE_TARGET})
target_include_directories(text_demo_core PUBLIC ${CMAKE_SOURCE_DIR})

target_compile_options(text_demo_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(text_demo_core PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(text_demo_core PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            TextDemoConfig.cppm
    )
endif()

if(TARGET project_sanitizers)
    target_link_libraries(text_demo_core PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(text_demo_core PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(text_demo_core PRIVATE project_size)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(text_demo_core)
endif()

# ── The example executable ─────────────────────────────────────────────
# No SHADERS: the example relies entirely on the engine's built-in shader set
# (the standard PBR mesh shader for its occluder and the two text pipelines).
add_app(text_demo
    APP_TITLE       "MazeGenerator Text Demo"
    MODULE_PREFIX   Examples.TextDemo
    SHADER_NAMESPACE TextDemo
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
)

target_link_libraries(text_demo PRIVATE text_demo_core)

# ── Font asset ─────────────────────────────────────────────────────────
# The one vendored test font, sourced from tests/assets/fonts instead of being
# duplicated per example. configure_app_asset_copy copies the directory, so
# OFL.txt (the SIL Open Font License) stays discoverable next to the binary at
# fonts/OFL.txt, alongside the font the example loads from fonts/.
include(CopyAppAssets)
configure_app_asset_copy(text_demo
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/tests/assets/fonts"
    DEST_SUBDIR "fonts"
)
