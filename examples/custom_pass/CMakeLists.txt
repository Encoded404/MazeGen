# ── Shared pass definitions ─────────────────────────────────────────────
# The custom passes live in their own static library so the example and the
# device-free plan test import one definition of the pass graph.
add_library(custom_pass_core STATIC
    Passes.cpp
)
target_compile_features(custom_pass_core PUBLIC cxx_std_23)
target_link_libraries(custom_pass_core PUBLIC ${ENGINE_TARGET})
target_include_directories(custom_pass_core PUBLIC ${CMAKE_SOURCE_DIR})

target_compile_options(custom_pass_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(custom_pass_core PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(custom_pass_core PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            Passes.cppm
    )
endif()

if(TARGET project_sanitizers)
    target_link_libraries(custom_pass_core PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(custom_pass_core PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(custom_pass_core PRIVATE project_size)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(custom_pass_core)
endif()

# ── The example executable ──────────────────────────────────────────────
add_app(custom_pass
    OUTPUT_SUBDIR   examples
    APP_TITLE       "MazeGenerator Custom Pass"
    MODULE_PREFIX   Examples.CustomPass
    SHADER_NAMESPACE CustomPass
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
    SHADERS
        FullscreenVert fullscreen_vert.slang vertex   main
        CaptureFrag    capture_frag.slang    fragment main
        ToneMapFrag    tonemap_frag.slang    fragment main
        ExposureComp   exposure_comp.slang   compute  main
)

target_link_libraries(custom_pass PRIVATE custom_pass_core)
