set(RUNTIME_TARGET "EngineRuntime")

find_package(CLI11 CONFIG REQUIRED)

add_library(${RUNTIME_TARGET} STATIC)
target_compile_features(${RUNTIME_TARGET} PUBLIC cxx_std_20)
target_link_libraries(${RUNTIME_TARGET} PUBLIC ${ENGINE_TARGET} ${BACKEND_TARGET} CLI11::CLI11)
target_include_directories(${RUNTIME_TARGET} PUBLIC ${CMAKE_SOURCE_DIR})

target_compile_options(${RUNTIME_TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(${RUNTIME_TARGET} PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

target_sources(${RUNTIME_TARGET} PRIVATE
    Overrides.cpp
    ExampleCli.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(${RUNTIME_TARGET} PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            RunApplication.cppm
            Overrides.cppm
            ExampleCli.cppm
            FrameStepClock.cppm
            FrameProfiler.cppm
    )
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(${RUNTIME_TARGET})
endif()