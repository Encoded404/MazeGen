add_library(${BACKEND_TARGET} STATIC)
target_compile_features(${BACKEND_TARGET} PUBLIC cxx_std_20)
target_link_libraries(${BACKEND_TARGET} PUBLIC logiface::logiface VulkanShared)

find_package(SDL3 CONFIG REQUIRED)
find_package(Vulkan QUIET)
find_package(imgui CONFIG REQUIRED)

target_link_libraries(${BACKEND_TARGET} PRIVATE
    SDL3::SDL3
    imgui::imgui
    Vulkan::HppModule
)

target_include_directories(${BACKEND_TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_compile_options(${BACKEND_TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(${BACKEND_TARGET} PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

set_source_files_properties(
    vulkan/VulkanInstance.cpp
    vulkan/VulkanDevice.cpp
    vulkan/DispatchLoader.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

target_sources(${BACKEND_TARGET} PRIVATE
    event/Event.cpp
    platform/SdlWindow.cpp
    platform/SdlWindowBackend.cpp
    vulkan/FramePacer.cpp
    vulkan/VulkanBootstrap.cpp
    vulkan/DispatchLoader.cpp
    vulkan/VulkanBootstrapBackend.cpp
    vulkan/VulkanInstance.cpp
    vulkan/VulkanDevice.cpp
    vulkan/VulkanSwapchain.cpp
    vulkan/VulkanCapabilities.cpp
    vulkan/FormatUtils.cpp
    vulkan/MemoryUtils.cpp
    vulkan/ImageUtils.cpp
    vulkan/PipelineBuilder.cpp
    vulkan/DebugUtils.cpp
    vulkan/RenderGraphExecutor.cpp
    imgui/ImGuiVulkanBackend.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(${BACKEND_TARGET} PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            event/Event.cppm
            platform/SdlWindow.cppm
            platform/SdlWindowBackend.cppm
            vulkan/GraphExecutionContext.cppm
            vulkan/RenderGraphContext.cppm
            vulkan/FramePacer.cppm
            vulkan/CommonTypes.cppm
            vulkan/FrameSubmissionRecord.cppm
            vulkan/VulkanBootstrap.cppm
            vulkan/VulkanBootstrapBackend.cppm
            vulkan/VulkanInstance.cppm
            vulkan/VulkanDevice.cppm
            vulkan/VulkanCapabilities.cppm
            vulkan/FormatUtils.cppm
            vulkan/VulkanSwapchain.cppm
            vulkan/MemoryUtils.cppm
            vulkan/ImageUtils.cppm
            vulkan/PipelineBuilder.cppm
            vulkan/DebugUtils.cppm
            vulkan/GpuPassTimer.cppm
            vulkan/RenderGraphExecutor.cppm
            imgui/ImGuiVulkanBackend.cppm
    )
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(${BACKEND_TARGET})
endif()
