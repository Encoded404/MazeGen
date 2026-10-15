add_library(VulkanShared STATIC)
target_compile_features(VulkanShared PUBLIC cxx_std_23)

# Vulkan targets (Vulkan::HppModule, Vulkan::Vulkan) are configured by the
# top-level CMakeLists; on Windows targets the loader is provided by vcpkg's
# vulkan-loader port, so this must not hard-require module-mode discovery.
find_package(Vulkan QUIET)
target_link_libraries(VulkanShared PUBLIC Vulkan::HppModule)

target_include_directories(VulkanShared PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_compile_options(VulkanShared PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(VulkanShared PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            CallbackList.cppm
            DeviceScope.cppm
            FileIO.cppm
            Jobs.cppm
            ScopedTimer.cppm
            Storage.cppm
            Teardown.cppm
            Timer.cppm
            UserPaths.cppm
            render_graph/RenderGraphTypes.cppm
    )
endif()

target_sources(VulkanShared PRIVATE
    Jobs.cpp
    Timer.cpp
)

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(VulkanShared)
endif()
