add_library(${ENGINE_TARGET} STATIC)
target_compile_features(${ENGINE_TARGET} PUBLIC cxx_std_23)
target_link_libraries(${ENGINE_TARGET} PUBLIC logiface::logiface)

target_link_libraries(${ENGINE_TARGET} PRIVATE
    FileLoader
    ${BACKEND_TARGET}
    VulkanShared
)

# Engine-level data encryption (secrets, at-rest blobs). PUBLIC so anything
# consuming the engine can `import VulkanEngine.DataCipher`.
target_link_libraries(${ENGINE_TARGET} PUBLIC VulkanSecurity)

find_package(Vulkan QUIET)
find_package(Ktx CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

include(PatchFastgltfSubmodule)
patch_fastgltf_submodule(
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/fastgltf"
    PATCHED_DIR_VAR _patched_fastgltf_dir
)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "")
set(FASTGLTF_ENABLE_CPP_MODULES ON CACHE BOOL "")
set(FASTGLTF_USE_STD_MODULE ON CACHE BOOL "")
add_subdirectory(${_patched_fastgltf_dir} ${CMAKE_CURRENT_BINARY_DIR}/fastgltf)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(efsw CONFIG REQUIRED)

# ── Text stack ──────────────────────────────────────────────────────────
# HarfBuzz shapes runs, FreeType rasterizes hinted greyscale glyphs for
# screen-space text, and msdfgen generates the multi-channel distance fields
# used by world-space text.
#
# harfbuzz::harfbuzz-gpu is NOT an upstream export: HarfBuzz 14 builds the GPU
# glyph encoder as a separate library and libharfbuzz defines none of the
# hb_gpu_* symbols, so the project overlay
# (vcpkg-overlays/ports/harfbuzz) adds that imported target. The same overlay
# raises the port above the registry baseline, which predates hb-gpu entirely;
# see the port for both reasons.
#
# msdfgen is requested with default-features: false in vcpkg.json, so this is
# the dependency-free core library. The engine adapts HarfBuzz outlines into
# msdfgen::Shape itself instead of pulling msdfgen's FreeType/PNG/Skia
# extensions, which keeps the generated fields on the same outline source as
# the rest of the text pipeline.
#
# freetype is also requested without its default features. Those are container
# decompression (brotli, bzip2, png, zlib) for WOFF/WOFF2 and PNG-embedded
# bitmap glyph tables; this engine rasterizes TTF/OTF outlines, so they would
# only widen the shipped closure. Its "subpixel-rendering" feature is not a
# default and is deliberately left off too: the UI path uses greyscale
# antialiasing, which does not depend on the panel's subpixel geometry.
#
# libunibreak supplies the line breaking rules layout wraps with. Where a line
# may break is a Unicode question -- CJK breaks between characters, Latin only
# at spaces and hyphens, and neither at a non-breaking space -- and the answer
# is UAX#14's pair table, which is exactly what this port is. The alternative is
# a whitespace check, which is wrong the moment the text is not English, and
# hand-rolling the table is a Unicode data file to keep in sync forever. Only
# the line breaker is used here; the same port's word and grapheme breakers stay
# unreferenced.
find_package(freetype CONFIG REQUIRED)
find_package(harfbuzz CONFIG REQUIRED)
find_package(msdfgen CONFIG REQUIRED)
find_package(libunibreak CONFIG REQUIRED)

target_link_libraries(${ENGINE_TARGET} PRIVATE
    freetype
    harfbuzz::harfbuzz
    harfbuzz::harfbuzz-gpu
    msdfgen::msdfgen
    libunibreak::libunibreak
)

# Runtime shader compilation links the Slang SDK. Two constraints shape when it
# is available:
#
#   * Windows/MinGW: the only available Slang build is the MSVC-produced release
#     archive, which is not usable as a MinGW link input here, so hot reload is
#     off entirely for that target.
#   * Release: a shipped client never recompiles shaders, and libslang-compiler
#     is a 31 MB shared library that is itself built against libstdc++. Linking
#     it would force both it and libstdc++ into the client's runtime closure
#     (see the vcpkg-overlays/ports/shader-slang notes). Hot reload is therefore
#     confined to the debuggable configurations, Debug and RelWithDebInfo.
#
# In the configurations without it, ShaderWatcher still runs and reloads the
# precompiled .spv from disk, so editing a shader still takes effect once the
# build's shader step has re-emitted the .spv — only in-process Slang
# recompilation is lost. Shaders are compiled ahead of time by the host
# slang-spirv-compiler tool (which runs on the build machine) in every
# configuration.
if(WIN32)
    set(_vkengine_hot_reload_default OFF)
else()
    set(_vkengine_hot_reload_default ON)
endif()
option(VKENGINE_HOT_RELOAD "Enable Slang SDK for runtime shader compilation (Debug/RelWithDebInfo)" ${_vkengine_hot_reload_default})
if(VKENGINE_HOT_RELOAD)
    find_package(slang CONFIG REQUIRED)
    # The macro selects between the real CompilerEngine::Compile and the stub,
    # so the define and the link must be gated by the exact same condition: a
    # configuration compiled against the Slang API needs the shared library,
    # and one compiled without it must not acquire a DT_NEEDED on it.
    target_link_libraries(${ENGINE_TARGET} PRIVATE
        "$<$<CONFIG:Debug,RelWithDebInfo>:slang::slang>")
    target_compile_definitions(${ENGINE_TARGET} PRIVATE
        "$<$<CONFIG:Debug,RelWithDebInfo>:VKENGINE_HOT_RELOAD=1>")
endif()

# ── PhysicalCamera (webcam) support ─────────────────────────────────────
# Compile-time switch: when OFF, no camera code is built, no SDL camera
# symbols are referenced and the API is absent (all uses are macro-guarded).
option(VKENGINE_PHYSICAL_CAMERA "Enable PhysicalCamera (webcam) support" ON)
if(VKENGINE_PHYSICAL_CAMERA)
    find_package(SDL3 CONFIG REQUIRED)
    target_link_libraries(${ENGINE_TARGET} PRIVATE SDL3::SDL3)
    target_compile_definitions(${ENGINE_TARGET} PUBLIC VKENGINE_PHYSICAL_CAMERA=1)
    target_sources(${ENGINE_TARGET} PRIVATE
        physical_camera/PhysicalCameraSystem.cpp
    )
endif()

target_link_libraries(${ENGINE_TARGET} PRIVATE
    Vulkan::HppModule
    KTX::ktx
    glm::glm
    fastgltf::module
    tinyobjloader::tinyobjloader
    imgui::imgui
    efsw::efsw
    MikkTSpace
)

target_include_directories(${ENGINE_TARGET} PUBLIC ${Stb_INCLUDE_DIR})
target_include_directories(${ENGINE_TARGET} PUBLIC ${CMAKE_SOURCE_DIR})

target_compile_definitions(${ENGINE_TARGET} PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
)

target_compile_definitions(${ENGINE_TARGET} PRIVATE
    VKENGINE_SLANG_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/shaders/"
)

target_compile_options(${ENGINE_TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(${ENGINE_TARGET} PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

target_sources(${ENGINE_TARGET} PRIVATE
    input/InputSystem.cpp
    gpu/BlockArray.cpp
    assets/FileLoaders/Mesh/MeshLoaderBase.cpp
    assets/FileLoaders/Mesh/GltfMaterialImport.cpp
    assets/FileLoaders/Mesh/GltfMeshAssembler.cpp
    assets/FileLoaders/Textures/TextureLoaders.cpp
    assets/FileLoaders/Textures/TextureLoaders.cppm
    assets/FileLoaders/Fonts/FontLoader.cpp
    assets/TextureFormat.cpp
    assets/ResourceCache.cpp
    assets/TextureResource.cpp
    assets/FontResource.cpp
    ui/ImGuiSystem.cpp
    render/Pipeline.cpp
    render/PipelinePass.cpp
    render/techniques/BaseTechnique.cpp
    gpu/Buffer.cpp
    gpu/Texture.cpp
    gpu/DescriptorSet.cpp
    gpu/StagingPool.cpp
    gpu/SamplerCache.cpp
    gpu/BufferHeap.cpp
    gpu/ImageHeap.cpp
    gpu/TransientAllocator.cpp
    gpu/TlsfAllocator.cpp
    assets/ShaderLoader.cpp
    assets/DefaultTextures.cpp
    render/MeshUploadManager.cpp
    render/MeshGatherSystem.cpp
    render/MeshPipeline.cpp
    assets/SceneImport.cpp
    assets/MeshCache.cpp
    components/DynamicMesh.cpp
    render/Renderer.cpp
    render/techniques/TechniqueManager.cpp
    render/materials/MaterialManager.cpp
    render/SceneRenderer.cpp
    render/SceneRendererPipeline.cpp
    render/SceneRendererFrame.cpp
    render/ShaderManager.cpp
    render/CompilerEngine.cpp
    render/PipelineFactory.cpp
    render/ShaderWatcher.cpp
    render/bindless/BindlessManager.cpp
    render/bindless/TextureUploader.cpp
    render/bindless/TextureResidency.cpp
    render/bindless/TextureReloader.cpp
    render/TextureWatcher.cpp
    text/FontLibrary.cpp
    text/Font.cpp
    text/Shaping.cpp
    text/Layout.cpp
    text/Atlas.cpp
    text/GlyphRaster.cpp
    text/Blob.cpp
    text/GlyphAtlasGpu.cpp
    text/GpuTextBlobBuffer.cpp
    text/Msdf.cpp
    text/FontWatcher.cpp
    text/FontReloader.cpp
    text/TextSystem.cpp
    GameEngine.cpp
    bootstrap/EngineBootstrap.cpp
    bootstrap/Crash.cpp
    render/passes/ExpandPass.cpp
    render/passes/HiZPass.cpp
    render/passes/OcclusionPass.cpp
    render/passes/CollectPass.cpp
    render/passes/OccluderSelectPass.cpp
    render/passes/OccluderPrePass.cpp
    render/passes/PreCullPass.cpp
    render/passes/DepthPrePass.cpp
    render/passes/MainPass.cpp
    render/passes/ImGuiPass.cpp
    render/passes/TextPass.cpp
    render/passes/WorldTextPass.cpp
    render/passes/ExpandPass.cppm
    render/passes/HiZPass.cppm
    render/passes/OcclusionPass.cppm
    render/passes/CollectPass.cppm
    render/passes/OccluderSelectPass.cppm
    render/passes/OccluderPrePass.cppm
    render/passes/PreCullPass.cppm
    render/passes/DepthPrePass.cppm
    render/passes/MainPass.cppm
    render/passes/ImGuiPass.cppm
    render_graph/RenderGraph.cpp
    render_graph/RenderGraphCompile.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/SlangSpriVCompilerHelper/cmake")
include(SlangSpirVCompiler)
set(_shader_gen_dir "${CMAKE_BINARY_DIR}/generated/shaders/engine")
set(_shader_opt_level "$<$<CONFIG:Debug>:0>$<$<CONFIG:RelWithDebInfo>:2>$<$<CONFIG:Release>:3>")
set(_engine_shaders
    ExpandComp              expand.slang              compute  main
    OcclusionCullComp       occlusion_cull.slang       compute  main
    OccluderSelectComp      occluder_select.slang      compute  main
    PreCullComp             pre_cull.slang             compute  main
    HizGenComp              hiz_gen.slang              compute  main
    CollectCountCompactComp collect_count_compact.slang compute  main
    CollectWriteComp        collect_write.slang        compute  main
    MainIndirVert           main_indir.slang           vertex   main
    MainIndirVertUV1        main_indir.slang           vertex   main_uv1
    DepthIndirVert          depth_indir.slang          vertex   main
    StandardMeshFrag        standard_mesh.slang        fragment main
    StandardMeshFragUV1     standard_mesh.slang        fragment main_uv1
    DepthPrepassFrag        depth_prepass.slang        fragment main
    UnlitFrag               unlit.slang                fragment main
    UiTextVert              ui_text.slang              vertex   vs_main
    UiTextFrag              ui_text.slang              fragment ps_main
    WorldTextVert           world_text.slang           vertex   vs_main
    WorldTextFrag           world_text.slang           fragment ps_main
    # The Slug backend's entry points. The names are distinct because the
    # compiler derives an entry's SPIR-V stage from its function; two functions
    # both called `main` in one file both compile as vertex modules.
    SlugTextVert            slug_text.slang            vertex   slug_vs_main
    SlugTextFrag            slug_text.slang            fragment slug_ps_main
)
if(VKENGINE_PHYSICAL_CAMERA)
    list(APPEND _engine_shaders
        PhysicalCameraCompositeVert camera_composite_vert.slang vertex   main
        PhysicalCameraCompositeFrag camera_composite.slang       fragment main
    )
endif()
add_slang_shaders(
    TARGET      ${ENGINE_TARGET}_shaders
    OUTPUT_DIR  ${_shader_gen_dir}
    NAMESPACE   Engine
    SHADER_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    COMPILER    ${VKENGINE_SLANG_SPIRV_COMPILER}
    COMPILER_DEPENDS ${VKENGINE_SLANG_COMPILER_DEPENDS}
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene_entries.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene_block_layout.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/technique_packing.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hiz_helpers.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/slug_gpu.slang"
    OPT_LEVEL   ${_shader_opt_level}
    SHADERS
        ${_engine_shaders}
)
add_dependencies(${ENGINE_TARGET} ${ENGINE_TARGET}_shaders)

# Expose where the engine .spv files are generated so per-example deploy steps
# (add_engine_example) can copy them next to each example executable.
set_property(GLOBAL PROPERTY VKENGINE_ENGINE_SHADER_DIR "${_shader_gen_dir}")

get_target_property(_gen_shader_mods ${ENGINE_TARGET}_shaders SLANG_CPPM_FILES)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(${ENGINE_TARGET} PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            input/InputSystem.cppm
            assets/MeshTypes.cppm
            assets/NormalEncoding.cppm
            assets/TextureTypes.cppm
            assets/TextureFormat.cppm
            assets/TextureUploadScheduler.cppm
            assets/ResourceCache.cppm
            assets/TextureResource.cppm
            assets/FontResource.cppm
            bootstrap/Startup.cppm
            ApplicationContext.cppm
            ecs/Entity.cppm
            ecs/ComponentRegistry.cppm
            ecs/SystemSchedule.cppm
            components/Transform.cppm
            components/Camera.cppm
            components/MeshReference.cppm
            components/Text.cppm
            components/DynamicMesh.cppm
            components/OrmOverride.cppm
            components/MaterialOverride.cppm
            components/TransformOverride.cppm
            assets/FileLoaders/Mesh/BinMeshAssembler.cppm
            assets/FileLoaders/Mesh/GltfMaterialImport.cppm
            assets/FileLoaders/Mesh/GltfMeshAssembler.cppm
            assets/FileLoaders/Mesh/MeshLoaderBase.cppm
            assets/FileLoaders/Mesh/ObjMeshAssembler.cppm
            assets/FileLoaders/Mesh/MeshMagicLoader.cppm
            assets/FileLoaders/Mesh/TangentGenerator.cppm
            assets/FileLoaders/Textures/TextureLoaders.cppm
            assets/FileLoaders/Fonts/FontLoader.cppm
            ui/ImGuiSystem.cppm
            text/FontLibrary.cppm
            text/Font.cppm
            text/Shaping.cppm
            text/Layout.cppm
            text/Atlas.cppm
            text/GlyphRaster.cppm
            text/Blob.cppm
            text/GlyphAtlasGpu.cppm
            text/GpuTextBlobBuffer.cppm
            text/Msdf.cppm
            text/FontWatcher.cppm
            text/FontReloader.cppm
            text/TextSystem.cppm
            render/Pipeline.cppm
            render/PipelinePass.cppm
            render/DescriptorDecl.cppm
            gpu/GpuResources.cppm
            gpu/Buffer.cppm
            gpu/BlockArray.cppm
            gpu/Texture.cppm
            gpu/DescriptorSet.cppm
            gpu/StagingPool.cppm
            gpu/SamplerCache.cppm
            gpu/BufferHeap.cppm
            gpu/ImageHeap.cppm
            gpu/FrameRing.cppm
            gpu/TlsfAllocator.cppm
            gpu/TransientAllocator.cppm
            gpu/MeshData.cppm
            assets/ShaderLoader.cppm
            assets/DefaultTextures.cppm
            render/SceneLimits.cppm
            render/MeshUploadManager.cppm
            render/MeshGatherSystem.cppm
            render/MeshPipeline.cppm
            assets/SceneImport.cppm
            assets/MeshCache.cppm
            render/Renderer.cppm
            render/techniques/TechniqueManager.cppm
            render/techniques/BaseTechnique.cppm
            render/techniques/DefaultMeshTechnique.cppm
            render/techniques/UnlitTextureTechnique.cppm
            render/techniques/TechniqueId.cppm
            render/materials/MaterialManager.cppm
            render/materials/MaterialId.cppm
            render/materials/MaterialHandle.cppm
            render/SceneRenderer.cppm
            render/DrawMode.cppm
            render/bindless/BindlessManager.cppm
            render/bindless/TextureSlot.cppm
            render/bindless/TextureUploader.cppm
            render/bindless/TextureResidency.cppm
            render/bindless/TextureReloader.cppm
            render/TextureWatcher.cppm
            render/ResidencyPolicy.cppm
            render/ShaderManager.cppm
            render/CompilerEngine.cppm
            render/GplPolicy.cppm
            render/GpuStats.cppm
            render/PipelineFactory.cppm
            render/PipelineTemplates.cppm
            render/ShaderWatcher.cppm
            render/ShaderRegistration.cppm
            GameEngine.cppm
            EngineContext.cppm
            bootstrap/EngineBootstrap.cppm
            render_graph/RenderGraph.cppm
            render/passes/ExpandPass.cppm
            render/passes/OccluderSelectPass.cppm
            render/passes/OccluderPrePass.cppm
            render/passes/PreCullPass.cppm
            render/passes/HiZPass.cppm
            render/passes/OcclusionPass.cppm
            render/passes/CollectPass.cppm
            render/passes/DepthPrePass.cppm
            render/passes/MainPass.cppm
            render/passes/ImGuiPass.cppm
            render/passes/TextPass.cppm
            render/passes/WorldTextPass.cppm
    )
    if(VKENGINE_PHYSICAL_CAMERA)
        target_sources(${ENGINE_TARGET} PUBLIC
            FILE_SET CXX_MODULES
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES
                physical_camera/PhysicalCameraTypes.cppm
                physical_camera/PhysicalCameraSystem.cppm
        )
    endif()
    if(_gen_shader_mods)
        target_sources(${ENGINE_TARGET} PUBLIC
            FILE_SET CXX_MODULES
            BASE_DIRS ${_shader_gen_dir}
            FILES ${_gen_shader_mods}
        )
    endif()
endif()

target_link_libraries(${ENGINE_TARGET} PUBLIC slang_shared_reflection)

# ═══════════════════════════════════════════════════════════════════════════
# Engine-owned entry point
# ═══════════════════════════════════════════════════════════════════════════
# Provides the real process entry point (main, and WinMain on Windows) and
# calls the application's VulkanEngine::AppMain hook after installing crash
# handling and crash-safe log capture. Applications therefore never implement
# the shim themselves.
#
# It lives in its own static library on purpose: an executable only pulls it in
# when its own `main` is unresolved. A target that defines its own main (tests,
# gtest, a game that opts out) links this as an archive member that is never
# selected, so there is no duplicate-symbol conflict.
add_library(${ENTRY_TARGET} STATIC)
target_compile_features(${ENTRY_TARGET} PUBLIC cxx_std_23)
target_sources(${ENTRY_TARGET} PRIVATE bootstrap/EntryPoint.cpp)
target_link_libraries(${ENTRY_TARGET} PUBLIC ${ENGINE_TARGET})

target_compile_options(${ENTRY_TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(TARGET project_sanitizers)
    target_link_libraries(${ENTRY_TARGET} PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(${ENTRY_TARGET} PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(${ENTRY_TARGET} PRIVATE project_size)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(${ENGINE_TARGET})
    enable_target_clang_tidy(${ENTRY_TARGET})
endif()
