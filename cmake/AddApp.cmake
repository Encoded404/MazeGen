include_guard(GLOBAL)

#[=======================================================================
# add_app
#
#   add_app(<name>
#     [APP_TITLE <title>]            # window/CLI app title (default: <name>)
#     [ORG_ID <id>]                  # storage org segment (default: VKENGINE_DEFAULT_ORG_ID)
#     [APP_ID <id>]                  # storage app segment (default: <name>)
#     [MODULE_PREFIX <prefix>]       # C++ module prefix (default: Apps.<name>)
#     [SHADER_NAMESPACE <ns>]        # shader namespace segment (default: <name>)
#     [OUTPUT_SUBDIR <dir>]          # role folder under bin (default: apps)
#     [FLAT_OUTPUT]                  # omit the per-target output subdirectory
#     SOURCES <file...>              # .cpp implementation files
#     MODULES <file...>              # .cppm interface units (relative to the app dir)
#     SHADERS <stem> <file> <stage> <entry> ...
#     ASSET_DIRS <dir...>            # asset folders copied next to the exe
#   )
#
# Creates a fully-configured application executable target that links the
# engine via EngineRuntime. Each app is self-contained:
#   - output goes to bin/<OUTPUT_SUBDIR>/<Config>[/<name>]/
#   - engine + app shaders are deployed to <exe_dir>/shaders
#   - asset dirs are copied to <exe_dir>/<dir>
#   - modules are C++23 std-module based (Clang)
#   - sanitizers/sections glue and clang-tidy are applied like the engine libs
#
# ORG_ID/APP_ID are exported to the target as VKENGINE_ORG_ID/VKENGINE_APP_ID.
# They are the application's storage identity: the two path components under the
# per-user root that settings, saves, caches and logs live in, so every app gets
# its own directory instead of sharing one. They are compile-time defaults only
# - the application receives them as runtime values (Runtime::Cli ->
# ApplicationConfig) so --user-dir and VKENGINE_USER_DIR can still override the
# location.
#=======================================================================]
function(add_app NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ADD_APP
        "FLAT_OUTPUT"
        "MODULE_PREFIX;SHADER_NAMESPACE;APP_TITLE;ORG_ID;APP_ID;OUTPUT_SUBDIR"
        "SOURCES;MODULES;SHADERS;ASSET_DIRS"
    )

    if(NOT ADD_APP_MODULE_PREFIX)
        set(ADD_APP_MODULE_PREFIX "Apps.${NAME}")
    endif()
    if(NOT ADD_APP_SHADER_NAMESPACE)
        set(ADD_APP_SHADER_NAMESPACE "${NAME}")
    endif()
    if(NOT ADD_APP_APP_TITLE)
        set(ADD_APP_APP_TITLE "${NAME}")
    endif()
    if(NOT ADD_APP_ORG_ID)
        set(ADD_APP_ORG_ID "${VKENGINE_DEFAULT_ORG_ID}")
    endif()
    # The target name is already a stable, ASCII, separator-free slug, which is
    # exactly what a directory component needs - unlike APP_TITLE, which is UI
    # copy that may be reworded.
    if(NOT ADD_APP_APP_ID)
        set(ADD_APP_APP_ID "${NAME}")
    endif()
    if(NOT ADD_APP_OUTPUT_SUBDIR)
        set(ADD_APP_OUTPUT_SUBDIR "apps")
    endif()

    add_executable(${NAME} ${ADD_APP_SOURCES})

    # Apps are grouped by role (app/, examples/, ...) under bin/. The per-target
    # directory keeps each app's shaders and assets isolated; FLAT_OUTPUT drops
    # it for role folders that hold exactly one executable. The shader and asset
    # deploys below follow automatically via $<TARGET_FILE_DIR>.
    if(OUTPUT_PATH)
        set(_add_app_output_dir "${OUTPUT_PATH}/${ADD_APP_OUTPUT_SUBDIR}/$<CONFIG>")
        if(NOT ADD_APP_FLAT_OUTPUT)
            string(APPEND _add_app_output_dir "/${NAME}")
        endif()
        set_target_properties(${NAME} PROPERTIES
            OUTPUT_NAME ${NAME}
            RUNTIME_OUTPUT_DIRECTORY "${_add_app_output_dir}"
        )
        unset(_add_app_output_dir)
    endif()

    # Windows: build as a GUI-subsystem executable so no console window appears
    # when the app is double-clicked. The engine entry point supplies WinMain and
    # re-attaches to a parent console when launched from a terminal, so CLI
    # output (e.g. -l/--log-level) is still visible there.
    if(WIN32)
        set_target_properties(${NAME} PROPERTIES WIN32_EXECUTABLE TRUE)
    endif()

    target_link_libraries(${NAME} PRIVATE
        ${RUNTIME_TARGET}
        ${ENTRY_TARGET}
        slang_shared_reflection
    )

    find_package(SDL3 CONFIG REQUIRED)
    find_package(CLI11 CONFIG REQUIRED)
    target_link_libraries(${NAME} PRIVATE
        SDL3::SDL3
        CLI11::CLI11
        Vulkan::HppModule
    )

    # ---- compile app slang shaders to SPIR-V + C++20 modules ----
    # Optional: an app may rely solely on the engine's built-in shader set
    # (e.g. the standard PBR mesh shader). When SHADERS is empty there is no
    # per-app shader target, but the engine shader deploy below still runs.
    if(ADD_APP_SHADERS)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/SlangSpriVCompilerHelper/cmake")
        include(SlangSpirVCompiler)
        set(_shader_gen_dir "${CMAKE_BINARY_DIR}/generated/shaders/apps/${NAME}")
        set(_shader_opt_level
            "$<$<CONFIG:Debug>:0>$<$<CONFIG:RelWithDebInfo>:2>$<$<CONFIG:Release>:3>")
        add_slang_shaders(
            TARGET      ${NAME}_shaders
            OUTPUT_DIR  ${_shader_gen_dir}
            NAMESPACE   ${ADD_APP_SHADER_NAMESPACE}
            SHADER_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            COMPILER    ${VKENGINE_SLANG_SPIRV_COMPILER}
            COMPILER_DEPENDS ${VKENGINE_SLANG_COMPILER_DEPENDS}
            OPT_LEVEL   ${_shader_opt_level}
            SHADERS
                ${ADD_APP_SHADERS}
        )
        add_dependencies(${NAME} ${NAME}_shaders)

        # Deploy this app's shaders next to the executable.
        add_custom_command(
            TARGET ${NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${NAME}>/shaders"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_shader_gen_dir}/*.spv" "$<TARGET_FILE_DIR:${NAME}>/shaders"
            COMMENT "Deploying ${NAME} SPIR-V shaders"
        )
    endif()

    # Deploy the engine's shared shader set next to the executable too, so the
    # app is runnable on its own.
    get_property(_engine_shader_dir GLOBAL PROPERTY VKENGINE_ENGINE_SHADER_DIR)
    if(_engine_shader_dir)
        add_custom_command(
            TARGET ${NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${NAME}>/shaders"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_engine_shader_dir}/*.spv" "$<TARGET_FILE_DIR:${NAME}>/shaders"
            COMMENT "Deploying engine SPIR-V shaders to ${NAME}"
        )
    endif()

    # ---- Windows runtime DLL deployment ----
    # MinGW/libc++ links against shared runtimes (libc++.dll, libunwind.dll,
    # libwinpthread-1.dll) and the vcpkg dependencies are dynamic, so the
    # app executable needs those DLLs next to it to run. One POST_BUILD step
    # resolves the executable's whole import closure across every candidate
    # directory and copies just those DLLs - globbing the directories would
    # also ship the gtest/gmock, ASan and OpenMP runtimes that no app links
    # (~3 MB per app). Release deliverables get the copies stripped;
    # Debug/RelWithDebInfo keep the dependency DWARF so those builds stay
    # steppable into.
    if(WIN32)
        # The candidate directories have to be escaped before they go on a
        # command line: a raw ';'-separated list is written into the generated
        # shell command verbatim, so /bin/sh splits it and runs everything
        # after the first semicolon as a separate command (CMake then starts
        # without -P, and the rule fails with status 126). Escaping the
        # separators makes the shell pass the script a single argument that
        # still reads as a list inside CMake.
        string(REPLACE ";" "\\;" _vkengine_dll_dirs "${VKENGINE_RUNTIME_DLL_DIRS}")
        add_custom_command(
            TARGET ${NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DDLL_DIRS=${_vkengine_dll_dirs}"
                "-DEXE=$<TARGET_FILE:${NAME}>"
                "-DOBJDUMP=${CMAKE_OBJDUMP}"
                "-DSTRIP=$<IF:$<CONFIG:Release>,${CMAKE_STRIP},>"
                "-DDEST=$<TARGET_FILE_DIR:${NAME}>"
                -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDlls.cmake"
            COMMENT "Deploying Windows runtime DLLs for ${NAME}"
        )
    endif()


    # ---- C++ module registration ----
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_sources(${NAME} PUBLIC
            FILE_SET CXX_MODULES
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES ${ADD_APP_MODULES}
        )

        # Register generated slang shader module files (only when this app
        # declared its own shaders; engine-only apps have no shader target).
        if(TARGET ${NAME}_shaders)
            get_target_property(_gen_shader_mods ${NAME}_shaders SLANG_CPPM_FILES)
            if(_gen_shader_mods)
                target_sources(${NAME} PUBLIC
                    FILE_SET CXX_MODULES
                    BASE_DIRS ${_shader_gen_dir}
                    FILES ${_gen_shader_mods}
                )
            endif()
        endif()
    endif()

    target_compile_features(${NAME} PRIVATE cxx_std_20)

    target_compile_definitions(${NAME}
        PUBLIC
            $<$<BOOL:${ENABLE_LOGGING}>:LOGIFACE_ENABLE_LOGGING=1>
            $<$<NOT:$<BOOL:${ENABLE_LOGGING}>>:LOGIFACE_ENABLE_LOGGING=0>
            LOGIFACE_DEFAULT_TO_FUNCTION=1
    )

    target_compile_definitions(${NAME} PRIVATE
        VKENGINE_SLANG_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/shaders/"
        VKENGINE_ORG_ID="${ADD_APP_ORG_ID}"
        VKENGINE_APP_ID="${ADD_APP_APP_ID}"
    )

    target_compile_options(${NAME} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    )

    if(CMAKE_CONFIGURATION_TYPES)
        target_compile_options(${NAME} PRIVATE
            "$<$<CONFIG:Debug>:-g>"
            "$<$<CONFIG:Debug>:-O0>"
            "$<$<CONFIG:RelWithDebInfo>:-g>"
            "$<$<CONFIG:RelWithDebInfo>:-O2>"
            "$<$<CONFIG:Release>:-O3>"
        )
    endif()

    if(TARGET project_sanitizers)
        target_link_libraries(${NAME} PRIVATE project_sanitizers)
    endif()
    if(TARGET project_sections)
        target_link_libraries(${NAME} PRIVATE project_sections)
    endif()
    if(TARGET project_size)
        target_link_libraries(${NAME} PRIVATE project_size)
    endif()
    # Statically linked libc++/libc++abi for the slang-free configurations
    # (see project_libcpp_static). Kept last on purpose: the static
    # libc++abi archive item should trail the other link inputs so archive
    # members are pulled for all their pending users.
    if(TARGET project_libcpp_static)
        target_link_libraries(${NAME} PRIVATE project_libcpp_static)
    endif()
    # Build-order: the sanitized libc++ archives must exist before any link
    # that consumes them (see libcpp-sanitize in the top-level CMakeLists).
    if(TARGET project_libcpp_sanitized)
        add_dependencies(${NAME} project_libcpp_sanitized)
    endif()

    # Release deliverables ship without a symbol table: .symtab/.strtab alone
    # is ~1.5 MB on an app binary and carries nothing the program needs at
    # runtime. The flag is applied at link time so the table is never written,
    # rather than stripping the file afterwards. Debug and RelWithDebInfo are
    # deliberately left untouched - they are the configurations used for
    # debugging, and their DWARF (compressed by project_size) is what makes a
    # crash trace useful.
    target_link_options(${NAME} PRIVATE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-Wl,--strip-all>)

    if(COMMAND enable_target_clang_tidy)
        enable_target_clang_tidy(${NAME})
    endif()

    # ---- asset copies ----
    foreach(_asset_dir IN LISTS ADD_APP_ASSET_DIRS)
        include(CopyAppAssets)
        configure_app_asset_copy(${NAME}
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${_asset_dir}"
            DEST_SUBDIR "${_asset_dir}"
        )
    endforeach()
endfunction()
