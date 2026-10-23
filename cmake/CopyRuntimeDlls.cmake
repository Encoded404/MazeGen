# Copies the MinGW/Windows runtime DLLs a built executable depends on into its
# output directory.
#
# Invoked as a script (cmake -P) once per executable from the POST_BUILD step
# in add_app:
#
#   cmake -DDLL_DIRS=<dir>[;<dir>...] -DEXE=<exe> -DOBJDUMP=<tool>
#         [-DSTRIP=<tool>] -DDEST=<exe dir> -P CopyRuntimeDlls.cmake
#
# DLL_DIRS lists the directories that may hold those DLLs:
#   * <build>/vcpkg_installed/<triplet>/bin            (release dependencies)
#   * <build>/vcpkg_installed/<triplet>/debug/bin      (debug dependencies)
#   * <llvm-mingw>/<triple>/bin                        (libc++, libunwind, ...)
#
# When EXE and OBJDUMP are supplied, only DLLs in the executable's transitive
# import closure are deployed. Globbing the directories instead ships several
# megabytes the program never loads: the vcpkg bin directory also holds the
# gtest/gmock runtimes and the llvm-mingw one the sanitizer and OpenMP
# runtimes, none of which an example links (measured: 3.1 MB of 14 MB, 23%, on
# the RelWithDebInfo minimal example). The closure is read from the PE import
# tables, so a dependency that is only ever reached through LoadLibrary would
# need to be listed explicitly.
#
# STRIP, when set, strips each DLL right after it is copied. The vcpkg release
# DLLs ship with DWARF and a COFF symbol table, which is about a quarter of
# their size; exports live in a separate section and are unaffected (verified:
# identical export counts before and after stripping). Stripping is applied to
# every deployed DLL rather than only to ones copied here, because vcpkg's
# applocal step runs earlier in the same POST_BUILD command and may have put
# them in place already; stripping twice is a no-op.
#
# Files are only copied when missing or newer than the destination, so
# incremental builds do not re-copy unchanged DLLs.
#
# Run with -P, so no cmake_minimum_required from the surrounding project is in
# effect; state it here to get current policy defaults (notably CMP0057, the
# if() IN_LIST operator used below).
cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED DEST OR DEST STREQUAL "")
    message(FATAL_ERROR "CopyRuntimeDlls.cmake: DEST is required")
endif()

# Accept the pre-refactor single-directory spelling too.
if(NOT DEFINED DLL_DIRS AND DEFINED DLL_DIR)
    set(DLL_DIRS "${DLL_DIR}")
endif()
if(NOT DEFINED DLL_DIRS OR DLL_DIRS STREQUAL "")
    message(FATAL_ERROR "CopyRuntimeDlls.cmake: DLL_DIRS is required")
endif()

file(MAKE_DIRECTORY "${DEST}")

# Index every candidate DLL by lowercased file name. The lookup has to be
# case-insensitive because PE import names are, while the build host's
# filesystem generally is not: the executable imports "SDL3.dll" but a naive
# "${dir}/sdl3.dll" test misses the file on Linux and silently drops it.
set(_dll_names "")
set(_dll_paths "")
foreach(_dir IN LISTS DLL_DIRS)
    if(EXISTS "${_dir}")
        file(GLOB _dir_dlls "${_dir}/*.dll")
        foreach(_dll IN LISTS _dir_dlls)
            get_filename_component(_base "${_dll}" NAME)
            string(TOLOWER "${_base}" _low)
            list(APPEND _dll_names "${_low}")
            list(APPEND _dll_paths "${_dll}")
        endforeach()
    endif()
endforeach()

# Resolve a lowercased DLL name to a path, or an empty string.
function(_crd_find_in_dirs name out)
    list(FIND _dll_names "${name}" _index)
    set(_found "")
    if(NOT _index EQUAL -1)
        list(GET _dll_paths ${_index} _found)
    endif()
    set(${out} "${_found}" PARENT_SCOPE)
endfunction()

# Lowercased DLL names imported by a PE file.
function(_crd_imports file out)
    set(_names "")
    if(EXISTS "${file}")
        execute_process(
            COMMAND "${OBJDUMP}" -p "${file}"
            OUTPUT_VARIABLE _dump
            ERROR_QUIET
            RESULT_VARIABLE _rc
        )
        if(_rc EQUAL 0)
            string(REGEX MATCHALL "DLL Name: [^ \t\r\n]+" _matches "${_dump}")
            foreach(_match IN LISTS _matches)
                string(REPLACE "DLL Name: " "" _match "${_match}")
                string(STRIP "${_match}" _match)
                string(TOLOWER "${_match}" _match)
                list(APPEND _names "${_match}")
            endforeach()
        endif()
    endif()
    set(${out} "${_names}" PARENT_SCOPE)
endfunction()

# Decide what to deploy.
set(_deploy "")
if(DEFINED EXE AND NOT EXE STREQUAL "" AND DEFINED OBJDUMP AND NOT OBJDUMP STREQUAL "")
    # Walk the import graph breadth-first, recursing only into DLLs we ship.
    set(_queue "${EXE}")
    set(_visited "")
    while(_queue)
        list(POP_FRONT _queue _current)
        _crd_imports("${_current}" _imports)
        foreach(_name IN LISTS _imports)
            if(_name IN_LIST _visited)
                continue()
            endif()
            list(APPEND _visited "${_name}")
            _crd_find_in_dirs("${_name}" _path)
            if(_path)
                list(APPEND _deploy "${_path}")
                list(APPEND _queue "${_path}")
            endif()
        endforeach()
    endwhile()
else()
    # No executable given: fall back to deploying every DLL in DLL_DIRS.
    foreach(_dir IN LISTS DLL_DIRS)
        if(EXISTS "${_dir}")
            file(GLOB _dlls "${_dir}/*.dll")
            list(APPEND _deploy ${_dlls})
        endif()
    endforeach()
endif()

set(_copied 0)
foreach(_dll IN LISTS _deploy)
    get_filename_component(_name "${_dll}" NAME)
    set(_dest_file "${DEST}/${_name}")
    # IS_NEWER_THAN is true when the destination does not exist.
    if("${_dll}" IS_NEWER_THAN "${_dest_file}")
        file(COPY_FILE "${_dll}" "${_dest_file}")
        math(EXPR _copied "${_copied} + 1")
    endif()
    # Strip unconditionally rather than only after our own copy: vcpkg's
    # applocal step (VCPKG_APPLOCAL_DEPS) deploys the same DLLs earlier in the
    # same POST_BUILD command, so a "did we copy it?" guard would silently skip
    # exactly the files that need stripping. Stripping an already-stripped DLL
    # is a no-op.
    if(DEFINED STRIP AND NOT STRIP STREQUAL "")
        execute_process(COMMAND "${STRIP}" --strip-all "${_dest_file}" ERROR_QUIET)
    endif()
endforeach()

if(_copied GREATER 0)
    message(STATUS "CopyRuntimeDlls: deployed ${_copied} DLL(s) to ${DEST}")
endif()
