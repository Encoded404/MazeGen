cmake_minimum_required(VERSION 3.30)

# ── import std: CMake gates it behind a per-release activation value ────────
#
# `import std;` is still an experimental CMake feature. It is enabled by setting
# CMAKE_EXPERIMENTAL_CXX_IMPORT_STD to the value documented in *that* CMake
# release's Help/dev/experimental.rst under "C++ ``import std`` support", and
# CMake deliberately changes the value from release to release to keep the
# feature experimental. It must be set before the CXX toolchain is discovered,
# which is why this block sits above project().
#
# A stale value is a nasty failure: compiler detection records no std-module
# support, and the build dies later in the *generate* step with
#   The "CXX_MODULE_STD" property on target "Vulkan-hppModule" requires
#   toolchain support, but it was not provided.  Reason: Experimental `import
#   std` support not enabled when detecting toolchain
# which names neither the CMake version nor the real cause. So each band below
# names the release it was verified against, and a release outside every band
# stops with instructions instead of silently reusing another release's value.
#
# Verified values: 3.31.11, 4.3.1 and 4.4.0. When adding a band, verify it by
# configuring with that CMake; passing -DCMAKE_EXPERIMENTAL_CXX_IMPORT_STD=<v>
# from the command line still overrides everything here.
if( NOT CMAKE_EXPERIMENTAL_CXX_IMPORT_STD )
    if( CMAKE_VERSION VERSION_GREATER_EQUAL 4.4 AND CMAKE_VERSION VERSION_LESS 4.5 )
        message("enabling import std for CMake 4.4 (verified on 4.4.0)")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
    elseif( CMAKE_VERSION VERSION_GREATER_EQUAL 4.3 AND CMAKE_VERSION VERSION_LESS 4.4 )
        message("enabling import std for CMake 4.3 (verified on 4.3.1)")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
    elseif( CMAKE_VERSION VERSION_GREATER_EQUAL 3.30 AND CMAKE_VERSION VERSION_LESS 3.32 )
        message("enabling import std for CMake 3.30-3.31 (verified on 3.31.11)")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
    elseif( CMAKE_VERSION VERSION_GREATER_EQUAL 4.5 )
        message(FATAL_ERROR
            "CMake ${CMAKE_VERSION} is newer than any release whose `import std` activation "
            "value this project has verified. Set CMAKE_EXPERIMENTAL_CXX_IMPORT_STD to the "
            "value documented for CxxImportStd in this CMake's Help/dev/experimental.rst:\n"
            "  cmake -DCMAKE_EXPERIMENTAL_CXX_IMPORT_STD=<value> ...\n"
            "then add a band for CMake ${CMAKE_VERSION} to this block.")
    else()
        message(FATAL_ERROR
            "CMake ${CMAKE_VERSION} is outside the verified range for `import std;`, which "
            "this project requires. Use CMake 3.30-3.31, 4.3 or 4.4, or set "
            "CMAKE_EXPERIMENTAL_CXX_IMPORT_STD from this CMake's "
            "Help/dev/experimental.rst.")
    endif()
endif()

set(PROJECT_NAME "MazeGenerator")
set(BACKEND_TARGET "VulkanBackend")
set(ENGINE_TARGET "VulkanEngine")
set(RUNTIME_TARGET "EngineRuntime")
set(ENTRY_TARGET "EngineEntryPoint")

# Org segment of the per-user storage path (<root>/<org>/<app>/...). One value
# for the whole project so every executable built here shares a directory
# group; a fork shipping its own product changes this in one place. Forks must
# also change it rather than inherit, or two products would share user data.
set(VKENGINE_DEFAULT_ORG_ID "MazeGenerator" CACHE STRING
    "Organization id used as the first component of the per-user storage path")

set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Use standard C++" FORCE)

set(USE_LIBCPP OFF CACHE BOOL "use libc++ instead of libstdc++, note: requires libc++ to have the std module supported.")

# ---- IMPORTANT: stdlib flags must be set BEFORE project() ----
#
# CMake's built-in std module detection (the __cmake_cxx23 target) runs
# during project(), where it discovers the compiler frontend variant and
# the <source>/<bits>/std.cc file to use for `import std;`.
#
# On Fedora with Clang, the `/etc/clang/x86_64-redhat-linux-gnu-clang++.cfg`
# file passes `--gcc-triple=x86_64-redhat-linux`.  This makes the compiler
# (and CMake's module scanner) find libstdc++'s bits/std.cc even when
# -stdlib=libc++ is specified, because --gcc-install-dir still points at
# the GCC installation.
#
# The fix: push -stdlib=libc++ AND --gcc-install-dir= into the *initial*
# compiler flags so CMake's probe and module detection see a Clang that
# uses libc++ everywhere and does not discover the GCC installation tree.
#
if(USE_LIBCPP)
	# Make the initial compiler probe use libc++ headers/module.
	# NOTE: Do NOT add --gcc-install-dir= here — that flag silences the
	# GCC installation entirely, which breaks Clang's ability to find
	# libc++.modules.json for module discovery.  The module scanner
	# also fails to detect `import std;` in source files (e.g. fastgltf).
	# We handle the std module source selection separately via
	# CMAKE_CXX_MODULE_STD below.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# if this is not set by the user cmake will decide
set(OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# ── Strip ccache shim directories from PATH before compiler detection ──
# Distros like Fedora put the ccache wrapper directory first in PATH, so a
# bare `clang++` can resolve to the ccache shim. ccache's cache key does
# not cover C++ module BMI/IFC contents: after a module interface changes
# it can return objects compiled against the stale interface (ODR
# corruption). Compiler detection runs inside project(), so sanitize PATH
# here — CMake then finds the real compiler automatically, with no
# per-configure flags and no platform-specific preset. This is a no-op on
# systems without ccache shims; the guard after project() still rejects
# explicit launcher/compiler usage.
if(DEFINED ENV{PATH})
	if(CMAKE_HOST_WIN32)
		set(_ccache_path_sep ";")
	else()
		set(_ccache_path_sep ":")
	endif()
	string(REPLACE "${_ccache_path_sep}" ";" _ccache_path_list "$ENV{PATH}")
	set(_ccache_clean_path "")
	foreach(_dir IN LISTS _ccache_path_list)
		if(NOT _dir MATCHES "(^|[/\\])[^/\\]*ccache$")
			list(APPEND _ccache_clean_path "${_dir}")
		endif()
	endforeach()
	list(JOIN _ccache_clean_path "${_ccache_path_sep}" _ccache_sanitized_path)
	set(ENV{PATH} "${_ccache_sanitized_path}")
	unset(_ccache_path_sep)
	unset(_ccache_path_list)
	unset(_ccache_clean_path)
	unset(_ccache_sanitized_path)
endif()

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

# ── ccache is unsupported (C++20/23 modules) ────────────────────────────
# Safety net for the PATH sanitization above: reject ccache in the compiler
# invocation outright. ccache's cache key does not cover module BMI/IFC
# contents, so it can serve objects compiled against a stale module
# interface — silent ODR corruption. Covers an explicit
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache and a compiler explicitly pointing at
# a ccache wrapper (checked by name, resolved symlinks, and directory).
if(CMAKE_CXX_COMPILER_LAUNCHER MATCHES "ccache")
	message(FATAL_ERROR
		"ccache is unsupported in this C++ modules project (stale module-object "
		"hazard). Configure without the launcher, e.g.:\n"
		"  cmake --preset default\n"
		"or unset CMAKE_CXX_COMPILER_LAUNCHER and reconfigure.")
endif()
get_filename_component(_no_ccache_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_no_ccache_name "${CMAKE_CXX_COMPILER}" NAME)
file(REAL_PATH "${CMAKE_CXX_COMPILER}" _no_ccache_real)
get_filename_component(_no_ccache_real_name "${_no_ccache_real}" NAME)
if(_no_ccache_name MATCHES "ccache"
		OR _no_ccache_real_name MATCHES "ccache"
		OR _no_ccache_dir MATCHES "(^|[/\\])[^/\\]*ccache$")
	message(FATAL_ERROR
		"The C++ compiler resolves to a ccache wrapper (${CMAKE_CXX_COMPILER}).\n"
		"ccache is unsupported in this C++ modules project (see above). Configure "
		"with the real compiler, e.g.:\n"
		"  cmake --preset default\n"
		"or -DCMAKE_CXX_COMPILER=/usr/bin/clang++ (adjust for your platform).")
endif()
unset(_no_ccache_dir)
unset(_no_ccache_name)
unset(_no_ccache_real)
unset(_no_ccache_real_name)

# CMAKE_CXX_MODULE_STD was already set before project() to point at libc++'s
# module files.  We still set it to ON here: the ON flag tells CMake to
# propagate the __cmake_cxx23 module target reference to all consumer
# targets via their .modmap files (-fmodule-file=std=.../std.pcm).
# Without the ON flag, consumer targets won't know where to find the
# std module and will fail with "module 'std' not found".
#
# When USE_LIBCPP is OFF, we simply set ON and let CMake auto-detect
# libstdc++'s bits/std.cc — which is the correct default.
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Require C++23 for import std; and vulkan module support
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Hot-reload (in-process Slang) configuration ─────────────────────────
# Declared here instead of in engine/core because the static-libc++ decision
# (the project_libcpp_static target below) needs these values at configure
# time: whichever configurations ship slang/hot-reload must keep the dynamic
# libc++ link, because slang's official shared libraries are built against
# libstdc++ and pulling it into a process that also carries a statically
# linked libc++ would be exactly the "more than one C++ runtime copy in the
# process" situation Steam's Linux runtime forbids. Release therefore never
# gets slang (shaders are precompiled to .spv by the build-time host tool in
# every configuration); RelWithDebInfo gets it only while
# VKENGINE_HOT_RELOAD_RELWITHDEBINFO is ON, so a RelWithDebInfo build can
# also be shipped slang-free.
if(WIN32)
    set(_vkengine_hot_reload_default OFF)
else()
    set(_vkengine_hot_reload_default ON)
endif()
option(VKENGINE_HOT_RELOAD
    "Enable Slang SDK for runtime shader compilation (Debug and, when VKENGINE_HOT_RELOAD_RELWITHDEBINFO is ON, RelWithDebInfo)"
    ${_vkengine_hot_reload_default})
unset(_vkengine_hot_reload_default)
option(VKENGINE_HOT_RELOAD_RELWITHDEBINFO
    "Keep hot reload (in-process slang) enabled in RelWithDebInfo as well (OFF = RelWithDebInfo is slang-free, like Release)"
    ON)

# Verify libc++ availability (only valid after project() — we now know the compiler)
if(USE_LIBCPP)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		include(CheckCXXSourceCompiles)

		set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")

		check_cxx_source_compiles("
			#include <string>
			int main() {
				std::string s = \"hello\";
				return s.size();
			}
		" HAS_LIBCXX)

		unset(CMAKE_REQUIRED_FLAGS)

		if (HAS_LIBCXX)
			message(STATUS "libc++ detected and working — using it")

			# Force -stdlib=libc++ onto the auto-generated std-module targets so
			# the module BMI is compiled against the same libc++ headers that
			# the rest of the project uses.
			if(TARGET __cmake_cxx23)
				set_property(TARGET __cmake_cxx23 APPEND PROPERTY COMPILE_OPTIONS "-stdlib=libc++")
			endif()
			if(TARGET __cmake_cxx26)
				set_property(TARGET __cmake_cxx26 APPEND PROPERTY COMPILE_OPTIONS "-stdlib=libc++")
			endif()

			# Linker flag is not automatically propagated from CMAKE_CXX_FLAGS.
			add_link_options(-stdlib=libc++)

			# ── Statically linked libc++ (Steam Linux Runtime readiness) ──
			#
			# Steam's Linux runtime combines application libraries with the
			# host's graphics drivers, and libstdc++/libgcc sit inside the
			# driver dependency stack: bundling (or letting several shared
			# objects each require) a copy of a C++ runtime in the process is
			# how the well-known driver crashes happen. Valve's guidance for
			# game developers is to statically link such libraries instead.
			#
			# This project targets libc++, whose Mesa/NVIDIA crash-mode does
			# not exist (drivers use libstdc++), so statically embedding
			# libc++ and libc++abi satisfies the "one C++ runtime copy" rule
			# vacuously: the executable carries its copy, and no shared
			# libc++ exists in the process at all. License-wise this is clean
			# (libc++ is Apache-2.0-with-LLVM-exception).
			#
			# It is applied exactly to the slang-free configurations
			# (_static_libcpp_configs): Debug keeps the dynamic link (Address
			# Sanitizer cannot intercept a statically linked libc++), and
			# configurations that load slang's libstdc++-based shared
			# libraries keep it too, so no C++ runtime is multiplied.
			#
			# Mechanics:
			#   * -static-libstdc++ is clang's unified flag; with
			#     -stdlib=libc++ the driver emits "-Bstatic -lc++ -Bdynamic"
			#     after every user library (verified via -###) and adds no
			#     dynamic -lc++ beside it. The driver never adds -lc++abi, so
			#     the archive is supplied explicitly.
			#   * -Wl,-Bstatic,-lc++abi,-Bdynamic is carried by the
			#     project_libcpp_static INTERFACE target, which every C++
			#     executable links LAST. Position matters: a static archive
			#     only satisfies undefined symbols pending at its position
			#     (except that modern GNU ld rescan and lld's lazy archive
			#     resolution fix up a too-early position; a too-late
			#     position is always safe).
			#   * build-time host tools (the slang-spirv-compiler tool links
			#     slang itself) deliberately do NOT link this target: they
			#     stay fully dynamic.
			option(VKENGINE_STATIC_LIBCPP
				"Statically link libc++ and libc++abi into slang-free configurations (Release, and RelWithDebInfo when VKENGINE_HOT_RELOAD_RELWITHDEBINFO is off)"
				ON)
			set(_static_libcpp_configs "Release")
			if(NOT (VKENGINE_HOT_RELOAD AND VKENGINE_HOT_RELOAD_RELWITHDEBINFO))
				list(APPEND _static_libcpp_configs "RelWithDebInfo")
			endif()
			list(JOIN _static_libcpp_configs "," _static_libcpp_configs_gx)

			# Always provide the hook target so executable-creating helpers
			# can link it unconditionally; it is a no-op unless static
			# linking was verified below.
			add_library(project_libcpp_static INTERFACE)

			if(VKENGINE_STATIC_LIBCPP)
				set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")
				set(CMAKE_REQUIRED_LINK_OPTIONS
					"-static-libstdc++" "-Wl,-Bstatic,-lc++abi,-Bdynamic")
				check_cxx_source_compiles("
					#include <string>
					#include <stdexcept>
					int main() {
						try {
							throw std::runtime_error(\"probe\");
						} catch (const std::exception& e) {
							return (int)std::string(e.what()).size();
						}
						return -1;
					}
				" HAS_STATIC_LIBCXX)
				unset(CMAKE_REQUIRED_FLAGS)
				unset(CMAKE_REQUIRED_LINK_OPTIONS)

				if(HAS_STATIC_LIBCXX)
					message(STATUS
						"static libc++ working — statically linking libc++ and libc++abi in configs: ${_static_libcpp_configs}")
					target_link_libraries(project_libcpp_static INTERFACE
						"$<$<CONFIG:${_static_libcpp_configs_gx}>:-static-libstdc++>"
						"$<$<CONFIG:${_static_libcpp_configs_gx}>:-Wl,-Bstatic,-lc++abi,-Bdynamic>")
				else()
					message(WARNING
						"Static libc++ could not be linked — keeping the dynamic libc++ link "
						"in all configurations. On Fedora install the static archives "
						"(\"dnf install libcxx-static\"; libc++abi.a comes with libcxxabi-static) "
						"and reconfigure.")
				endif()
			endif()
			# _static_libcpp_configs and _static_libcpp_configs_gx are kept:
			# the libcpp-sanitize block further down (after the lld probe)
			# derives its own remap genex from them.
		else()
			message(STATUS "libc++ NOT available — falling back to libstdc++")
			# Undo the pre-project() changes so the default libstdc++ is used
			string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
			unset(CMAKE_CXX_MODULE_STD)
		endif()
	else()
		message(WARNING "USE_LIBCPP=ON but compiler is not Clang — ignoring")
		string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
		unset(CMAKE_CXX_MODULE_STD)
	endif()
endif()

# Require compiler support for modules
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  message(FATAL_ERROR "C++ modules required but the compiler (${CMAKE_CXX_COMPILER_ID}) is not supported. Use Clang or GCC.")
endif()

# find vulkan
#
# The Vulkan loader comes from vcpkg's vulkan-loader port, which exports a
# CONFIG package (VulkanLoader) with proper per-configuration imported
# locations: Debug links the debug loader (debug/lib), Release and
# RelWithDebInfo link the release one (lib). Prefer that target over the
# module-mode FindVulkan below: FindVulkan resolves a single generic library
# location at configure time, and vcpkg's multi-config prefix ordering makes
# it pick the *debug* loader for every configuration (including Release),
# which also creates a cycle in CMake's runtime-search-path ordering together
# with release-only packages such as shader-slang ("Cannot generate a safe
# runtime search path" warnings).
#
# Other packages reference the loader by the name Vulkan::Vulkan (e.g. imgui's
# CMake config hard-links it), so re-export the per-configuration target under
# that name. The module-mode find_package(Vulkan) below is kept for the
# headers and the Vulkan::HppModule C++ module target; it skips creating
# Vulkan::Vulkan because the target already exists.
#
# On the Windows/MinGW target the loader is supplied by vcpkg's `vulkan-loader`
# port. When module mode cannot resolve the import library by name, fall back to
# that port's CONFIG package and re-export its Vulkan::Loader target.
find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
	find_package(VulkanLoader CONFIG REQUIRED)
	find_path(Vulkan_INCLUDE_DIR NAMES vulkan/vulkan.h REQUIRED)
	if(NOT TARGET Vulkan::Vulkan)
		add_library(Vulkan::Vulkan INTERFACE IMPORTED GLOBAL)
		set_property(TARGET Vulkan::Vulkan PROPERTY
			INTERFACE_LINK_LIBRARIES Vulkan::Loader)
	endif()
endif()

find_package( Vulkan 1.3.0 QUIET )

# manually create module target if it was not previously provided
if( NOT TARGET Vulkan::HppModule )
	message("could not find vulkan target with find, adding manually")
	add_library( Vulkan-HppModule OBJECT )
	add_library( Vulkan::HppModule ALIAS Vulkan-HppModule )
	target_sources( Vulkan-HppModule PUBLIC
			FILE_SET CXX_MODULES
			BASE_DIRS ${Vulkan_INCLUDE_DIR}
			FILES
			${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
			# ${Vulkan_INCLUDE_DIR}/vulkan/vulkan_video.cppm # no vulkan_video.cppm used and it sometimes gives errors so begone
	)
	target_compile_features( Vulkan-HppModule PUBLIC cxx_std_23 )
	set_target_properties( Vulkan-HppModule PROPERTIES CXX_MODULE_STD ON )
	target_include_directories( Vulkan-HppModule PUBLIC ${Vulkan_INCLUDE_DIR} )
endif()

target_compile_definitions( Vulkan-HppModule PUBLIC
	VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
	# Silence the "experimental named module" warning inside vulkan.cppm (new
	# headers, >= 1.4.363); the module is used deliberately here.
	VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING=1 )

# Enable CTest at the top level so `ctest` discovers tests
include(CTest)

# Include custom CMake modules path (for other helpers)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Sanitizers interface for Debug builds (applied to the engine libs and to
# every app via add_app).
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug mode" ON)
# Windows/MinGW targets: llvm-mingw's default runtimes do not include the
# sanitizer runtimes (there is no UBSan for MinGW, and the Windows ASan runtime
# is only available in the MSVC/ASan configuration), so force it off rather
# than emitting flags the linker cannot satisfy.
if(ENABLE_SANITIZERS AND WIN32)
    message(STATUS "Sanitizers are not available for Windows/MinGW targets — forcing ENABLE_SANITIZERS=OFF")
    set(ENABLE_SANITIZERS OFF)
endif()
if(ENABLE_SANITIZERS)
    add_library(project_sanitizers INTERFACE)
    target_compile_options(project_sanitizers INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(project_sanitizers INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
endif()

# ── Linker selection: use LLD when the host has it ─────────────────────
# LLD is optional, but when present it unlocks two size knobs the default GNU
# ld on most distros cannot offer:
#   * zstd-compressed debug sections (this host's binutils is built without
#     zstd support, so GNU ld only has zlib), and
#   * identical-code folding (--icf), which GNU ld does not implement at all.
#
# The check is a real link test: `-fuse-ld=lld` fails to link if lld is not
# installed, so a host without it simply keeps its default linker. On the
# Windows target the llvm-mingw wrapper already selects lld via its .cfg, so
# the probe succeeds there too and the extra flag is a no-op.
#
# The probe caches its result, so drop any memoized value first: installing
# lld after the first configure must be picked up by the next configure
# without having to delete the build directory by hand.
include(CheckLinkerFlag)
unset(VKENGINE_HAVE_LLD CACHE)
check_linker_flag(CXX "-fuse-ld=lld" VKENGINE_HAVE_LLD)
if(VKENGINE_HAVE_LLD)
    message(STATUS "lld found — linking with -fuse-ld=lld")
    add_link_options(-fuse-ld=lld)
else()
    message(STATUS "lld not found — using the default linker")
endif()

# ── Sanitized static libc++ archives (silences ICF noise, nothing else) ─
#
# Context: the slang-free configurations statically embed libc++ and
# libc++abi (see project_libcpp_static above). The distro-shipped archives
# (on Fedora: libcxx/libcxxabi RPMs) are stripped by the packager, and the
# stripping tool zeroes the `sh_link` field of every `.llvm_addrsig`
# (address-significance table) section. lld's `--icf=safe` then prints
#
#   ld.lld: warning: .../libc++.a(member.cpp.o): --icf=safe conservatively
#      ignores SHT_LLVM_ADDRSIG [index N] with sh_link=0 ...
#
# once per affected member per linked executable. Behaviour is unchanged and
# correct — lld treats a broken table as if none existed — so the fix is to
# remove exactly those broken tables from the archive members.
#
# Mechanism, designed to be foolproof:
#
#   1. Discovery: a probe link with `-Wl,-t` (same driver, same
#      -static-libstdc++ / -lc++abi flags the real executables use) reports
#      the exact archive paths lld resolves under *this* toolchain. Custom
#      `--gcc-install-dir`/`-L` setups are therefore handled by
#      construction: we ask the linker what it would use instead of
#      guessing at directories.
#   2. Generation: for every discovered archive `path/name.a`, a Ninja
#      custom command writes `<build>/libcpp-sanitized/name.a` by removing
#      the broken addrsig tables member-by-member (cmake/
#      SanitizeStaticCppArchive.cmake; healthy members are copied through
#      byte-for-byte, so nothing else about the archive changes). The
#      command's DEPENDS list contains the source archive, so Ninja
#      re-sanitizes automatically whenever the RPM was updated (mtime
#      changed) — the copies can never go stale. The result is re-verified
#      inside the script; a failed verification fails the build with a
#      loud message instead of silently continuing.
#   3. Substitution: the real link keeps every flag it has today
#      (`-static-libstdc++`, `-Wl,-Bstatic,-lc++abi,-Bdynamic`) and gains
#      only `-Wl,--remap-inputs=<resolved-path>=<sanitized-copy>` — lld
#      reads the sanitized copy in place of the distro archive. With the
#      broken sections gone, lld is quiet and folds nothing extra: absent
#      table and sh_link==0 are the same "conservative" case, so the
#      resulting binaries are bit-identical to yesterday's.
#
# Skipped silently-with-status when: no lld (remap is lld-only), Windows
# (lld-link target has no such warning), sanitization already applied or
# the probe cannot run — everything then works exactly as before the
# sanitization existed, just with the warning noise.
set(VKENGINE_LIBCPP_SANITIZED_DIR "${CMAKE_BINARY_DIR}/libcpp-sanitized")
if(VKENGINE_HAVE_LLD
        AND NOT WIN32
        AND HAS_STATIC_LIBCXX
        AND VKENGINE_STATIC_LIBCPP)
    # Toolchain bits for the member surgery.
    find_program(VKENGINE_READELF_TOOL
        NAMES llvm-readelf-${CMAKE_CXX_COMPILER_VERSION} llvm-readelf readelf)
    if(NOT VKENGINE_READELF_TOOL OR NOT CMAKE_OBJCOPY OR NOT CMAKE_AR)
        message(STATUS
            "libcpp-sanitize: readelf/objcopy/ar not available — keeping the "
            "dynamic ICF warnings (linking is unaffected)")
    else()
        file(MAKE_DIRECTORY "${VKENGINE_LIBCPP_SANITIZED_DIR}")

        # 1) Trace-probe: which libc++/libc++abi archives does lld resolve?
        set(_san_probe_src "${VKENGINE_LIBCPP_SANITIZED_DIR}/icf_trace_probe.cpp")
        file(WRITE "${_san_probe_src}" "
#include <string>
#include <stdexcept>
int main() {
    try { throw std::runtime_error(\"probe\"); }
    catch (const std::exception& e) { return (int)std::string(e.what()).size(); }
    return -1;
}
")
        set(_san_probe_obj "${VKENGINE_LIBCPP_SANITIZED_DIR}/icf_trace_probe.o")
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}"
                -c "${_san_probe_src}" -o "${_san_probe_obj}" -stdlib=libc++
            RESULT_VARIABLE _san_compile_result)
        set(_san_probe_out "${VKENGINE_LIBCPP_SANITIZED_DIR}/icf_trace_probe.bin")
        # CMAKE_EXE_LINKER_FLAGS may carry -L/-B switches from whoever
        # configured the build; spread it so they influence the archive
        # resolution exactly as they would in a real link.
        separate_arguments(_san_user_ldflags NATIVE_COMMAND "${CMAKE_EXE_LINKER_FLAGS}")
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" "${_san_probe_obj}"
                -o "${_san_probe_out}" -stdlib=libc++ -static-libstdc++
                -fuse-ld=lld -Wl,-t -Wl,-Bstatic,-lc++abi,-Bdynamic
                ${_san_user_ldflags}
            OUTPUT_VARIABLE _san_trace
            ERROR_QUIET
            RESULT_VARIABLE _san_link_result)
        file(REMOVE "${_san_probe_out}")

        set(_san_keys "")
        if(_san_compile_result EQUAL 0 AND _san_link_result EQUAL 0)
            # Distinct archive paths from trace lines of the form
            #   <dir>/libc++.a(<member>.o)
            string(REPLACE "\n" ";" _san_trace_lines "${_san_trace}")
            foreach(_san_line IN LISTS _san_trace_lines)
                if(_san_line MATCHES "^([^ ]*libc\\+\\+(abi)?\\.a)\\([^)]*\\)")
                    list(APPEND _san_keys "${CMAKE_MATCH_1}")
                endif()
            endforeach()
            if(_san_keys)
                list(REMOVE_DUPLICATES _san_keys)
            endif()
        else()
            message(STATUS
                "libcpp-sanitize: probe link failed — keeping the ICF warnings")
        endif()

        # 2) Per discovered archive: sanitized copy + remap flag.
        set(_san_outputs "")
        set(_san_remapped "")
        set(_san_basenames "")
        foreach(_san_key IN LISTS _san_keys)
            get_filename_component(_san_name "${_san_key}" NAME)
            if(_san_name IN_LIST _san_basenames)
                # The same archive name resolved from two different dirs in
                # one link cannot happen for a single -lc++/lc++abi; if it
                # ever does, skip remapping that one rather than guess.
                continue()
            endif()
            list(APPEND _san_basenames "${_san_name}")
            set(_san_out "${VKENGINE_LIBCPP_SANITIZED_DIR}/${_san_name}")
            add_custom_command(
                OUTPUT "${_san_out}"
                COMMAND "${CMAKE_COMMAND}"
                    -DSAN_IN=${_san_key}
                    -DSAN_OUT=${_san_out}
                    -DSAN_AR=${CMAKE_AR}
                    -DSAN_OBJCOPY=${CMAKE_OBJCOPY}
                    -DSAN_READELF=${VKENGINE_READELF_TOOL}
                    -P "${CMAKE_SOURCE_DIR}/cmake/SanitizeStaticCppArchive.cmake"
                DEPENDS "${_san_key}"
                    "${CMAKE_SOURCE_DIR}/cmake/SanitizeStaticCppArchive.cmake"
                COMMENT "Sanitizing ${_san_name} (removing broken addrsig tables)"
                VERBATIM)
            list(APPEND _san_outputs "${_san_out}")
            list(APPEND _san_remapped "${_san_key}")
            target_link_libraries(project_libcpp_static INTERFACE
                "$<$<CONFIG:${_static_libcpp_configs_gx}>:-Wl,--remap-inputs=${_san_key}=${_san_out}>")
        endforeach()

        if(_san_outputs)
            add_custom_target(project_libcpp_sanitized
                DEPENDS ${_san_outputs})
            list(JOIN _san_remapped "', '" _san_remapped_msg)
            message(STATUS
                "libcpp-sanitize: remapping '{${_san_remapped_msg}}' to "
                "sanitized copies in ${VKENGINE_LIBCPP_SANITIZED_DIR}")
        elseif(_san_link_result EQUAL 0 AND _san_compile_result EQUAL 0)
            message(STATUS
                "libcpp-sanitize: trace found no static libc++ archives "
                "(nothing to sanitize)")
        endif()
        # Executables that link project_libcpp_static must also depend on the
        # sanitizer outputs; the helper functions that create executables
        # (add_app, setup_test_target, and the standalone targets)
        # do this next to their link hook.
    endif()
endif()

# Sections optimization for dead-code stripping (gc-sections)
add_library(project_sections INTERFACE)
target_compile_options(project_sections INTERFACE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffunction-sections>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fdata-sections>
)
target_link_options(project_sections INTERFACE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--gc-sections>
)

# ── Size knobs that cost nothing in debuggability ──────────────────────
# Linked next to project_sections, i.e. onto the engine libs, the tests and
# every app.
#
# Debug info is 60-85% of the bytes in the Debug and RelWithDebInfo binaries,
# so compressing it is the single largest win available on ELF: gdb, lldb and
# perf decompress SHF_COMPRESSED sections transparently, so those
# configurations stay exactly as debuggable as before. zstd compresses better
# and is only reachable through lld; a GNU ld without zstd falls back to zlib.
#
# The Windows target cannot compress its DWARF at all — lld's COFF driver
# rejects --compress-debug-sections and llvm-objcopy reports "option is not
# supported for COFF" — so it gets the structural equivalent instead: split
# DWARF. The detailed DIEs move into per-object .dwo files and the executable
# keeps only skeleton compile units, line tables and the
# .debug_gnu_pubnames/pubtypes indexes that let gdb resolve symbols without
# them. Each skeleton records its .dwo relative to the configure-time build
# directory, so gdb resolves them straight from the build tree and debug
# fidelity is unchanged — the build tree simply has to survive for
# symbol-level debugging. Release carries no DWARF on this triplet, so this is
# limited to the two configurations that do.
#
# No --icf here: measured against this project's Windows binary it folded
# 336 bytes, because the dependencies are already-linked DLLs and cannot
# participate.
add_library(project_size INTERFACE)
if(WIN32)
    target_compile_options(project_size INTERFACE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-gsplit-dwarf>)
else()
    if(VKENGINE_HAVE_LLD)
        target_link_options(project_size INTERFACE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--compress-debug-sections=zstd>
        )
        # Safe identical-code folding, Release only. --icf=safe consults the
        # .llvm_addrsig address-significance tables clang emits on ELF, so it
        # never merges a function whose address is observable. It is kept out
        # of Debug/RelWithDebInfo on purpose: a folded function shares one
        # address with its twin, which makes breakpoints and backtraces
        # ambiguous in the configurations people actually debug with.
        target_link_options(project_size INTERFACE
            $<$<CONFIG:Release>:-Wl,--icf=safe>
        )
    else()
        target_link_options(project_size INTERFACE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--compress-debug-sections=zlib>
        )
    endif()
endif()

# -----------------
# Logging Interface
# -----------------
message("finding logging interface")

set(DEFAULT_LOGGING_INTERFACE_DIR "${CMAKE_SOURCE_DIR}/external/cppLoggingInterface")
set(LOGIFACE_PATH "" CACHE PATH "Optional path to the cppLoggingInterface repo")

if(LOGIFACE_PATH)
	set(LOGGING_INTERFACE_DIR "${LOGIFACE_PATH}")
else()
	set(LOGGING_INTERFACE_DIR "${DEFAULT_LOGGING_INTERFACE_DIR}")
endif()

# Build the logging module (provides import logiface;), but use
# #include headers instead of import std; (avoids Clang bug LLVM #138558).
set(LOGIFACE_BUILD_MODULE ON CACHE BOOL "" FORCE)
set(LOGIFACE_USE_STD_MODULE OFF CACHE BOOL "" FORCE)

if(EXISTS "${LOGGING_INTERFACE_DIR}/CMakeLists.txt")
	add_subdirectory("${LOGGING_INTERFACE_DIR}" cppLoggingInterface)
else()
	message(FATAL_ERROR "Missing logging interface at ${LOGGING_INTERFACE_DIR}; provide the repo there or set LOGIFACE_PATH to another directory.")
endif()

# Make the logging interface path available to subprojects that declare
# `LOGIFACE_PATH` in the cache (e.g. external/cppFileLoader). This ensures
# they look up the same logging interface instead of expecting a nested copy.
set(LOGIFACE_PATH "${LOGGING_INTERFACE_DIR}" CACHE PATH "Optional path to the cppLoggingInterface repo" FORCE)

# ----------
# FileLoader
# ----------
message("finding FileLoader")

set(FILELOADER_PATH "${CMAKE_SOURCE_DIR}/external/cppFileLoader" CACHE PATH "Optional path to the FileLoader repo")

# Enable module support for FileLoader (import std; + module files).
set(CPPFILELOADER_USE_STD_MODULE OFF CACHE BOOL "" FORCE)
set(CPPFILELOADER_USE_LOGIFACE_MODULE ON CACHE BOOL "" FORCE)
set(CPPFILELOADER_BUILD_AS_MODULE ON CACHE BOOL "" FORCE)

if(FILELOADER_PATH)
	add_subdirectory("${FILELOADER_PATH}" FileLoader)
else()
	message(FATAL_ERROR "Missing FileLoader at ${FILELOADER_PATH}; provide the repo there or set FILELOADER_PATH to another directory.")
endif()

# ----------------------------
# Slang/SPIR-V Compiler Helper
# ----------------------------
#
# `slang-spirv-compiler` is a *build-time host tool*: it runs on the machine
# performing the build and only emits target-independent artefacts (SPIR-V
# bytes plus a generated .cppm). When cross-compiling it must therefore be
# built for the HOST, never for the target — a Windows .exe cannot run on the
# build machine.
#
#   * Native build : build it in-tree (host == target compiler).
#   * Cross build  : build it in a nested configure for the host compiler via
#                    ExternalProject. Its Slang dependency comes from the host
#                    vcpkg triplet (the helper's own vcpkg.json requires
#                    shader-slang, which is installed for the host triplet).
#
# Controlled by VKENGINE_SLANG_COMPILER_MODE = auto | host | target.
#
# VKENGINE_SLANG_SPIRV_COMPILER holds either a CMake target name (native) or
# an absolute path to the host executable (cross); add_slang_shaders accepts
# both. VKENGINE_SLANG_COMPILER_DEPENDS carries the file-level dependency that
# shader custom commands must wait for (the ExternalProject build stamp).
message("finding SlangSpriVCompilerHelper")

if(NOT DEFINED VKENGINE_SLANG_SPIRV_COMPILER)
	set(VKENGINE_SLANG_SPIRV_COMPILER "slang-spirv-compiler")
endif()
set(VKENGINE_SLANG_COMPILER_DEPENDS "")

set(SLANG_HELPER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/SlangSpriVCompilerHelper")

# When to build slang-spirv-compiler. "auto" builds it for the host whenever
# cross-compiling, which is the only correct choice (the tool runs on the build
# machine). This deliberately is NOT derived through option(): an option's
# default is applied only when the variable is absent from the cache, so a value
# recorded by an earlier, non-cross configure would otherwise stick forever.
set(VKENGINE_SLANG_COMPILER_MODE "auto" CACHE STRING
	"When to build slang-spirv-compiler: auto (host when cross-compiling), host, or target")
set_property(CACHE VKENGINE_SLANG_COMPILER_MODE PROPERTY STRINGS auto host target)

# Drop the pre-release option whose cached value could be wrong.
unset(VKENGINE_SLANG_COMPILER_HOST_BUILD CACHE)

if(VKENGINE_SLANG_COMPILER_MODE STREQUAL "host")
	set(_vkengine_slang_build_host TRUE)
elseif(VKENGINE_SLANG_COMPILER_MODE STREQUAL "target")
	set(_vkengine_slang_build_host FALSE)
else()
	set(_vkengine_slang_build_host ${CMAKE_CROSSCOMPILING})
endif()

if(_vkengine_slang_build_host)
	set(VKENGINE_HOST_VCPKG_TRIPLET "x64-linux" CACHE STRING
		"vcpkg triplet used to build the host slang-spirv-compiler tool")

	# The helper hardcodes OUTPUT_PATH to <its CMAKE_BINARY_DIR>/bin, so the
	# host tool lands inside the nested build tree (which also keeps its RPATH
	# pointing at the nested vcpkg install).
	set(_host_build_dir "${CMAKE_BINARY_DIR}/host-tools/build")
	if(CMAKE_HOST_WIN32)
		set(_host_exe_suffix ".exe")
	else()
		set(_host_exe_suffix "")
	endif()
	set(_host_tool "${_host_build_dir}/bin/slang-spirv-compiler${_host_exe_suffix}")

	set(_host_vcpkg_args "")
	if(DEFINED ENV{VCPKG_ROOT})
		list(APPEND _host_vcpkg_args
			"-DCMAKE_TOOLCHAIN_FILE=$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
			"-DVCPKG_TARGET_TRIPLET=${VKENGINE_HOST_VCPKG_TRIPLET}"
			# Use this project's overlay ports for the host build too. The
			# helper's own vcpkg.json only requires shader-slang >= 2025.14.3, so
			# without this the nested build silently resolves the older registry
			# release, whose Slang lacks built-ins the shaders use (CDataLayout).
			# The host tool must be built against the same Shader-Slang release as
			# the target (pinned by vcpkg-overlays/ports/shader-slang).
			"-DVCPKG_OVERLAY_PORTS=${CMAKE_SOURCE_DIR}/vcpkg-overlays/ports")
	endif()

	find_program(_vkengine_host_cc NAMES clang gcc cc REQUIRED)
	find_program(_vkengine_host_cxx NAMES clang++ g++ c++ REQUIRED)

	include(ExternalProject)
	ExternalProject_Add(slang_spirv_compiler_host
		SOURCE_DIR "${SLANG_HELPER_SOURCE_DIR}"
		BINARY_DIR "${_host_build_dir}"
		CMAKE_GENERATOR "Ninja"
		CMAKE_ARGS
			-DCMAKE_BUILD_TYPE=Release
			-DCLANG_TIDY_ENABLED=OFF
			-DCMAKE_C_COMPILER=${_vkengine_host_cc}
			-DCMAKE_CXX_COMPILER=${_vkengine_host_cxx}
			${_host_vcpkg_args}
		BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target slang-spirv-compiler
		INSTALL_COMMAND ""
		BUILD_BYPRODUCTS "${_host_tool}"
		USES_TERMINAL_CONFIGURE TRUE
		USES_TERMINAL_BUILD TRUE
	)

	set(VKENGINE_SLANG_SPIRV_COMPILER "${_host_tool}")
	# No separate stamp dependency is needed: "${_host_tool}" is declared via
	# BUILD_BYPRODUCTS above, and add_slang_shaders already puts the COMPILER
	# path into each shader command's DEPENDS, which is enough for Ninja to
	# order the shader steps after this ExternalProject. (Depending on an
	# ExternalProject stamp file directly is not portable: it lives under
	# <name>-prefix/src/<name>-stamp/$<CONFIG>/ for multi-config generators.)
	set(VKENGINE_SLANG_COMPILER_DEPENDS "")
else()
	add_subdirectory("${SLANG_HELPER_SOURCE_DIR}" SlangSpriVCompilerHelper)
endif()

# ----------------------------
# MikkTSpace (tangent generation)
# ----------------------------
# Upstream ships no CMakeLists; build the two standalone C files as a library.
# Kept unaltered per the upstream zlib-style license; the engine wraps them
# behind VulkanEngine.Mesh.TangentGenerator.
if(EXISTS "${CMAKE_SOURCE_DIR}/external/MikkTSpace/mikktspace.c")
	# Project enables CXX only, so compile the C file as C++ — the extern "C"
	# block in mikktspace.h keeps the ABI identical.
	set_source_files_properties("${CMAKE_SOURCE_DIR}/external/MikkTSpace/mikktspace.c"
		PROPERTIES LANGUAGE CXX)
	add_library(MikkTSpace STATIC EXCLUDE_FROM_ALL
		"${CMAKE_SOURCE_DIR}/external/MikkTSpace/mikktspace.c"
	)
	target_include_directories(MikkTSpace PUBLIC "${CMAKE_SOURCE_DIR}/external/MikkTSpace")
else()
	message(FATAL_ERROR "Missing MikkTSpace at external/MikkTSpace; run: git submodule update --init external/MikkTSpace")
endif()

# ----------------------------
# secrets-gen (build-time secrets sealer)
# ----------------------------
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/Monocypher/src/monocypher.c")
    message(FATAL_ERROR "Missing Monocypher at external/Monocypher; run: git submodule update --init external/Monocypher")
endif()

# Runs on the build machine to seal each per-app secrets folder into a
# generated C++ module. Like slang-spirv-compiler it is a host tool: when
# cross-compiling it must be built for the host, never the target.
set(VKENGINE_SECRETS_GEN_COMMAND "")
set(VKENGINE_SECRETS_GEN_DEPENDS "")

if(CMAKE_CROSSCOMPILING)
    set(_secrets_gen_dir "${CMAKE_BINARY_DIR}/host-tools/secrets-gen")
    find_program(_vkengine_secrets_host_cc NAMES clang gcc cc REQUIRED)
    find_program(_vkengine_secrets_host_cxx NAMES clang++ g++ c++ REQUIRED)
    if(CMAKE_HOST_WIN32)
        set(_secrets_gen_exe "${_secrets_gen_dir}/bin/secrets-gen.exe")
    else()
        set(_secrets_gen_exe "${_secrets_gen_dir}/bin/secrets-gen")
    endif()

    include(ExternalProject)
    ExternalProject_Add(secrets_gen_host
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/tools/secrets-gen"
        BINARY_DIR "${_secrets_gen_dir}"
        CMAKE_GENERATOR "Ninja"
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${_vkengine_secrets_host_cc}
            -DCMAKE_CXX_COMPILER=${_vkengine_secrets_host_cxx}
        BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${_secrets_gen_exe}"
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
    )
    set(VKENGINE_SECRETS_GEN_COMMAND "${_secrets_gen_exe}")
    set(VKENGINE_SECRETS_GEN_DEPENDS "${_secrets_gen_exe}")
    unset(_secrets_gen_dir)
    unset(_secrets_gen_exe)
    unset(_vkengine_secrets_host_cc)
    unset(_vkengine_secrets_host_cxx)
else()
    add_subdirectory("${CMAKE_SOURCE_DIR}/tools/secrets-gen" secrets-gen)
    set(VKENGINE_SECRETS_GEN_COMMAND "$<TARGET_FILE:secrets-gen>")
    set(VKENGINE_SECRETS_GEN_DEPENDS secrets-gen)
endif()

include(EmbedSecrets)

option(CLANG_TIDY_ENABLED "Enable Clang-Tidy static analysis" ON)
# disable Clang-Tidy if not using Clang or GCC
if(CLANG_TIDY_ENABLED AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(CLANG_TIDY_ENABLED OFF)
endif()
if(CLANG_TIDY_ENABLED)
	# Enable Clang-Tidy if available
	include(ConfigureClangTidy)
endif()

# ----------------------------
# Windows runtime DLL deployment
# ----------------------------
# Directories scanned by add_app (cmake/CopyRuntimeDlls.cmake) to
# place the shared runtime and dependency DLLs next to each built executable:
#   * vcpkg dependencies (all dynamic on the MinGW triplet)
#   * the llvm-mingw runtimes (libc++.dll, libunwind.dll, libwinpthread-1.dll)
if(WIN32)
	# vcpkg keeps Debug and Release binaries in separate trees, so select the
	# matching one per configuration. The llvm-mingw runtimes are single-config.
	set(_vkengine_vcpkg_runtime_dir "$<$<CONFIG:Debug>:${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/bin>$<$<NOT:$<CONFIG:Debug>>:${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin>")
	set(VKENGINE_RUNTIME_DLL_DIRS "${_vkengine_vcpkg_runtime_dir}")
	if(DEFINED VKENGINE_LLVM_MINGW_ROOT AND VKENGINE_LLVM_MINGW_ROOT)
		list(APPEND VKENGINE_RUNTIME_DLL_DIRS
			"${VKENGINE_LLVM_MINGW_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32/bin")
	endif()
	unset(_vkengine_vcpkg_runtime_dir)
endif()

# Add subdirectories for the engine, the primary app, examples, the headless
# maze benchmark and tests
option(VKENGINE_BUILD_EXAMPLES "Build the example applications" OFF)

include(AddApp)
add_subdirectory(engine)
add_subdirectory(app)
if(VKENGINE_BUILD_EXAMPLES)
	add_subdirectory(examples)
else()
	message(STATUS "Examples are disabled (set VKENGINE_BUILD_EXAMPLES=ON to build them)")
endif()
add_subdirectory(bench)

if(BUILD_TESTING)
	add_subdirectory(tests)
endif()