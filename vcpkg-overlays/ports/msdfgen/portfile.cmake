# Overlay port (see vcpkg-overlays/ports/msdfgen in MazeGenerator).
#
# Upstream vcpkg msdfgen port with ONE addition: the linkage check below. The
# rest is verbatim from upstream at 1.13+new-skia-api.
#
# Why: upstream does not pin the linkage, so on a dynamic triplet (x64-mingw-
# libcxx) msdfgen is built as a DLL. That DLL exports nothing -- its import
# library carries only the import descriptor and no msdfgen symbol at all,
# because msdfgen's CMake build never defines the dllexport half of
# MSDFGEN_PUBLIC that its installed config tells consumers to treat as
# dllimport. Every consumer then fails to link with undefined symbols for the
# whole msdfgen:: namespace, which is exactly what the Windows cross build does.
#
# Forcing a static library is the smallest correct fix and matches how this
# engine consumes msdfgen: it uses the C++ core API directly and wants the code
# linked in, not a DLL to deploy. Verified: the static archive defines the
# symbols (libmsdfgen-core.a) while the shared import library defined none.
#
# To retire this overlay: move the registry baseline past a msdfgen port that
# pins the linkage itself (or that fixes the exports), then delete this
# directory.

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Chlumsky/msdfgen
    REF "v${VERSION}"
    SHA512 5a136996de6ae013d223cd246548613d2928adcd6c3357333447086817351816c1b49e3eb119f3fe299745a50684d312db9410adf7671120b9e38fee8b96ff29
    HEAD_REF master
)

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        openmp MSDFGEN_USE_OPENMP
        geometry-preprocessing MSDFGEN_USE_SKIA
        tools MSDFGEN_BUILD_STANDALONE
    INVERTED_FEATURES
        extensions MSDFGEN_CORE_ONLY
)

if (VCPKG_CRT_LINKAGE STREQUAL dynamic)
    set(MSDFGEN_DYNAMIC_RUNTIME ON)
else()
    set(MSDFGEN_DYNAMIC_RUNTIME OFF)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DMSDFGEN_USE_VCPKG=ON
        -DMSDFGEN_VCPKG_FEATURES_SET=ON
        -DMSDFGEN_INSTALL=ON
        -DMSDFGEN_DYNAMIC_RUNTIME="${MSDFGEN_DYNAMIC_RUNTIME}"
        ${FEATURE_OPTIONS}
    MAYBE_UNUSED_VARIABLES
        MSDFGEN_VCPKG_FEATURES_SET
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/msdfgen)

# move exe to tools
if("tools" IN_LIST FEATURES)
    vcpkg_copy_tools(TOOL_NAMES msdfgen AUTO_CLEAN)
endif()

# cleanup
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
