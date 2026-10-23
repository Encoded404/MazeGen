set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/../toolchains/clang-libcxx.cmake")

# Give every port the same section granularity the project's own targets get
# from the project_sections interface target. vcpkg ports default to a single
# monolithic .text/.rodata per object file, which means --gc-sections can only
# discard whole objects: one live symbol drags in everything its object
# defines. Splitting functions and data into their own sections lets the
# linker's --gc-sections drop the unreachable remainder.
#
# What this buys, measured on the Release app binary:
#   * the BasisU encoder inside libktx (basisu_frontend/comp/uastc_enc/...) is
#     pulled in only because lib/miniz_wrapper.cpp.o references
#     buminiz::mz_compress2, which lives in basisu_comp.cpp.o; with per-function
#     sections that reference disappears and the whole encoder chain is
#     collectable (~650 KB before gc-sections),
#   * the imgui demo/editor helpers and unused SDL3 subsystems likewise.
#
# NOTE: clang-libcxx.cmake (chainloaded below and by the consuming project)
# folds ${VCPKG_CXX_FLAGS} into CMAKE_CXX_FLAGS_INIT, so these flags also land
# on the project's own compile lines. That is harmless — project_sections adds
# the identical flags — and keeps port and project objects consistent.
set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} -ffunction-sections -fdata-sections")
set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} -ffunction-sections -fdata-sections")
