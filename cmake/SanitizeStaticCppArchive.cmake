# Rebuilds a static archive with broken `.llvm_addrsig` sections removed.
#
# What this is for
# ----------------
# `--icf=safe` (lld) consults clang's address-significance tables
# (`.llvm_addrsig`, SHT_LLVM_ADDRSIG) to decide which identical sections may
# be merged. Each such section must link to its object's symbol table
# (sh_link != 0). Distributions that post-process their shipped .a files
# (objcopy/strip/ld -r) zero that field; lld then warns for every affected
# member and conservatively skips folding for it. Both the warning and the
# skip are correct behaviour — the only fix is to remove the broken tables.
#
# What this script does
# ---------------------
# Copies <SAN_IN> to <SAN_OUT>, removing `.llvm_addrsig` only from members
# whose table is broken (sh_link == 0). Members with a healthy table (or no
# table at all) are copied through untouched, so archive behaviour is exactly
# the same as before: lld treated sh_link==0 as "no usable table" either way.
#
# Invoked at build time as a custom command:
#   cmake -DSAN_IN=<archive> -DSAN_OUT=<path> \
#         -DSAN_AR=<ar> -DSAN_OBJCOPY=<objcopy> -DSAN_READELF=<readelf> \
#         -P SanitizeStaticCppArchive.cmake
#
# After writing SAN_OUT the script re-verifies it: no member may contain a
# .llvm_addrsig section with sh_link == 0.

if(NOT SAN_IN OR NOT SAN_OUT)
    message(FATAL_ERROR "SAN_IN and SAN_OUT are required")
endif()
cmake_policy(SET CMP0007 NEW)
foreach(_tool AR OBJCOPY READELF)
    if(NOT SAN_${_tool})
        message(FATAL_ERROR "SAN_${_tool} is required")
    endif()
endforeach()

get_filename_component(_san_in_abs "${SAN_IN}" ABSOLUTE)
if(NOT EXISTS "${_san_in_abs}")
    message(FATAL_ERROR "input archive not found: ${_san_in_abs}")
endif()

# Returns the sh_link of the first `.llvm_addrsig` row in a readelf -SW
# dump: the third-from-last token of the matching row ("Flg Lk Inf Al").
# Empty result when the member has no such section.
function(sanitize_read_sh_link _san_text _san_out)
    set(${_san_out} "" PARENT_SCOPE)
    string(REPLACE "\n" ";" _san_lines "${_san_text}")
    foreach(_san_line IN LISTS _san_lines)
        if(NOT _san_line MATCHES "\\.llvm_addrsig")
            continue()
        endif()
        string(REGEX MATCHALL "[^ \t]+" _san_toks "${_san_line}")
        list(LENGTH _san_toks _san_len)
        if(_san_len LESS 3)
            message(FATAL_ERROR
                "unexpected readelf section row: ${_san_line}")
        endif()
        math(EXPR _san_lk_idx "${_san_len} - 3")
        list(GET _san_toks ${_san_lk_idx} _san_sh_link)
        set(${_san_out} "${_san_sh_link}" PARENT_SCOPE)
        return()
    endforeach()
endfunction()

# Runs readelf -SW on a file and returns its first addrsig sh_link.
function(sanitize_member_sh_link _san_file _san_out)
    execute_process(
        COMMAND "${SAN_READELF}" -SW "${_san_file}"
        OUTPUT_VARIABLE _san_sections
        RESULT_VARIABLE _san_rc)
    if(NOT _san_rc EQUAL 0)
        message(FATAL_ERROR "readelf failed on ${_san_file}")
    endif()
    sanitize_read_sh_link("${_san_sections}" _san_sh_link)
    set(${_san_out} "${_san_sh_link}" PARENT_SCOPE)
endfunction()

set(_san_tmp "${SAN_OUT}.work")
file(REMOVE_RECURSE "${_san_tmp}")
file(MAKE_DIRECTORY "${_san_tmp}")

# ── list members (original order is preserved on rebuild) ──────────────
execute_process(
    COMMAND "${SAN_AR}" t "${_san_in_abs}"
    OUTPUT_VARIABLE _san_members_raw
    RESULT_VARIABLE _san_ar_list_result)
if(NOT _san_ar_list_result EQUAL 0)
    message(FATAL_ERROR "ar t failed on ${_san_in_abs}")
endif()
string(REPLACE "\n" ";" _san_members "${_san_members_raw}")
list(REMOVE_ITEM _san_members "")

# ── extract each member, repair the broken ones ────────────────────────
set(_san_broken 0)
set(_san_healthy 0)
set(_san_member_files "")
foreach(_san_member IN LISTS _san_members)
    set(_san_member_file "${_san_tmp}/${_san_member}")
    execute_process(
        COMMAND "${SAN_AR}" p "${_san_in_abs}" "${_san_member}"
        OUTPUT_FILE "${_san_member_file}"
        RESULT_VARIABLE _san_ar_p_result)
    if(NOT _san_ar_p_result EQUAL 0)
        message(FATAL_ERROR "ar p ${_san_member} failed on ${_san_in_abs}")
    endif()

    sanitize_member_sh_link("${_san_member_file}" _san_sh_link)

    if(NOT _san_sh_link STREQUAL "0")
        # No .llvm_addrsig at all, or a healthy one (sh_link != 0): pass the
        # member through as-is.
        math(EXPR _san_healthy "${_san_healthy} + 1")
    else()
        execute_process(
            COMMAND "${SAN_OBJCOPY}"
                --remove-section=.llvm_addrsig
                "${_san_member_file}" "${_san_member_file}"
            RESULT_VARIABLE _san_objcopy_result)
        if(NOT _san_objcopy_result EQUAL 0)
            message(FATAL_ERROR
                "objcopy --remove-section failed on ${_san_member}")
        endif()
        math(EXPR _san_broken "${_san_broken} + 1")
    endif()
    list(APPEND _san_member_files "${_san_member_file}")
endforeach()

# ── write the result archive ───────────────────────────────────────────
if(_san_broken EQUAL 0)
    # Nothing to fix: the copy is content-identical to the input. (The link
    # still goes through the copy so the remap wiring stays unconditional
    # and the outputs never depend on a system file's state at link time.)
    file(COPY_FILE "${_san_in_abs}" "${SAN_OUT}")
else()
    # `ar qc` keeps member order; `ar s` then rebuilds the symbol index.
    execute_process(
        COMMAND "${SAN_AR}" qc "${SAN_OUT}" ${_san_member_files}
        RESULT_VARIABLE _san_ar_qc_result)
    if(NOT _san_ar_qc_result EQUAL 0)
        message(FATAL_ERROR "ar qc failed writing ${SAN_OUT}")
    endif()
    execute_process(
        COMMAND "${SAN_AR}" s "${SAN_OUT}"
        RESULT_VARIABLE _san_ar_s_result)
    if(NOT _san_ar_s_result EQUAL 0)
        message(FATAL_ERROR "ar s failed on ${SAN_OUT}")
    endif()
endif()

# ── verify: no broken table may survive in the output ──────────────────
set(_san_left 0)
foreach(_san_member IN LISTS _san_members)
    execute_process(
        COMMAND "${SAN_AR}" p "${SAN_OUT}" "${_san_member}"
        OUTPUT_FILE "${_san_tmp}/${_san_member}.check"
        RESULT_VARIABLE _san_ar_p2_result)
    if(NOT _san_ar_p2_result EQUAL 0)
        message(FATAL_ERROR
            "verification: member ${_san_member} missing from ${SAN_OUT}")
    endif()
    sanitize_member_sh_link("${_san_tmp}/${_san_member}.check" _san_sh_link2)
    if(_san_sh_link2 STREQUAL "0")
        math(EXPR _san_left "${_san_left} + 1")
    endif()
endforeach()
file(REMOVE_RECURSE "${_san_tmp}")

message(STATUS
    "libcpp-sanitize: ${_san_in_abs} -> ${SAN_OUT} "
    "(members: ${_san_healthy} intact, ${_san_broken} broken addrsig tables removed)")
if(NOT _san_left EQUAL 0)
    message(FATAL_ERROR
        "libcpp-sanitize: ${_san_left} broken addrsig tables remain in ${SAN_OUT}")
endif()