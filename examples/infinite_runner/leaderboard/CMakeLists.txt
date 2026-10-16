# Leaderboard: wire protocol, placeholder TCP transport, session crypto, score
# store, and the async client. Engine-independent apart from VulkanSecurity, so
# the headless server does not pull in the renderer.
add_library(infinite_runner_leaderboard STATIC
    Config.cpp
    Account.cpp
    AccountStore.cpp
    Log.cpp
    Protocol.cpp
    Transport.cpp
    SessionCrypto.cpp
    Store.cpp
    Client.cpp
    Server.cpp
)
target_compile_features(infinite_runner_leaderboard PUBLIC cxx_std_23)

target_link_libraries(infinite_runner_leaderboard PUBLIC
    infinite_runner_core
    infinite_runner_secrets
    VulkanSecurity
)

target_include_directories(infinite_runner_leaderboard PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_compile_options(infinite_runner_leaderboard PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(infinite_runner_leaderboard PRIVATE
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:Release>:-O3>"
    )
endif()

if(WIN32)
    # Winsock, used by the placeholder transport.
    target_link_libraries(infinite_runner_leaderboard PUBLIC ws2_32)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_sources(infinite_runner_leaderboard PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            Config.cppm
            Account.cppm
            AccountStore.cppm
            Log.cppm
            Protocol.cppm
            Transport.cppm
            SessionCrypto.cppm
            Store.cppm
            Client.cppm
            Server.cppm
    )
endif()

if(TARGET project_sanitizers)
    target_link_libraries(infinite_runner_leaderboard PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(infinite_runner_leaderboard PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(infinite_runner_leaderboard PRIVATE project_size)
endif()

if(COMMAND enable_target_clang_tidy)
    enable_target_clang_tidy(infinite_runner_leaderboard)
endif()

# ── Headless server executable ──────────────────────────────────────────
# Plain console executable: it defines its own main and deliberately does not
# link the engine entry point, the renderer or any GPU dependency.
add_executable(leaderboard_server
    server_main.cpp
)
target_compile_features(leaderboard_server PRIVATE cxx_std_23)
target_link_libraries(leaderboard_server PRIVATE infinite_runner_leaderboard)

# Build-time check: BalanceConfig is fingerprinted at runtime (its Hash() is
# deliberately not constexpr), so the earliest gate is running the freshly built
# server against the committed list. A balance change that forgets to record the
# new fingerprint fails the build instead of shipping a server that rejects its
# own clients.
if(NOT CMAKE_CROSSCOMPILING)
    add_custom_command(TARGET leaderboard_server POST_BUILD
        COMMAND "$<TARGET_FILE:leaderboard_server>" --check-configs
                "${CMAKE_SOURCE_DIR}/examples/infinite_runner/leaderboard/accepted_configs.txt"
        COMMENT "Verifying the current ruleset is listed in accepted_configs.txt"
        VERBATIM
    )
endif()

target_compile_options(leaderboard_server PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(OUTPUT_PATH)
    set_target_properties(leaderboard_server PROPERTIES
        OUTPUT_NAME leaderboard_server
        RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_PATH}/server/$<CONFIG>"
    )
endif()

if(TARGET project_sanitizers)
    target_link_libraries(leaderboard_server PRIVATE project_sanitizers)
endif()
if(TARGET project_sections)
    target_link_libraries(leaderboard_server PRIVATE project_sections)
endif()
if(TARGET project_size)
    target_link_libraries(leaderboard_server PRIVATE project_size)
endif()
# Statically linked libc++/libc++abi for the slang-free configurations (see
# project_libcpp_static); keeps the shipped server's runtime linkage identical
# to the examples'.
if(TARGET project_libcpp_static)
    target_link_libraries(leaderboard_server PRIVATE project_libcpp_static)
endif()
# Build-order: sanitized libc++ archives exist before the link consumes them.
if(TARGET project_libcpp_sanitized)
    add_dependencies(leaderboard_server project_libcpp_sanitized)
endif()
