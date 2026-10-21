add_executable(runtime_tests
  overrides_tests.cpp
  frame_step_clock_tests.cpp
  frame_profiler_tests.cpp
)
target_link_libraries(runtime_tests PRIVATE
  ${ENGINE_TARGET}
  ${BACKEND_TARGET}
  ${RUNTIME_TARGET}
  slang_shared_reflection
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_sources(runtime_tests PRIVATE
    FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      test_logging.cppm
  )
endif()

setup_test_target(runtime_tests)