add_executable(shared_tests
  jobs_tests.cpp
  teardown_tests.cpp
  user_paths_tests.cpp
  storage_tests.cpp
)
target_link_libraries(shared_tests PRIVATE
  VulkanShared
)

setup_test_target(shared_tests)
