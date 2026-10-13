add_executable(examples_tests
  balance_tests.cpp
  account_tests.cpp
  profile_store_tests.cpp
  leaderboard_tests.cpp
  custom_pass_tests.cpp
  text_demo_tests.cpp
)
target_link_libraries(examples_tests PRIVATE
  custom_pass_core
  text_demo_core
  infinite_runner_core
  infinite_runner_secrets
  infinite_runner_leaderboard
  infinite_runner_account
  VulkanSecurity
)
setup_test_target(examples_tests)
