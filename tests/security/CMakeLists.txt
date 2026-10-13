add_executable(security_tests
  data_cipher_tests.cpp
  password_hash_tests.cpp
  key_exchange_tests.cpp
)
target_link_libraries(security_tests PRIVATE VulkanSecurity)
# cipher_format.hpp (a public VulkanSecurity header) includes monocypher.h,
# whose include dir is private to VulkanSecurity; the tests read the format
# header's layout constants, so they need the include path too.
target_include_directories(security_tests PRIVATE "${CMAKE_SOURCE_DIR}/external/Monocypher/src")
setup_test_target(security_tests)
