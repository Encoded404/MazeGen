add_executable(physics_tests
  physics_tests.cpp
  physics_debug_overlay_tests.cpp
)
target_link_libraries(physics_tests PRIVATE ${ENGINE_TARGET})

# No `gpu` label on purpose: physics needs no Vulkan device, so the default test
# presets cover it.
setup_test_target(physics_tests)
