include(AddApp)

add_subdirectory(minimal)
# The physics examples import the Jolt-backed facade, which is opt-in.
if(VKENGINE_PHYSICS)
	add_subdirectory(physics_vehicle)
endif()
add_subdirectory(basic_scene)
add_subdirectory(infinite_runner)
add_subdirectory(custom_pass)
add_subdirectory(text_demo)
if(VKENGINE_PHYSICS)
	add_subdirectory(physics_composed)
endif()
