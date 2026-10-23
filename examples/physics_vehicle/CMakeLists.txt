# A drivable four-wheel vehicle: the physics facade's `WheeledVehicle` driven
# through the engine's fixed-step frame contract, with a chase camera, meshed
# wheels, a static ground plane with ramps and crates, and the physics debug
# overlay drawn through ImGui.
add_app(physics_vehicle
    APP_TITLE        "MazeGenerator Physics Vehicle"
    MODULE_PREFIX    Examples.PhysicsVehicle
    SHADER_NAMESPACE PhysicsVehicle
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
    SHADERS
        CheckerFrag  checker.slang  fragment  main
)