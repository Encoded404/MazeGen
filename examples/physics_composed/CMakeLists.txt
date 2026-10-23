# The composed-path example (rung 4 of the engine modularization design): the
# scene lives in engine modules registered by the game, per-frame work registers
# through the frame-phase seam, and physics comes entirely from the engine's
# default preset module. No custom shaders: the engine's built-in PBR set covers
# the fallback-quad crates.
add_app(physics_composed
    APP_TITLE        "MazeGenerator Composed Physics"
    MODULE_PREFIX    Examples.PhysicsComposed
    SHADER_NAMESPACE PhysicsComposed
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
)